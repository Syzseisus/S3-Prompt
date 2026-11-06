import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from torch.utils.data import DataLoader
from utils.dataset import S3PromptDataset, S3PromptCollatorWLabel
from utils.evaluation import extract_pred, eval_regression, eval_classification
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from utils.prompter import Prompter
from models.s3prompt import S3PromptModel
from utils.dataset_loader import load_dataset
from utils.scaffold_decompose import MoleculeDecomposer
from utils.constants import DATA_ROOT, REGRESSION, CLASSIFICATION
from utils.helper import log_df, load_json, save_json, parse_args, format_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Special Tokens
SCAF_TOKEN = "<|scaffold|>"
SIDE_TOKEN = "<|decoration|>"


# ==========================================
# 1. Helper & Visualization Functions
# ==========================================
def suggest_oom_fix(args, oom_count, total_len):
    """우선순위(GC -> Batch -> Quant -> Truncation)에 따라 해결책을 제안합니다."""
    suggestions = []

    # 1순위: Gradient Checkpointing (성능 손실 없음)
    if not args.gradient_checkpointing:
        suggestions.append("Enable Gradient Checkpointing: --gradient_checkpointing")

    # 2순위: Batch Size (성능 손실 없음)
    if args.batch_size > 1:
        suggest_bs = args.batch_size // 2
        suggestions.append(f"Reduce Batch Size to {suggest_bs}: --batch_size 1 (Current: {args.batch_size})")

    # 3순위: Quantization (메모리 대폭 절감)
    current_bit = args.quant_bit
    if current_bit is None:
        suggestions.append("Apply 4-bit Quantization: --quant_bit 4 (Current: FP16) ⚡")
        suggestions.append("   (Or try 8-bit first: --quant_bit 8)")
    elif current_bit == 8:
        suggestions.append("Switch to 4-bit Quantization: --quant_bit 4 (Current: 8-bit) ⚡")

    # 4순위: Truncation (데이터 손실 위험 - 최후의 수단)
    current_len = args.model_max_length
    if current_len == -1 or current_len > 4096:
        suggestions.append(f"Enable Truncation: --model_max_length 4096 (Current: {current_len}) ⚠️")
    elif current_len > 2048:
        suggestions.append(f"Stricter Truncation: --model_max_length 2048 ⚠️")

    print(f"\n\n{'='*60}")
    print(f"🛑 [CRITICAL ERROR] OOM Count Exceeded Limit!")
    print(f"   - Current OOM Count: {oom_count}/{total_len}")
    print(f"{'='*60}")

    if suggestions:
        print("💡 [Recommended Actions in Order]")
        for i, tip in enumerate(suggestions, 1):
            print(f"{i}. {tip}")
    else:
        print("💡 [No more easy fixes...]")
        print("   You are already using Batch=1, GC=On, 4-bit, and Truncation.")
        print("   Your data sequence might be extremely long.")

    print(f"{'='*60}\n")


def load_predictor(args, **kwargs):
    """
    Load Llama model with optional quantization support via kwargs.

    Args:
        args: ArgumentParser object containing model_name, etc.
        **kwargs: Additional arguments passed to from_pretrained (e.g., quantization_config, trust_remote_code).
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    print(f"Loading LLaMA model ({args.model_name}) on {device}...")

    # 1. Tokenizer Load
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # 2. Model Load
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
        **kwargs,
    )
    model.eval()

    # Llama-3 specific terminators
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    print("Model loaded successfully")
    return tokenizer, model, terminators


def setup_model_and_tokenizer(args):
    """
    Main 함수를 깔끔하게 유지하기 위해 모델 로드 및 설정 로직을 분리함.
    양자화, 길이 제한, Gradient Checkpointing 설정을 모두 처리.
    """
    # 1. Quantization Config 생성
    bnb_config = None
    if args.quant_bit == 4:
        print("⚡ [Info] Config: 4-bit Quantization (NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.quant_bit == 8:
        print("⚡ [Info] Config: 8-bit Quantization")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # 2. Load Model & Tokenizer
    tokenizer, llm_model, terminators = load_predictor(args, quantization_config=bnb_config)

    # 3. Apply Max Length (Truncation)
    if args.model_max_length > 0:
        tokenizer.model_max_length = args.model_max_length
        print(f"[Info] Enforcing Tokenizer Max Length: {args.model_max_length}")

    # 4. Apply Gradient Checkpointing
    if args.gradient_checkpointing:
        if hasattr(llm_model, "gradient_checkpointing_enable"):
            print("[Info] Enabling Gradient Checkpointing for VRAM efficiency...")
            llm_model.gradient_checkpointing_enable()
            if hasattr(llm_model, "enable_input_require_grads"):
                llm_model.enable_input_require_grads()
    else:
        print("[Info] Gradient Checkpointing is DISABLED.")

    # 5. Add Special Tokens
    special_tokens = {"additional_special_tokens": [SCAF_TOKEN, SIDE_TOKEN]}
    tokenizer.add_special_tokens(special_tokens)
    llm_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    return tokenizer, llm_model, terminators


def gen_method_tag(args):
    if args.method == "pool_clustering":
        if args.sample_strategy == "random":
            return "pc-rand"
        elif args.sample_strategy == "similarity":
            return "pc-sim"
    return args.method


def generate_random_shots(query_df, train_df, k):
    """Method='random'일 때 즉석에서 랜덤 샷 생성"""
    shot_map = {}
    train_ids = train_df["raw_ID"].tolist()

    for _, row in query_df.iterrows():
        rid = str(row["raw_ID"])
        selected_shots = np.random.choice(train_ids, k, replace=False).tolist()
        shot_map[rid] = {"shot_idx_list": [str(s) for s in selected_shots]}

    return shot_map


def visualize_training_sample(metadata_sample):
    print("\n" + "=" * 60)
    print("  🔍 [Training Visualization] Activity Cliff Mining Process")
    print("=" * 60)

    q_smiles = metadata_sample.get("smiles", "N/A")
    q_label = metadata_sample.get("label", "N/A")
    shots_data = metadata_sample.get("shots", [])
    mining_info = metadata_sample.get("mining")

    print(f"🔹 Query Molecule")
    print(f"   - SMILES: {q_smiles}...")
    print(f"   - Label : {q_label}")
    print(f"   - Total Candidates (Shots): {len(shots_data)}")
    print("-" * 60)

    if mining_info:
        p_idx = mining_info["p_idx"]
        p_sim = mining_info["p_sim"]
        n_idx = mining_info["n_idx"]
        n_sim = mining_info["n_sim"]

        if 0 <= p_idx < len(shots_data):
            pos_shot = shots_data[p_idx]
            print(f"✅ Selected Positive (Easiest Pos) [Idx={p_idx}]")
            print(f"   - SMILES: {pos_shot['smiles']}")
            print(f"   - Label: {pos_shot['label']}")
            print(f"   - Sim: {p_sim:.4f}")
        else:
            print(f"⚠️ Positive Index Error (Idx={p_idx})")

        print("-" * 30)

        if 0 <= n_idx < len(shots_data):
            neg_shot = shots_data[n_idx]
            print(f"❌ Selected Negative (Hardest Neg) [Idx={n_idx}]")
            print(f"   - SMILES: {neg_shot['smiles']}...")
            print(f"   - Label: {neg_shot['label']}")
            print(f"   - Sim Weight: {n_sim:.4f} (Used for Loss)")
        else:
            print(f"⚠️ Negative Index Error (Idx={n_idx})")

    else:
        print("⚠️ No Mining Info (Skipped or conditions not met)")
    print("=" * 60 + "\n")


def visualize_eval_sample(tokenizer, raw_id, true_lbl, score, gen_tokens_cpu, dataset_type="classification"):
    print("\n" + "=" * 60)
    print("  🧐 [Evaluation Visualization] Generation & Scoring")
    print("=" * 60)

    gen_text = tokenizer.decode(gen_tokens_cpu, skip_special_tokens=True)
    print(f"🔹 Sample ID: {raw_id}")
    print(f"   - Ground Truth: {true_lbl}")

    if dataset_type in CLASSIFICATION:
        print(f"   - Predicted Score (Yes): {score:.4f}")
        pred_class = 1 if score > 0.5 else 0
        result_icon = "🟢 Correct" if pred_class == true_lbl else "🔴 Wrong"
        print(f"   - Final Prediction: {pred_class} ({result_icon})")
    else:
        pred_val = score
        if pred_val is not None:
            abs_err = abs(pred_val - true_lbl)
            if abs_err < 0.5:
                err_icon = "🟢 Excellent (< 0.5)"
            elif abs_err < 1.0:
                err_icon = "🟡 Close (< 1.0)"
            elif abs_err < 2.0:
                err_icon = "🟠 Moderate (< 2.0)"
            else:
                err_icon = "🔴 Far (>= 2.0)"
            print(f"   - Predicted Value: {pred_val:.4f}")
            print(f"   - Absolute Error: {abs_err:.4f} {err_icon}")
        else:
            print(f"   - Predicted Value: PARSE FAILED")

    print("-" * 60)
    print(f"🔤 Full Generated Text:")
    print(f"   {gen_text.replace(chr(10), ' ')}")
    print("=" * 60 + "\n")


def plot_history(history, tag, save_dir):
    epochs = range(1, len(history["train_loss_epoch"]) + 1)

    # 1. Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss_epoch"], "b-", label="Train Loss")
    # Valid Loss는 None이 섞여있을 수 있음
    valid_loss_arr = np.array(history["valid_loss"], dtype=np.float64)
    mask = ~np.isnan(valid_loss_arr)
    valid_epochs = epochs[mask]
    valid_values = valid_loss_arr[mask]
    plt.plot(valid_epochs, valid_values, "r-", label="Valid Total Loss")
    plt.title(f"Training and Validation Loss\n{tag}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()

    # 2. Detailed Loss Plot (CE vs Triplet)
    if "train_loss_ce" in history and "train_loss_triplet" in history:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history["train_loss_ce"], "g-", label="CE Loss")
        plt.plot(epochs, history["train_loss_triplet"], "m-", label="Triplet Loss")
        plt.title(f"Detailed Training Components (CE vs Triplet)\n{tag}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "training_detailed_loss.png"))
        plt.close()

    # 3. Step-wise Loss
    if "train_loss_step" in history and len(history["train_loss_step"]) > 0:
        plt.figure(figsize=(12, 4))
        steps = range(1, len(history["train_loss_step"]) + 1)
        plt.plot(steps, history["train_loss_step"], "c-", alpha=0.3, label="Step Loss")
        if len(steps) > 20:
            w = 20
            avg = np.convolve(history["train_loss_step"], np.ones(w) / w, mode="valid")
            plt.plot(steps[w - 1 :], avg, "b-", label="Moving Avg")
        plt.title(f"Step-wise Training Loss\n{tag}")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "training_step_loss.png"))
        plt.close()


# ==========================================
# 2. Loss Functions
# ==========================================
def find_pattern_indices(input_ids, pattern_str="[ANSWER]", tokenizer=None, debug=False):
    """
    [Robust Search]
    ID 매칭 대신, 토큰들을 슬라이딩 윈도우로 묶어 디코딩하며
    pattern_str(예: "[ANSWER]")가 형성되는 지점(마지막 토큰 인덱스)을 찾습니다.
    """
    batch_indices = []
    input_ids_list = input_ids.cpu().tolist()
    max_window_size = 6

    for batch_idx, seq_list in enumerate(input_ids_list):
        seq_indices = []
        seq_len = len(seq_list)

        for i in range(seq_len):
            for w in range(1, max_window_size + 1):
                if i + w > seq_len:
                    break
                chunk = seq_list[i : i + w]
                decoded_chunk = tokenizer.decode(chunk)
                if pattern_str in decoded_chunk:
                    found_idx = i + w - 1
                    if not seq_indices or found_idx > seq_indices[-1]:
                        seq_indices.append(found_idx)
                    break
        batch_indices.append(seq_indices)
    return batch_indices


# 디버깅용: 배치 첫 번째 데이터의 토큰을 전수 조사
def inspect_ids(input_ids, tokenizer):
    print("\n🔍 Token Inspection (First Sequence):")
    seq = input_ids[0].cpu().tolist()
    for i, tid in enumerate(seq):
        # ID와 디코딩 결과를 같이 출력
        token_str = tokenizer.decode([tid])
        print(f"idx={i:04d} | ID={tid:<6} | Str='{token_str}'")

        # [ANSWER] 근처면 멈춰서 강조
        if "ANSWER" in token_str or "[" in token_str or "]" in token_str:
            print(f"      ^^^^ POTENTIAL TARGET PART ^^^^")


def compute_context_aware_triplet_loss(v_q, v_p, v_n, w_sim, margin=1.0, device="cuda"):
    """
    Decision-Time Vector로 Triplet Loss 계산
    """
    triplet_loss_fct = nn.TripletMarginLoss(margin=margin, p=2, reduction="none")
    # v_q, v_p, v_n: [dim] -> unsqueeze -> [1, dim]
    loss = triplet_loss_fct(v_q.unsqueeze(0), v_p.unsqueeze(0), v_n.unsqueeze(0))
    return loss * w_sim


def precompute_activity_cliff_info(
    shot_map, query_df, train_df, k_shot, embed="fp", query_df_fp=None, train_df_fp=None,
    dataset=None, label_std=None,
):
    """
    [Optimization]
    학습 루프 전에 모든 Query에 대해 p(Easy Pos)와 n(Hard Neg)의 인덱스 및 가중치를 미리 계산합니다.
    Embedding 타입에 따라 적절한 DataFrame(ECFP 포함)을 선택하여 캐싱합니다.
    """
    print(f"[Info] Pre-computing Activity Cliff indices for {len(shot_map)} queries...")

    # 1. Select Source DataFrames
    if embed == "fp":
        target_train_df = train_df
        target_query_df = query_df
    elif train_df_fp is not None and query_df_fp is not None:
        target_train_df = train_df_fp
        target_query_df = query_df_fp
    else:
        print(f"  [Error] ECFP source missing! embed='{embed}' but *_df_fp is None.")
        return shot_map

    # 2. Cache Construction
    # raw_ID는 str로 통일하여 매핑
    train_cache = dict(
        zip(target_train_df["raw_ID"].astype(str), zip(target_train_df["embed"], target_train_df["label"]))
    )
    query_cache = dict(
        zip(target_query_df["raw_ID"].astype(str), zip(target_query_df["embed"], target_query_df["label"]))
    )

    # 3. Mining Process
    processed_count = 0
    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Mining"):
        qid = str(row["raw_ID"])
        if qid not in shot_map:
            continue

        info = shot_map[qid]
        shot_ids = info["shot_idx_list"]  # List of raw_ids
        if len(shot_ids) < k_shot:
            info["mining"] = None
            continue

        # Query Info Lookup
        q_data = query_cache.get(str(qid))
        if not q_data:
            info["mining"] = None
            continue

        q_fp, q_label_val = q_data
        q_label = str(q_label_val)
        if q_fp is None:
            info["mining"] = None
            continue

        pos_candidates = []
        neg_candidates = []
        all_deltas = []
        is_regression = dataset is not None and dataset in REGRESSION

        for i, s_rid in enumerate(shot_ids):
            # Train Info Lookup
            s_data = train_cache.get(str(s_rid))
            if not s_data:
                continue

            s_fp, s_label_val = s_data
            s_label = str(s_label_val)
            if s_fp is None:
                continue

            # Tanimoto Sim
            sim = DataStructs.TanimotoSimilarity(q_fp, s_fp)

            if is_regression and label_std is not None:
                delta_y = abs(float(q_label_val) - float(s_label_val))
                all_deltas.append((i, sim, delta_y))
                if delta_y < 0.5 * label_std:
                    pos_candidates.append((i, sim))
                elif delta_y > 1.0 * label_std:
                    neg_candidates.append((i, sim))
            else:
                if s_label == q_label:
                    pos_candidates.append((i, sim))
                else:
                    neg_candidates.append((i, sim))

        # Select P & N
        best_p = max(pos_candidates, key=lambda x: x[1]) if pos_candidates else None
        best_n = max(neg_candidates, key=lambda x: x[1]) if neg_candidates else None

        # Regression fallback: use min/max delta pair if gap > 0.3σ
        if is_regression and label_std is not None and (not best_p or not best_n) and all_deltas:
            all_deltas.sort(key=lambda x: x[2])
            fb_pos = all_deltas[0]
            fb_neg = all_deltas[-1]
            if abs(fb_neg[2] - fb_pos[2]) > 0.3 * label_std:
                if not best_p:
                    best_p = (fb_pos[0], fb_pos[1])
                if not best_n:
                    best_n = (fb_neg[0], fb_neg[1])

        if best_p and best_n:
            info["mining"] = {"p_idx": best_p[0], "p_sim": best_p[1], "n_idx": best_n[0], "n_sim": best_n[1]}
            processed_count += 1
        else:
            info["mining"] = None

    print(f"[Info] Mining completed. {processed_count}/{len(shot_map)} queries have valid triplets.")
    return shot_map


# ==========================================
# 3. Train & Eval Loops
# ==========================================
def process_batch(args, model, batch, tokenizer, device, is_train=True):
    input_ids = batch["input_ids"].to(device)
    scaf_batch = batch["scaf_batch"].to(device)
    side_batch = batch["side_batch"].to(device)

    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
    labels = batch["labels"].to(device)

    scaf_mask = input_ids == tokenizer.convert_tokens_to_ids(SCAF_TOKEN)
    side_mask = input_ids == tokenizer.convert_tokens_to_ids(SIDE_TOKEN)

    outputs = model(
        input_ids,
        scaf_batch,
        side_batch,
        scaf_mask,
        side_mask,
        labels=labels,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    ce_loss = outputs.loss

    triplet_loss = torch.tensor(0.0, device=device)
    valid_count = 0

    if is_train and "metadata" in batch:
        hidden_states = outputs.hidden_states[-1]  # Last Layer: [Batch, Seq, Dim]

        # inspect_ids(input_ids, tokenizer)
        indices_list_1 = find_pattern_indices(input_ids, pattern_str="[Answer]", tokenizer=tokenizer, debug=False)
        indices_list_2 = find_pattern_indices(input_ids, pattern_str="[ANSWER]", tokenizer=tokenizer, debug=False)
        answer_indices_batch = []
        for l1, l2 in zip(indices_list_1, indices_list_2):
            combined = sorted(list(set(l1 + l2)))
            answer_indices_batch.append(combined)

        batch_meta = batch["metadata"]
        total_triplet_loss = torch.tensor(0.0, device=device)
        for i, meta in enumerate(batch_meta):
            mining_info = meta.get("mining")
            if not mining_info:
                continue

            indices = answer_indices_batch[i]
            k_shots = args.k

            # Safety Check: Expected indices count (Shots + Query)
            if len(indices) < k_shots + 1:
                print(f"\n[WARN] Token Count Mismatch! Expected {k_shots+1}, Found {len(indices)}")
                print(f"       Mining Info Exists? {bool(mining_info)}")
                continue

            # Extract Indices
            try:
                idx_p = indices[mining_info["p_idx"]]
                idx_n = indices[mining_info["n_idx"]]
                idx_q = indices[k_shots]
            except IndexError as e:
                print(f"\n[WARN] IndexError during Vector Extraction!")
                print(f"       Total [ANSWER] tokens found: {len(indices)}")
                print(f"       Attempted to access -> P_idx: {mining_info['p_idx']}, N_idx: {mining_info['n_idx']}, Q_idx: {k_shots}")  # fmt: skip
                print(f"       Error Detail: {e}")
                continue

            # Extract Vectors and normalize
            v_p = F.normalize(hidden_states[i, idx_p, :], p=2, dim=0)
            v_n = F.normalize(hidden_states[i, idx_n, :], p=2, dim=0)
            v_q = F.normalize(hidden_states[i, idx_q, :], p=2, dim=0)

            # Compute S2 Loss
            w_sim = mining_info["n_sim"]
            loss = compute_context_aware_triplet_loss(v_q, v_p, v_n, w_sim, device=device)

            total_triplet_loss += loss.mean()
            valid_count += 1

        if valid_count > 0:
            triplet_loss = total_triplet_loss / valid_count

    total_loss = ce_loss + args.lambda_loss * triplet_loss
    return ce_loss, triplet_loss, total_loss


def train_epoch(args, model, dataloader, tokenizer, optimizer, device, accumulation_steps=1):
    model.train()
    logs = {"ce": [], "triplet": [], "total": [], "step_total": []}
    optimizer.zero_grad()
    last_metadata_sample = None

    oom_count = 0
    oom_raw_ids = []
    total_processed = 0
    OOM_RATIO_THOLD = 0.10
    total_len = len(dataloader)
    max_allowed_oom = max(int(total_len * OOM_RATIO_THOLD), 5)

    print(f"\n[Info] OOM Tolerance Strategy:")
    print(f"       - Total Batches: {total_len}")
    print(f"       - Max Allowed OOMs: {max_allowed_oom} (Threshold: {OOM_RATIO_THOLD*100}%)")
    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        total_processed += 1
        try:
            # 1. Forward & loss
            ce, tri, tot = process_batch(args, model, batch, tokenizer, device, is_train=True)

            # 2. Backward with Gradient Accumulation
            loss_norm = tot / accumulation_steps
            loss_norm.backward()

            # 3. Optimizer step
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Logging
            logs["ce"].append(ce.item())
            logs["triplet"].append(tri.item())
            logs["total"].append(tot.item())
            logs["step_total"].append(tot.item())

            if "metadata" in batch and len(batch["metadata"]) > 0:
                last_metadata_sample = batch["metadata"][0]

            pbar.set_postfix(oom=f"{oom_count}/{max_allowed_oom}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_count += 1
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                pbar.set_postfix(oom=f"{oom_count}/{max_allowed_oom}")

                oom_raw_ids.extend(batch["raw_ids"])
                print(f" [OOM] Batch {i} skipped. (Count: {oom_count})")
                if oom_count > max_allowed_oom:
                    suggest_oom_fix(args, oom_count, total_len)
                    raise RuntimeError("Training Terminated due to excessive OOM.")
                continue
            else:
                raise e

    # 4. Handle remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    if last_metadata_sample:
        visualize_training_sample(last_metadata_sample)

    torch.cuda.empty_cache()

    def safe_mean(l):
        return np.mean(l) if len(l) > 0 else 0.0

    return {
        "ce": safe_mean(logs["ce"]),
        "triplet": safe_mean(logs["triplet"]),
        "total": safe_mean(logs["total"]),
        "step_history": logs["step_total"],
        "oom_count": oom_count,
        "oom_raw_ids": oom_raw_ids,
        "total_batches": total_len,
    }


def evaluate_s3prompt(args, model, dataloader, tokenizer, terminators, device, dataset_type="classification", label_stats=None):
    model.eval()
    results = {}

    yes_candidates = ["Yes", " Yes", "yes", " yes"]
    no_candidates = ["No", " No", "no", " no"]

    def get_ids(candidates):
        ids = []
        for cand in candidates:
            enc = tokenizer.encode(cand, add_special_tokens=False)
            if len(enc) >= 1:
                ids.append(enc[-1])
        return list(set(ids))

    yes_ids_list = get_ids(yes_candidates)
    no_ids_list = get_ids(no_candidates)
    val_losses = {"ce": [], "triplet": [], "total": []}

    last_vis_data = None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Loss
            ce, tri, tot = process_batch(args, model, batch, tokenizer, device, is_train=False)
            val_losses["ce"].append(ce.item())
            val_losses["triplet"].append(tri.item())
            val_losses["total"].append(tot.item())

            # Generation Setup
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", (input_ids != tokenizer.pad_token_id).long()).to(device)
            scaf_batch = batch["scaf_batch"].to(device)
            side_batch = batch["side_batch"].to(device)
            raw_ids = batch["raw_ids"]
            true_labels = np.array(batch["q_labels"])

            scaf_mask = input_ids == tokenizer.convert_tokens_to_ids(SCAF_TOKEN)
            side_mask = input_ids == tokenizer.convert_tokens_to_ids(SIDE_TOKEN)
            input_len = input_ids.shape[1]

            # Generation
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                scaffold_batch=scaf_batch,
                sidechain_batch=side_batch,
                scaf_mask=scaf_mask,
                side_mask=side_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
            )

            new_tokens = gen_out
            if new_tokens.shape[1] == 0:
                default_true = 0 if dataset_type in CLASSIFICATION else 0.0
                for rid in raw_ids:
                    results[rid] = {"true": default_true, "pred": None, "text": "", "score": 0.0, "entropy": None}
                continue

            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            if dataset_type in CLASSIFICATION:
                # Classification: Surface Form Competition scoring
                first_gen_token = new_tokens[:, 0:1]
                check_input = torch.cat([input_ids, first_gen_token], dim=1)
                full_scaf_mask = F.pad(scaf_mask, (0, 1), value=False)
                full_side_mask = F.pad(side_mask, (0, 1), value=False)

                outputs = model(check_input, scaf_batch, side_batch, full_scaf_mask, full_side_mask)
                logits = outputs.logits
                target_logits = logits[:, input_len - 1, :]

                score_yes = torch.logsumexp(target_logits[:, yes_ids_list], dim=1)
                score_no = torch.logsumexp(target_logits[:, no_ids_list], dim=1)
                probs = torch.softmax(torch.stack([score_no, score_yes], dim=1), dim=1)
                prob_yes = probs[:, 1].float().cpu().numpy()

                for i, rid in enumerate(raw_ids):
                    pred_text = texts[i]
                    pred = extract_pred(pred_text, args.dataset)
                    if pred is None:
                        pred = 1 if prob_yes[i] > 0.5 else 0

                    results[rid] = {
                        "true": int(true_labels[i]),
                        "pred": pred,
                        "text": pred_text,
                        "score": float(prob_yes[i]),
                        "entropy": None,
                    }

                last_vis_data = {
                    "raw_id": raw_ids[0],
                    "true_label": int(true_labels[0]),
                    "score": float(prob_yes[0]),
                    "gen_tokens": gen_out[0].cpu().tolist(),
                }
            else:
                # Regression: extract float from text, de-standardize
                for i, rid in enumerate(raw_ids):
                    pred_text = texts[i]
                    pred = extract_pred(pred_text, args.dataset)

                    if pred is not None and label_stats is not None:
                        pred = pred * label_stats["std"] + label_stats["mean"]

                    results[rid] = {
                        "true": float(true_labels[i]),
                        "pred": pred,
                        "text": pred_text,
                        "score": float(pred) if pred is not None else None,
                        "entropy": None,
                    }

                first_pred = results[raw_ids[0]].get("pred")
                last_vis_data = {
                    "raw_id": raw_ids[0],
                    "true_label": float(true_labels[0]),
                    "score": float(first_pred) if first_pred is not None else 0.0,
                    "gen_tokens": gen_out[0].cpu().tolist(),
                }

    if last_vis_data:
        visualize_eval_sample(
            tokenizer,
            last_vis_data["raw_id"],
            last_vis_data["true_label"],
            last_vis_data["score"],
            last_vis_data["gen_tokens"],
            dataset_type=dataset_type,
        )

    torch.cuda.empty_cache()

    fail_set = {k for k, v in results.items() if v["pred"] is None}

    valid_results = [res for row, res in results.items() if row not in fail_set]

    result_log = {}
    result_log["true_list"] = [result["true"] for result in valid_results]
    result_log["pred_list"] = [result["pred"] for result in valid_results]

    if valid_results and "score" in valid_results[0]:
        result_log["score_list"] = [result["score"] for result in valid_results]

    result_log["results"] = results
    result_log["total"] = len(results)
    result_log["fail_set"] = list(fail_set)
    result_log["fail_count"] = len(fail_set)

    # Entropy 통계
    entropy_list = [res["entropy"] for res in valid_results if res.get("entropy") is not None]
    if entropy_list:
        result_log["entropy_mean"] = np.mean(entropy_list)
        result_log["entropy_std"] = np.std(entropy_list)

    if dataset_type in CLASSIFICATION:
        eval_metrics = eval_classification(result_log)
    else:
        eval_metrics = eval_regression(result_log)

    loss_metrics = {
        "loss_ce": np.mean(val_losses["ce"]),
        "loss_triplet": np.mean(val_losses["triplet"]),
        "loss_total": np.mean(val_losses["total"]),
    }

    return {**loss_metrics, **eval_metrics}


# ==========================================
# 4. Main Execution
# ==========================================
def load_shot_wrt_method(args, train_df, valid_df, test_df):
    embed_tag = f"molt5-{args.embedding_type}" if args.embedding_type != "fp" else "fp"
    if args.method == "random":
        print(f"[Info] Generating random shots on-the-fly (k={args.k})...")
        valid_shots = generate_random_shots(valid_df, train_df, args.k)
        test_shots = generate_random_shots(test_df, train_df, args.k)

    elif args.method == "pool_clustering":
        eps_tag = f"eps={args.eps}" if args.eps else "no_merge"
        cluster_tag_dir = f"{embed_tag}_{eps_tag}_{args.sample_strategy}-sampling"
        cache_dir = os.path.join(args.cache_root, args.dataset, cluster_tag_dir)

        valid_shot_path = os.path.join(cache_dir, f"query_to_shot_valid_k={args.k}.json")
        test_shot_path = os.path.join(cache_dir, f"query_to_shot_test_k={args.k}.json")

        if os.path.exists(valid_shot_path) and os.path.exists(test_shot_path):
            print(f"[Info] Loading Pool Clustering shots from {cache_dir}")
            valid_shots = load_json(valid_shot_path)
            test_shots = load_json(test_shot_path)
        else:
            raise FileNotFoundError(f"[Error] Pool Clustering files not found at {cache_dir}.")

    elif args.method == "topk":
        tag = f"topk_{embed_tag}"
        cache_dir = os.path.join(args.cache_root, args.dataset, tag)

        def load_topk_file(prefix):
            target_file = None
            if os.path.exists(cache_dir):
                for fname in os.listdir(cache_dir):
                    if fname.startswith(f"query_to_shot_{prefix}_topk="):
                        target_file = os.path.join(cache_dir, fname)
                        break
            if not target_file:
                raise FileNotFoundError(f"TopK file for {prefix} not found in {cache_dir}.")

            print(f"[Info] Loaded TopK file: {os.path.basename(target_file)}")
            data = load_json(target_file)
            for qid, info in data.items():
                if len(info["shot_idx_list"]) < args.k:
                    pass
                info["shot_idx_list"] = info["shot_idx_list"][: args.k]
            return data

        print(f"[Info] Loading Top-K shots from {cache_dir}")
        valid_shots = load_topk_file("valid")
        test_shots = load_topk_file("test")

    return valid_shots, test_shots


def load_shot(args, train_df, valid_df, test_df, label_stats=None):
    # [Mining Cache Path Definition]
    mining_cache_dir = os.path.join(args.cache_root, args.dataset, "mining")
    os.makedirs(mining_cache_dir, exist_ok=True)
    mining_fname = f"mining_info_{args.embedding_type}_{args.method}_k={args.k}.json"
    valid_cache_path = os.path.join(mining_cache_dir, f"valid_{mining_fname}")
    test_cache_path = os.path.join(mining_cache_dir, f"test_{mining_fname}")

    print("\n" + "=" * 40)
    print(" >>> Activity Cliff Mining Check")
    print("=" * 40)

    # 1. Check if Cache Exists
    valid_cached = os.path.exists(valid_cache_path)
    test_cached = os.path.exists(test_cache_path)

    if valid_cached and test_cached:
        print(f"[Cache Hit] Loading pre-computed mining info...")
        print(f"  - Valid: {valid_cache_path}")
        print(f"  - Test : {test_cache_path}")
        valid_shots = load_json(valid_cache_path)
        test_shots = load_json(test_cache_path)

    else:
        print("[Cache Miss] Computing Activity Cliff indices...")
        valid_shots, test_shots = load_shot_wrt_method(args, train_df, valid_df, test_df)

        # 2. Load FP Dataset (Only required for computation)
        if args.embedding_type == "fp":
            train_df_fp, valid_df_fp, test_df_fp = train_df, valid_df, test_df
        else:
            print("[Info] Loading FP dataset for mining calculations...")
            train_df_fp, valid_df_fp, test_df_fp = load_dataset(args.dataset, "fp", verbose=False)

        # 3. Compute & Save (Valid)
        if valid_cached:
            print(f"  - Valid cache exists, loading...")
            valid_shots = load_json(valid_cache_path)
        else:
            valid_shots = precompute_activity_cliff_info(
                valid_shots,
                valid_df,
                train_df,
                args.k,
                embed=args.embedding_type,
                query_df_fp=valid_df_fp,
                train_df_fp=train_df_fp,
                dataset=args.dataset,
                label_std=label_stats["std"] if label_stats else None,
            )
            save_json(valid_shots, valid_cache_path)
            print(f"  - Saved valid mining info to {valid_cache_path}")

        # 4. Compute & Save (Test)
        if test_cached:
            print(f"  - Test cache exists, loading...")
            test_shots = load_json(test_cache_path)
        else:
            test_shots = precompute_activity_cliff_info(
                test_shots,
                test_df,
                train_df,
                args.k,
                embed=args.embedding_type,
                query_df_fp=test_df_fp,
                train_df_fp=train_df_fp,
                dataset=args.dataset,
                label_std=label_stats["std"] if label_stats else None,
            )
            save_json(test_shots, test_cache_path)
            print(f"  - Saved test mining info to {test_cache_path}")

    return valid_shots, test_shots


def main():
    try:
        args, categories = parse_args()

        if args.method == "zero_shot":
            raise ValueError("Zero-shot unsupported.")

        accumulation_steps = args.grad_accum_steps
        device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
        print(f"Using device: {device}")
        print(f"Physical Batch Size: {args.batch_size}")
        print(f"Accumulation Steps: {accumulation_steps}")
        print(f"Effective Batch Size: {args.batch_size * accumulation_steps}")

        # PATH
        sub_path_config = f"temp={args.temperature}/eps={args.eps}"
        base_ckpt_path = os.path.join(args.ckpt_root, args.dataset, sub_path_config)
        base_save_path = os.path.join(args.save_root, args.dataset, sub_path_config)

        # Resume
        start_epoch = getattr(args, "start_epoch", 1)
        if start_epoch > 1:
            # Resume: Find latest timestamp directory in the config path
            if not os.path.exists(base_save_path):
                raise FileNotFoundError(f"Cannot resume: Base path {base_save_path} does not exist.")

            # Find all timestamp directories then use the lastest run
            timestamps = sorted([d for d in os.listdir(base_save_path) if os.path.isdir(os.path.join(base_save_path, d))])  # fmt: skip
            if not timestamps:
                raise FileNotFoundError(f"Cannot resume: No experiment directories found in {base_save_path}")
            args.now = timestamps[-1]
            print(f"[Info] Auto-detected latest experiment for resume: {args.now}")
        else:
            args.now = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Final Paths
        args.ckpt_dir = os.path.join(base_ckpt_path, args.now)
        args.save_dir = os.path.join(base_save_path, args.now)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)

        embed_tag = f"molt5-{args.embedding_type}" if args.embedding_type != "fp" else "fp"
        method_tag = gen_method_tag(args)
        args.exp_tag = f"{args.dataset} - {embed_tag}_{method_tag}_k={args.k}"

        categories.setdefault("Experiment Settings", []).extend(["now", "exp_tag", "ckpt_dir", "save_dir"])
        print(format_args(args, categories))
        save_json(vars(args), os.path.join(args.save_dir, "args.json"))

        # Dataset
        print("Loading Dataset...")
        train_df, valid_df, test_df = load_dataset(args.dataset, args.embedding_type, verbose=True)

        # Detect task type and compute standardization stats for regression
        is_cls = args.dataset in CLASSIFICATION
        label_stats = None
        if not is_cls:
            _lbl_mean = float(train_df["label"].mean())
            _lbl_std = float(train_df["label"].std())
            label_stats = {"mean": _lbl_mean, "std": _lbl_std}
            print(f"[Info] Regression Label Stats: mean={_lbl_mean:.4f}, std={_lbl_std:.4f}")

        valid_shots, test_shots = load_shot(args, train_df, valid_df, test_df, label_stats=label_stats)

        # Model & Optimizer
        args.mode = "llama"
        tokenizer, llm_model, terminators = setup_model_and_tokenizer(args)

        model = S3PromptModel(llm_model, tokenizer, terminators, gnn_hidden=args.gnn_hidden).to(device)

        optimizer = torch.optim.AdamW(
            [
                {"params": model.gnn.parameters()},
                {"params": model.scaf_proj.parameters()},
                {"params": model.side_proj.parameters()},
            ],
            lr=args.lr,
        )

        # fmt: off
        history = {
            "train_loss_epoch": [], "train_loss_step": [], "train_loss_ce": [], "train_loss_triplet": [],
            "valid_loss": [], "valid_acc": [], "valid_auc": [], "valid_f1": [], "valid_mcc": [], "valid_mse": [], "valid_rmse": [],
            "oom_count": [], "oom_raw_ids": [],
        }
        # fmt: on

        if start_epoch > 1:
            resume_epoch = start_epoch - 1
            ckpt_file = os.path.join(args.ckpt_dir, f"epoch_{resume_epoch:02}.pt")

            if os.path.exists(ckpt_file):
                print(f"[Info] Loading checkpoint from {ckpt_file}...")
                checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
                model.gnn.load_state_dict(checkpoint["gnn_state_dict"])
                model.scaf_proj.load_state_dict(checkpoint["scaf_proj_state_dict"])
                model.side_proj.load_state_dict(checkpoint["side_proj_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # Load History if available
                if "history" in checkpoint:
                    history = checkpoint["history"]
            else:
                raise FileNotFoundError(f"Checkpoint for epoch {resume_epoch} not found at {ckpt_file}")

        # Dataset & Loader
        with open(args.inst_json, "r") as f:
            full_inst = json.load(f)
            instruction = full_inst["full"][args.dataset] if args.use_inst else full_inst["short"][args.dataset]

        prompter = Prompter(args.dataset, instruction)
        decomposer = MoleculeDecomposer()
        collator = S3PromptCollatorWLabel(decomposer, tokenizer, is_cls=is_cls)

        train_ds = S3PromptDataset(valid_df, train_df, valid_shots, tokenizer, prompter, decomposer, args.k, is_train=True, is_cls=is_cls, label_stats=label_stats)  # fmt: skip
        test_ds = S3PromptDataset(test_df, train_df, test_shots, tokenizer, prompter, decomposer, args.k, is_train=False, is_cls=is_cls, label_stats=label_stats)  # fmt: skip

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=4)  # fmt: skip
        eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)  # fmt: skip
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)  # fmt: skip

        print(f">>> Start Training S3-Prompt from Epoch {start_epoch}...")
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\n[Epoch {epoch}/{args.epochs}]")

            # Train
            train_metrics = train_epoch(args, model, train_loader, tokenizer, optimizer, device, accumulation_steps)
            print(f"  Train Loss | Total: {train_metrics['total']:.4f}, CE: {train_metrics['ce']:.4f}, Tri: {train_metrics['triplet']:.4f}")  # fmt: skip

            oom_cnt = train_metrics["oom_count"]
            oom_ids = train_metrics["oom_raw_ids"]
            total_b = train_metrics["total_batches"]
            oom_rate = (oom_cnt / total_b) * 100 if total_b > 0 else 0
            print(f"  [OOM Stats] {oom_cnt}/{total_b} batches skipped ({oom_rate:.2f}%)")
            if oom_cnt > 0:
                print(f"  [OOM Samples] First 3 IDs: {oom_ids[:3]}...")

            history["train_loss_epoch"].append(train_metrics["total"])
            history["train_loss_step"].extend(train_metrics["step_history"])
            history["train_loss_ce"].append(train_metrics["ce"])
            history["train_loss_triplet"].append(train_metrics["triplet"])
            history["oom_count"].append(oom_cnt)
            history["oom_raw_ids"].append(oom_ids)

            # Eval
            # TODO: train / valid 구분하기
            if False:
            # if epoch % args.eval_epoch == 0 or epoch == args.epochs:
                print(f"  >>> Running Evaluation at Epoch {epoch}...")
                val_metrics = evaluate_s3prompt(args, model, eval_loader, tokenizer, terminators, device, args.dataset, label_stats=label_stats)
                print(f"  Valid Loss | Total: {val_metrics['loss_total']:.4f}")

                history["valid_loss"].append(val_metrics["loss_total"])
                if args.dataset in CLASSIFICATION:
                    print(f"  Valid Metric | ACC: {val_metrics['acc']:.4f}, AUC: {val_metrics['auc']:.4f}")
                    history["valid_acc"].append(val_metrics["acc"])
                    history["valid_auc"].append(val_metrics["auc"])
                    history["valid_f1"].append(val_metrics.get("f1", 0))
                    history["valid_mcc"].append(val_metrics.get("mcc", 0))
                else:
                    history["valid_mse"].append(val_metrics.get("mse", 0))
                    history["valid_rmse"].append(val_metrics.get("rmse", 0))

                save_json(val_metrics, os.path.join(args.save_dir, f"result_valid_epoch={epoch:02}.json"))

            else:
                print(f"  [Info] Skipping evaluation (next eval at epoch {epoch + (args.eval_epoch - epoch % args.eval_epoch)})")  # fmt: skip
                history["valid_loss"].append(None)
                if args.dataset in CLASSIFICATION:
                    history["valid_acc"].append(None)
                    history["valid_auc"].append(None)
                    history["valid_f1"].append(None)
                    history["valid_mcc"].append(None)
                else:
                    history["valid_mse"].append(None)
                    history["valid_rmse"].append(None)

            ckpt_payload = {
                "epoch": epoch,
                "gnn_state_dict": model.gnn.state_dict(),
                "scaf_proj_state_dict": model.scaf_proj.state_dict(),
                "side_proj_state_dict": model.side_proj.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }
            torch.save(ckpt_payload, os.path.join(args.ckpt_dir, f"epoch_{epoch:02}.pt"))

            # JSON Saving
            result_json_path = os.path.join(args.save_dir, "result.json")
            save_json(history, result_json_path)
            try:
                plot_history(history, args.exp_tag, args.save_dir)
            except Exception as e:
                # 혹시 plot 함수가 None을 처리 못할 경우를 대비해 로그만 남기고 학습 계속 진행
                print(f"[Warning] Plotting failed: {e}")

        print("\n>>> Final Testing...")
        test_metrics = evaluate_s3prompt(args, model, test_loader, tokenizer, terminators, device, args.dataset, label_stats=label_stats)
        history["test_results"] = test_metrics
        save_json(history, result_json_path)
        print("Done.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
