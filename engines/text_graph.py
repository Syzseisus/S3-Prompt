import os
import sys
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader

# [FIX] utils/dataset.py에서 필요한 클래스 Import
from utils.dataset import S3PromptDataset, S3PromptCollatorWLabel

# Local Imports
from utils.evaluation import extract_pred, eval_regression, eval_classification
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

# PEFT Imports (LoRA)
from peft import TaskType, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from utils.prompter import Prompter
from models.s3prompt import S3PromptModel
from utils.dataset_loader import load_dataset
from utils.scaffold_decompose import MoleculeDecomposer
from utils.constants import DATA_ROOT, REGRESSION, CLASSIFICATION
from utils.helper import log_df, load_json, save_json, parse_args, format_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# [Log Filter] Right Padding Warning 제거
class RightPaddingFilter(logging.Filter):
    def filter(self, record):
        return "right-padding was detected" not in record.getMessage()


logging.getLogger("transformers.generation.utils").addFilter(RightPaddingFilter())

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
    + PEFT(LoRA) 적용 - args.lora_r, args.lora_alpha로 조절 가능
    """
    # 1. Quantization Config
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

    # 2. Load Base Model
    tokenizer, model, terminators = load_predictor(args, quantization_config=bnb_config)

    # 3. Apply PEFT (LoRA)
    print("🛠️ [Info] Applying PEFT (LoRA)...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=12,
        lora_alpha=24,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Attention Q/V
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Settings
    if args.model_max_length > 0:
        tokenizer.model_max_length = args.model_max_length
        print(f"[Info] Enforcing Tokenizer Max Length: {args.model_max_length}")

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            print("[Info] Enabling Gradient Checkpointing...")
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    # 5. Add Special Tokens
    special_tokens = {"additional_special_tokens": [SCAF_TOKEN, SIDE_TOKEN]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    return tokenizer, model, terminators


def gen_method_tag(args):
    if args.method == "pool_clustering":
        if args.sample_strategy == "random":
            return "pc-rand"
        elif args.sample_strategy == "similarity":
            return "pc-sim"
    return args.method


def generate_random_shots(query_df, train_df, k):
    shot_map = {}
    train_ids = train_df["raw_ID"].tolist()
    for _, row in query_df.iterrows():
        rid = str(row["raw_ID"])
        selected_shots = np.random.choice(train_ids, k, replace=False).tolist()
        shot_map[rid] = {"shot_idx_list": [str(s) for s in selected_shots]}
    return shot_map


def plot_history(history, tag, save_dir):
    epochs = range(1, len(history["train_loss_epoch"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss_epoch"], "b-", label="Train Loss")
    valid_loss_arr = np.array(history["valid_loss"], dtype=np.float64)
    mask = ~np.isnan(valid_loss_arr)
    valid_epochs = np.array(epochs)[mask]
    valid_values = valid_loss_arr[mask]
    plt.plot(valid_epochs, valid_values, "r-", label="Valid Total Loss")
    plt.title(f"Training Loss\n{tag}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
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


def compute_context_aware_triplet_loss(v_q, v_p, v_n, w_sim, margin=1.0, device="cuda"):
    """
    Decision-Time Vector로 Triplet Loss 계산
    """
    triplet_loss_fct = nn.TripletMarginLoss(margin=margin, p=2, reduction="none")
    # v_q, v_p, v_n: [dim] -> unsqueeze -> [1, dim]
    loss = triplet_loss_fct(v_q.unsqueeze(0), v_p.unsqueeze(0), v_n.unsqueeze(0))
    return loss * w_sim


def precompute_activity_cliff_info(
    shot_map, query_df, train_df, k_shot, embed="fp", query_df_fp=None, train_df_fp=None
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

            if s_label == q_label:
                pos_candidates.append((i, sim))
            else:
                neg_candidates.append((i, sim))

        # Select P & N
        best_p = max(pos_candidates, key=lambda x: x[1]) if pos_candidates else None
        best_n = max(neg_candidates, key=lambda x: x[1]) if neg_candidates else None

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
def process_batch(args, model, batch, tokenizer, device, need_triplet=False, return_outputs=False):
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
        output_hidden_states=need_triplet,
    )
    ce_loss = outputs.loss

    triplet_loss = torch.tensor(0.0, device=device)
    valid_count = 0

    if need_triplet and "metadata" in batch:
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

    # 3. Return Logic (return_outputs=True면 outputs도 반환)
    if return_outputs:
        return ce_loss, triplet_loss, total_loss, outputs

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
            ce, tri, tot = process_batch(args, model, batch, tokenizer, device, need_triplet=True, return_outputs=False)

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


# TODO: 다른 곳에도 valid와 test를 동시에 고려한 함수로 변경할 것
def evaluate_s3prompt(
    args, model, dataloader, tokenizer, terminators, device, dataset_type="classification", is_test=False
):
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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 1. Forward -> Loss + Logits
            need_triplet = not is_test
            ce, tri, tot, outputs = process_batch(
                args, model, batch, tokenizer, device, need_triplet=need_triplet, return_outputs=True
            )
            # 1-1. Loss
            val_losses["ce"].append(ce.item())
            val_losses["triplet"].append(tri.item())
            val_losses["total"].append(tot.item())

            # 1-2. Scoring
            # Valid: 입력이 [... Answer, Yes] -> 'Answer'(-2)가 'Yes'를 예측했는지 확인, !! input_len - 1과 다름 !!
            # Test : 입력이 [... Answer]      -> 'Answer'(-1)가 무엇을 예측하는지 확인, input_len - 1과 동일
            target_idx = -1 if is_test else -2

            # outputs = model(check_input, scaf_batch, side_batch, full_scaf_mask, full_side_mask)  # 이미 얻음
            logits = outputs.logits
            target_logits = logits[:, target_idx, :]

            score_yes = torch.logsumexp(target_logits[:, yes_ids_list], dim=1)
            score_no = torch.logsumexp(target_logits[:, no_ids_list], dim=1)
            probs = torch.softmax(torch.stack([score_no, score_yes], dim=1), dim=1)
            prob_yes = probs[:, 1].float().cpu().numpy()

            # 2. Generation
            input_ids = batch["input_ids"].to(device)
            # [Data Leakage 방지] Valid Set은 정답 토큰을 자르고 생성 시작
            if not is_test:
                curr_input_ids = input_ids[:, :-1]
                if "attention_mask" in batch:
                    curr_att_mask = batch["attention_mask"].to(device)[:, :-1]
                else:
                    curr_att_mask = (curr_input_ids != tokenizer.pad_token_id).long()
            else:
                curr_input_ids = input_ids
                if "attention_mask" in batch:
                    curr_att_mask = batch["attention_mask"].to(device)
                else:
                    curr_att_mask = (curr_input_ids != tokenizer.pad_token_id).long()

            scaf_batch = batch["scaf_batch"].to(device)
            side_batch = batch["side_batch"].to(device)
            raw_ids = batch["raw_ids"]
            true_labels = np.array(batch["q_labels"])

            scaf_mask = input_ids == tokenizer.convert_tokens_to_ids(SCAF_TOKEN)
            side_mask = input_ids == tokenizer.convert_tokens_to_ids(SIDE_TOKEN)
            input_len = input_ids.shape[1]

            # Generation
            gen_out = model.generate(
                input_ids=curr_input_ids,
                attention_mask=curr_att_mask,
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
                for rid in raw_ids:
                    results[rid] = {"true": 0, "pred": None, "text": "", "score": 0.0, "entropy": None}
                continue

            # Scoring
            first_gen_token = new_tokens[:, 0:1]
            full_scaf_mask = F.pad(scaf_mask, (0, 1), value=False)
            full_side_mask = F.pad(side_mask, (0, 1), value=False)

            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for i, rid in enumerate(raw_ids):
                pred_text = texts[i]
                pred = extract_pred(pred_text, args.dataset)
                if pred is None and dataset_type in CLASSIFICATION:
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
                "true_label": true_labels[0],
                "score": prob_yes[0],
                "gen_tokens": gen_out[0].cpu().tolist(),
            }

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


def load_shot(args, train_df, valid_df, test_df):
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
        sub_path_config = f"temp={args.temperature}/eps={args.eps}/graph_peft"
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
        valid_shots, test_shots = load_shot(args, train_df, valid_df, test_df)

        # Model & Optimizer
        args.mode = "llama"
        tokenizer, llm_model, terminators = setup_model_and_tokenizer(args)

        # Wrap LoRA model with S3PromptModel
        model = S3PromptModel(llm_model, tokenizer, terminators, gnn_hidden=args.gnn_hidden).to(device)

        # Optimizer: GNN + Projectors + LoRA Parameters
        optimizer = torch.optim.AdamW(
            [
                {"params": model.gnn.parameters()},
                {"params": model.scaf_proj.parameters()},
                {"params": model.side_proj.parameters()},
                {"params": model.llm.parameters()},  # LoRA Params
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
                set_peft_model_state_dict(model.llm, checkpoint["peft_state_dict"], adapter_name="default")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # Load History if available
                if "history" in checkpoint:
                    history = checkpoint["history"]
            else:
                raise FileNotFoundError(f"Checkpoint for epoch {resume_epoch} not found at {ckpt_file}")

        # Datasets
        with open(args.inst_json, "r") as f:
            full_inst = json.load(f)
            instruction = full_inst["full"][args.dataset] if args.use_inst else full_inst["short"][args.dataset]
        prompter = Prompter(args.dataset, instruction)
        decomposer = MoleculeDecomposer()
        collator = S3PromptCollatorWLabel(decomposer, tokenizer)

        is_cls = args.dataset in CLASSIFICATION
        train_ds = S3PromptDataset(valid_df, train_df, valid_shots, tokenizer, prompter, decomposer, args.k, is_train=True, is_cls=is_cls)  # fmt: skip
        test_ds = S3PromptDataset(test_df, train_df, test_shots, tokenizer, prompter, decomposer, args.k, is_train=False, is_cls=is_cls)  # fmt: skip

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
                val_metrics = evaluate_s3prompt(args, model, eval_loader, tokenizer, terminators, device, args.dataset)
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

            # Save GNN/Projectors + LoRA weights separately
            peft_state_dict = get_peft_model_state_dict(model.llm, state_dict=None, adapter_name="default")

            ckpt_payload = {
                "epoch": epoch,
                "gnn_state_dict": model.gnn.state_dict(),
                "scaf_proj_state_dict": model.scaf_proj.state_dict(),
                "side_proj_state_dict": model.side_proj.state_dict(),
                "peft_state_dict": peft_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }
            torch.save(ckpt_payload, os.path.join(args.ckpt_dir, f"epoch_{epoch:02}.pt"))
            # TODO: 이 snippet은 필요한가???
            if epoch > 1:
                prev_ckpt = os.path.join(args.ckpt_dir, f"epoch_{epoch-1:02}.pt")
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
            save_json(history, os.path.join(args.save_dir, "result.json"))

        print("\n>>> Final Testing...")
        test_metrics = evaluate_s3prompt(
            args, model, test_loader, tokenizer, terminators, device, args.dataset, is_test=True
        )
        save_json(test_metrics, os.path.join(args.save_dir, "result_test.json"))
        print("Done.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
