"""
run_text_graph_reg.py
=====================
Text+Graph Modality for Continuous Regression (Configs D, E, F, G).
Uses S3PromptModel (GNN + Projector + LLM) with an --use_lora toggle.

Usage:
    # Config D – GNN/Proj only (LLM frozen), no LoRA
    python run_text_graph_reg.py llama --dataset ESOL --epochs 10

    # Config E – GNN/Proj + LoRA
    python run_text_graph_reg.py llama --dataset ESOL --use_lora --epochs 10
"""

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

from utils.dataset import S3PromptDataset, S3PromptCollatorWLabel
from utils.evaluation import extract_pred, eval_regression
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import TaskType, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from utils.prompter import Prompter
from models.s3prompt import S3PromptModel
from utils.dataset_loader import load_dataset
from utils.scaffold_decompose import MoleculeDecomposer
from utils.constants import DATA_ROOT, REGRESSION, CLASSIFICATION
from utils.helper import log_df, load_json, save_json, parse_args, format_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress noisy right-padding warning during generation
class _RightPadFilter(logging.Filter):
    def filter(self, record):
        return "right-padding was detected" not in record.getMessage()

logging.getLogger("transformers.generation.utils").addFilter(_RightPadFilter())

SCAF_TOKEN = "<|scaffold|>"
SIDE_TOKEN = "<|decoration|>"


# ==========================================
# 1. Helper Functions
# ==========================================
def suggest_oom_fix(args, oom_count, total_len):
    suggestions = []
    if not args.gradient_checkpointing:
        suggestions.append("Enable Gradient Checkpointing: --gradient_checkpointing")
    if args.batch_size > 1:
        suggestions.append(f"Reduce Batch Size: --batch_size 1 (Current: {args.batch_size})")
    if args.quant_bit is None:
        suggestions.append("Apply 4-bit Quantization: --quant_bit 4")
    elif args.quant_bit == 8:
        suggestions.append("Switch to 4-bit: --quant_bit 4 (Current: 8-bit)")
    current_len = args.model_max_length
    if current_len == -1 or current_len > 4096:
        suggestions.append(f"Enable Truncation: --model_max_length 4096 (Current: {current_len})")

    print(f"\n{'='*60}")
    print(f"[CRITICAL] OOM Count Exceeded: {oom_count}/{total_len}")
    print(f"{'='*60}")
    for i, tip in enumerate(suggestions, 1):
        print(f"  {i}. {tip}")
    print(f"{'='*60}\n")


def load_predictor(args, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    print(f"Loading LLaMA model ({args.model_name}) on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
        **kwargs,
    )
    model.eval()
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    print("Model loaded successfully")
    return tokenizer, model, terminators


def setup_base_model(args):
    """Load LLM with quantization + special tokens.  LoRA is applied later in main()."""
    bnb_config = None
    if args.quant_bit == 4:
        print("[Info] Config: 4-bit Quantization (NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.quant_bit == 8:
        print("[Info] Config: 8-bit Quantization")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer, model, terminators = load_predictor(args, quantization_config=bnb_config)

    if args.model_max_length > 0:
        tokenizer.model_max_length = args.model_max_length
        print(f"[Info] Enforcing Tokenizer Max Length: {args.model_max_length}")

    special_tokens = {"additional_special_tokens": [SCAF_TOKEN, SIDE_TOKEN]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    return tokenizer, model, terminators


def gen_method_tag(args):
    if args.method == "pool_clustering":
        return "pc-rand" if args.sample_strategy == "random" else "pc-sim"
    return args.method


def generate_random_shots(query_df, train_df, k):
    shot_map = {}
    train_ids = train_df["raw_ID"].tolist()
    for _, row in query_df.iterrows():
        rid = str(row["raw_ID"])
        selected = np.random.choice(train_ids, k, replace=False).tolist()
        shot_map[rid] = {"shot_idx_list": [str(s) for s in selected]}
    return shot_map


def plot_history(history, tag, save_dir):
    epochs = range(1, len(history["train_loss_epoch"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss_epoch"], "b-", label="Train Loss")
    if "valid_loss" in history:
        arr = np.array(history["valid_loss"], dtype=np.float64)
        mask = ~np.isnan(arr)
        if mask.any():
            plt.plot(np.array(list(epochs))[mask], arr[mask], "r.-", label="Valid Loss")
    plt.title(f"Training Loss\n{tag}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()


# ==========================================
# 2. Regression Triplet Mining (sigma-based)
# ==========================================
def precompute_activity_cliff_info(
    shot_map, query_df, train_df, k_shot,
    embed="fp", query_df_fp=None, train_df_fp=None,
    label_std=None,
):
    """
    sigma-based Activity Cliff mining for regression.
      Positive: |delta_y| < 0.5 * sigma   (similar property despite structural difference)
      Negative: |delta_y| > 1.0 * sigma   (different property despite structural similarity)
      Fallback: min/max delta pair IF gap > 0.3 * sigma.
    """
    print(f"[Info] Pre-computing Regression Activity Cliff indices for {len(shot_map)} queries...")

    if embed == "fp":
        target_train_df = train_df
        target_query_df = query_df
    elif train_df_fp is not None and query_df_fp is not None:
        target_train_df = train_df_fp
        target_query_df = query_df_fp
    else:
        print(f"  [Error] ECFP source missing! embed='{embed}' but *_df_fp is None.")
        return shot_map

    train_cache = dict(
        zip(target_train_df["raw_ID"].astype(str),
            zip(target_train_df["embed"], target_train_df["label"]))
    )
    query_cache = dict(
        zip(target_query_df["raw_ID"].astype(str),
            zip(target_query_df["embed"], target_query_df["label"]))
    )

    processed_count = 0
    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Mining (Reg)"):
        qid = str(row["raw_ID"])
        if qid not in shot_map:
            continue
        info = shot_map[qid]
        shot_ids = info["shot_idx_list"]
        if len(shot_ids) < k_shot:
            info["mining"] = None
            continue

        q_data = query_cache.get(str(qid))
        if not q_data:
            info["mining"] = None
            continue
        q_fp, q_label_val = q_data
        if q_fp is None:
            info["mining"] = None
            continue

        pos_candidates = []
        neg_candidates = []
        all_deltas = []

        for i, s_rid in enumerate(shot_ids):
            s_data = train_cache.get(str(s_rid))
            if not s_data:
                continue
            s_fp, s_label_val = s_data
            if s_fp is None:
                continue

            sim = DataStructs.TanimotoSimilarity(q_fp, s_fp)
            delta_y = abs(float(q_label_val) - float(s_label_val))
            all_deltas.append((i, sim, delta_y))

            if label_std is not None and label_std > 0:
                if delta_y < 0.5 * label_std:
                    pos_candidates.append((i, sim))
                elif delta_y > 1.0 * label_std:
                    neg_candidates.append((i, sim))
            else:
                pos_candidates.append((i, sim))

        best_p = max(pos_candidates, key=lambda x: x[1]) if pos_candidates else None
        best_n = max(neg_candidates, key=lambda x: x[1]) if neg_candidates else None

        # Fallback: use min/max delta pair if strict candidates are missing
        if label_std is not None and label_std > 0 and (not best_p or not best_n) and all_deltas:
            all_deltas_sorted = sorted(all_deltas, key=lambda x: x[2])
            fb_pos = all_deltas_sorted[0]
            fb_neg = all_deltas_sorted[-1]
            if abs(fb_neg[2] - fb_pos[2]) > 0.3 * label_std:
                if not best_p:
                    best_p = (fb_pos[0], fb_pos[1])
                if not best_n:
                    best_n = (fb_neg[0], fb_neg[1])

        if best_p and best_n:
            info["mining"] = {
                "p_idx": best_p[0], "p_sim": best_p[1],
                "n_idx": best_n[0], "n_sim": best_n[1],
            }
            processed_count += 1
        else:
            info["mining"] = None

    print(f"[Info] Mining completed. {processed_count}/{len(shot_map)} queries have valid triplets.")
    return shot_map


# ==========================================
# 3. Loss Functions
# ==========================================
def find_pattern_indices(input_ids, pattern_str="[ANSWER]", tokenizer=None, debug=False):
    """Sliding-window token decode to locate `pattern_str` in each sequence."""
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
                decoded = tokenizer.decode(seq_list[i: i + w])
                if pattern_str in decoded:
                    found_idx = i + w - 1
                    if not seq_indices or found_idx > seq_indices[-1]:
                        seq_indices.append(found_idx)
                    break
        batch_indices.append(seq_indices)
    return batch_indices


def compute_context_aware_triplet_loss(v_q, v_p, v_n, w_sim, margin=1.0, device="cuda"):
    triplet_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction="none")
    loss = triplet_fn(v_q.unsqueeze(0), v_p.unsqueeze(0), v_n.unsqueeze(0))
    return loss * w_sim


# ==========================================
# 4. Train & Eval Loops
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
        input_ids, scaf_batch, side_batch, scaf_mask, side_mask,
        labels=labels, attention_mask=attention_mask,
        output_hidden_states=need_triplet,
    )
    ce_loss = outputs.loss

    triplet_loss = torch.tensor(0.0, device=device)
    valid_count = 0

    if need_triplet and "metadata" in batch:
        hidden_states = outputs.hidden_states[-1]

        indices_1 = find_pattern_indices(input_ids, "[Answer]", tokenizer=tokenizer)
        indices_2 = find_pattern_indices(input_ids, "[ANSWER]", tokenizer=tokenizer)
        answer_indices_batch = []
        for l1, l2 in zip(indices_1, indices_2):
            answer_indices_batch.append(sorted(set(l1 + l2)))

        total_tri = torch.tensor(0.0, device=device)
        for i, meta in enumerate(batch["metadata"]):
            mining_info = meta.get("mining")
            if not mining_info:
                continue
            indices = answer_indices_batch[i]
            k_shots = args.k
            if len(indices) < k_shots + 1:
                continue
            try:
                idx_p = indices[mining_info["p_idx"]]
                idx_n = indices[mining_info["n_idx"]]
                idx_q = indices[k_shots]
            except IndexError:
                continue

            v_p = F.normalize(hidden_states[i, idx_p, :], p=2, dim=0)
            v_n = F.normalize(hidden_states[i, idx_n, :], p=2, dim=0)
            v_q = F.normalize(hidden_states[i, idx_q, :], p=2, dim=0)

            w_sim = mining_info["n_sim"]
            loss_t = compute_context_aware_triplet_loss(v_q, v_p, v_n, w_sim, device=device)
            total_tri += loss_t.mean()
            valid_count += 1

        if valid_count > 0:
            triplet_loss = total_tri / valid_count

    total_loss = ce_loss + args.lambda_loss * triplet_loss

    if return_outputs:
        return ce_loss, triplet_loss, total_loss, outputs
    return ce_loss, triplet_loss, total_loss


def train_epoch(args, model, dataloader, tokenizer, optimizer, device, accumulation_steps=1):
    model.train()
    logs = {"ce": [], "triplet": [], "total": [], "step_total": []}
    optimizer.zero_grad()

    oom_count = 0
    oom_raw_ids = []
    OOM_RATIO_THOLD = 0.10
    total_len = len(dataloader)
    max_allowed_oom = max(int(total_len * OOM_RATIO_THOLD), 5)

    print(f"\n[Info] OOM Tolerance: {max_allowed_oom}/{total_len} batches ({OOM_RATIO_THOLD*100:.0f}%)")
    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        try:
            ce, tri, tot = process_batch(
                args, model, batch, tokenizer, device,
                need_triplet=True, return_outputs=False,
            )
            (tot / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            logs["ce"].append(ce.item())
            logs["triplet"].append(tri.item())
            logs["total"].append(tot.item())
            logs["step_total"].append(tot.item())
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
                    raise RuntimeError("Training terminated: excessive OOM.")
                continue
            raise

    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.empty_cache()

    safe_mean = lambda lst: np.mean(lst) if lst else 0.0
    return {
        "ce": safe_mean(logs["ce"]),
        "triplet": safe_mean(logs["triplet"]),
        "total": safe_mean(logs["total"]),
        "step_history": logs["step_total"],
        "oom_count": oom_count,
        "oom_raw_ids": oom_raw_ids,
        "total_batches": total_len,
    }


def evaluate_s3prompt_reg(
    args, model, dataloader, tokenizer, terminators, device,
    label_stats=None, is_test=False,
):
    """Regression eval: generate text -> parse float -> de-standardize -> metrics."""
    model.eval()
    results = {}
    val_losses = {"ce": [], "triplet": [], "total": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 1. Loss (full input including ground truth for validation)
            need_triplet = not is_test
            ce, tri, tot = process_batch(
                args, model, batch, tokenizer, device,
                need_triplet=need_triplet, return_outputs=False,
            )
            val_losses["ce"].append(ce.item())
            val_losses["triplet"].append(tri.item())
            val_losses["total"].append(tot.item())

            # 2. Generation input (strip GT token for validation to avoid leakage)
            input_ids = batch["input_ids"].to(device)
            if not is_test:
                curr_input_ids = input_ids[:, :-1]
                curr_att_mask = batch["attention_mask"].to(device)[:, :-1]
            else:
                curr_input_ids = input_ids
                curr_att_mask = batch["attention_mask"].to(device)

            scaf_batch = batch["scaf_batch"].to(device)
            side_batch = batch["side_batch"].to(device)
            raw_ids = batch["raw_ids"]
            true_labels = batch["q_labels"].numpy()

            scaf_mask = curr_input_ids == tokenizer.convert_tokens_to_ids(SCAF_TOKEN)
            side_mask = curr_input_ids == tokenizer.convert_tokens_to_ids(SIDE_TOKEN)

            # 3. Generate
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
                    results[rid] = {
                        "true": 0.0, "pred": None, "text": "",
                        "score": None, "entropy": None,
                    }
                continue

            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            # 4. Parse floats and de-standardize
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

                if i == 0 and len(results) <= 1:
                    print(f"\n  [Preview ID: {rid}] Pred={pred} | True={true_labels[i]:.4f}")
                    print(f"  Generated: {pred_text[:120]}...")

    torch.cuda.empty_cache()

    fail_set = {k for k, v in results.items() if v["pred"] is None}
    valid_results = [res for rid, res in results.items() if rid not in fail_set]

    result_log = {
        "true_list": [r["true"] for r in valid_results],
        "pred_list": [r["pred"] for r in valid_results],
        "results": results,
        "total": len(results),
        "fail_set": list(fail_set),
        "fail_count": len(fail_set),
    }

    eval_metrics = eval_regression(result_log)

    loss_metrics = {
        "loss_ce": np.mean(val_losses["ce"]),
        "loss_triplet": np.mean(val_losses["triplet"]),
        "loss_total": np.mean(val_losses["total"]),
    }

    return {**loss_metrics, **eval_metrics}


# ==========================================
# 5. Shot Loading (with mining)
# ==========================================
def load_shot_wrt_method(args, train_df, valid_df, test_df):
    embed_tag = f"molt5-{args.embedding_type}" if args.embedding_type != "fp" else "fp"
    if args.method == "random":
        print(f"[Info] Generating random shots (k={args.k})...")
        valid_shots = generate_random_shots(valid_df, train_df, args.k)
        test_shots = generate_random_shots(test_df, train_df, args.k)
    elif args.method == "pool_clustering":
        eps_tag = f"eps={args.eps}" if args.eps else "no_merge"
        cluster_tag_dir = f"{embed_tag}_{eps_tag}_{args.sample_strategy}-sampling"
        cache_dir = os.path.join(args.cache_root, args.dataset, cluster_tag_dir)
        vp = os.path.join(cache_dir, f"query_to_shot_valid_k={args.k}.json")
        tp = os.path.join(cache_dir, f"query_to_shot_test_k={args.k}.json")
        if os.path.exists(vp) and os.path.exists(tp):
            print(f"[Info] Loading Pool Clustering shots from {cache_dir}")
            valid_shots = load_json(vp)
            test_shots = load_json(tp)
        else:
            raise FileNotFoundError(f"Pool Clustering files not found at {cache_dir}.")
    elif args.method == "topk":
        tag = f"topk_{embed_tag}"
        cache_dir = os.path.join(args.cache_root, args.dataset, tag)

        def _load_topk(prefix):
            target_file = None
            if os.path.exists(cache_dir):
                for fname in os.listdir(cache_dir):
                    if fname.startswith(f"query_to_shot_{prefix}_topk="):
                        target_file = os.path.join(cache_dir, fname)
                        break
            if not target_file:
                raise FileNotFoundError(f"TopK file for {prefix} not found in {cache_dir}.")
            print(f"[Info] Loaded TopK: {os.path.basename(target_file)}")
            data = load_json(target_file)
            for qid, info in data.items():
                info["shot_idx_list"] = info["shot_idx_list"][: args.k]
            return data

        print(f"[Info] Loading Top-K shots from {cache_dir}")
        valid_shots = _load_topk("valid")
        test_shots = _load_topk("test")
    return valid_shots, test_shots


def load_shot(args, train_df, valid_df, test_df, label_stats=None):
    mining_cache_dir = os.path.join(args.cache_root, args.dataset, "mining_reg")
    os.makedirs(mining_cache_dir, exist_ok=True)
    mining_fname = f"mining_info_{args.embedding_type}_{args.method}_k={args.k}.json"
    valid_cache_path = os.path.join(mining_cache_dir, f"valid_{mining_fname}")
    test_cache_path = os.path.join(mining_cache_dir, f"test_{mining_fname}")

    print("\n" + "=" * 40)
    print(" >>> Regression Activity Cliff Mining Check")
    print("=" * 40)

    valid_cached = os.path.exists(valid_cache_path)
    test_cached = os.path.exists(test_cache_path)

    label_std = label_stats["std"] if label_stats else None

    if valid_cached and test_cached:
        print(f"[Cache Hit] Loading pre-computed mining info...")
        valid_shots = load_json(valid_cache_path)
        test_shots = load_json(test_cache_path)
    else:
        print("[Cache Miss] Computing Activity Cliff indices (sigma-based)...")
        valid_shots, test_shots = load_shot_wrt_method(args, train_df, valid_df, test_df)

        if args.embedding_type == "fp":
            train_df_fp, valid_df_fp, test_df_fp = train_df, valid_df, test_df
        else:
            print("[Info] Loading FP dataset for mining calculations...")
            train_df_fp, valid_df_fp, test_df_fp = load_dataset(args.dataset, "fp", verbose=False)

        if not valid_cached:
            valid_shots = precompute_activity_cliff_info(
                valid_shots, valid_df, train_df, args.k,
                embed=args.embedding_type,
                query_df_fp=valid_df_fp, train_df_fp=train_df_fp,
                label_std=label_std,
            )
            save_json(valid_shots, valid_cache_path)
            print(f"  - Saved valid mining: {valid_cache_path}")

        if not test_cached:
            test_shots = precompute_activity_cliff_info(
                test_shots, test_df, train_df, args.k,
                embed=args.embedding_type,
                query_df_fp=test_df_fp, train_df_fp=train_df_fp,
                label_std=label_std,
            )
            save_json(test_shots, test_cache_path)
            print(f"  - Saved test mining: {test_cache_path}")

    return valid_shots, test_shots


# ==========================================
# 6. Main
# ==========================================
def main():
    try:
        _use_lora = "--use_lora" in sys.argv
        if _use_lora:
            sys.argv.remove("--use_lora")

        args, categories = parse_args()
        args.use_lora = _use_lora

        # ----------------------------------------------------------------------
        # Resume override: if --resume_from is set, ignore all CLI args (except
        # --epochs, --resume_from, --start_epoch) and reload the original
        # training config from the saved args.json. This guarantees fairness:
        # the resumed run is identical to the original except for the new
        # epoch budget.
        # ----------------------------------------------------------------------
        args.orig_epochs = None
        if getattr(args, "resume_from", None):
            ts = args.resume_from
            candidates = []
            for root, dirs, files in os.walk(args.save_root):
                if os.path.basename(root) == ts and "args.json" in files:
                    candidates.append(root)
            if not candidates:
                raise FileNotFoundError(
                    f"--resume_from '{ts}': no args.json found under {args.save_root}"
                )
            if len(candidates) > 1:
                raise RuntimeError(
                    f"--resume_from '{ts}' is ambiguous. Matches:\n  "
                    + "\n  ".join(candidates)
                )
            found_dir = candidates[0]
            with open(os.path.join(found_dir, "args.json"), "r") as f:
                orig = json.load(f)

            # Keep these from CLI / current run; replace everything else.
            cli_keep = {"epochs", "resume_from", "start_epoch"}
            # Don't overwrite runtime-only fields either.
            runtime_fields = {"now", "ckpt_dir", "save_dir", "exp_tag",
                              "start_time", "end_time", "duration"}
            preserved_cli = {k: getattr(args, k) for k in cli_keep if hasattr(args, k)}
            args.orig_epochs = orig.get("epochs")
            for k, v in orig.items():
                if k in cli_keep or k in runtime_fields:
                    continue
                setattr(args, k, v)
            for k, v in preserved_cli.items():
                setattr(args, k, v)
            # Force auto-detect of start_epoch from disk unless user explicitly set it.
            if args.start_epoch == 1:
                args.start_epoch = 0
            print(f"[Resume] Loaded original args from {found_dir}/args.json")
            print(f"[Resume] Only --epochs={args.epochs} (was {args.orig_epochs}) differs.")

        if args.method == "zero_shot":
            raise ValueError("Zero-shot unsupported for graph modality.")

        accumulation_steps = args.grad_accum_steps
        device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
        print(f"Using device: {device}")
        print(f"Mode: {'GNN+Proj+LoRA' if args.use_lora else 'GNN+Proj only (LLM frozen)'}")
        print(f"Physical Batch Size: {args.batch_size}")
        print(f"Effective Batch Size: {args.batch_size * accumulation_steps}")

        # Paths
        lora_tag = "graph_lora_reg" if args.use_lora else "graph_frozen_reg"
        sub_path_config = f"temp={args.temperature}/eps={args.eps}/{lora_tag}"
        base_ckpt_path = os.path.join(args.ckpt_root, args.dataset, sub_path_config)
        base_save_path = os.path.join(args.save_root, args.dataset, sub_path_config)

        start_epoch = getattr(args, "start_epoch", 1)
        resume_from = getattr(args, "resume_from", None)
        is_resume = (start_epoch != 1) or (resume_from is not None)

        if is_resume:
            if not os.path.exists(base_save_path):
                raise FileNotFoundError(f"Cannot resume: {base_save_path} not found.")

            if resume_from is not None:
                ts_save = os.path.join(base_save_path, resume_from)
                ts_ckpt = os.path.join(base_ckpt_path, resume_from)
                if not (os.path.isdir(ts_save) and os.path.isdir(ts_ckpt)):
                    raise FileNotFoundError(
                        f"Cannot resume: --resume_from='{resume_from}' not found under\n"
                        f"  save: {base_save_path}\n  ckpt: {base_ckpt_path}"
                    )
                args.now = resume_from
            else:
                timestamps = sorted([d for d in os.listdir(base_save_path)
                                     if os.path.isdir(os.path.join(base_save_path, d))])
                if not timestamps:
                    raise FileNotFoundError(f"Cannot resume: no dirs in {base_save_path}")
                args.now = timestamps[-1]

            # Auto-detect start_epoch from the surviving epoch_NN.pt file
            # when user passes --start_epoch 0 (or leaves it implicit with --resume_from).
            ts_ckpt_dir = os.path.join(base_ckpt_path, args.now)
            auto_needed = (start_epoch == 0) or (resume_from is not None and start_epoch == 1)
            if auto_needed:
                ck_files = [f for f in os.listdir(ts_ckpt_dir)
                            if f.startswith("epoch_") and f.endswith(".pt")]
                if not ck_files:
                    raise FileNotFoundError(
                        f"Cannot auto-detect start_epoch: no epoch_*.pt in {ts_ckpt_dir}"
                    )
                last_ep = max(int(f[len("epoch_"):-len(".pt")]) for f in ck_files)
                start_epoch = last_ep + 1
                args.start_epoch = start_epoch
                print(f"[Resume] Auto-detected start_epoch={start_epoch} from {ts_ckpt_dir}")

            print(f"[Resume] now={args.now}  start_epoch={start_epoch}  target_epochs={args.epochs}")
        else:
            args.now = datetime.now().strftime("%Y%m%d_%H%M%S")

        args.ckpt_dir = os.path.join(base_ckpt_path, args.now)
        args.save_dir = os.path.join(base_save_path, args.now)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)

        embed_tag = f"molt5-{args.embedding_type}" if args.embedding_type != "fp" else "fp"
        method_tag = gen_method_tag(args)
        args.exp_tag = f"{args.dataset} - {embed_tag}_{method_tag}_k={args.k}_GRAPH_REG"
        if args.use_lora:
            args.exp_tag += "_LoRA"

        categories.setdefault("Experiment Settings", []).extend(["now", "exp_tag", "ckpt_dir", "save_dir"])
        print(format_args(args, categories))
        save_json(vars(args), os.path.join(args.save_dir, "args.json"))

        # --- Data ---
        print("Loading Dataset...")
        train_df, valid_df, test_df = load_dataset(args.dataset, args.embedding_type, verbose=True)

        assert args.dataset in REGRESSION, f"This script is for regression datasets only. Got: {args.dataset}"
        _lbl_mean = float(train_df["label"].mean())
        _lbl_std = float(train_df["label"].std())
        label_stats = {"mean": _lbl_mean, "std": _lbl_std}
        print(f"[Info] Regression Label Stats: mean={_lbl_mean:.4f}, std={_lbl_std:.4f}")

        valid_shots, test_shots = load_shot(args, train_df, valid_df, test_df, label_stats=label_stats)

        # --- Model ---
        args.mode = "llama"
        tokenizer, llm_model, terminators = setup_base_model(args)

        model = S3PromptModel(llm_model, tokenizer, terminators, gnn_hidden=args.gnn_hidden).to(device)

        if args.use_lora:
            print("[Info] Applying PEFT (LoRA) to LLM inside S3PromptModel...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            model.llm = get_peft_model(model.llm, peft_config)
            model.llm.print_trainable_parameters()

            if args.gradient_checkpointing:
                if hasattr(model.llm, "gradient_checkpointing_enable"):
                    print("[Info] Enabling Gradient Checkpointing...")
                    model.llm.gradient_checkpointing_enable()
                    if hasattr(model.llm, "enable_input_require_grads"):
                        model.llm.enable_input_require_grads()

        # --- Optimizer ---
        opt_params = [
            {"params": model.gnn.parameters()},
            {"params": model.scaf_proj.parameters()},
            {"params": model.side_proj.parameters()},
        ]
        if args.use_lora:
            lora_params = [p for p in model.llm.parameters() if p.requires_grad]
            if lora_params:
                opt_params.append({"params": lora_params})
        optimizer = torch.optim.AdamW(opt_params, lr=args.lr)

        history = {
            "train_loss_epoch": [], "train_loss_step": [],
            "train_loss_ce": [], "train_loss_triplet": [],
            "valid_loss": [], "valid_mse": [], "valid_rmse": [],
            "oom_count": [], "oom_raw_ids": [],
        }

        if start_epoch > 1:
            ckpt_file = os.path.join(args.ckpt_dir, f"epoch_{start_epoch - 1:02}.pt")
            if os.path.exists(ckpt_file):
                print(f"[Info] Loading checkpoint from {ckpt_file}...")
                ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
                model.gnn.load_state_dict(ckpt["gnn_state_dict"])
                model.scaf_proj.load_state_dict(ckpt["scaf_proj_state_dict"])
                model.side_proj.load_state_dict(ckpt["side_proj_state_dict"])
                if args.use_lora and ckpt.get("peft_state_dict") is not None:
                    set_peft_model_state_dict(model.llm, ckpt["peft_state_dict"], adapter_name="default")
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "history" in ckpt:
                    history = ckpt["history"]
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        # --- Datasets ---
        with open(args.inst_json, "r") as f:
            full_inst = json.load(f)
            instruction = full_inst["full"][args.dataset] if args.use_inst else full_inst["short"][args.dataset]

        prompter = Prompter(args.dataset, instruction)
        decomposer = MoleculeDecomposer()
        collator = S3PromptCollatorWLabel(decomposer, tokenizer, is_cls=False)

        train_ds = S3PromptDataset(
            valid_df, train_df, valid_shots, tokenizer, prompter, decomposer, args.k,
            is_train=True, is_cls=False, label_stats=label_stats,
        )
        test_ds = S3PromptDataset(
            test_df, train_df, test_shots, tokenizer, prompter, decomposer, args.k,
            is_train=False, is_cls=False, label_stats=label_stats,
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=4)
        eval_loader  = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)
        test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)

        # --- Training ---
        print(f"\n>>> Start Training (Graph + Reg) from Epoch {start_epoch}...")
        result_json_path = os.path.join(args.save_dir, "result.json")

        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\n[Epoch {epoch}/{args.epochs}]")
            train_metrics = train_epoch(args, model, train_loader, tokenizer, optimizer, device, accumulation_steps)
            print(f"  Train Loss | Total: {train_metrics['total']:.4f}, CE: {train_metrics['ce']:.4f}, Tri: {train_metrics['triplet']:.4f}")

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
            history["valid_loss"].append(None)
            history["valid_mse"].append(None)
            history["valid_rmse"].append(None)

            # Checkpoint
            peft_sd = None
            if args.use_lora:
                peft_sd = get_peft_model_state_dict(model.llm, state_dict=None, adapter_name="default")

            ckpt_payload = {
                "epoch": epoch,
                "gnn_state_dict": model.gnn.state_dict(),
                "scaf_proj_state_dict": model.scaf_proj.state_dict(),
                "side_proj_state_dict": model.side_proj.state_dict(),
                "peft_state_dict": peft_sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "label_stats": label_stats,
            }
            curr_ckpt = os.path.join(args.ckpt_dir, f"epoch_{epoch:02}.pt")
            torch.save(ckpt_payload, curr_ckpt)
            if epoch > 1:
                prev = os.path.join(args.ckpt_dir, f"epoch_{epoch-1:02}.pt")
                if os.path.exists(prev):
                    os.remove(prev)

            save_json(history, result_json_path)
            try:
                plot_history(history, args.exp_tag, args.save_dir)
            except Exception as e:
                print(f"[Warning] Plotting failed: {e}")

            # --- Periodic Test (every TEST_EVERY epochs and on the final epoch) ---
            TEST_EVERY = 10
            do_test = (epoch % TEST_EVERY == 0) or (epoch == args.epochs)
            if do_test:
                print(f"\n>>> [Epoch {epoch}] Testing...")
                test_metrics = evaluate_s3prompt_reg(
                    args, model, test_loader, tokenizer, terminators, device,
                    label_stats=label_stats, is_test=True,
                )
                history["test_results"] = test_metrics
                save_json(history, result_json_path)

                # result_test.json: always list-of-runs so periodic + resumed
                # entries accumulate. Each entry is tagged with its epoch.
                result_test_path = os.path.join(args.save_dir, "result_test.json")
                new_entry = {"epoch": epoch, **test_metrics}
                existing = None
                if os.path.exists(result_test_path):
                    try:
                        with open(result_test_path, "r") as f:
                            existing = json.load(f)
                    except Exception:
                        existing = None
                if isinstance(existing, list):
                    out = existing + [new_entry]
                elif isinstance(existing, dict):
                    prev = dict(existing)
                    if "epoch" not in prev:
                        prev["epoch"] = (
                            getattr(args, "orig_epochs", None) or "previous"
                        )
                    out = [prev, new_entry]
                else:
                    out = [new_entry]
                save_json(out, result_test_path)
                rmse_val = test_metrics.get("rmse")
                rmse_str = f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else "?"
                print(f"  [Test @ epoch {epoch}] RMSE={rmse_str}  "
                      f"(result_test.json has {len(out)} entries)")

        print("Done.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
