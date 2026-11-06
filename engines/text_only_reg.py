"""
run_text_only_reg.py
====================
Text-Only Modality for Continuous Regression (Configs A, B, C).
Combines Frozen-LLM inference (run_text_only.py) and LoRA fine-tuning
(run_text_only_peft.py) behind a single --use_lora toggle.

Usage:
    # Config A – Frozen LLM, zero-shot or few-shot inference only
    python run_text_only_reg.py llama --dataset ESOL --epochs 0

    # Config B – LoRA fine-tuning
    python run_text_only_reg.py llama --dataset ESOL --use_lora --epochs 5
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
from torch.utils.data import Dataset, DataLoader
from utils.evaluation import extract_pred, eval_regression
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import TaskType, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from utils.dataset_loader import load_dataset
from utils.prompter import Prompter as BasePrompter
from utils.scaffold_decompose import MoleculeDecomposer
from utils.constants import DATA_ROOT, REGRESSION, CLASSIFICATION
from utils.helper import log_df, load_json, save_json, parse_args, format_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Suppress noisy right-padding warning during generation
class _RightPadFilter(logging.Filter):
    def filter(self, record):
        return "right-padding was detected" not in record.getMessage()


logging.getLogger("transformers.generation.utils").addFilter(_RightPadFilter())


# ==========================================
# 1. Helper & Visualization
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


def setup_model_and_tokenizer(args):
    """Load LLM with optional LoRA wrapping controlled by args.use_lora."""
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

    if args.use_lora:
        print("[Info] Applying PEFT (LoRA)...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("[Info] Frozen LLM mode (no LoRA)")

    if args.model_max_length > 0:
        tokenizer.model_max_length = args.model_max_length
        print(f"[Info] Enforcing Tokenizer Max Length: {args.model_max_length}")

    if args.use_lora and args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            print("[Info] Enabling Gradient Checkpointing...")
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

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
        valid_shot_path = os.path.join(cache_dir, f"query_to_shot_valid_k={args.k}.json")
        test_shot_path = os.path.join(cache_dir, f"query_to_shot_test_k={args.k}.json")
        if os.path.exists(valid_shot_path) and os.path.exists(test_shot_path):
            print(f"[Info] Loading Pool Clustering shots from {cache_dir}")
            valid_shots = load_json(valid_shot_path)
            test_shots = load_json(test_shot_path)
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
            print(f"[Info] Loaded TopK file: {os.path.basename(target_file)}")
            data = load_json(target_file)
            for qid, info in data.items():
                info["shot_idx_list"] = info["shot_idx_list"][: args.k]
            return data

        print(f"[Info] Loading Top-K shots from {cache_dir}")
        valid_shots = _load_topk("valid")
        test_shots = _load_topk("test")
    elif args.method == "zero_shot":
        valid_shots = {}
        test_shots = {}
    else:
        raise ValueError(f"Unknown method: {args.method}")
    return valid_shots, test_shots


def load_shot(args, train_df, valid_df, test_df):
    return load_shot_wrt_method(args, train_df, valid_df, test_df)


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
# 2. Prompter & Dataset (Regression)
# ==========================================
class TextScaffoldPrompter(BasePrompter):
    """Text-only variant: references SMILES strings instead of graph embeddings."""

    def generate(self, query_formatted_str, shots_str=None, mode="llama"):
        system_content = (
            "## System\n\n"
            f"You are an expert AI chemist specializing in Structure-Activity Relationship (SAR) analysis for the {self.dataset} task.\n"
            "You will be provided with a molecule's full structure as a **SMILES string**.\n"
            "Crucially, the molecule's core [Scaffold] and [Decorations] (side chains) are provided as **SMILES strings** to explicitly highlight structural components.\n"
            "Your task is to:\n"
            "1. Interpret the full SMILES string.\n"
            "2. Integrate the explicit structural features provided in [Scaffold] and [Decorations].\n"
            "3. Analyze how these structural components influence the target property to predict the label.\n"
            "Follow the Task Instruction CAREFULLY.\n"
            "Your output MUST contain exactly two sections: [Answer] and [Reasoning].\n"
            "No additional text is allowed.\n"
        )

        user_parts = []
        if self.instruction:
            user_parts.append(
                "## Task Instruction\n\n"
                "### Domain Context & Objective\n\n"
                f"{self.instruction}\n\n\n"
                "### Analysis Strategy\n\n"
                "Based on the domain context above, analyze the target molecule to predict the property.\n"
                "Follow these steps strictly:\n"
                "1. Global Structure: Interpret the 'Input' SMILES string to understand the overall molecular topology.\n"
                "2. Explicit Feature Integration: Pay close attention to the `[Scaffold]` and `[Decorations]` fields.\n"
                "These provide decomposed structural views of the molecule.\n"
                "3. SAR Synthesis: Combine the biological context (from the Domain Context section) with the structural features to infer the label.\n"
                "4. Output Format:\n"
                f"    - [Answer] — {self.strict_answer}\n"
                "    - [Reasoning] — Provide step-by-step reasoning. Keep it concise (ideally under 150 words).\n"
                "No additional text before and after these sections is allowed.\n"
            )

        if shots_str:
            user_parts.append("## Examples\n\n" + shots_str.strip())

        user_parts.append(f"## Query\n\n{query_formatted_str.strip()}\n- Output:")
        user_content = "\n\n".join(user_parts)

        if mode == "llama":
            return [
                {"role": "system", "content": system_content.strip()},
                {"role": "user", "content": user_content.strip()},
            ]
        raise ValueError(f"Unsupported mode: {mode}")


class TextOnlyRegDataset(Dataset):
    """
    Text-only dataset for regression tasks.
    Standardises labels via z-score and embeds the normalised value in prompts.
    """

    def __init__(
        self,
        query_df,
        shot_df,
        query_to_shot,
        tokenizer,
        prompter,
        decomposer,
        k_shot,
        is_train=False,
        label_stats=None,
    ):
        self.query_df = query_df
        shot_df = shot_df.copy()
        shot_df["raw_ID"] = shot_df["raw_ID"].astype(str)
        self.shot_df = shot_df.set_index("raw_ID")

        self.query_to_shot = query_to_shot
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.decomposer = decomposer
        self.k_shot = k_shot
        self.is_train = is_train

        self.label_stats = label_stats
        self.label_mean = label_stats["mean"] if label_stats else None
        self.label_std = label_stats["std"] if label_stats else None

        self.mol_cache = {}
        self._precompute_all_molecules_text()

        self.data_list = []
        print(f"[Info] Pre-building dataset from {len(self.query_df)} queries...")
        self._build_dataset()
        print(f"[Info] Final Dataset Size: {len(self.data_list)}")

    def _standardize(self, val):
        if self.label_mean is not None and self.label_std is not None:
            return (val - self.label_mean) / self.label_std
        return val

    def _precompute_all_molecules_text(self):
        all_query_ids = set(self.query_df["raw_ID"].astype(str))
        all_shot_ids = set(self.shot_df.index)
        all_ids = all_query_ids.union(all_shot_ids)
        print(f"[Info] Decomposing {len(all_ids)} unique molecules (Text Only)...")

        combined_smiles = {}
        combined_smiles.update(self.shot_df["smiles"].to_dict())
        query_dict = dict(zip(self.query_df["raw_ID"].astype(str), self.query_df["smiles"]))
        combined_smiles.update(query_dict)

        for rid in tqdm(all_ids, desc="Text Gen"):
            smiles = combined_smiles.get(rid)
            if not smiles:
                continue
            result = self.decomposer.decompose(smiles)
            if result is None:
                orig, scaf_s, dec_s = smiles, "None", "None"
            else:
                orig = result["original_smiles"]
                scaf_s = result["scaffold_smiles"]
                if result["decorations"]:
                    dec_list = [r["smiles"] for r in result["decorations"]]
                    dec_s = "[" + ", ".join(dec_list) + "]"
                else:
                    dec_s = "None"
            self.mol_cache[rid] = {"orig": orig, "scaf_s": scaf_s, "dec_s": dec_s}

    def _build_dataset(self):
        for idx in tqdm(range(len(self.query_df)), desc="Tokenizing"):
            row = self.query_df.iloc[idx]
            raw_id = str(row["raw_ID"])
            q_label = row["label"]
            q_smiles = row["smiles"]

            if raw_id not in self.query_to_shot:
                continue
            shot_info = self.query_to_shot[raw_id]
            shot_ids = shot_info["shot_idx_list"][: self.k_shot]
            if any(sid not in self.mol_cache for sid in shot_ids):
                continue
            if raw_id not in self.mol_cache:
                continue

            shots_meta_list = []
            shots_parts = []

            for i, sid in enumerate(shot_ids):
                s_data = self.mol_cache[sid]
                s_label = self.shot_df.loc[sid, "label"]
                shots_meta_list.append({"smiles": s_data["orig"], "label": s_label})

                s_label_std = self._standardize(float(s_label))
                s_label_str = f"{s_label_std:.4f}"
                shots_parts.append(
                    f"### Example {i+1}\n\n"
                    f"- Input: {s_data['orig']}\n"
                    f"[Scaffold]: {s_data['scaf_s']}\n"
                    f"[Decorations]: {s_data['dec_s']}\n"
                    f"- Output:\n"
                    f"[Answer] {s_label_str}\n\n"
                )

            shots_str = "".join(shots_parts)

            q_data = self.mol_cache[raw_id]
            query_struct_str = (
                f"- Input: {q_data['orig']}\n" f"[Scaffold]: {q_data['scaf_s']}\n" f"[Decorations]: {q_data['dec_s']}\n"
            )

            messages = self.prompter.generate(query_struct_str, shots_str, mode="llama")
            prompt_only = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_only += "[Answer]"
            full_prompt = prompt_only

            if self.is_train:
                q_label_std = self._standardize(float(q_label))
                label_str = f" {q_label_std:.4f}"
                full_prompt += f"{label_str}{self.tokenizer.eos_token}"

            enc_full = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc_full.input_ids.squeeze(0)
            if self.is_train:
                enc_prompt = self.tokenizer(prompt_only, return_tensors="pt", add_special_tokens=False)
                source_len = enc_prompt.input_ids.shape[1]
            else:
                source_len = 0

            self.data_list.append(
                {
                    "input_ids": input_ids,
                    "source_len": source_len,
                    "label": q_label,
                    "raw_id": raw_id,
                    "prompt_text": full_prompt,
                    "num_shots": len(shot_ids),
                    "metadata": {"smiles": q_smiles, "label": q_label, "shots": shots_meta_list},
                }
            )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn_text_reg(batch, tokenizer):
    """Left-padding collator for regression (no graphs). q_labels as float32."""
    input_ids_list = [item["input_ids"] for item in batch]
    max_len = max(len(t) for t in input_ids_list)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    raw_ids, q_labels, prompt_texts, batch_metadata = [], [], [], []
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        source_len = item.get("source_len", 0)
        padded_ids[i, -seq_len:] = item["input_ids"]
        attention_mask[i, -seq_len:] = 1
        labels[i, -seq_len:] = item["input_ids"]
        if source_len > 0:
            labels[i, -seq_len : -seq_len + source_len] = -100

        raw_ids.append(item["raw_id"])
        q_labels.append(item["label"])
        prompt_texts.append(item["prompt_text"])
        batch_metadata.append(item["metadata"])

    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "q_labels": torch.tensor(q_labels, dtype=torch.float32),
        "raw_ids": raw_ids,
        "prompt_texts": prompt_texts,
        "metadata": batch_metadata,
    }


# ==========================================
# 3. Train & Eval Loops
# ==========================================
def process_batch(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels_lm = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_lm)
    return outputs.loss


def train_epoch(args, model, dataloader, optimizer, device, accumulation_steps=1):
    model.train()
    logs = {"total": [], "step_total": []}
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
            loss = process_batch(model, batch, device)
            (loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            logs["total"].append(loss.item())
            logs["step_total"].append(loss.item())
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
        "total": safe_mean(logs["total"]),
        "step_history": logs["step_total"],
        "oom_count": oom_count,
        "oom_raw_ids": oom_raw_ids,
        "total_batches": total_len,
    }


def evaluate_reg(args, model, dataloader, tokenizer, terminators, device, label_stats=None):
    """Pure-generative regression evaluation: generate → parse float → de-standardize → metrics."""
    model.eval()
    results = {}
    val_loss = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            loss = process_batch(model, batch, device)
            val_loss.append(loss.item())

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            raw_ids = batch["raw_ids"]
            true_labels = batch["q_labels"].numpy()

            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
                    results[rid] = {"true": 0.0, "pred": None, "text": "", "score": None, "entropy": None}
                continue

            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

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
    return {"loss_total": np.mean(val_loss) if val_loss else 0.0, **eval_metrics}


# ==========================================
# 4. Main
# ==========================================
def main():
    try:
        # Extract --use_lora before parse_args (which uses strict argparse)
        _use_lora = "--use_lora" in sys.argv
        if _use_lora:
            sys.argv.remove("--use_lora")

        args, categories = parse_args()
        args.use_lora = _use_lora

        accumulation_steps = args.grad_accum_steps
        device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
        print(f"Using device: {device}")
        print(f"Mode: {'LoRA' if args.use_lora else 'Frozen'}")

        # Paths
        lora_tag = "text_only_lora_reg" if args.use_lora else "text_only_frozen_reg"
        sub_path_config = f"temp={args.temperature}/eps={args.eps}/{lora_tag}"
        base_ckpt_path = os.path.join(args.ckpt_root, args.dataset, sub_path_config)
        base_save_path = os.path.join(args.save_root, args.dataset, sub_path_config)

        start_epoch = getattr(args, "start_epoch", 1)
        if start_epoch > 1:
            if not os.path.exists(base_save_path):
                raise FileNotFoundError(f"Cannot resume: {base_save_path} not found.")
            timestamps = sorted(
                [d for d in os.listdir(base_save_path) if os.path.isdir(os.path.join(base_save_path, d))]
            )
            if not timestamps:
                raise FileNotFoundError(f"Cannot resume: no experiment dirs in {base_save_path}")
            args.now = timestamps[-1]
        else:
            args.now = datetime.now().strftime("%Y%m%d_%H%M%S")

        args.ckpt_dir = os.path.join(base_ckpt_path, args.now)
        args.save_dir = os.path.join(base_save_path, args.now)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)

        embed_tag = f"molt5-{args.embedding_type}" if args.embedding_type != "fp" else "fp"
        method_tag = gen_method_tag(args)
        args.exp_tag = f"{args.dataset} - {embed_tag}_{method_tag}_k={args.k}_TEXT_REG"
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

        valid_shots, test_shots = load_shot(args, train_df, valid_df, test_df)

        # --- Model ---
        args.mode = "llama"
        tokenizer, model, terminators = setup_model_and_tokenizer(args)

        # --- Prompter & Datasets ---
        with open(args.inst_json, "r") as f:
            full_inst = json.load(f)
            instruction = full_inst["full"][args.dataset] if args.use_inst else full_inst["short"][args.dataset]

        prompter = TextScaffoldPrompter(args.dataset, instruction)
        decomposer = MoleculeDecomposer()
        mk_loader = lambda ds, shuffle: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_fn_text_reg(b, tokenizer),
            num_workers=4,
        )

        if args.use_lora and args.epochs > 0:
            # ---------- Training path ----------
            train_ds = TextOnlyRegDataset(
                valid_df,
                train_df,
                valid_shots,
                tokenizer,
                prompter,
                decomposer,
                args.k,
                is_train=True,
                label_stats=label_stats,
            )
            test_ds = TextOnlyRegDataset(
                test_df,
                train_df,
                test_shots,
                tokenizer,
                prompter,
                decomposer,
                args.k,
                is_train=False,
                label_stats=label_stats,
            )

            train_loader = mk_loader(train_ds, shuffle=True)
            test_loader = mk_loader(test_ds, shuffle=False)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

            history = {
                "train_loss_epoch": [],
                "train_loss_step": [],
                "valid_loss": [],
                "valid_mse": [],
                "valid_rmse": [],
                "oom_count": [],
                "oom_raw_ids": [],
            }

            if start_epoch > 1:
                ckpt_file = os.path.join(args.ckpt_dir, f"epoch_{start_epoch - 1:02}.pt")
                if os.path.exists(ckpt_file):
                    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
                    set_peft_model_state_dict(model, ckpt["model_state_dict"], adapter_name="default")
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if "history" in ckpt:
                        history = ckpt["history"]
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

            print(f"\n>>> Start LoRA Training (Text-Only Reg) from Epoch {start_epoch}...")
            for epoch in range(start_epoch, args.epochs + 1):
                print(f"\n[Epoch {epoch}/{args.epochs}]")
                train_metrics = train_epoch(args, model, train_loader, optimizer, device, accumulation_steps)
                print(f"  Train Loss: {train_metrics['total']:.4f} (OOM: {train_metrics['oom_count']})")

                history["train_loss_epoch"].append(train_metrics["total"])
                history["train_loss_step"].extend(train_metrics["step_history"])
                history["oom_count"].append(train_metrics["oom_count"])
                history["oom_raw_ids"].append(train_metrics["oom_raw_ids"])
                history["valid_loss"].append(None)
                history["valid_mse"].append(None)
                history["valid_rmse"].append(None)

                peft_sd = get_peft_model_state_dict(model, state_dict=None, adapter_name="default")
                ckpt_payload = {
                    "epoch": epoch,
                    "model_state_dict": peft_sd,
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

                save_json(history, os.path.join(args.save_dir, "result.json"))
                try:
                    plot_history(history, args.exp_tag, args.save_dir)
                except Exception as e:
                    print(f"[Warning] Plotting failed: {e}")

            print("\n>>> Final Evaluation...")
            test_metrics = evaluate_reg(args, model, test_loader, tokenizer, terminators, device, label_stats)
            history["test_results"] = test_metrics
            save_json(history, os.path.join(args.save_dir, "result.json"))
            save_json(test_metrics, os.path.join(args.save_dir, "result_test.json"))
            print("Done.")
        else:
            # ---------- Inference-only path (Frozen LLM) ----------
            test_ds = TextOnlyRegDataset(
                test_df,
                train_df,
                test_shots,
                tokenizer,
                prompter,
                decomposer,
                args.k,
                is_train=False,
                label_stats=label_stats,
            )
            test_loader = mk_loader(test_ds, shuffle=False)

            print("\n>>> Start Text-Only Regression Evaluation (Frozen)...")
            test_metrics = evaluate_reg(args, model, test_loader, tokenizer, terminators, device, label_stats)

            save_json(test_metrics, os.path.join(args.save_dir, "result_text_only_reg.json"))
            print("Done.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
