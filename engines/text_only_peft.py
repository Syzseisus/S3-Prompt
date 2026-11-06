import os
import sys
import json
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from rdkit import Chem
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import TaskType, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

# Local Imports
from utils.evaluation import extract_pred, eval_regression, eval_classification
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from utils.dataset_loader import load_dataset
from utils.prompter import Prompter as BasePrompter
from utils.scaffold_decompose import MoleculeDecomposer
from utils.constants import DATA_ROOT, REGRESSION, CLASSIFICATION
from utils.helper import log_df, load_json, save_json, parse_args, format_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    # no need special tokens

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
    valid_shots, test_shots = load_shot_wrt_method(args, train_df, valid_df, test_df)
    return valid_shots, test_shots


def plot_history(history, tag, save_dir):
    epochs = range(1, len(history["train_loss_epoch"]) + 1)

    # 1. Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss_epoch"], "b-", label="Train Loss")

    if "valid_loss" in history:
        valid_loss_arr = np.array(history["valid_loss"], dtype=np.float64)
        mask = ~np.isnan(valid_loss_arr)
        if mask.any():
            valid_epochs = np.array(epochs)[mask]
            valid_values = valid_loss_arr[mask]
            plt.plot(valid_epochs, valid_values, "r.-", label="Valid Loss")

    plt.title(f"Training Loss\n{tag}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()


# ==========================================
# 2. Dataset & Prompter (Text Only with Train Support)
# ==========================================
class TextScaffoldPrompter(BasePrompter):
    """
    기존 Prompter와 동일한 포맷을 유지하되,
    System Prompt에서 'graph embeddings'가 아닌 'SMILES strings'라고 명시.
    """

    def generate(self, query_formatted_str, shots_str=None, mode="llama"):
        # 1. System Prompt (Modified)
        system_content = (
            "## System\n\n"
            f"You are an expert AI chemist specializing in Structure-Activity Relationship (SAR) analysis for the {self.dataset} task.\n"
            "You will be provided with a molecule's full structure as a **SMILES string**.\n"
            # [MODIFIED] Graph Embeddings -> SMILES strings
            "Crucially, the molecule's core [Scaffold] and [Decorations] (side chains) are provided as **SMILES strings** to explicitly highlight structural components.\n"
            "Your task is to:\n"
            "1. Interpret the full SMILES string.\n"
            "2. Integrate the explicit structural features provided in [Scaffold] and [Decorations].\n"
            "3. Analyze how these structural components influence the target property to predict the label.\n"
            "Follow the Task Instruction CAREFULLY.\n"
            "Your output MUST contain exactly two sections: [Answer] and [Reasoning].\n"
            "No additional text is allowed.\n"
        )

        # 2. User Prompt
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
                # [MODIFIED] Latent Feature -> Explicit Feature
                "2. Explicit Feature Integration: Pay close attention to the `[Scaffold]` and `[Decorations]` fields.\n"
                "These provide decomposed structural views of the molecule.\n"
                "3. SAR Synthesis: Combine the biological context (from the Domain Context section) with the structural features to infer the label.\n"
                "4. Output Format:\n"
                f"    - [Answer] — {self.strict_answer}\n"
                "    - [Reasoning] — Provide step-by-step reasoning. Keep it concise (ideally under 150 words).\n"
                "No additional text before and after these sections is allowed.\n"
            )

        # Examples with proper headers
        if shots_str:
            user_parts.append("## Examples\n\n" + shots_str.strip())

        # Query
        user_parts.append(f"## Query\n\n{query_formatted_str.strip()}\n- Output:")

        user_content = "\n\n".join(user_parts)

        if mode == "llama":
            return [
                {"role": "system", "content": system_content.strip()},
                {"role": "user", "content": user_content.strip()},
            ]
        else:
            raise ValueError(f"Unsupported mode: {mode}")


class TextOnlyDataset(Dataset):
    """
    S3PromptDataset (utils/dataset.py)와 구조적으로 동일하게 동작합니다.
    차이점:
        - Graph 객체(PyG)를 생성하지 않음.
        - <|scaffold|> 토큰 대신 실제 분해된 SMILES 문자열을 넣음.
    """

    def __init__(
        self, query_df, shot_df, query_to_shot, tokenizer, prompter, decomposer, k_shot, is_train=True, is_cls=True
    ):
        self.query_df = query_df
        # Shot lookup 최적화: 문자열 인덱스 변환
        shot_df["raw_ID"] = shot_df["raw_ID"].astype(str)
        self.shot_df = shot_df.set_index("raw_ID")

        self.query_to_shot = query_to_shot
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.decomposer = decomposer
        self.k_shot = k_shot
        self.is_train = is_train
        self.is_cls = is_cls

        # [Optimization 1] 모든 분자(Query + Shot) 미리 분해하여 텍스트 캐싱
        self.mol_cache = {}
        self._precompute_all_molecules_text()

        # [Optimization 2] 유효한 샘플만 리스트에 미리 저장 (Pre-build)
        # dict 대신 list를 쓰되, 유효성 검사를 통과한 데이터만 append 하므로 안전합니다.
        self.data_list = []
        print(f"[Info] Pre-building dataset from {len(self.query_df)} queries...")
        self._build_dataset()
        print(f"[Info] Final Dataset Size: {len(self.data_list)}")

    def _precompute_all_molecules_text(self):
        """
        데이터셋에 존재하는 모든 Unique ID(Query & Shot)에 대해
        RDKit Decomposition 및 Text 변환을 수행하고 캐시에 저장합니다.
        """
        # 1. 분해해야 할 모든 ID 수집
        all_query_ids = set(self.query_df["raw_ID"].astype(str))
        all_shot_ids = set(self.shot_df.index)
        all_ids = all_query_ids.union(all_shot_ids)

        print(f"[Info] Decomposing {len(all_ids)} unique molecules (Text Only)...")

        # 2. DataFrame 통합 (빠른 조회를 위해)
        combined_smiles = {}
        # Shot 우선
        combined_smiles.update(self.shot_df["smiles"].to_dict())
        # Query로 덮어쓰기
        query_dict = dict(zip(self.query_df["raw_ID"].astype(str), self.query_df["smiles"]))
        combined_smiles.update(query_dict)

        for rid in tqdm(all_ids, desc="Text Gen"):
            smiles = combined_smiles.get(rid)
            if not smiles:
                continue

            # Decompose
            result = self.decomposer.decompose(smiles)

            if result is None:
                orig, scaf_s, dec_s = smiles, "None", "None"
                # don't use graph
            else:
                orig = result["original_smiles"]
                # Scaffold & Side Chain Text Logic
                scaf_s = result["scaffold_smiles"]

                if result["decorations"]:
                    # Clean decorations from list of dicts
                    dec_list = [r["smiles"] for r in result["decorations"]]
                    dec_s = "[" + ", ".join(dec_list) + "]"
                else:
                    dec_s = "None"

            self.mol_cache[rid] = {
                "orig": orig,
                "scaf_s": scaf_s,
                "dec_s": dec_s,
            }

    def _build_dataset(self):
        for idx in tqdm(range(len(self.query_df)), desc="Tokenizing"):
            row = self.query_df.iloc[idx]
            raw_id = str(row["raw_ID"])
            q_label = row["label"]
            q_smiles = row["smiles"]

            # 1. Retrieve Shots (유효성 검사)
            if raw_id not in self.query_to_shot:
                continue

            shot_info = self.query_to_shot[raw_id]
            shot_ids = shot_info["shot_idx_list"][: self.k_shot]

            # Shot이 부족하거나, Cache에 없는 ID가 있으면 스킵할 수도 있음 (Safety)
            if any(sid not in self.mol_cache for sid in shot_ids):
                continue
            if raw_id not in self.mol_cache:
                continue

            # don't use graphs
            shots_meta_list = []

            shots_parts = ["## Examples\n\n"]

            # 2. Build Shot String with Headers
            for i, sid in enumerate(shot_ids):
                s_data = self.mol_cache[sid]
                s_label = self.shot_df.loc[sid, "label"]

                shots_meta_list.append({"smiles": s_data["orig"], "label": s_label})

                s_label_str = "Yes" if s_label == 1 else "No"
                shot_text = (
                    f"### Example {i+1}\n\n"
                    f"- Input: {s_data['orig']}\n"
                    f"[Scaffold]: {s_data['scaf_s']}\n"
                    f"[Decorations]: {s_data['dec_s']}\n"
                    f"- Output:\n"
                    f"[Answer] {s_label_str}\n\n"
                )
                shots_parts.append(shot_text)

                # don't use graphs

            shots_str = "".join(shots_parts)

            # 3. Build Query
            q_data = self.mol_cache[raw_id]
            query_struct_str = (
                f"- Input: {q_data['orig']}\n" f"[Scaffold]: {q_data['scaf_s']}\n" f"[Decorations]: {q_data['dec_s']}\n"
            )

            # don't use graphs

            # 4. Tokenize
            messages = self.prompter.generate(query_struct_str, shots_str, mode="llama")
            prompt_only = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_only += "[Answer]"

            full_prompt = prompt_only

            # 5. Add Groundtruth if is train
            if self.is_train:
                if self.is_cls:
                    label_str = "Yes" if q_label == 1 else "No"
                else:
                    label_str = f"{q_label}"
                full_prompt += f"{label_str}{self.tokenizer.eos_token}"

            # 6. Encoding
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
                    "metadata": {
                        "smiles": q_smiles,
                        "label": q_label,
                        "shots": shots_meta_list,
                    },
                }
            )

    def __len__(self):
        # 유효한 데이터의 개수만 반환
        return len(self.data_list)

    def __getitem__(self, idx):
        # DataLoader가 0 ~ len-1 사이의 idx를 주면 바로 꺼내줌 (O(1))
        return self.data_list[idx]


def collate_fn_text(batch, tokenizer):
    """Text only collator (No Graphs)"""
    input_ids_list = [item["input_ids"] for item in batch]
    max_len = max(len(t) for t in input_ids_list)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    # don't use graphs
    raw_ids = []
    q_labels = []
    prompt_texts = []

    batch_metadata = []
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        source_len = item.get("source_len", 0)
        # Left Padding
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
        "q_labels": q_labels,
        "labels": labels,
        "raw_ids": raw_ids,
        "prompt_texts": prompt_texts,
        "metadata": batch_metadata,
    }


# ==========================================
# 3. Train & Eval Loops
# ==========================================
def process_batch(model, batch, device):
    input_ids = batch["input_ids"].to(device)

    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"].to(device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # Return: ce_loss
    loss = outputs.loss
    return loss


def train_epoch(args, model, dataloader, optimizer, device, accumulation_steps=1):
    model.train()
    logs = {"total": [], "step_total": []}
    optimizer.zero_grad()

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
        try:
            # 1. Forward
            loss = process_batch(model, batch, device)

            # 2. Backward
            loss_norm = loss / accumulation_steps
            loss_norm.backward()

            # 3. Step
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Logging
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
                    raise RuntimeError("Training Terminated due to excessive OOM.")
                continue
            else:
                raise e

    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.empty_cache()

    def safe_mean(l):
        return np.mean(l) if len(l) > 0 else 0.0

    return {
        "total": safe_mean(logs["total"]),
        "step_history": logs["step_total"],
        "step_history": logs["step_total"],
        "oom_count": oom_count,
        "oom_raw_ids": oom_raw_ids,
        "total_batches": total_len,
    }


def evaluate_text_only(args, model, dataloader, tokenizer, terminators, device, dataset_type):
    print("Just ignore right padding warning ... It can't be helped ...")

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

    val_loss = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Loss
            loss = process_batch(model, batch, device)
            val_loss.append(loss.item())

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            raw_ids = batch["raw_ids"]
            true_labels = np.array(batch["q_labels"])
            input_len = input_ids.shape[1]

            # Generation
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
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
            check_input = torch.cat([input_ids, first_gen_token], dim=1)
            outputs = model(check_input)
            target_logits = outputs.logits[:, input_len - 1, :]

            score_yes = torch.logsumexp(target_logits[:, yes_ids_list], dim=1)
            score_no = torch.logsumexp(target_logits[:, no_ids_list], dim=1)
            probs = torch.softmax(torch.stack([score_no, score_yes], dim=1), dim=1)
            prob_yes = probs[:, 1].float().cpu().numpy()

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

    loss_total = np.mean(val_loss)

    return {"loss_total": loss_total, **eval_metrics}


# ==========================================
# 4. Main Execution
# ==========================================
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
        sub_path_config = f"temp={args.temperature}/eps={args.eps}/text_only_peft"
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
        args.exp_tag = f"{args.dataset} - {embed_tag}_{method_tag}_k={args.k}_TEXT_PEFT"

        categories.setdefault("Experiment Settings", []).extend(["now", "exp_tag", "ckpt_dir", "save_dir"])
        print(format_args(args, categories))
        save_json(vars(args), os.path.join(args.save_dir, "args.json"))

        # 1. Dataset Loading
        print("Loading Dataset...")
        train_df, valid_df, test_df = load_dataset(args.dataset, args.embedding_type, verbose=True)
        valid_shots, test_shots = load_shot(args, train_df, valid_df, test_df)

        # 3. Model Setup (PEFT)
        args.mode = "llama"
        tokenizer, model, terminators = setup_model_and_tokenizer(args)

        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # fmt: off
        history = {
            "train_loss_epoch": [], "train_loss_step": [],
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
                set_peft_model_state_dict(model, checkpoint["model_state_dict"], adapter_name="default")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # Load History if available
                if "history" in checkpoint:
                    history = checkpoint["history"]
            else:
                raise FileNotFoundError(f"Checkpoint for epoch {resume_epoch} not found at {ckpt_file}")

        # 4. Prompter & Dataset
        with open(args.inst_json, "r") as f:
            full_inst = json.load(f)
            instruction = full_inst["full"][args.dataset] if args.use_inst else full_inst["short"][args.dataset]

        prompter = TextScaffoldPrompter(args.dataset, instruction)
        decomposer = MoleculeDecomposer()

        # Training Dataset (is_train=True, Appends label)
        is_cls = args.dataset in CLASSIFICATION
        train_ds = TextOnlyDataset(valid_df, train_df, valid_shots, tokenizer, prompter, decomposer, args.k, is_train=True, is_cls=is_cls)  # fmt: skip
        test_ds = TextOnlyDataset(test_df, train_df, test_shots, tokenizer, prompter, decomposer, args.k, is_train=False, is_cls=is_cls)  # fmt: skip

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn_text(b, tokenizer), num_workers=4)  # fmt: skip
        eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn_text(b, tokenizer), num_workers=4)  # fmt: skip
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn_text(b, tokenizer), num_workers=4)  # fmt: skip

        print(f">>> Start PEFT Training (Text-Only) from Epoch {args.start_epoch}...")
        for epoch in range(args.start_epoch, args.epochs + 1):
            print(f"\n[Epoch {epoch}/{args.epochs}]")

            # Train
            train_metrics = train_epoch(args, model, train_loader, optimizer, device, args.grad_accum_steps)
            print(f"  Train Loss: {train_metrics['total']:.4f} (OOM: {train_metrics['oom_count']})")

            oom_cnt = train_metrics["oom_count"]
            oom_ids = train_metrics["oom_raw_ids"]
            total_b = train_metrics["total_batches"]
            oom_rate = (oom_cnt / total_b) * 100 if total_b > 0 else 0
            print(f"  [OOM Stats] {oom_cnt}/{total_b} batches skipped ({oom_rate:.2f}%)")
            if oom_cnt > 0:
                print(f"  [OOM Samples] First 3 IDs: {oom_ids[:3]}...")

            history["train_loss_epoch"].append(train_metrics["total"])
            history["train_loss_step"].extend(train_metrics["step_history"])
            history["oom_count"].append(oom_cnt)
            history["oom_raw_ids"].append(oom_ids)

            # Eval
            # TODO: train / valid 구분하기
            # if epoch % args.eval_epoch == 0 or epoch == args.epochs:
            if False:
                print(f"  >>> Running Evaluation at Epoch {epoch}...")
                val_metrics = evaluate_text_only(args, model, eval_loader, tokenizer, terminators, device, args.dataset)
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

            peft_state_dict = get_peft_model_state_dict(model, state_dict=None, adapter_name="default")
            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": peft_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }
            curr_ckpt = os.path.join(args.ckpt_dir, f"epoch_{epoch:02}.pt")
            torch.save(ckpt_payload, curr_ckpt)
            print(f"  [Info] Current checkpoint saved: {curr_ckpt} ({datetime.now().strftime('%H:%M:%S')})")
            if epoch > 1:
                prev_ckpt = os.path.join(args.ckpt_dir, f"epoch_{epoch-1:02}.pt")
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
                    print(f"  [Info] Previous checkpoint removed: {prev_ckpt} ({datetime.now().strftime('%H:%M:%S')})")

            # JSON Saving
            result_json_path = os.path.join(args.save_dir, "result.json")
            save_json(history, result_json_path)
            try:
                plot_history(history, args.exp_tag, args.save_dir)
            except Exception as e:
                # 혹시 plot 함수가 None을 처리 못할 경우를 대비해 로그만 남기고 학습 계속 진행
                print(f"[Warning] Plotting failed: {e}")

        # Final Eval
        print("\n>>> Final Evaluation...")
        test_metrics = evaluate_text_only(args, model, test_loader, tokenizer, terminators, device, args.dataset)
        history["test_results"] = test_metrics
        save_json(history, result_json_path)
        print("Done.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
