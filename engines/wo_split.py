import os
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from rdkit import Chem
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Local Imports (engines/s3prompt_w_label.py와 동일)
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
    Text-Only Inference 모드이므로 Gradient Checkpointing 및 Special Token Resize는 수행하지 않음.
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
            # k개만 자르기
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
    # engines/s3prompt_w_label.py와 동일한 로직 구조 유지
    valid_shots, test_shots = load_shot_wrt_method(args, train_df, valid_df, test_df)
    return valid_shots, test_shots


# ==========================================
# 2. Dataset & Prompter (Text Only Version)
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
            "Your task is to:\n"
            "1. Interpret the full SMILES string.\n"
            "2. Predict the label.\n"
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
                "1. Global Structure: Analyze the 'Input' SMILES string to identify key functional groups and the core structure.\n"
                "2. SAR Synthesis: Combine the biological context (from the Domain Context section) with the structural features to infer the label.\n"
                "3. Output Format:\n"
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

    def __init__(self, query_df, shot_df, query_to_shot, tokenizer, prompter, decomposer, k_shot):
        self.query_df = query_df
        # Shot lookup 최적화: 문자열 인덱스 변환
        shot_df["raw_ID"] = shot_df["raw_ID"].astype(str)
        self.shot_df = shot_df.set_index("raw_ID")

        self.query_to_shot = query_to_shot
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.decomposer = decomposer
        self.k_shot = k_shot

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
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_prompt += "[Answer]"

            enc = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc.input_ids.squeeze(0)

            self.data_list.append(
                {
                    "input_ids": input_ids,
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

    # don't use graphs
    raw_ids = []
    labels = []
    prompt_texts = []

    batch_metadata = []
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        # Left Padding
        padded_ids[i, -seq_len:] = item["input_ids"]
        attention_mask[i, -seq_len:] = 1

        raw_ids.append(item["raw_id"])
        labels.append(item["label"])
        prompt_texts.append(item["prompt_text"])

        batch_metadata.append(item["metadata"])

    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels),
        "raw_ids": raw_ids,
        "prompt_texts": prompt_texts,
        "metadata": batch_metadata,
    }


# ==========================================
# 3. Evaluation Function (Simplified for Text)
# ==========================================
def evaluate_text_only(args, model, dataloader, tokenizer, terminators, device, dataset_type):
    model.eval()
    results = {}

    # Candidates for Scoring
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

    print(f"Start Inference on {device}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            raw_ids = batch["raw_ids"]
            true_labels = np.array(batch["labels"])
            input_len = input_ids.shape[1]

            # 1. Generation
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

                if i == 0 and len(results) == 1:
                    print(f"\n[Preview ID: {rid}]")
                    print(f"Pred: {pred} (Score: {prob_yes[i]:.4f}) | True: {true_labels[i]}")
                    print(f"Generated: {pred_text}...")

    # Metrics
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

    return eval_metrics


# ==========================================
# 4. Main Execution
# ==========================================
def main():
    try:
        args, categories = parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
        print(f"Using device: {device}")

        # Path Setup (Same as S3-Prompt)
        base_save_path = os.path.join(args.save_root, args.dataset, f"temp={args.temperature}/eps={args.eps}")
        args.now = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(base_save_path, args.now)
        os.makedirs(args.save_dir, exist_ok=True)

        embed_tag = f"molt5-{args.embedding_type}" if args.embedding_type != "fp" else "fp"
        method_tag = gen_method_tag(args)
        args.exp_tag = f"{args.dataset} - {embed_tag}_{method_tag}_k={args.k}"

        categories.setdefault("Experiment Settings", []).extend(["now", "exp_tag", "ckpt_dir", "save_dir"])
        print(format_args(args, categories))
        save_json(vars(args), os.path.join(args.save_dir, "args.json"))

        # 1. Dataset Loading
        print("Loading Dataset...")
        train_df, valid_df, test_df = load_dataset(args.dataset, args.embedding_type, verbose=True)

        # 2. Shot Loading (Reusing logic)
        valid_shots, test_shots = load_shot(args, train_df, valid_df, test_df)

        # 3. Model Setup (Llama Pure)
        args.mode = "llama"
        tokenizer, model, terminators = setup_model_and_tokenizer(args)

        # 4. Prompter & Dataset
        with open(args.inst_json, "r") as f:
            full_inst = json.load(f)
            instruction = full_inst["full"][args.dataset] if args.use_inst else full_inst["short"][args.dataset]

        prompter = TextScaffoldPrompter(args.dataset, instruction)
        decomposer = MoleculeDecomposer()

        # Use TextOnlyDataset
        test_ds = TextOnlyDataset(test_df, train_df, test_shots, tokenizer, prompter, decomposer, args.k)

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn_text(b, tokenizer),
            num_workers=4,
        )

        # 5. Evaluation
        print("\n>>> Start Text-Only Evaluation...")
        test_metrics = evaluate_text_only(args, model, test_loader, tokenizer, terminators, device, args.dataset)

        print("\n>>> Results:")
        results = test_metrics["results"]
        random_ids = random.sample(list(results.keys()), 3)
        sample = {k: results[k] for k in random_ids}
        sample.update({k: test_metrics[k] for k in ["auc", "acc", "f1", "mcc"]})
        print(json.dumps(sample, indent=4, ensure_ascii=False, default=str))

        save_path = os.path.join(args.save_dir, "result_text_only.json")
        save_json(test_metrics, save_path)
        print(f"Done. Saved to {save_path}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
