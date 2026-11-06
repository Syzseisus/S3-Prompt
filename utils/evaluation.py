import os

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_CPP_LOG_LEVEL"] = "NONE"
import gc
import re
import json
import math
import time
import pickle
import random
import string
import asyncio
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from datetime import datetime

import torch

# from google import genai
import torch.nn.functional as F

# from google.genai import types
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from torch.nn.functional import cosine_similarity
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from sklearn.metrics import (
    f1_score,
    r2_score,
    roc_auc_score,
    accuracy_score,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
)

# [Updated] Ensure utils.prompter contains the NEW Prompter class we designed
from utils.prompter import Prompter
from utils.scaffold_decompose import MoleculeDecomposer
from utils.dataset_loader import load_dataset, get_molt5_model
from utils.constants import DATA_ROOT, REGRESSION, CLASSIFICATION, POSSIBLE_DATASETS
from utils.helper import log_df, load_json, save_json, head_print, parse_args, format_args, print_debug
from utils.clustering import (
    find_cluster,
    load_cluster,
    pool_clustering,
    get_scaffold_smiles,
    retrieve_whole_query,
)

RDLogger.DisableLog("rdApp.*")


# ===================================
# 0. Shot Selection
# ===================================
def select_shot(args, row, smiles_dict, label_dict, emb_dict, query_to_shot, decomposer=None):
    """
    Select k-shot examples and return them as a formatted string.
    [Updated] Now omits reasoning to prevent parroting, matching the paper's design.
    """
    if args.method == "zero_shot":
        if args.debug:
            print_debug("Strategy: Zero-Shot")
            print(f"No shots selected", "#" * 70, sep="")
        return None

    if args.method == "random":
        shot_idxs = random.sample(list(smiles_dict.keys()), k=args.k)
        if args.debug:
            print_debug("Strategy: Random")
            print(f"Selected shot indices: {shot_idxs}\n", "#" * 70, sep="")
    elif args.method == "topk":
        if args.embedding_type.lower() in {"small", "base", "large"}:
            shot_idxs = topk_similarity_molt5(row["embed"], emb_dict, k=args.k)
        elif args.embedding_type.lower() == "fp":
            shot_idxs = topk_similarity_fp(row["embed"], emb_dict, k=args.k)
        if args.debug:
            print_debug("Strategy: Top-K")
            print(f"Selected shot indices: {shot_idxs}\n", "#" * 70, sep="")
    elif args.method == "pool_clustering":
        # Use precomputed cluster-based shots
        raw_id = str(row["raw_ID"])
        query_info = query_to_shot[raw_id]
        assert row["smiles"] == query_info["query_smiles"], "SMILES mismatch"
        shot_idxs = query_info["shot_idx_list"]
        if isinstance(shot_idxs, pd.Series):
            shot_idxs = shot_idxs.tolist()
        if args.debug:
            print_debug(f"Strategy: Pool Clustering - {args.sample_strategy}-sampling")
            print(f"Selected shot indices: {shot_idxs}\n", "#" * 70, sep="")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # --- Construct Few-shot String ---
    # [Updated] Follows the format:
    # ## Examples
    # ### Example N
    # - Input: ...
    # - Output:
    # [Answer] ...
    shots = "## Examples\n\n"
    for i, idx in enumerate(shot_idxs, 1):
        smiles_a = smiles_dict[str(idx)]
        label_a = label_dict[str(idx)]
        if args.dataset in CLASSIFICATION:
            label_a = "Yes" if label_a == 1 else "No"

        # NOTE: Reasoning omitted to avoid parroting
        if args.split_scaffold:
            shots += decomposer.format_prompt(i, smiles_a, label_a)
        else:
            shots += f"### Example {i}\n" f"- Input: {smiles_a}\n" f"- Output:\n" f"[Answer] {label_a}\n\n"

    return shots


def load_query_to_shot(args):
    if args.method == "pool_clustering":
        # [Updated] Path construction matching the new cluster code
        embed_tag = f"molt5-{args.embedding_type}" if args.embedding_type != "fp" else "fp"
        eps_tag = f"eps={args.eps}" if args.eps else "no_merge"
        tag = f"{embed_tag}_{eps_tag}_{args.sample_strategy}-sampling"

        # Cache dir is now: root/dataset/tag/
        cache_dir = os.path.join(args.cache_root, args.dataset, tag)

        # File name is simple: query_to_shot_k={k}.json
        query_shot_path = os.path.join(cache_dir, f"query_to_shot_k={args.k}.json")

        if os.path.exists(query_shot_path):
            print(f"Load cache from {query_shot_path}")
            query_to_shot = load_json(query_shot_path)
        else:
            raise FileNotFoundError(
                f"No cached query-to-shot mapping found in {query_shot_path}. Run clustering and shot selection first with `cluster.py`."
            )
        return query_to_shot
    else:
        return None


def topk_similarity_fp(query_fp, emb_dict, k=8):
    sims = {idx: DataStructs.TanimotoSimilarity(query_fp, fp) for idx, fp in emb_dict.items()}
    topk_idxs = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)[:k]
    return topk_idxs


def topk_similarity_molt5(query_emb, emb_dict, k=8):
    if query_emb.ndim == 1:
        query_emb_expand = query_emb.unsqueeze(0)
    embs = torch.stack([emb for emb in emb_dict.values()], dim=0)
    keys = list(emb_dict.keys())
    sims = cosine_similarity(query_emb_expand, embs)
    topk_vals, topk_idxs = torch.topk(sims, k=k, largest=True)
    topk_idxs = [keys[idx] for idx in topk_idxs.squeeze().tolist()]
    return topk_idxs


# ===================================
# 1. Evaluation
# ===================================
def extract_yes_no(text):
    text = text.strip().lower()
    # Updated extraction logic based on [Answer] format
    # The model should output: [Answer] Yes/No
    # But we check robustly
    if "[answer]" in text:
        ans_part = text.split("[answer]")[-1]
    else:
        ans_part = ""

    if "yes" in ans_part:
        return 1
    elif "no" in ans_part:
        return 0
    return None


def extract_float(text):
    text = text.strip().lower()
    # Updated to handle [Answer] tag
    if "[answer]" in text:
        ans_part = text.split("[answer]")[-1]
    else:
        ans_part = text

    # Regex for float
    matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", ans_part)
    # matches = re.findall(r"^\s*([-+]?\d*\.\d+|[-+]?\d+)", ans_part)  # 맨처음 숫자만 Parsing
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


def extract_pred(text, dataset):
    pred = None
    if dataset in CLASSIFICATION:
        pred = extract_yes_no(text)
    elif dataset in REGRESSION:
        pred = extract_float(text)
    return pred


def eval_classification(result_log):
    true_list = result_log["true_list"]
    pred_list = result_log["pred_list"]
    score_list = result_log["score_list"]  # prob list

    result_log["acc"] = accuracy_score(true_list, pred_list)
    result_log["f1"] = f1_score(true_list, pred_list)
    result_log["mcc"] = matthews_corrcoef(true_list, pred_list)
    try:
        result_log["auc"] = roc_auc_score(true_list, score_list)
    except ValueError:
        result_log["auc"] = 0.0  # Handle case with only one class

    print(f"[EVAL] Total: {result_log['total']}, Invalid: {result_log['fail_count']}")
    print(f"  ACC: {result_log['acc']:.4f}")
    print(f"  F1 : {result_log['f1']:.4f}")
    print(f"  MCC: {result_log['mcc']:.4f}")
    print(f"  AUC: {result_log['auc']:.4f}")

    return result_log


def eval_regression(result_log):
    true_list = result_log["true_list"]
    pred_list = result_log["pred_list"]

    mse = mean_squared_error(true_list, pred_list)
    result_log["mae"] = mean_absolute_error(true_list, pred_list)
    result_log["mse"] = mse
    result_log["rmse"] = math.sqrt(mse)
    result_log["r2"] = r2_score(true_list, pred_list)

    fail_ratio = result_log["fail_count"] / result_log["total"] if result_log["total"] > 0 else 0
    result_log["fail_ratio"] = fail_ratio

    print(f"[EVAL] Total: {result_log['total']}, Invalid: {result_log['fail_count']} ({fail_ratio*100:.1f}%)")
    if fail_ratio > 0.1:
        print(
            f"  [WARNING] High parse failure rate ({fail_ratio*100:.1f}%). Check prompt constraints or generation config."
        )
    print(f"  MAE : {result_log['mae']:.4f}")
    print(f"  MSE : {result_log['mse']:.4f}")
    print(f"  RMSE: {result_log['rmse']:.4f}")
    print(f"  R2  : {result_log['r2']:.4f}")

    return result_log


def evaluate(results, fail_set, save_dir, dataset, debug=None):
    result_log = {}
    valid_results = [res for row, res in results.items() if row not in fail_set]

    result_log["true_list"] = [result["true"] for result in valid_results]
    result_log["pred_list"] = [result["pred"] for result in valid_results]

    # Collect Scores if available
    # Check if 'score' key exists in the first valid result
    if valid_results and "score" in valid_results[0]:
        result_log["score_list"] = [result["score"] for result in valid_results]

    result_log["results"] = results
    result_log["total"] = len(results)
    result_log["fail_set"] = list(fail_set)
    result_log["fail_count"] = len(fail_set)

    entropy_list = [res["entropy"] for res in valid_results if res.get("entropy") is not None]
    if entropy_list:
        ent_mean = np.mean(entropy_list)
        ent_std = np.std(entropy_list)
        result_log["entropy_mean"] = ent_mean
        result_log["entropy_std"] = ent_std
        entropy_msg = f"{ent_mean:.4f} ± {ent_std:.4f}"
    else:
        entropy_msg = "N/A"

    print("=" * 70)
    print(f"[EVAL] Dataset: {dataset}")
    print(f"  Total: {result_log['total']}, Failed: {result_log['fail_count']}")

    if dataset in CLASSIFICATION:
        result_log = eval_classification(result_log)  # Updated to use score_list
    elif dataset in REGRESSION:
        result_log = eval_regression(result_log)

    print(f"  Entropy: {entropy_msg}")
    print("=" * 70)

    save_path = os.path.join(save_dir, "result.json")
    save_json(result_log, save_path)
    print(f"Completely saved to {save_path}")

    if debug:
        valid_keys = [k for k in results.keys() if k not in fail_set]
        if valid_keys:
            sample_idx = random.choice(valid_keys)
            sample_result = results[sample_idx]
            print_debug(f"print sample idx={sample_idx}")
            sample_prompt = sample_result["prompt"]

            if isinstance(sample_prompt, (list, dict)):
                pprint(sample_prompt)
            else:
                print(sample_prompt)

            head_print("Output")
            print(sample_result["text"])
            head_print("")
            print(f" -> Pred: {sample_result['pred']}, Label: {sample_result['true']}")
            print(f" -> Entropy: {sample_result.get('entropy', 'N/A'):.4f}")
            print(f" -> Score (P_Yes): {sample_result.get('score', 'N/A')}")
            is_correct = sample_result["pred"] == sample_result["true"]
            print(" -> Result:", "CORRECT" if is_correct else "WRONG")
            print("#" * 70)


# ===================================
# 2. Predicton
# ===================================
def load_predictor(args):
    if args.mode == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        # if api_key is None:
        #     print("No ENVIRON VAR found for 'GEMINI_API_KEY'. Try to read `.api.key`:")
        #     try:
        #         with open("./.api.key") as f:
        #             api_key = f.read().strip()
        #     except FileNotFoundError:
        #         raise ValueError("API Key not found.")
        # client = genai.Client(api_key=api_key)
        # return client

    elif args.mode == "llama":
        device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

        print(f"Loading LLaMA model ({args.model_name}) on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        model.eval()
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        print("Model loaded successfully")
        return tokenizer, model, terminators


# def predict_gemini(args, prompt, client):
#     pred, text = None, None
#     system_content, user_content = prompt  # Prompter(mode="gemini") returns tuple

#     for attempt in range(args.max_retries):
#         try:
#             response = client.models.generate_content(
#                 model="gemini-2.5-flash-lite",
#                 contents=user_content,  # Now includes [Reasoning] at end
#                 config=types.GenerateContentConfig(
#                     system_instruction=system_content,
#                     temperature=args.temperature,
#                     topP=args.top_p,
#                 ),
#             )
#             text = response.text.strip()
#             # Append [Reasoning] to text for completeness since model output continues from it
#             # text = "[Reasoning] " + text
#             pred = extract_pred(text, args.dataset)

#             if pred is not None:
#                 return pred, text
#             else:
#                 # If extraction failed, maybe try again or log error
#                 # print(f"Extraction failed: {text}")
#                 pass

#         except Exception as e:
#             print(f"Attempt {attempt+1} failed with error: {type(e).__name__}")
#             time.sleep(1)

#     return pred, text


# def evaluate_gemini(
#     args,
#     client,
#     instruction,
#     smiles_dict,
#     label_dict,
#     emb_dict,
#     test_df,
#     clusters_idx,
#     clusters_scaf,
#     scaf_emb_dict,
# ):
#     # [Updated] Use new Prompter
#     prompter = Prompter(args.dataset, instruction)
#     query_to_shot = load_query_to_shot(args)
#     decomposer = None
#     if args.split_scaffold:
#         decomposer = MoleculeDecomposer()

#     results = {}
#     fail_set = set()
#     for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predict ..."):
#         q_id, q_smiles, q_label = str(row["raw_ID"]), row["smiles"], row["label"]
#         shots = select_shot(args, row, smiles_dict, label_dict, emb_dict, query_to_shot, decomposer)

#         # [Updated] Generate prompt (returns tuple for Gemini)
#         prompt = prompter.generate(q_smiles, shots, mode="gemini")

#         pred, text = predict_gemini(args, prompt, client)
#         if pred is None:
#             fail_set.add(q_id)
#             print(f"[ERROR] Failed to process {q_smiles} ({q_id}) with text: {text}")

#         # Save results
#         system_content, user_content = prompt
#         _prompt = {"system_content": system_content, "user_content": user_content}
#         results[q_id] = {"true": q_label, "pred": pred, "prompt": _prompt, "text": text}

#     evaluate(results, fail_set, args.save_dir, args.dataset, args.debug)


def process_llama(args, index_list, prompt_list, label_list, tokenizer, model, terminators):
    """
    LLaMA prediction in batches with Entropy (NLL) calculation & Detailed Debugging.
    Includes "Surface Form Competition" for Robust AUC Scoring.
    """
    bs = args.eval_batch_size
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    # -------------------------------------------------------------------------
    # [Pre-computation] Token Candidates for Surface Form Competition
    # Reference: Holtzman et al., 2021, "Surface Form Competition: Why the Highest Probability Answer Isn't Always Right"
    # -------------------------------------------------------------------------
    yes_candidates = ["Yes", " Yes", "yes", " yes"]
    no_candidates = ["No", " No", "no", " no"]

    def get_ids(candidates):
        ids = []
        for cand in candidates:
            # add_special_tokens=False is crucial to avoid adding BOS token
            enc = tokenizer.encode(cand, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        return ids

    yes_ids_list = get_ids(yes_candidates)
    no_ids_list = get_ids(no_candidates)

    # Convert to Tensor for efficient indexing
    yes_ids_tensor = torch.tensor(yes_ids_list, device=model.device)
    no_ids_tensor = torch.tensor(no_ids_list, device=model.device)
    all_target_ids = set(yes_ids_list + no_ids_list)
    # -------------------------------------------------------------------------

    results = {}
    retry_set = set()

    # Inference mode
    with torch.no_grad():
        for i in tqdm(range(0, len(prompt_list), bs), desc=f"Running Batch (bs={bs})..."):
            batch_idx = index_list[i : i + bs]
            batch_prompt = prompt_list[i : i + bs]
            batch_label = label_list[i : i + bs]

            # 1. Encode Prompts
            inputs = tokenizer(batch_prompt, return_tensors="pt", padding=True, truncation=False).to(model.device)
            input_len = inputs.input_ids.shape[-1]  # Prompt Length

            # 2. Generate
            batch_output_ids = model.generate(
                **inputs,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                generation_config=generation_config,
            )

            # 3. Calculate Entropy & Extract Logits
            outputs = model(batch_output_ids)
            logits = outputs.logits  # [bs, seq_len, vocab_size]

            # Shift for autoregressive prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_output_ids[..., 1:].contiguous()

            # Create mask to ignore prompt
            mask_labels = shift_labels.clone()
            if input_len > 1:
                mask_labels[:, : input_len - 1] = -100

            # EOS(Padding) Masking
            for b_idx in range(batch_output_ids.shape[0]):
                gen_ids = batch_output_ids[b_idx, input_len:]
                eos_indices = []
                for term_id in terminators:
                    inds = (gen_ids == term_id).nonzero(as_tuple=True)[0]
                    if len(inds) > 0:
                        eos_indices.append(inds[0].item())

                if eos_indices:
                    first_eos_idx = min(eos_indices)
                    mask_start = (input_len - 1) + first_eos_idx + 1
                    if mask_start < mask_labels.shape[1]:
                        mask_labels[b_idx, mask_start:] = -100

            # Compute NLL for Entropy
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            nll_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), mask_labels.view(-1))
            nll_per_token = nll_per_token.view(shift_labels.size())

            valid_tokens = (mask_labels != -100).float()
            sum_nll = (nll_per_token * valid_tokens).sum(dim=1)
            num_tokens = valid_tokens.sum(dim=1)
            entropy = sum_nll / (num_tokens + 1e-9)

            # ---------------------------------------------------------
            # 4. Robust AUC Score Calculation (Surface Form Competition)
            # ---------------------------------------------------------
            batch_scores = []
            debug_info = {}  # Store info for the first sample for debugging

            for b_idx in range(batch_output_ids.shape[0]):
                gen_ids = batch_output_ids[b_idx, input_len:]
                gen_ids_list = gen_ids.tolist()

                # Find the first occurrence of any Yes/No token
                found_idx = -1
                found_token_id = -1

                for t_idx, token_id in enumerate(gen_ids_list):
                    if token_id in all_target_ids:
                        found_idx = t_idx
                        found_token_id = token_id
                        break

                if found_idx != -1:
                    # Logit Index: (input_len + found_idx) is the token ITSELF
                    # We want the logit that PREDICTED this token -> index - 1
                    logit_idx = input_len + found_idx - 1

                    # Extract logits for all candidates
                    token_logits = logits[b_idx, logit_idx, :]  # [vocab_size]

                    yes_group_logits = token_logits[yes_ids_tensor]
                    no_group_logits = token_logits[no_ids_tensor]

                    # LogSumExp to aggregate probabilities
                    total_yes_logit = torch.logsumexp(yes_group_logits, dim=0)
                    total_no_logit = torch.logsumexp(no_group_logits, dim=0)

                    # Binary Softmax: P(Yes) = exp(Yes) / (exp(Yes) + exp(No))
                    binary_logits = torch.stack([total_no_logit, total_yes_logit])
                    probs = F.softmax(binary_logits, dim=0)
                    prob_yes = probs[1].item()

                    batch_scores.append(prob_yes)

                    # Debug Info Collection (Only for the first sample in batch 0)
                    if args.debug and i == 0 and b_idx == 0:
                        debug_info = {
                            "found_token": tokenizer.decode([found_token_id]),
                            "found_id": found_token_id,
                            "logit_idx": logit_idx,
                            "yes_logits": yes_group_logits.tolist(),
                            "no_logits": no_group_logits.tolist(),
                            "total_yes": total_yes_logit.item(),
                            "total_no": total_no_logit.item(),
                            "final_score": prob_yes,
                        }
                else:
                    batch_scores.append(0.5)  # Default neutral score

            # ================= [DEBUG START] =================
            if args.debug and i == 0:
                print_debug("\n" + "=" * 60)
                print_debug(f"[DEBUG] Entropy & Score Analysis (Sample 0)")

                # 1. Entropy Part (Concise)
                print_debug(f"  [Entropy] Generated Len: {batch_output_ids.shape[1] - input_len}")
                print_debug(f"  [Entropy] Valid Tokens : {num_tokens[0]}")
                print_debug(f"  [Entropy] Value        : {entropy[0]:.4f}")

                # 2. AUC Score Part
                if debug_info:
                    print_debug("-" * 40)
                    print_debug(f"  [Score] Found Token: '{debug_info['found_token']}' (ID: {debug_info['found_id']})")

                    # Print Candidates Detail
                    print_debug("  [Score] Yes Candidates:")
                    for y_id, y_logit in zip(yes_ids_list, debug_info["yes_logits"]):
                        print_debug(f"    - '{tokenizer.decode([y_id])}' ({y_id}): {y_logit:.4f}")
                    print_debug(f"    -> LogSumExp(Yes): {debug_info['total_yes']:.4f}")

                    print_debug("  [Score] No Candidates:")
                    for n_id, n_logit in zip(no_ids_list, debug_info["no_logits"]):
                        print_debug(f"    - '{tokenizer.decode([n_id])}' ({n_id}): {n_logit:.4f}")
                    print_debug(f"    -> LogSumExp(No) : {debug_info['total_no']:.4f}")

                    print_debug(f"  [Score] Final P(Yes): {debug_info['final_score']:.6f}")
                else:
                    print_debug("  [Score] No Yes/No token found in Sample 0.")

                print_debug("=" * 60 + "\n")
            # ================= [DEBUG END] =================

            # 5. Decode and Store Results
            new_tokens = batch_output_ids[:, input_len:]
            batch_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for j, (raw_id, text, label, prompt) in enumerate(zip(batch_idx, batch_text, batch_label, batch_prompt)):
                pred = extract_pred(text, args.dataset)
                if pred is None:
                    retry_set.add(raw_id)

                results[raw_id] = {
                    "true": label,
                    "pred": pred,
                    "prompt": prompt,
                    "text": text,
                    "entropy": entropy[j].item(),
                    "score": batch_scores[j],  # Robust AUC Score
                }

            torch.cuda.empty_cache()

    return results, retry_set


def evaluate_llama(
    args,
    instruction,
    smiles_dict,
    label_dict,
    emb_dict,
    test_df,
    clusters_idx,
    clusters_scaf,
    scaf_emb_dict,
):
    tokenizer, model, terminators = load_predictor(args)
    decomposer = None
    if args.split_scaffold:
        decomposer = MoleculeDecomposer()

    # [Updated] Use new Prompter
    prompter = Prompter(args.dataset, instruction)
    query_to_shot = load_query_to_shot(args)

    retry_list = []
    test_label_dict = {}
    test_prompt_dict = {}

    # Prepare prompts
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Prepare prompt ..."):
        q_id, q_smiles, q_label = str(row["raw_ID"]), row["smiles"], row["label"]
        shots = select_shot(args, row, smiles_dict, label_dict, emb_dict, query_to_shot, decomposer)

        # [Updated] Get messages list
        message_list = prompter.generate(q_smiles, shots, mode="llama")

        retry_list.append(q_id)
        test_label_dict[q_id] = q_label

        # [Updated] Apply chat template AND append [Reasoning] manually
        # This is the CRITICAL STEP for Llama 3 Pre-fill
        prompt_str = tokenizer.apply_chat_template(message_list, add_generation_prompt=True, tokenize=False)
        # Manually force the assistant to start with [Reasoning]
        prompt_str += "[Reasoning]"

        test_prompt_dict[q_id] = prompt_str

    # Prediction Loop
    retry = 0
    results = {}
    while retry_list and retry <= args.max_retries:
        if retry == 0:
            print(f"Starting prediction for {len(retry_list)} samples...")
        else:
            print(f"Retrying {len(retry_list)} failed samples (Retry {retry}/{args.max_retries})...")

        new_results, retry_set = process_llama(
            args,
            retry_list,
            [test_prompt_dict[i] for i in retry_list],
            [test_label_dict[i] for i in retry_list],
            tokenizer,
            model,
            terminators,
        )
        results.update(new_results)
        retry += 1
        retry_list = list(retry_set)

    evaluate(results, retry_list, args.save_dir, args.dataset, args.debug)


# ===================================
# 3. Main Section
# ===================================
if __name__ == "__main__":
    try:
        args, categories = parse_args()

        # [Updated] Tag generation logic consistency
        embed_tag = f"molt5-{args.embedding_type}" if args.embedding_type != "fp" else "fp"
        eps_tag = f"eps={args.eps}" if args.eps else "no_merge"
        inst_tag = "w-inst" if args.use_inst else "wo-inst"

        if args.method == "pool_clustering":
            sam_strat = "rand" if args.sample_strategy == "random" else "sim"
            # Cluster tag for identifying the specific clustering setup
            cluster_tag_dir = f"{embed_tag}_{eps_tag}_{args.sample_strategy}-sampling"
            method_tag = f"pc-{sam_strat}"
        else:
            cluster_tag_dir = ""  # Not used
            method_tag = args.method

        # Experiment logging tag
        args.tag = f"{args.k}-shot_{inst_tag}_{embed_tag}_{method_tag}_{eps_tag}"
        args.now = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(args.save_root, args.dataset, args.tag, args.now)
        categories.setdefault("Experiment Settings", []).extend(["tag", "now"])

        print(format_args(args, categories))
        os.makedirs(args.save_dir, exist_ok=True)
        save_json(vars(args), os.path.join(args.save_dir, "args.json"))

        # Load dataset embedding
        if args.embedding_type == "fp":
            if args.debug:
                print_debug("Load Fingerprint Embedding")
            train_df, test_df = load_dataset(
                dataset_name=args.dataset,
                embed_model="fp",
                radius=args.radius,
                n_bits=args.n_bits,
                verbose=args.verbose,
            )
        elif args.embedding_type in {"small", "base", "large"}:
            if args.debug:
                print_debug(f"Load MolT5-{args.embedding_type} Embedding")
            train_df, test_df = load_dataset(
                dataset_name=args.dataset,
                embed_model=args.embedding_type,
                batch_size=args.batch_size,
                device="cpu" if args.use_cpu else "cuda",
                verbose=args.verbose,
            )

        if args.debug:
            if args.method != "pool_clustering":
                train_df = train_df.head(100)
            test_df = test_df.head(8)
            print_debug("sample train/test data")
            log_df("Train Data", df=train_df)
            log_df("Test Data", df=test_df)

        emb_dict = {str(row.raw_ID): row.embed for row in train_df.itertuples()}
        smiles_dict = {str(row.raw_ID): row.smiles for row in train_df.itertuples()}
        label_dict = {str(row.raw_ID): row.label for row in train_df.itertuples()}

        with open(args.inst_json, "r") as f:
            instruction = json.load(f)
        if args.use_inst:
            instruction = instruction["full"][args.dataset]
        else:
            instruction = instruction["short"][args.dataset]

        # Load cached clustering
        if args.method == "pool_clustering":
            cache_dir = os.path.join(args.cache_root, args.dataset, cluster_tag_dir)

            if os.path.exists(cache_dir):
                # load_cluster now takes just the directory path
                cluster_idx, cluster_scaf, scaf_emb_dict, meta = load_cluster(cache_dir)
                print(f"Cached clustering results loaded from {cache_dir}")
            else:
                raise FileNotFoundError(f"No cached clustering found in {cache_dir}")
        else:
            cluster_idx, cluster_scaf, scaf_emb_dict = None, None, None

        # Run evaluation
        if args.mode == "gemini":
            client = load_predictor(args)
            evaluate_gemini(
                args,
                client,
                instruction,
                smiles_dict,
                label_dict,
                emb_dict,
                test_df,
                cluster_idx,
                cluster_scaf,
                scaf_emb_dict,
            )

        elif args.mode == "llama":
            evaluate_llama(
                args,
                instruction,
                smiles_dict,
                label_dict,
                emb_dict,
                test_df,
                cluster_idx,
                cluster_scaf,
                scaf_emb_dict,
            )
    finally:
        gc.collect()
        torch.cuda.empty_cache()
