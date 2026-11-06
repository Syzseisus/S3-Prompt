import io
import os
import json
import pickle
import random
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict as ddict

import torch
from transformers import AutoTokenizer

from utils.constants import CKPT_ROOT, DATA_ROOT, SAVE_ROOT, REGRESSION, CLASSIFICATION, POSSIBLE_DATASETS


# ===================================
# 0. Seed Setting
# ===================================
# Contains:
#   - set_all_seeds
# ===================================
def set_all_seeds(seed, cudnn=False):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn:
        # cudnn deterministic settings (may slow down)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ===================================
# 1. File Utils
# ===================================
# Contains:
#   - Save/load
#     - pickle: save_pickle, load_pickle
#     - json: save_json, load_json
# ===================================
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===================================
# 2. Print Utils
# ===================================
# Contains:
#   - head_print
#   - print_debug
#   - verbose_print
#   - log_df: Display the DataFrame with the given msg
#             + Separating sections (optional)
# ===================================
def head_print(text, *args, **kwargs):
    if text:
        print(f"{f' {text} ':=^70}", *args, **kwargs)
    else:
        print("=" * 70, *args, **kwargs)


def print_debug(text, *args, **kwargs):
    print(f"{f'### [DEBUG] - {text} ':#<70}", *args, **kwargs)


def verbose_print(*args, verbose=True, **kwargs):
    if verbose:
        print(*args, **kwargs)


def log_df(*args, df=None, verbose=True, sep_line=True, **kwargs):
    verbose_print(*args, verbose=verbose, **kwargs)
    if df is not None and verbose:
        df.info()
    if sep_line:
        verbose_print("=" * 70, verbose=verbose)


# ===================================
# 3. ArgumentParser Utils
# ===================================
# Utility for categorizing and displaying argparse command-line arguments
# in a clean, human-readable table.
# Contains:
#   - add_arg_group: Wraps parser.add_argument() to record argument categories
#   - get_args: Parses arguments grouped by category
#   - format_args: Returns a formatted string of arguments grouped by category
# ===================================
def add_arg_group(parser, group_name, category_dict):
    """
    Wraps `parser.add_argument()` to automatically group arguments by category.

    Args:
        - parser (argparse.ArgumentParser): The argument parser instance
        - group_name (str): Name of the argument group/category
        - category_dict (dict): Dictionary that maps category names to a list of argument names
    """

    def add_argument(*args, **kwargs):
        dest = kwargs.get("dest")
        if not dest:
            for arg in args:
                if arg.startswith("--"):
                    dest = arg.lstrip("-").replace("-", "_")
                    break
        if dest:
            category_dict.setdefault(group_name, []).append(dest)
        return parser.add_argument(*args, **kwargs)

    return add_argument


def add_common_args(parser, categories):
    """
    Defines categorized command-line arguments and parses them.

    Returns (tuple):
        - argparse.Namespace: Parsed arguments.
        - dict: Mapping of category names to lists of argument names.
    """

    # fmt: off
    # === Experiment Settings ===
    add = add_arg_group(parser, "Experiment Settings", categories)
    add("--dataset", type=str, default="BBBP", choices=POSSIBLE_DATASETS, help="Name of the dataset (e.g., BBBP, BACE, etc.)")
    add("--k", type=int, default=8, help="Number of shots to select per query")
    add("--num_cand", type=int, default=30, help="ConE전, TopK를 뽑을 개수.")
    add("--embedding_type", type=str, default="fp", choices=["fp", "small", "base", "large"], help="Type of molecule embedding")
    add("--method", type=str, default="pool_clustering", choices=["zero_shot", "random", "topk", "pc", "pool_clustering", "random_cone", "topk_cone", "pc_cone"], help="Method to select shots")  # pc == pool_clustering
    add("--sample_strategy", type=str, default="random", choices=["random", "similarity"], help="Strategy to sample shots from selected clusters (only used for `pool_clsutering` method)")
    add("--eps", type=float, default=0.6, help="Similarity threshold for Bunita merging (0 < eps <= 1)")
    add("--inst_json", type=str, default="inst.json", help="Path to JSON file with task instructions")
    add("--use_inst", action="store_true", default=False, help="Where use instructions")
    add("--split_scaffold", action="store_true", default=False, help="Whether split scaffold and decorations")

    # === Molecule Embedding ===
    add = add_arg_group(parser, "Molecule Embedding", categories)
    add("--radius", type=int, default=2, help="Radius parameter for Morgan Fingerprint")
    add("--n_bits", type=int, default=1024, help="Number of bits for Morgan Fingerprint")

    # === LLM Inference ===
    add = add_arg_group(parser, "LLM Inference", categories)
    add("--temperature", type=float, default=0.0, help="LLM temperature setting")
    add("--top_p", type=float, default=0.9, help="LLM top_p setting")

    # === Runtime / Logging ===
    add = add_arg_group(parser, "Runtime / Logging", categories)
    add("--max_retries", type=int, default=3, help="Maximum number of retreis for failure")
    add("--infer_mode", type=str, default="single", choices=["single", "batch"], help="LLM으로 inference할 때 single vs batch")
    add("--use_cpu", action="store_true", default=False, help="Use CPU instead of GPU")
    add("--debug", action="store_true", default=False, help="Enable debug mode: use only a small number of test samples and print extra information")
    add("--verbose", action="store_true", default=False, help="Enable verbose mode for loading data and clusters")

    # === Paths / Files ===
    add = add_arg_group(parser, "Paths / Files", categories)
    add("--save_root", type=str, default=SAVE_ROOT, help="Directory to save results (predictions, metrics, etc.)")
    add("--ckpt_root", type=str, default=CKPT_ROOT, help="Directory to save results (predictions, metrics, etc.)")
    add("--cache_root", type=str, default=DATA_ROOT, help="Directory to store cached clustering data")

    # === S3-Prompt Arguments (Add these lines) ===
    add = add_arg_group(parser, "S3-Prompt", categories)
    add("--lr", type=float, default=1e-4, help="Learning rate")
    add("--epochs", type=int, default=10, help="Number of training epochs")
    add("--start_epoch", type=int, default=1, help="어디서부터 시작할지. 0 = 디스크의 epoch_NN.pt 자동 탐지 후 (N+1) 사용.")
    add("--resume_from", type=str, default=None, help="resume 대상 timestamp(YYYYMMDD_HHMMSS). 미지정 시 가장 최근 timestamp 사용. start_epoch>1 이거나 0 일 때만 의미 있음.")
    add("--gnn_hidden", type=int, default=300, help="Number of hidden units in the GNN")
    add("--patience", type=int, default=3, help="Early stopping patience")
    add("--eval_epoch", type=int, default=5, help="Evaluation frequency")
    add("--lambda_loss", type=float, default=0.5, help="Weight coefficient for the Activity Cliff-Contrastive (Triplet) Loss.")
    # fmt: on

    return parser, categories


def parse_args():
    parser = argparse.ArgumentParser(description="Molecular Few-Shot Evaluation with Clustering")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    categories_other = {}
    parser_other = subparsers.add_parser("other")
    parser_other, categories_other = add_common_args(parser_other, categories_other)
    # fmt: off
    add = add_arg_group(parser_other, "Other LLM", categories_other)
    # fmt: on

    categories_llama = {}
    parser_llama = subparsers.add_parser("llama")
    parser_llama, categories_llama = add_common_args(parser_llama, categories_llama)
    # fmt: off
    add = add_arg_group(parser_llama, "LLaMA", categories_llama)
    add("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Name of the LLM model to use")
    add("--batch_size", type=int, default=4, help="Batch size for LLaMA train/inference")
    add("--grad_accum_steps", type=int, default=16, help="")
    add("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens the LLM can generate")
    add("--model_max_length", type=int, default=-1, help="Maximum sequence length. -1 means use model default. Set to 4096 or 2048 to prevent OOM.")
    add("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save memory at the cost of slower backward pass.")
    add("--no_grad_ckpting", dest="gradient_checkpointing", action="store_false", help="Disable gradient checkpointing for faster training.")
    add("--quant_bit", type=int, choices=[4, 8], default=8, help="Quantization bit depth. Choose 4 or 8. Default is None (FP16/BF16).")
    
    add = add_arg_group(parser_llama, "LoRA", categories_llama)
    add("--lora_r", type=int, default=8)
    add("--lora_alpha", type=int, default=16)
    # fmt: on

    args = parser.parse_args()

    # TODO: (gemini, batch) -> assert, (llama, single) -> warning, (batch, eval_batch_size=1) -> warning
    args.infer_mode = "single" if args.mode == "other" else "batch"

    args.verbose = args.debug or args.verbose
    if args.mode == "llama":
        args.do_sample = args.temperature > 0
        try:
            _ = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            raise ValueError(f"Failed to load model '{args.model_name}': {e}")

    # fmt: off
    # sanity checks
    if (args.embedding_type == "fp" and args.eps != 0.6) or (args.embedding_type != "fp" and args.eps != 0.8):
        print(
            f"[WARN] Currently, eps is fixed: 0.6 for FP and 0.8 for MolT5.",
            "Changing eps requires re-caching using `clustering.py`.",
            f"(Got embedding_type={args.embedding_type} with eps={args.eps})",
            sep="\n"
        )
    if (args.method == "zero_shot") != (args.k == 0):
        raise AssertionError(f"Inconsistent setup: If method is 'zero_shot', k must be 0; otherwise, k must be > 0. (Got method='{args.method}', k={args.k})")
    if "cone" in args.method:
        v_mode = args.mode != "llama"
        v_num = (args.k > args.num_cand)
        if v_mode or v_num:
            violate = "all" if v_mode and v_num else "# 1" if v_mode else "# 2"
            raise AssertionError(
                f"Inconsistent setup: If method is 'topk_cone', must hold following conditions:\n"
                f"1. LLM mode must be 'llama' (Got mode={args.mode})\n"
                f"2. The number of shots `k` must not exceed candidates `num_cand` (Got k={args.k}, num_cand={args.num_cand})\n"
                f"You violated: {violate}."
            )

    assert args.k >= 0, f"Invalid value: k cannot be negative. (Got k={args.k})"
    assert 0 <= args.eps <= 1.0, f"Invalid value: eps must be in [0.0, 1.0]. (Got eps={args.eps})"
    assert 0.0 <= args.temperature <= 2.0, f"Invalid value: temperature should be in [0.0, 2.0]. (Got temperature={args.temperature})"
    assert 0.0 <= args.top_p <= 1.0, f"Invalid value: top_p should be in [0.0, 1.0]. (Got top_p={args.top_p})"
    # fmt: on

    categories = categories_other if args.mode == "other" else categories_llama if args.mode == "llama" else {}

    return args, categories


def format_args(args, categories):
    """
    Formats parsed arguments into a neatly aligned table grouped by category.

    Args:
        - args (argparse.Namespace): Parsed arguments
        - categories (dict): Mapping of category names to their respective argument names

    Returns (str):
        - A formatted string displaying all arguments and their values in a box-style layout

    Sample:
        ┌────────────────────┬────────────────┐
        │ Hyperparameter     │     Value      │
        ├────────────────────┴────────────────┤
        │       [ Project Config ]            │
        ├────────────────────┬────────────────┤
        │ project            │ "demo_project" │
        │ save_dir           │ "./save"       │
        ├────────────────────┴────────────────┤
        │         [ Training ]                │
        ├────────────────────┬────────────────┤
        │ batch_size         │ 32             │
        │ epochs             │ 20             │
        │ lr                 │ 0.0005         │
        ├────────────────────┴────────────────┤
        │          [ Model ]                  │
        ├────────────────────┬────────────────┤
        │ dropout            │ 0.1            │
        │ hidden_dim         │ 256            │
        └────────────────────┴────────────────┘
    """

    args_dict = vars(args)
    arg_to_cat = {arg: cat for cat, arg_list in categories.items() for arg in arg_list}
    grouped = ddict(list)

    for key, val in args_dict.items():
        cat = arg_to_cat.get(key, "Other")
        val_str = f'"{val}"' if isinstance(val, str) else str(val)
        grouped[cat].append((key, val_str))

    for cat in grouped:
        grouped[cat] = sorted(grouped[cat], key=lambda x: x[0])

    all_items = [(k, v) for cat in categories if cat in grouped for (k, v) in grouped[cat]]
    max_key_len = max(len(k) for k, _ in all_items + [("Hyperparameter", "")])
    max_val_len = max(len(v) for _, v in all_items + [("", "Value")])

    horz = "─"
    vert = "│"
    tl, tr, bl, br = "┌┐└┘"
    l_sep, r_sep = "├┤"
    t_sep, b_sep = "┬┴"
    h_key, h_val = horz * (max_key_len + 2), horz * (max_val_len + 2)

    out = io.StringIO()
    w = lambda line="": out.write(line + "\n")

    w(f"{tl}{h_key}{t_sep}{h_val}{tr}")
    w(f"{vert} {'Hyperparameter':<{max_key_len}} {vert} {'Value'.center(max_val_len)} {vert}")

    for cat in categories:
        if cat not in grouped:
            continue
        w(f"{l_sep}{h_key}{b_sep}{h_val}{r_sep}")
        w(f"{vert}{f'[ {cat} ]':^{max_key_len + max_val_len + 5}}{vert}")
        w(f"{l_sep}{h_key}{t_sep}{h_val}{r_sep}")
        for k, v_ in grouped[cat]:
            w(f"{vert} {k:<{max_key_len}} {vert} {v_:<{max_val_len}} {vert}")

    w(f"{bl}{h_key}{b_sep}{h_val}{br}")
    return out.getvalue()


if __name__ == "__main__":
    # test args
    args, categories = parse_args()

    embed_tag = f"molt5_{args.embedding_type}" if args.embedding_type != "fp" else "fp"
    eps_tag = f"eps={args.eps}" if args.eps else "no-merge"
    inst_tag = "w-inst" if args.inst_json else "wo-inst"
    if args.method == "pool_clustering":
        sam_strat = "rand" if args.sample_strategy == "random" else "sim"
        method_tag = f"pc-{sam_strat}"
    else:
        method_tag = args.method

    args.tag = f"{args.k}-shot_{inst_tag}_{embed_tag}_{method_tag}_{eps_tag}"
    args.now = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_root, args.dataset, args.tag, args.now)

    # Update categories
    categories.setdefault("Experiment Settings", []).append("tag")
    categories.setdefault("Runtime Generated Fields", []).extend(["now", "start_time", "end_time", "duration"])

    print(format_args(args, categories))
    os.makedirs(args.save_dir, exist_ok=True)
    save_json(vars(args), os.path.join(args.save_dir, "args.json"))
