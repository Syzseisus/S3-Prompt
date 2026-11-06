"""
Unified entry point for S3-Prompt training / inference.

This is a thin, faithful front-end over the per-variant implementations: it routes a single
``--decomposition`` / dataset / flag combination to the exact script that produced the
corresponding paper results, then hands off unchanged. The implementations themselves live at
the repository root (``run_*.py``) and are not modified, so reproduced numbers are identical to
the original per-script runs. Post-hoc analysis and abandoned variants live under ``legacy/``.

Ablation index -> flags (see README):
    A  none / frozen / --lambda_loss 0
    B  text / frozen / --lambda_loss 0
    C  text / LoRA   / --lambda_loss 0
    D  soft / frozen / --lambda_loss 0
    E  soft / frozen / --lambda_loss >0
    F  soft / LoRA   / --lambda_loss 0
    G  soft / LoRA   / --lambda_loss >0

Examples:
    # Index E on BACE (soft, frozen, triplet, fingerprint retrieval)
    python train.py --dataset BACE --decomposition soft --lambda_loss 0.1 \
        --k 8 --embedding_type fp --eps 0.6 --method topk --use_inst

    # Index G on ESOL (soft, LoRA, triplet, semantic retrieval)
    python train.py --dataset ESOL --decomposition soft --use_lora --lambda_loss 0.1 \
        --k 8 --embedding_type small --eps 0.8 --method topk --use_inst

    # Inspect the dispatch without launching the model:
    python train.py --dataset BACE --decomposition soft --use_lora --dispatch_only ...
"""

import os
import sys
import runpy
import argparse

from utils.constants import CLASSIFICATION, REGRESSION

# (decomposition, task, lora) -> implementation module at the repository root.
# Classification selects the script by (decomposition, lora); the classification scripts do NOT
# expose a --use_lora flag, so it is stripped before hand-off. Regression scripts DO take
# --use_lora, so a single script per decomposition handles both frozen and LoRA.
_CLS = {
    ("none", False): "engines.wo_split",
    ("none", True): "engines.wo_split",        # no LoRA variant for the no-decomposition baseline
    ("text", False): "engines.text_only",
    ("text", True): "engines.text_only_peft",
    ("soft", False): "engines.s3prompt_w_label",
    ("soft", True): "engines.text_graph",
}
_REG = {
    "none": "engines.wo_split_reg",
    "text": "engines.text_only_reg",
    "soft": "engines.text_graph_reg",
}


def _peek(flag, argv, takes_value=False, default=None):
    """Read a flag's value (or presence) from a raw argv list without consuming it."""
    if flag not in argv:
        return default if takes_value else False
    if not takes_value:
        return True
    i = argv.index(flag)
    return argv[i + 1] if i + 1 < len(argv) else default


def resolve(argv):
    """Return (module_name, forwarded_argv) for a raw passthrough argv (no --decomposition)."""
    dataset = _peek("--dataset", argv, takes_value=True)
    if dataset is None:
        raise SystemExit("train.py: --dataset is required")
    if dataset in CLASSIFICATION:
        task = "classification"
    elif dataset in REGRESSION:
        task = "regression"
    else:
        raise SystemExit(f"train.py: unknown dataset '{dataset}'")

    decomposition = resolve.decomposition
    use_lora = _peek("--use_lora", argv)

    if task == "classification":
        module = _CLS[(decomposition, use_lora)]
        # classification scripts do not define --use_lora; drop it (routing already encodes it)
        forwarded = [a for a in argv if a != "--use_lora"]
    else:
        module = _REG[decomposition]
        forwarded = list(argv)

    return module, ["llama"] + forwarded


def main():
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Unified S3-Prompt launcher (routes to the per-variant implementation).",
        add_help=True,
    )
    parser.add_argument(
        "--decomposition", required=True, choices=["none", "text", "soft"],
        help="S1 structural decomposition: none (SMILES only) | text | soft (GNN soft tokens).",
    )
    parser.add_argument(
        "--dispatch_only", action="store_true",
        help="Print the resolved implementation and argument list, then exit (no model launch).",
    )
    known, rest = parser.parse_known_args()
    resolve.decomposition = known.decomposition

    module, forwarded = resolve(rest)

    if known.dispatch_only:
        print(f"decomposition : {known.decomposition}")
        print(f"module        : {module}.py")
        print(f"argv          : python {module}.py {' '.join(forwarded)}")
        return

    # Hand off exactly as if the user had launched `python {module}.py llama ...`.
    sys.argv = [f"{module}.py"] + forwarded
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
