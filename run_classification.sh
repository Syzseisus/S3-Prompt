#!/bin/bash
# Reproduces the classification results of S3-Prompt (BACE, BBBP).
#
#   Table 3  component ablation, Index A-G, dataset-optimal retrieval (BACE: FP, BBBP: Sem)
#   Table 2  main results: S3-Prompt (FP) = Index E + fp, S3-Prompt (Sem) = Index G + small
#
# Usage:  bash run_classification.sh [GPU_BATCH_SIZE]      # default 4 -> grad accum 8 (effective 32)
#         EPOCHS=30 bash run_classification.sh 8
#         DRY_RUN=1 bash run_classification.sh             # list every cell without launching a model
# Prerequisite: preprocess.sh and preprocess_topk.sh (or the released preprocessed data).

cd "$(dirname "$0")" || exit 1
source ./sweep_common.sh "$@"

init_sweep "classification" 16

# --- Table 3: Index A-G ------------------------------------------------------
for idx in A B C D E F G; do run_cell "${idx}" BACE fp;    done
for idx in A B C D E F G; do run_cell "${idx}" BBBP small; done

# --- Table 2: the other retrieval modality for each dataset ------------------
run_cell G BACE small      # S3-Prompt (Sem) on BACE
run_cell E BBBP fp         # S3-Prompt (FP)  on BBBP

echo "================================================================================"
echo "Classification sweep finished. Logs: ${LOG_DIR}"
