#!/bin/bash
# Reproduces the regression results of S3-Prompt (LIPOPHILICITY, ESOL).
#
#   Table 7  component ablation, Index A-G, dataset-optimal retrieval (LIPO: FP, ESOL: Sem)
#   Table 6  main results: S3-Prompt (FP) = Index E + fp, S3-Prompt (Sem) = Index G + small
#
# Usage:  bash run_regression.sh [GPU_BATCH_SIZE]          # default 4 -> grad accum 8 (effective 32)
#         EPOCHS=30 bash run_regression.sh 8
#         DRY_RUN=1 bash run_regression.sh                 # list every cell without launching a model
# Prerequisite: preprocess.sh and preprocess_topk.sh (or the released preprocessed data).

cd "$(dirname "$0")" || exit 1
source ./sweep_common.sh "$@"

init_sweep "regression" 16

# --- Table 7: Index A-G ------------------------------------------------------
for idx in A B C D E F G; do run_cell "${idx}" LIPOPHILICITY fp;    done
for idx in A B C D E F G; do run_cell "${idx}" ESOL          small; done

# --- Table 6: the other retrieval modality for each dataset ------------------
run_cell G LIPOPHILICITY small   # S3-Prompt (Sem) on LIPO
run_cell E ESOL          fp      # S3-Prompt (FP)  on ESOL

echo "================================================================================"
echo "Regression sweep finished. Logs: ${LOG_DIR}"
