#!/bin/bash
# Shared helpers for run_classification.sh / run_regression.sh. Not meant to be run directly.
#
# Paper settings (S3-Prompt, CIKM'26, Sec. 3.7 / 7):
#   LLaMA-3.1-8B-Instruct, 8-bit, AdamW lr=1e-4, effective batch size 32,
#   LoRA r=12 / alpha=24 on q_proj,v_proj, triplet weight lambda=0.1, k=8, full instruction.
#
# Environment overrides:
#   EPOCHS    training epochs for trainable cells (C-G). Not stated in the paper.  (default 30)
#   DRY_RUN=1 print the resolved engine + arguments for every cell without launching a model
#   GPU_BS    per-step batch size; grad accumulation = 32 / GPU_BS               (default $1 or 4)

set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

EFFECTIVE_BS=32
GPU_BS="${1:-${GPU_BS:-4}}"
GRAD_ACCUM=$((EFFECTIVE_BS / GPU_BS))
if [ "${GRAD_ACCUM}" -lt 1 ]; then GRAD_ACCUM=1; fi
EPOCHS="${EPOCHS:-30}"
LAMBDA="0.1"
K=8
LORA_ARGS="--lora_r 12 --lora_alpha 24"   # engines/*_reg.py read these; classification engines hardcode 12/24

now() { date +"%Y-%m-%d %H:%M:%S"; }      # portable (GNU/BSD)

init_sweep() {   # init_sweep <tag> <total_cells>
    LOG_DIR="logs/$(date +"%y%m%d-%H%M%S")_$1"
    TOTAL_STEPS=$2
    CURRENT_STEP=0
    SUMMARY_FILE="${LOG_DIR}/summary.log"
    if [ "${DRY_RUN}" == "1" ]; then
        echo "[INFO] DRY_RUN=1 -> dispatch only, no model is launched"
        return 0
    fi
    mkdir -p "${LOG_DIR}"
    printf "%-5s | %-14s | %-5s | %-4s | %-5s | %-6s | %-5s | %-3s | %-7s\n" \
        "Index" "Dataset" "Decom" "LoRA" "Emb" "Lambda" "Epoch" "BS" "GradAcc" > "${SUMMARY_FILE}"
    echo "[INFO] GPU batch size ${GPU_BS} x grad accum ${GRAD_ACCUM} (effective ${EFFECTIVE_BS}), epochs ${EPOCHS}"
    echo "[INFO] Logs: ${LOG_DIR}"
}

notify() {       # optional e-mail; only when a credentials file is present
    if [ -f .email_cred.key ]; then python utils/mailing.py --subject "$1" --body "$2" --file "$3"; fi
}

# run_cell <Index A-G> <dataset> <embedding: fp|small>
run_cell() {
    local idx=$1 dataset=$2 emb=$3
    local decomposition lora="" lambda="0"
    case "${idx}" in
        A) decomposition="none" ;;
        B) decomposition="text" ;;
        C) decomposition="text"; lora="--use_lora" ;;
        D) decomposition="soft" ;;
        E) decomposition="soft"; lambda="${LAMBDA}" ;;
        F) decomposition="soft"; lora="--use_lora" ;;
        G) decomposition="soft"; lora="--use_lora"; lambda="${LAMBDA}" ;;
        *) echo "[ERROR] unknown index ${idx}"; return 1 ;;
    esac

    local eps="0.6"; if [ "${emb}" == "small" ]; then eps="0.8"; fi
    local emb_tag="fp"; if [ "${emb}" != "fp" ]; then emb_tag="molt5-${emb}"; fi

    local cmd="python train.py --decomposition ${decomposition} --dataset ${dataset} ${lora} \
--lambda_loss ${lambda} --k ${K} --embedding_type ${emb} --eps ${eps} --method topk --use_inst \
--batch_size ${GPU_BS} --grad_accum_steps ${GRAD_ACCUM} --epochs ${EPOCHS} ${LORA_ARGS}"
    cmd=$(echo "${cmd}" | tr -s ' ')

    CURRENT_STEP=$((CURRENT_STEP + 1))
    local tag="[Index ${idx}] ${dataset} k=${K} ${emb_tag}"
    echo "--------------------------------------------------------------------------------"
    echo "[$(now)] (${CURRENT_STEP}/${TOTAL_STEPS}) ${tag}"

    if [ "${DRY_RUN}" == "1" ]; then
        ${cmd} --dispatch_only | sed -n 's/^module *: */  engine : /p'
        echo "  cmd    : ${cmd}"
        return 0
    fi

    printf "%-5s | %-14s | %-5s | %-4s | %-5s | %-6s | %-5s | %-3s | %-7s\n" \
        "${idx}" "${dataset}" "${decomposition}" "${lora:+yes}" "${emb}" "${lambda}" "${EPOCHS}" "${GPU_BS}" "${GRAD_ACCUM}" >> "${SUMMARY_FILE}"

    mkdir -p "${LOG_DIR}/${dataset}"
    local logfile="${LOG_DIR}/${dataset}/${idx}_k=${K}_${emb_tag}.log"
    echo "=== ${tag} ===" | tee "${logfile}"
    echo "=== Command: ${cmd} ===" | tee -a "${logfile}"

    if ${cmd} 2>&1 | tee -a "${logfile}"; then
        echo "=== [DONE] $(now) ===" | tee -a "${logfile}"
        notify "[NOALERT] DONE: ${tag}" "Log attached." "${logfile}"
    else
        echo "=== [ERROR] $(now) ===" | tee -a "${logfile}"
        notify "[ERROR] FAILED: ${tag}" "Check log." "${logfile}"
    fi
}
