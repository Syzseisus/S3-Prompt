#!/bin/bash
# Step 2 of data preparation: cache the exact top-K nearest training neighbours of every
# valid/test query, for both retrieval modalities (fp = topological, small = MolT5 semantic).
# Requires preprocess.sh to have been run first.
# Output: ${DATA_ROOT}/{dataset}/topk_{fp|molt5-small}/query_to_shot_{valid,test}_topk=100.json
#
# Usage: bash preprocess_topk.sh             # all four paper datasets
#        bash preprocess_topk.sh BACE ESOL   # a subset

cd "$(dirname "$0")" || exit 1

now() { date +"%Y-%m-%d %H:%M:%S"; }
notify() { if [ -f .email_cred.key ]; then python utils/mailing.py --subject "$1" --body "$2" --file "$3"; fi; }

root=$(python -c "from utils.constants import DATA_ROOT; print(DATA_ROOT)")

if [ $# -gt 0 ]; then datasets=("$@"); else datasets=("BACE" "BBBP" "ESOL" "LIPOPHILICITY"); fi
models=("fp" "small")
MAX_K=100   # engines slice the first k (k=8 in the paper) from these 100 candidates

total=$(( ${#models[@]} * ${#datasets[@]} ))
i=1
n_fail=0
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        mkdir -p "${root}/${dataset}"
        if [ "${model}" == "fp" ]; then embed_tag="fp"; else embed_tag="molt5_${model}"; fi
        PROCESS_FILE="${root}/${dataset}/topk_${embed_tag}.log"

        echo "[$(now)] ($i/$total) python -m utils.preprocess_topk --dataset ${dataset} --embed_model ${model} --max_k ${MAX_K}"
        echo "=== [${dataset}/${embed_tag}] Top-K started at $(now) ===" > "${PROCESS_FILE}"
        start_time=$(date +%s)

        if python -m utils.preprocess_topk --dataset "${dataset}" --embed_model "${model}" --max_k "${MAX_K}" >> "${PROCESS_FILE}" 2>&1; then
            echo ">> DONE"
            echo "=== Completed at $(now) (Duration: $(( $(date +%s) - start_time ))s) ===" >> "${PROCESS_FILE}"
            notify "[NOALERT] DONE Top-K ${dataset} ${model}" "" "${PROCESS_FILE}"
        else
            echo ">> ERROR (see ${PROCESS_FILE})"; n_fail=$((n_fail + 1))
            echo "=== ERROR at $(now) (Duration: $(( $(date +%s) - start_time ))s) ===" >> "${PROCESS_FILE}"
            notify "ERROR Top-K ${dataset} ${model}" "Failed for dataset=${dataset}, model=${model}" "${PROCESS_FILE}"
        fi
        i=$((i + 1))
    done
done

TREE_FILE="${root}/preprocess_tree.txt"
if command -v tree &> /dev/null; then tree "${root}" -L 3 > "${TREE_FILE}"; else ls -R "${root}" > "${TREE_FILE}"; fi

echo "ALL DONE (failures: ${n_fail})"
notify "ALL DONE (Top-K)" "All requested preprocessing tasks are finished." "${TREE_FILE}"
exit "${n_fail}"
