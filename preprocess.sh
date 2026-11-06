#!/bin/bash
# Step 1 of data preparation: download MoleculeNet, clean, embed (fp + MolT5-small), split.
# Output: ${DATA_ROOT}/{dataset}/  (DATA_ROOT is set in utils/constants.py)
#
# Usage: bash preprocess.sh                  # all four paper datasets
#        bash preprocess.sh BACE ESOL        # a subset

cd "$(dirname "$0")" || exit 1

now() { date +"%Y-%m-%d %H:%M:%S"; }
notify() { if [ -f .email_cred.key ]; then python utils/mailing.py --subject "$1" --body "$2" --file "$3"; fi; }

root=$(python -c "from utils.constants import DATA_ROOT; print(DATA_ROOT)")

if [ $# -gt 0 ]; then datasets=("$@"); else datasets=("BACE" "BBBP" "ESOL" "LIPOPHILICITY"); fi

first_time=$(date +%s)
n_fail=0
for dataset in "${datasets[@]}"; do
    mkdir -p "${root}/${dataset}"
    PROCESS_FILE="${root}/${dataset}/preprocess.log"
    VERIFY_FILE="${root}/${dataset}/preprocess_sampling.log"

    echo "[$(now)] Running: python -m utils.dataset_loader ${dataset}"
    echo "=== [${dataset}] Data generation started at $(now) ===" > "${PROCESS_FILE}"
    start_time=$(date +%s)

    if python -m utils.dataset_loader "${dataset}" >> "${PROCESS_FILE}" 2>&1; then
        echo "=== Completed at $(now) (Duration: $(( $(date +%s) - start_time ))s) ===" >> "${PROCESS_FILE}"
        echo ">> DONE"

        # Second pass reloads everything from the cache to verify the files are readable.
        echo "[$(now)] Verification: python -m utils.dataset_loader ${dataset}"
        echo "=== [${dataset}] Verification started at $(now) ===" > "${VERIFY_FILE}"
        if python -m utils.dataset_loader "${dataset}" >> "${VERIFY_FILE}" 2>&1; then
            echo ">> DONE"
            notify "[NOALERT] DONE ${dataset}" "" "${VERIFY_FILE}"
        else
            echo ">> ERROR (see ${VERIFY_FILE})"; n_fail=$((n_fail + 1))
            echo "=== ERROR at $(now) ===" >> "${VERIFY_FILE}"
            notify "ERROR ${dataset}" "Loading failed for dataset=${dataset}" "${VERIFY_FILE}"
        fi
    else
        echo ">> ERROR (see ${PROCESS_FILE})"; n_fail=$((n_fail + 1))
        echo "=== ERROR at $(now) (Duration: $(( $(date +%s) - start_time ))s) ===" >> "${PROCESS_FILE}"
        notify "ERROR ${dataset}" "Execution failed for dataset=${dataset}" "${PROCESS_FILE}"
    fi
done

total=$(( $(date +%s) - first_time ))
time_str="$((total / 3600))h $(((total % 3600) / 60))m $((total % 60))s"

TREE_FILE="${root}/preprocess_tree.txt"
if command -v tree &> /dev/null; then tree "${root}" -L 3 > "${TREE_FILE}"; else ls -R "${root}" > "${TREE_FILE}"; fi

echo "ALL DONE (${time_str}, failures: ${n_fail})"
notify "ALL DONE" "Total time consuming: ${time_str}" "${TREE_FILE}"
exit "${n_fail}"
