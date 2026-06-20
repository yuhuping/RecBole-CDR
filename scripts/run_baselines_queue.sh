#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/yuhp/.conda/envs/torch/bin/python"
LOG_DIR="${ROOT_DIR}/log/baseline_queue"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

run_one() {
    local model="$1"
    local stem="$2"
    local direction="$3"
    local configs="$4"
    local log_file="${LOG_DIR}/${direction}_${model}.log"

    printf '[%s] starting %s %s\n' \
        "$(date '+%F %T')" "${direction}" "${model}" | tee -a "${log_file}"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" run_recbole_cdr.py \
        --model="${model}" \
        --config_files="${configs}" \
        >>"${log_file}" 2>&1
    local status=$?
    printf '[%s] finished %s %s status=%d\n' \
        "$(date '+%F %T')" "${direction}" "${model}" "${status}" | tee -a "${log_file}"
    return "${status}"
}

models=(
    "CMF:cmf"
    "CoNet:conet"
    "DisenCDR:disencdr"
    "DRLCDR:drlcdr"
    "CUT_24:cut24"
    "UniCDR:unicdr"
)

overall_status=0
for pair in "${models[@]}"; do
    IFS=: read -r model stem <<<"${pair}"
    run_one \
        "${model}" \
        "${stem}" \
        "sport_cloth" \
        "configs/sport_cloth_${stem}_unicdr_data.yaml"
    status=$?
    (( status != 0 )) && overall_status=1

    run_one \
        "${model}" \
        "${stem}" \
        "cloth_sport" \
        "configs/sport_cloth_${stem}_unicdr_data.yaml configs/cloth_sport_unicdr_data_override.yaml"
    status=$?
    (( status != 0 )) && overall_status=1
done

printf '[%s] baseline queue complete status=%d\n' \
    "$(date '+%F %T')" "${overall_status}" | tee -a "${LOG_DIR}/queue.log"
exit "${overall_status}"
