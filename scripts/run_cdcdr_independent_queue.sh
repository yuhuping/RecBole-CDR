#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/yuhp/.conda/envs/torch/bin/python"
LOG_DIR="${ROOT_DIR}/log/cdcdr_independent"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

run_direction() {
    local name="$1"
    local config_files="$2"
    local log_file="${LOG_DIR}/${name}.log"

    printf '[%s] starting %s\n' "$(date '+%F %T')" "${name}" | tee -a "${log_file}"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" run_recbole_cdr.py \
        --model=CDCDR \
        --config_files="${config_files}" \
        >>"${log_file}" 2>&1
    local status=$?
    printf '[%s] finished %s status=%d\n' \
        "$(date '+%F %T')" "${name}" "${status}" | tee -a "${log_file}"
    return "${status}"
}

run_direction \
    "sport_cloth" \
    "configs/sport_cloth_cdcdr_independent.yaml"
sport_cloth_status=$?

run_direction \
    "cloth_sport" \
    "configs/sport_cloth_cdcdr_independent.yaml configs/cloth_sport_cdcdr_independent_override.yaml"
cloth_sport_status=$?

printf '[%s] queue complete sport_cloth=%d cloth_sport=%d\n' \
    "$(date '+%F %T')" "${sport_cloth_status}" "${cloth_sport_status}" \
    | tee -a "${LOG_DIR}/queue.log"

if (( sport_cloth_status != 0 || cloth_sport_status != 0 )); then
    exit 1
fi
