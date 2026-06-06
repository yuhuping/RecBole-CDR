#!/usr/bin/env bash

set -u

GPU_ID="${1:?usage: $0 <gpu_id>}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_LOG_DIR="${ROOT_DIR}/log/rerun_overlap"
QUEUE_LOG="${RUN_LOG_DIR}/gpu${GPU_ID}_queue.log"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate torch
cd "${ROOT_DIR}"
mkdir -p "${RUN_LOG_DIR}"

run_model() {
    local model="$1"
    local config="$2"
    local output="${RUN_LOG_DIR}/${model}.log"

    printf '%s START model=%s config=%s gpu=%s\n' "$(date '+%F %T')" "${model}" "${config}" "${GPU_ID}" >> "${QUEUE_LOG}"
    python run_recbole_cdr.py \
        --model="${model}" \
        --config_files="${config}" \
        --gpu_id="${GPU_ID}" > "${output}" 2>&1
    local status=$?
    printf '%s END model=%s status=%s\n' "$(date '+%F %T')" "${model}" "${status}" >> "${QUEUE_LOG}"
}

if [[ "${GPU_ID}" == "1" ]]; then
    run_model CUT_24 configs/sport_cloth_cut24_loo_uni999.yaml
    run_model DisenCDR configs/sport_cloth_disencdr_loo_uni999.yaml
    run_model CMF configs/sport_cloth_cmf_loo_uni999.yaml
elif [[ "${GPU_ID}" == "3" ]]; then
    run_model DRLCDR configs/sport_cloth_drlcdr_loo_uni999.yaml
    run_model CoNet configs/sport_cloth_conet_loo_uni999.yaml
else
    printf 'Unsupported GPU id: %s\n' "${GPU_ID}" >&2
    exit 2
fi
