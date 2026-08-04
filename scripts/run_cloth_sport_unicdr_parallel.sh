#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/log/cloth_sport_unicdr"
OVERRIDE_CONFIG="configs/cloth_sport_unicdr_data_override.yaml"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate torch
cd "${ROOT_DIR}"

"${ROOT_DIR}/scripts/build_unicdr_cloth_sport.sh"
mkdir -p "${LOG_DIR}"

launch_model() {
    local model="$1"
    local base_config="$2"
    local gpu="$3"
    local log_file="${LOG_DIR}/${model}.log"
    local pid_file="${LOG_DIR}/${model}.pid"

    if [[ -f "${pid_file}" ]]; then
        local old_pid
        old_pid="$(cat "${pid_file}")"
        if kill -0 "${old_pid}" 2>/dev/null; then
            printf 'SKIP %-10s pid=%s is already running\n' "${model}" "${old_pid}"
            return
        fi
    fi

    # RecBole sets CUDA_VISIBLE_DEVICES from gpu_id during configuration.
    # Pass the physical GPU here instead of pre-setting CUDA_VISIBLE_DEVICES.
    nohup setsid python -u run_recbole_cdr.py \
        --model="${model}" \
        --config_files="${base_config} ${OVERRIDE_CONFIG}" \
        --gpu_id="${gpu}" > "${log_file}" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "${pid}" > "${pid_file}"
    printf 'START %-10s gpu=%s pid=%s log=%s\n' "${model}" "${gpu}" "${pid}" "${log_file}"
}

# Override these environment variables if a GPU is occupied.
launch_model CUT_24  configs/sport_cloth_cut24_unicdr_data.yaml  "${GPU_CUT:-0}"
launch_model CMF     configs/sport_cloth_cmf_unicdr_data.yaml    "${GPU_CMF:-0}"
launch_model UniCDR  configs/sport_cloth_unicdr_unicdr_data.yaml "${GPU_UNICDR:-1}"
launch_model CoNet   configs/sport_cloth_conet_unicdr_data.yaml  "${GPU_CONET:-2}"
launch_model DisenCDR configs/sport_cloth_disencdr_unicdr_data.yaml "${GPU_DISEN:-2}"
launch_model DRLCDR  configs/sport_cloth_drlcdr_unicdr_data.yaml "${GPU_DRL:-3}"

printf 'All launch requests submitted. Logs: %s\n' "${LOG_DIR}"
