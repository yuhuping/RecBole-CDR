#!/usr/bin/env bash

set -u

MODE="${1:-all}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_LOG_DIR="${ROOT_DIR}/log/rerun_overlap"
LAUNCH_LOG="${RUN_LOG_DIR}/parallel_launch.log"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate torch
cd "${ROOT_DIR}"
mkdir -p "${RUN_LOG_DIR}"

launch_model() {
    local model="$1"
    local config="$2"
    local gpu_id="$3"
    local output="${RUN_LOG_DIR}/${model}.log"
    local pid_file="${RUN_LOG_DIR}/${model}.pid"

    if [[ -f "${pid_file}" ]]; then
        local existing_pid
        existing_pid="$(cat "${pid_file}")"
        if kill -0 "${existing_pid}" 2>/dev/null; then
            printf '%s SKIP model=%s pid=%s already running\n' \
                "$(date '+%F %T')" "${model}" "${existing_pid}" | tee -a "${LAUNCH_LOG}"
            return
        fi
    fi

    nohup setsid python run_recbole_cdr.py \
        --model="${model}" \
        --config_files="${config}" \
        --gpu_id="${gpu_id}" > "${output}" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "${pid}" > "${pid_file}"
    printf '%s START model=%s gpu=%s pid=%s config=%s\n' \
        "$(date '+%F %T')" "${model}" "${gpu_id}" "${pid}" "${config}" | tee -a "${LAUNCH_LOG}"
}

if [[ "${MODE}" == "all" ]]; then
    launch_model CUT_24 configs/sport_cloth_cut24_loo_uni999.yaml 1
elif [[ "${MODE}" != "remaining" ]]; then
    printf 'Usage: %s [all|remaining]\n' "$0" >&2
    exit 2
fi

launch_model CMF configs/sport_cloth_cmf_loo_uni999.yaml 1
launch_model CoNet configs/sport_cloth_conet_loo_uni999.yaml 1
launch_model DisenCDR configs/sport_cloth_disencdr_loo_uni999.yaml 2
launch_model DRLCDR configs/sport_cloth_drlcdr_loo_uni999.yaml 3
