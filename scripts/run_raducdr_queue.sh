#!/usr/bin/env bash
set -uo pipefail

cd /home/yuhp/Rec/RecBole-CDR || exit 1

LOG_DIR="log/raducdr_queue"
mkdir -p "${LOG_DIR}"

PYTHON_BIN=/home/yuhp/.conda/envs/torch/bin/python
GPU_DEVICE=${GPU_DEVICE:-2}

run_one() {
  local tag="$1"
  local config_files="$2"
  local log_file="${LOG_DIR}/${tag}.log"

  printf '[%s] starting %s\n' "$(date '+%F %T')" "${tag}" \
    | tee -a "${LOG_DIR}/queue.log"
  CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" "${PYTHON_BIN}" run_recbole_cdr.py \
    --model RADUCDR \
    --config_files "${config_files}" \
    > "${log_file}" 2>&1
  local status=$?
  printf '[%s] finished %s status=%d\n' "$(date '+%F %T')" "${tag}" "${status}" \
    | tee -a "${LOG_DIR}/queue.log"
  return "${status}"
}

status=0

run_one "sport_cloth_RADUCDR_rank01" \
  "configs/sport_cloth_raducdr_rank01.yaml" || status=$?
run_one "cloth_sport_RADUCDR_rank01" \
  "configs/sport_cloth_raducdr_rank01.yaml configs/cloth_sport_unicdr_data_override.yaml" || status=$?
run_one "sport_cloth_RADUCDR_rank02" \
  "configs/sport_cloth_raducdr_rank02.yaml" || status=$?
run_one "cloth_sport_RADUCDR_rank02" \
  "configs/sport_cloth_raducdr_rank02.yaml configs/cloth_sport_unicdr_data_override.yaml" || status=$?

printf '[%s] RADUCDR queue complete status=%d\n' "$(date '+%F %T')" "${status}" \
  | tee -a "${LOG_DIR}/queue.log"
exit "${status}"
