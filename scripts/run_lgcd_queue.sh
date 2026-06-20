#!/usr/bin/env bash
set -euo pipefail

cd /home/yuhp/Rec/RecBole-CDR
mkdir -p log/lgcd_queue

PYTHON_BIN=/home/yuhp/.conda/envs/torch/bin/python
GPU_DEVICE=${GPU_DEVICE:-1}

run_one() {
  local name="$1"
  local config_files="$2"
  local log_file="log/lgcd_queue/${name}.log"
  echo "[$(date '+%F %T')] starting ${name}" | tee -a log/lgcd_queue/queue.log
  CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" "${PYTHON_BIN}" run_recbole_cdr.py \
    --model LGCD \
    --config_files "${config_files}" \
    > "${log_file}" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] finished ${name} status=${status}" | tee -a log/lgcd_queue/queue.log
  return "${status}"
}

run_one "sport_cloth_LGCD" "configs/sport_cloth_lgcd_unicdr_data.yaml"
run_one "cloth_sport_LGCD" "configs/sport_cloth_lgcd_unicdr_data.yaml configs/cloth_sport_unicdr_data_override.yaml"

echo "[$(date '+%F %T')] LGCD queue complete status=0" | tee -a log/lgcd_queue/queue.log
