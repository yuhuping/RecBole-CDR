#!/usr/bin/env bash
set -u

cd /home/yuhp/Rec/RecBole-CDR || exit 1
mkdir -p log/gducdr_full

PYTHON_BIN=/home/yuhp/.conda/envs/torch/bin/python
GPU_DEVICE=${GPU_DEVICE:-1}

run_one() {
  local tag="$1"
  local config_files="$2"
  local log_file="log/gducdr_full/${tag}.log"

  echo "[$(date '+%F %T')] starting ${tag} on GPU ${GPU_DEVICE}" | tee -a log/gducdr_full/queue.log
  CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" "${PYTHON_BIN}" run_recbole_cdr.py \
    --model GDUCDR \
    --config_files "${config_files}" \
    > "${log_file}" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] finished ${tag} status=${status}" | tee -a log/gducdr_full/queue.log
  return "${status}"
}

status=0
run_one "sport_cloth_GDUCDR" "configs/sport_cloth_gducdr_unicdr_data.yaml configs/gpu1_override.yaml" || status=$?
run_one "cloth_sport_GDUCDR" "configs/sport_cloth_gducdr_unicdr_data.yaml configs/cloth_sport_unicdr_data_override.yaml configs/gpu1_override.yaml" || status=$?

echo "[$(date '+%F %T')] GDUCDR full queue complete status=${status}" | tee -a log/gducdr_full/queue.log
exit "${status}"
