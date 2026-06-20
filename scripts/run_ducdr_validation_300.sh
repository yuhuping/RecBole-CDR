#!/usr/bin/env bash
set -u

cd /home/yuhp/Rec/RecBole-CDR || exit 1
mkdir -p log/ducdr_validation

PYTHON_BIN=/home/yuhp/.conda/envs/torch/bin/python
GPU_DEVICE=${GPU_DEVICE:-2}

run_one() {
  local tag="$1"
  local model="$2"
  local config_files="$3"
  local log_file="log/ducdr_validation/${tag}.log"

  echo "[$(date '+%F %T')] starting ${tag}" | tee -a log/ducdr_validation/queue.log
  CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" "${PYTHON_BIN}" run_recbole_cdr.py \
    --model "${model}" \
    --config_files "${config_files}" \
    > "${log_file}" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] finished ${tag} status=${status}" | tee -a log/ducdr_validation/queue.log
  return "${status}"
}

status=0

run_one "sport_cloth_DUCDRLite_300" "DUCDRLite" "configs/sport_cloth_ducdrlite_300.yaml" || status=$?
run_one "sport_cloth_DUCDRDirect_300" "DUCDRDirect" "configs/sport_cloth_ducdrdirect_300.yaml" || status=$?
run_one "sport_cloth_DADUCDR_300" "DADUCDR" "configs/sport_cloth_daducdr_300.yaml" || status=$?

echo "[$(date '+%F %T')] DUCDR validation queue complete status=${status}" | tee -a log/ducdr_validation/queue.log
exit "${status}"
