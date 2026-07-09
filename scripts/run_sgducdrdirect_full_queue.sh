#!/usr/bin/env bash
set -u

cd /home/yuhp/Rec/RecBole-CDR || exit 1
mkdir -p log/sgducdrdirect_full

PYTHON_BIN=/home/yuhp/.conda/envs/torch/bin/python
GPU_OVERRIDE=${GPU_OVERRIDE:-configs/gpu1_override.yaml}

run_one() {
  local tag="$1"
  local config_files="$2"
  local log_file="log/sgducdrdirect_full/${tag}.log"
  echo "[$(date '+%F %T')] starting ${tag}" | tee -a log/sgducdrdirect_full/queue.log
  "${PYTHON_BIN}" run_recbole_cdr.py --model SGDUCDRDirect --config_files "${config_files}" > "${log_file}" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] finished ${tag} status=${status}" | tee -a log/sgducdrdirect_full/queue.log
  return "${status}"
}

status=0
run_one "sport_cloth_SGDUCDRDirect" "configs/sport_cloth_sgducdrdirect_unicdr_data.yaml ${GPU_OVERRIDE}" || status=$?
run_one "cloth_sport_SGDUCDRDirect" "configs/sport_cloth_sgducdrdirect_unicdr_data.yaml configs/cloth_sport_unicdr_data_override.yaml ${GPU_OVERRIDE}" || status=$?
echo "[$(date '+%F %T')] SGDUCDRDirect full queue complete status=${status}" | tee -a log/sgducdrdirect_full/queue.log
exit "${status}"
