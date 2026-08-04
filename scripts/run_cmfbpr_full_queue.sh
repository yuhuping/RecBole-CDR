#!/usr/bin/env bash
set -u

cd /home/yuhp/Rec/RecBole-CDR || exit 1
mkdir -p log/cmfbpr_full

PYTHON_BIN=/home/yuhp/.conda/envs/torch/bin/python
GPU_OVERRIDE=${GPU_OVERRIDE:-configs/gpu0_override.yaml}

run_one() {
  local tag="$1"
  local config_files="$2"
  local log_file="log/cmfbpr_full/${tag}.log"
  echo "[$(date '+%F %T')] starting ${tag}" | tee -a log/cmfbpr_full/queue.log
  "${PYTHON_BIN}" run_recbole_cdr.py --model CMFBPR --config_files "${config_files}" > "${log_file}" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] finished ${tag} status=${status}" | tee -a log/cmfbpr_full/queue.log
  return "${status}"
}

status=0
run_one "sport_cloth_CMFBPR" "configs/sport_cloth_cmfbpr_unicdr_data.yaml ${GPU_OVERRIDE}" || status=$?
run_one "cloth_sport_CMFBPR" "configs/sport_cloth_cmfbpr_unicdr_data.yaml configs/cloth_sport_unicdr_data_override.yaml ${GPU_OVERRIDE}" || status=$?
echo "[$(date '+%F %T')] CMFBPR full queue complete status=${status}" | tee -a log/cmfbpr_full/queue.log
exit "${status}"
