#!/usr/bin/env bash
set -u

cd /home/yuhp/Rec/RecBole-CDR || exit 1
mkdir -p log/gducdrdirect_full

PYTHON_BIN=/home/yuhp/.conda/envs/torch/bin/python
GPU_OVERRIDE=${GPU_OVERRIDE:-configs/gpu0_override.yaml}

wait_for_ducdrdirect() {
  while pgrep -f "run_recbole_cdr.py --model DUCDRDirect" >/dev/null; do
    echo "[$(date '+%F %T')] waiting for existing DUCDRDirect process" | tee -a log/gducdrdirect_full/queue.log
    sleep 120
  done
}

run_one() {
  local tag="$1"
  local config_files="$2"
  local log_file="log/gducdrdirect_full/${tag}.log"

  echo "[$(date '+%F %T')] starting ${tag}" | tee -a log/gducdrdirect_full/queue.log
  "${PYTHON_BIN}" run_recbole_cdr.py \
    --model GDUCDRDirect \
    --config_files "${config_files}" \
    > "${log_file}" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] finished ${tag} status=${status}" | tee -a log/gducdrdirect_full/queue.log
  return "${status}"
}

wait_for_ducdrdirect

status=0
run_one "sport_cloth_GDUCDRDirect" "configs/sport_cloth_gducdrdirect_unicdr_data.yaml ${GPU_OVERRIDE}" || status=$?
run_one "cloth_sport_GDUCDRDirect" "configs/sport_cloth_gducdrdirect_unicdr_data.yaml configs/cloth_sport_unicdr_data_override.yaml ${GPU_OVERRIDE}" || status=$?

echo "[$(date '+%F %T')] GDUCDRDirect full queue complete status=${status}" | tee -a log/gducdrdirect_full/queue.log
exit "${status}"
