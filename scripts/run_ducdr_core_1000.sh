#!/usr/bin/env bash
set -u
cd /home/yuhp/Rec/RecBole-CDR || exit 1
mkdir -p log/ducdr_core_1000
PYTHON_BIN=/home/yuhp/.conda/envs/torch/bin/python
GPU_DEVICE=${GPU_DEVICE:-1}
run_one() {
  local tag="$1"
  local model="$2"
  local config_files="$3"
  local log_file="log/ducdr_core_1000/${tag}.log"
  echo "[$(date +%F %T)] starting ${tag}" | tee -a log/ducdr_core_1000/queue.log
  CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" "${PYTHON_BIN}" run_recbole_cdr.py --model "${model}" --config_files "${config_files}" > "${log_file}" 2>&1
  local status=$?
  echo "[$(date +%F %T)] finished ${tag} status=${status}" | tee -a log/ducdr_core_1000/queue.log
  return "${status}"
}
status=0
run_one "sport_cloth_DUCDRLite_1000" "DUCDRLite" "configs/sport_cloth_ducdrlite_1000.yaml" || status=$?
run_one "cloth_sport_DUCDRLite_1000" "DUCDRLite" "configs/sport_cloth_ducdrlite_1000.yaml configs/cloth_sport_unicdr_data_override.yaml" || status=$?
run_one "sport_cloth_DUCDRDirect_1000" "DUCDRDirect" "configs/sport_cloth_ducdrdirect_1000.yaml" || status=$?
run_one "cloth_sport_DUCDRDirect_1000" "DUCDRDirect" "configs/sport_cloth_ducdrdirect_1000.yaml configs/cloth_sport_unicdr_data_override.yaml" || status=$?
echo "[$(date +%F %T)] DUCDR core 1000 queue complete status=${status}" | tee -a log/ducdr_core_1000/queue.log
exit "${status}"
