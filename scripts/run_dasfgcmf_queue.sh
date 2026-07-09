#!/usr/bin/env bash
set -u

cd /home/yuhp/Rec/RecBole-CDR || exit 1
PY=/home/yuhp/.conda/envs/torch/bin/python
GPU_DEVICE=${GPU_DEVICE:-1}
LOGDIR=log/dasfgcmf_queue
mkdir -p "$LOGDIR"

BASE=configs/sport_cloth_dasfgcmf_unicdr_data.yaml
GPU_OVERRIDE=configs/gpu1_override.yaml

TASKS=(
  "sport_cloth|$BASE $GPU_OVERRIDE"
  "cloth_sport|$BASE configs/cloth_sport_unicdr_data_override.yaml $GPU_OVERRIDE"
  "cloth_electronic|$BASE configs/disen_cloth_electronic_data.yaml $GPU_OVERRIDE"
  "electronic_cloth|$BASE configs/disen_electronic_cloth_data.yaml $GPU_OVERRIDE"
  "electronic_phone|$BASE configs/disen_electronic_phone_data.yaml $GPU_OVERRIDE"
  "phone_electronic|$BASE configs/disen_phone_electronic_data.yaml $GPU_OVERRIDE"
  "phone_sport|$BASE configs/disen_phone_sport_data.yaml $GPU_OVERRIDE"
  "sport_phone|$BASE configs/disen_sport_phone_data.yaml $GPU_OVERRIDE"
)

status=0
for entry in "${TASKS[@]}"; do
  task="${entry%%|*}"
  config_files="${entry#*|}"
  log_file="$LOGDIR/${task}_DASFGCMF.log"
  echo "[$(date '+%F %T')] START gpu=${GPU_DEVICE} ${task} DASFGCMF -> ${log_file}" | tee -a "$LOGDIR/queue.log"
  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PY" run_recbole_cdr.py \
    --model DASFGCMF \
    --config_files "$config_files" \
    > "$log_file" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE  gpu=${GPU_DEVICE} ${task} DASFGCMF rc=${rc}" | tee -a "$LOGDIR/queue.log"
  if [ "$rc" -ne 0 ]; then
    status="$rc"
    break
  fi
done

echo "[$(date '+%F %T')] DASFGCMF queue complete status=${status}" | tee -a "$LOGDIR/queue.log"
exit "$status"
