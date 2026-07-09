#!/usr/bin/env bash
set -u

cd /home/yuhp/Rec/RecBole-CDR || exit 1

PY=/home/yuhp/.conda/envs/torch/bin/python
LOGDIR=log/targetonly_gcmf_check
mkdir -p "$LOGDIR"

GPU_OVERRIDE=${GPU_OVERRIDE:-}
WAIT_GPU_INDICES=${WAIT_GPU_INDICES:-"0 1 2 3"}
WAIT_FREE_MIB=${WAIT_FREE_MIB:-10000}
BASE_DIRECT=configs/sport_cloth_targetonlydirect_unicdr_data.yaml
BASE_GCMF=configs/sport_cloth_targetonlygcmf_unicdr_data.yaml

wait_for_gpu() {
  while true; do
    status=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
    best_gpu=""
    best_free=-1
    while IFS=, read -r idx used total; do
      idx=$(echo "$idx" | tr -d ' ')
      used=$(echo "$used" | tr -d ' ')
      total=$(echo "$total" | tr -d ' ')
      case " $WAIT_GPU_INDICES " in
        *" $idx "*) ;;
        *) continue ;;
      esac
      free=$((total - used))
      if [ "$free" -gt "$best_free" ]; then
        best_free="$free"
        best_gpu="$idx"
      fi
    done <<< "$status"

    if [ -n "$best_gpu" ] && [ "$best_free" -ge "$WAIT_FREE_MIB" ]; then
      GPU_OVERRIDE="$LOGDIR/gpu${best_gpu}_override.yaml"
      printf "gpu_id: %s\n" "$best_gpu" > "$GPU_OVERRIDE"
      echo "[$(date '+%F %T')] GPU${best_gpu} free=${best_free}MiB >= ${WAIT_FREE_MIB}MiB, starting queue with ${GPU_OVERRIDE}" | tee -a "$LOGDIR/queue.log"
      break
    fi
    echo "[$(date '+%F %T')] waiting any GPU in [${WAIT_GPU_INDICES}]: best=GPU${best_gpu:-NA} free=${best_free}MiB < ${WAIT_FREE_MIB}MiB" | tee -a "$LOGDIR/queue.log"
    sleep 300
  done
}

run_one() {
  local task="$1"
  local model="$2"
  local config_files="$3"
  local log_file="$LOGDIR/${task}_${model}.log"
  echo "[$(date '+%F %T')] START ${task} ${model} -> ${log_file}" | tee -a "$LOGDIR/queue.log"
  "$PY" run_recbole_cdr.py \
    --model "$model" \
    --config_files "$config_files" \
    > "$log_file" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE  ${task} ${model} rc=${rc}" | tee -a "$LOGDIR/queue.log"
  return "$rc"
}

TASKS=(
  "sport_cloth|$BASE_DIRECT|$BASE_GCMF"
  "cloth_sport|$BASE_DIRECT configs/cloth_sport_unicdr_data_override.yaml|$BASE_GCMF configs/cloth_sport_unicdr_data_override.yaml"
  "cloth_electronic|$BASE_DIRECT configs/disen_cloth_electronic_data.yaml|$BASE_GCMF configs/disen_cloth_electronic_data.yaml"
  "electronic_cloth|$BASE_DIRECT configs/disen_electronic_cloth_data.yaml|$BASE_GCMF configs/disen_electronic_cloth_data.yaml"
  "electronic_phone|$BASE_DIRECT configs/disen_electronic_phone_data.yaml|$BASE_GCMF configs/disen_electronic_phone_data.yaml"
  "phone_electronic|$BASE_DIRECT configs/disen_phone_electronic_data.yaml|$BASE_GCMF configs/disen_phone_electronic_data.yaml"
  "phone_sport|$BASE_DIRECT configs/disen_phone_sport_data.yaml|$BASE_GCMF configs/disen_phone_sport_data.yaml"
  "sport_phone|$BASE_DIRECT configs/disen_sport_phone_data.yaml|$BASE_GCMF configs/disen_sport_phone_data.yaml"
)

wait_for_gpu

status=0
for entry in "${TASKS[@]}"; do
  task="${entry%%|*}"
  rest="${entry#*|}"
  direct_cfg="${rest%%|*} $GPU_OVERRIDE"
  gcmf_cfg="${rest#*|} $GPU_OVERRIDE"
  run_one "$task" "TargetOnlyDirect" "$direct_cfg" || { status=$?; break; }
  run_one "$task" "TargetOnlyGCMF" "$gcmf_cfg" || { status=$?; break; }
done

echo "[$(date '+%F %T')] target-only GCMF check complete status=${status}" | tee -a "$LOGDIR/queue.log"
exit "$status"
