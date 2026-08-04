#!/usr/bin/env bash
# Run the 5 DUCDR/GDUCDR-family models on the 6 new DisenCDR directed tasks.
# Two GPU lanes (GPU0, GPU1) run in parallel; each lane is sequential.
set -u
cd /home/yuhp/Rec/RecBole-CDR
PY=/home/yuhp/.conda/envs/torch/bin/python
LOGDIR=log/disen_queue
mkdir -p "$LOGDIR"

# model -> template config (BOTH:1000, all hyperparams)
declare -A TMPL=(
  [DUCDRDirect]=configs/sport_cloth_ducdrdirect_unicdr_data.yaml
  [DUCDR]=configs/sport_cloth_ducdr_unicdr_data.yaml
  [DUCDRLite]=configs/sport_cloth_ducdrlite_unicdr_data.yaml
  [GDUCDR]=configs/sport_cloth_gducdr_unicdr_data.yaml
  [GDUCDRDirect]=configs/sport_cloth_gducdrdirect_unicdr_data.yaml
)
MODELS=(DUCDRDirect DUCDR DUCDRLite GDUCDR GDUCDRDirect)
TASKS=(cloth_electronic electronic_cloth electronic_phone phone_electronic phone_sport sport_phone)

run_one() {  # gpu task model
  local gpu="$1" task="$2" model="$3"
  local log="$LOGDIR/${task}_${model}.log"
  echo "[$(date '+%F %T')] START gpu=$gpu $task $model -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_recbole_cdr.py \
    --model "$model" \
    --config_files "${TMPL[$model]} configs/disen_${task}_data.yaml" \
    > "$log" 2>&1
  echo "[$(date '+%F %T')] DONE  gpu=$gpu $task $model rc=$?"
}

# Build the full 30-job list (task-major so each GPU sees mixed sizes)
JOBS=()
for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    JOBS+=("$task|$model")
  done
done

# Split round-robin into two lanes
lane0=(); lane1=()
for i in "${!JOBS[@]}"; do
  if (( i % 2 == 0 )); then lane0+=("${JOBS[$i]}"); else lane1+=("${JOBS[$i]}"); fi
done

lane() {  # gpu  jobs...
  local gpu="$1"; shift
  for j in "$@"; do
    run_one "$gpu" "${j%%|*}" "${j##*|}"
  done
  echo "[$(date '+%F %T')] LANE gpu=$gpu FINISHED"
}

lane 0 "${lane0[@]}" &
P0=$!
lane 1 "${lane1[@]}" &
P1=$!
wait $P0 $P1
echo "[$(date '+%F %T')] ALL DONE"
