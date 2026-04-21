#!/usr/bin/env bash
# Group B in parallel on 8 GPUs — two 4-GPU lanes, one seed per lane.
# Requires T_BASELINE.
set -euo pipefail

: "${T_BASELINE:?set T_BASELINE (tokens) from Group A DDP mean}"
: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

SCHEDULE="$(python3 -c "t=int('$T_BASELINE'); print(','.join(str(int(t*f)) for f in (0.25,0.5,0.75)))")"
echo "[parallel-group-b] crash thresholds: $SCHEDULE"

run_lane() {
  local lane=$1
  local seed=$2
  local gpus
  local port
  if [[ $lane -eq 0 ]]; then gpus="0,1,2,3"; port=29500; else gpus="4,5,6,7"; port=29600; fi
  {
    CUDA_VISIBLE_DEVICES=$gpus MASTER_PORT=$port bash scripts/run_ddp_crash.sh "$seed" "$SCHEDULE"
    for H in 10 50 100 500; do
      CUDA_VISIBLE_DEVICES=$gpus MASTER_PORT=$port bash scripts/run_diloco_crash.sh "$seed" "$H" "$SCHEDULE"
    done
  } 2>&1 | sed "s/^/[lane${lane}] /"
}

run_lane 0 0 &
LANE0_PID=$!
run_lane 1 1 &
LANE1_PID=$!

trap "kill $LANE0_PID $LANE1_PID 2>/dev/null; wait" EXIT
wait $LANE0_PID $LANE1_PID
echo "[parallel-group-b] both lanes done"
