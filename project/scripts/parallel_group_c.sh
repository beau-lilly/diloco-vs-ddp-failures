#!/usr/bin/env bash
# Group C (straggler) in parallel on 8 GPUs.
set -euo pipefail

: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

run_lane() {
  local lane=$1
  local seed=$2
  local gpus
  local port
  if [[ $lane -eq 0 ]]; then gpus="0,1,2,3"; port=29500; else gpus="4,5,6,7"; port=29600; fi
  {
    CUDA_VISIBLE_DEVICES=$gpus MASTER_PORT=$port bash scripts/run_one_clean.sh ddp straggler "" "$seed"
    CUDA_VISIBLE_DEVICES=$gpus MASTER_PORT=$port bash scripts/run_one_clean.sh diloco straggler 50 "$seed"
  } 2>&1 | sed "s/^/[lane${lane}] /"
}

run_lane 0 0 &
LANE0_PID=$!
run_lane 1 1 &
LANE1_PID=$!

trap "kill $LANE0_PID $LANE1_PID 2>/dev/null; wait" EXIT
wait $LANE0_PID $LANE1_PID
echo "[parallel-group-c] both lanes done"
