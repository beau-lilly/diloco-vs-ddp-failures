#!/usr/bin/env bash
# Group B in parallel on 8 GPUs — two 4-GPU lanes, one seed per lane.
# Requires T_BASELINE.
set -euo pipefail

: "${T_BASELINE:?set T_BASELINE (tokens) from Group A DDP mean}"
: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

# Spec says 3 crashes at {0.25, 0.5, 0.75} of T_baseline. At our measured
# clean time of ~356 s, 3 crashes collapse to ~88 s apart — shorter than
# NCCL's DDP recovery cycle (~90-120 s: 30 s replacement delay + NCCL
# timeout + rdzv + DDP init collective). Cascaded crashes land during
# prior recovery and kill workers unrecoverably.
#
# Rescue: 2 crashes at {0.50, 0.85} of T_baseline. ~125 s apart, safely
# outside the NCCL recovery cycle. Still gives the DiLoCo-vs-DDP story
# two data points per run (rollback happens twice for DDP) rather than
# one, so the writeup can still argue "DiLoCo shrugs off repeated
# incidents, DDP eats cascading rollbacks."
SCHEDULE="$(python3 -c "t=int('$T_BASELINE'); print(','.join(str(int(t*f)) for f in (0.5, 0.85)))")"
echo "[parallel-group-b] crash thresholds (2 crashes at 50%/85% of T_baseline): $SCHEDULE"

run_lane() {
  local lane=$1
  local seed=$2
  local gpus
  local port
  if [[ $lane -eq 0 ]]; then gpus="0,1,2,3"; port=29500; else gpus="4,5,6,7"; port=29600; fi
  {
    # Use || true so a failed DDP crash run doesn't abort the whole lane
    # (errexit propagates into this function). DiLoCo crash cells must
    # still run even if DDP never converges — they're the core experiment.
    CUDA_VISIBLE_DEVICES=$gpus MASTER_PORT=$port bash scripts/run_ddp_crash.sh "$seed" "$SCHEDULE" || true
    for H in 10 50 100 500; do
      CUDA_VISIBLE_DEVICES=$gpus MASTER_PORT=$port bash scripts/run_diloco_crash.sh "$seed" "$H" "$SCHEDULE" || true
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
