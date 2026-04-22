#!/usr/bin/env bash
# Re-run the Group B cells that failed in the parallel run, sequentially.
# Uses only GPUs 0-3 (single lane) to avoid the contention that caused the
# earlier OOM-kill + rdzv failures.
#
# Usage:
#   T_BASELINE=26738688 bash scripts/fill_missing_b.sh ddp_s0 ddp_s1 H50_s0 H50_s1 H100_s0
#
# Cell format: {ddp|Hxxx}_s{seed}

set -euo pipefail

: "${T_BASELINE:?set T_BASELINE (tokens)}"
: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

SCHEDULE="$(python3 -c "t=int('$T_BASELINE'); print(','.join(str(int(t*f)) for f in (0.25, 0.5, 0.75)))")"
echo "[fill-missing] crash schedule: $SCHEDULE"

# Force all runs onto the first 4 GPUs, single lane, distinct MASTER_PORT per
# run so we never have stale rdzv state colliding.
export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT_BASE=29700

i=0
for cell in "$@"; do
  port=$((PORT_BASE + i))
  i=$((i + 1))

  if [[ "$cell" == ddp_s* ]]; then
    seed="${cell#ddp_s}"
    runtime_dir="runtime/ddp_crash_s${seed}"
    echo "[fill-missing] >>> re-running DDP crash seed=$seed port=$port <<<"
    rm -rf "$runtime_dir"
    MASTER_PORT=$port bash scripts/run_ddp_crash.sh "$seed" "$SCHEDULE" || {
      echo "[fill-missing] FAILED: $cell (continuing to next)"
      continue
    }
  elif [[ "$cell" == H*_s* ]]; then
    H="${cell#H}"; H="${H%_s*}"
    seed="${cell##*_s}"
    runtime_dir="runtime/diloco_H${H}_crash_s${seed}"
    echo "[fill-missing] >>> re-running DiLoCo H=$H seed=$seed port=$port <<<"
    rm -rf "$runtime_dir"
    MASTER_PORT=$port bash scripts/run_diloco_crash.sh "$seed" "$H" "$SCHEDULE" || {
      echo "[fill-missing] FAILED: $cell (continuing to next)"
      continue
    }
  else
    echo "[fill-missing] unknown cell format: $cell (expected ddp_sN or HxxxxyN)"
    continue
  fi

  # Belt-and-suspenders: kill any stragglers between cells.
  sleep 3
  pkill -9 -f torchrun 2>/dev/null || true
  pkill -9 -f train.py 2>/dev/null || true
  sleep 3
done

echo "[fill-missing] all cells attempted"
