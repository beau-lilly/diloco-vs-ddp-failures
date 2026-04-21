#!/usr/bin/env bash
# Group B — crash H sweep: DDP + DiLoCo H ∈ {10, 50, 100, 500}, 2 seeds each.
# 10 runs total. Requires T_BASELINE env var (mean DDP tokens-to-target from A).
set -euo pipefail

: "${T_BASELINE:?set T_BASELINE (integer tokens, e.g. 30000000)}"
: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

SCHEDULE="$(python -c "t=int('$T_BASELINE'); print(','.join(str(int(t*f)) for f in (0.25,0.5,0.75)))")"
echo "[group-b] crash thresholds: $SCHEDULE"

for SEED in 0 1; do
  bash scripts/run_ddp_crash.sh "$SEED" "$SCHEDULE"
  for H in 10 50 100 500; do
    bash scripts/run_diloco_crash.sh "$SEED" "$H" "$SCHEDULE"
  done
done

echo "[group-b] done"
