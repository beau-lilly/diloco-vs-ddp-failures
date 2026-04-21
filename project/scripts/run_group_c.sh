#!/usr/bin/env bash
# Group C — straggler secondary cut: DDP + DiLoCo H=50, 2 seeds each.
# 4 runs total.
set -euo pipefail

: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

for SEED in 0 1; do
  bash scripts/run_one_clean.sh ddp straggler "" "$SEED"
  bash scripts/run_one_clean.sh diloco straggler 50 "$SEED"
done

echo "[group-c] done"
