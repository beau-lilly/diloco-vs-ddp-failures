#!/usr/bin/env bash
# Group A — clean H sweep: DDP + DiLoCo H ∈ {10, 50, 100, 500}, 2 seeds each.
# 10 runs total.
set -euo pipefail

: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

for SEED in 0 1; do
  bash scripts/run_one_clean.sh ddp none "" "$SEED"
  for H in 10 50 100 500; do
    bash scripts/run_one_clean.sh diloco none "$H" "$SEED"
  done
done

echo "[group-a] done. compute T_baseline from the DDP runs:"
echo "  python - <<'PY'"
echo "from pathlib import Path; import glob, json"
echo "# ... pull tokens_to_target from wandb or local logs"
echo "PY"
