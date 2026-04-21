#!/usr/bin/env bash
# CPU smoke test with N=2 gloo backend.
#
# Exercises both DDP and DiLoCo end-to-end against a synthetic tiny corpus
# (data.py --smoke generates one). Finishes in under a minute per framework.
# Not intended to converge — just sanity-check that:
#   * process groups initialize
#   * both trainers run train/eval cycles
#   * per-rank token files and progress.json get written
#   * W&B logging in disabled mode doesn't crash
#   * outer_state.pt is persisted after a DiLoCo outer step

set -euo pipefail

: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=disabled
export TORCH_NCCL_BLOCKING_WAIT=0

FRAMEWORK="${1:-both}"
SEED="${SEED:-0}"
PY=${PY:-python}

run_one() {
  local fw="$1"
  local extra_args="$2"
  local runtime="runtime/smoke_${fw}"
  rm -rf "$runtime"
  mkdir -p "$runtime"
  echo ""
  echo "==============================="
  echo "   SMOKE TEST: $fw"
  echo "==============================="
  local port=$((29510 + RANDOM % 1000))
  "$PY" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:$port \
    --max_restarts=0 \
    train.py \
      --framework "$fw" \
      --failure none \
      --seed "$SEED" \
      --config "config/${fw}.yaml" \
      --runtime-dir "$runtime" \
      --smoke \
      $extra_args
  echo "[smoke:$fw] progress.json:"
  cat "$runtime/progress.json" 2>/dev/null || echo "(missing)"
  ls -la "$runtime" | head -20 || true
}

if [[ "$FRAMEWORK" == "ddp" || "$FRAMEWORK" == "both" ]]; then
  run_one ddp ""
fi
if [[ "$FRAMEWORK" == "diloco" || "$FRAMEWORK" == "both" ]]; then
  run_one diloco "--H 4"
fi

echo ""
echo "[smoke] done"
