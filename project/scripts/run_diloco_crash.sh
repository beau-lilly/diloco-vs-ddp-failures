#!/usr/bin/env bash
# DiLoCo crash run — in-process virtual crashes (no sidecar, no SIGKILL, no
# destroy/reinit of the process group). At each token threshold the trainer
# picks a victim rank; that rank skips its inner loop for one outer step
# (contributes Δ=0), all ranks pause replacement_delay seconds, outer
# all-reduce proceeds normally.
#
# This replaces the direct-shell + control FileStore launcher from the
# earlier SIGKILL-based design. The destroy/reinit code remains in
# diloco_trainer.py for CPU smoke testing, just unused here.
#
# Args:
#   $1 seed
#   $2 H
#   $3 crash thresholds (comma-separated token counts)
set -euo pipefail

SEED="${1:?seed}"
H="${2:?H}"
SCHEDULE="${3:?comma-separated token thresholds}"

: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

RUNTIME_DIR="runtime/diloco_H${H}_crash_s${SEED}"
mkdir -p "$RUNTIME_DIR"

: "${NPROC:=4}"
: "${MASTER_PORT:=29500}"

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:$MASTER_PORT" \
  --max_restarts=0 \
  train.py \
    --framework diloco \
    --failure crash \
    --seed "$SEED" \
    --H "$H" \
    --config config/diloco.yaml \
    --runtime-dir "$RUNTIME_DIR" \
    --crash-schedule "$SCHEDULE"
