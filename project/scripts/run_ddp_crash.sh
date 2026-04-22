#!/usr/bin/env bash
# DDP crash run — in-process virtual crashes (no sidecar, no SIGKILL, no
# torchrun --max_restarts cascade). The trainer itself fires a synchronous
# "pause + roll-back-to-checkpoint" event at each token threshold.
#
# Args:
#   $1 seed
#   $2 crash thresholds (comma-separated token counts, e.g. "6684672,13369344,20054016")
set -euo pipefail

SEED="${1:?seed}"
SCHEDULE="${2:?comma-separated token thresholds}"

: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

RUNTIME_DIR="runtime/ddp_crash_s${SEED}"
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
    --framework ddp \
    --failure crash \
    --seed "$SEED" \
    --config config/ddp.yaml \
    --runtime-dir "$RUNTIME_DIR" \
    --crash-schedule "$SCHEDULE"
