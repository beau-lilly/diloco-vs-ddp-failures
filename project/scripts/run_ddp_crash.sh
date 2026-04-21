#!/usr/bin/env bash
# DDP crash run via torchrun --max_restarts + out-of-process sidecar.
# torchrun handles the kill-and-respawn cascade; the sidecar sends SIGKILL at
# token thresholds and the respawn path sleeps 30s (via env var) to simulate
# replacement delay.
#
# Args:
#   $1 seed
#   $2 crash thresholds (comma-separated token counts, e.g. "7500000,15000000,22500000")
set -euo pipefail

SEED="${1:?seed}"
SCHEDULE="${2:?comma-separated token thresholds}"

: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

RUNTIME_DIR="runtime/ddp_crash_s${SEED}"
mkdir -p "$RUNTIME_DIR"

: "${NPROC:=4}"
: "${MASTER_PORT:=29500}"
: "${REPLACEMENT_DELAY_SECONDS:=30}"
: "${MAX_RESTARTS:=10}"

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export DDP_REPLACEMENT_DELAY_SECONDS="$REPLACEMENT_DELAY_SECONDS"

# Launch torchrun and sidecar concurrently. Clean up on exit.
torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="localhost:$MASTER_PORT" \
  --max_restarts="$MAX_RESTARTS" \
  --monitor_interval=5 \
  train.py \
    --framework ddp \
    --failure crash \
    --seed "$SEED" \
    --config config/ddp.yaml \
    --runtime-dir "$RUNTIME_DIR" &
TORCHRUN_PID=$!

python sidecar_crash_controller.py \
    --framework ddp \
    --schedule "$SCHEDULE" \
    --seed "$SEED" \
    --runtime-dir "$RUNTIME_DIR" \
    --world-size "$NPROC" \
    --replacement-delay-seconds "$REPLACEMENT_DELAY_SECONDS" \
    --log "$RUNTIME_DIR/sidecar.jsonl" &
SIDECAR_PID=$!

cleanup() {
  kill "$SIDECAR_PID" 2>/dev/null || true
  kill "$TORCHRUN_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

wait "$TORCHRUN_PID"
kill "$SIDECAR_PID" 2>/dev/null || true
