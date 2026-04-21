#!/usr/bin/env bash
# Run a single clean / straggler cell via torchrun.
#
# Args:
#   $1 framework (ddp|diloco)
#   $2 failure   (none|straggler)
#   $3 H         (DiLoCo only; pass "" for DDP)
#   $4 seed
#   $5 runtime-dir (optional; default runtime/<cell>_s<seed>)
#
# Requires PROJECT_ROOT env var pointing at project/
set -euo pipefail

FRAMEWORK="${1:?framework}"
FAILURE="${2:?failure (none|straggler)}"
H_ARG="${3:-}"
SEED="${4:?seed}"
RUNTIME_DIR="${5:-}"

: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ -z "$RUNTIME_DIR" ]]; then
  CELL_TAG="${FRAMEWORK}_${FAILURE}"
  [[ -n "$H_ARG" ]] && CELL_TAG="${FRAMEWORK}_H${H_ARG}_${FAILURE}"
  RUNTIME_DIR="runtime/${CELL_TAG}_s${SEED}"
fi
mkdir -p "$RUNTIME_DIR"

CFG="config/${FRAMEWORK}.yaml"
EXTRA_ARGS=()
[[ -n "$H_ARG" && "$FRAMEWORK" == "diloco" ]] && EXTRA_ARGS+=(--H "$H_ARG")

: "${NPROC:=4}"
: "${MASTER_PORT:=29500}"
: "${NCCL_TIMEOUT_SECONDS:=120}"

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="localhost:$MASTER_PORT" \
  --max_restarts=0 \
  train.py \
    --framework "$FRAMEWORK" \
    --failure "$FAILURE" \
    --seed "$SEED" \
    --config "$CFG" \
    --runtime-dir "$RUNTIME_DIR" \
    "${EXTRA_ARGS[@]}"
