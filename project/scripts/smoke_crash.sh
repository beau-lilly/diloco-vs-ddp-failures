#!/usr/bin/env bash
# Crash-injection dry run on CPU (gloo backend).
#
# Arg $1: framework (ddp|diloco)
#
# Each variant uses a tiny model, a never-reached target loss (so training
# keeps going), and shortened crash thresholds so the first crash fires ~5s
# into the run. This exercises the full sidecar → SIGKILL → recovery path
# locally before committing to Lambda runs.

set -euo pipefail

FRAMEWORK="${1:?framework (ddp|diloco)}"
: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=disabled

: "${PY:=python}"

# Use tiny N=2 for CPU. The production runs will use N=4 on Lambda.
WORLD_SIZE=2
SEED=0
# 4096 / 8192 tokens ≈ 16 / 32 minibatches at B=4, T=64 → ~1-2s apart on CPU.
SCHEDULE="4096,8192"
# 5-second replacement delay so the dry-run finishes quickly.
REPLACEMENT_DELAY=5
RUNTIME_DIR="runtime/smoke_crash_${FRAMEWORK}"
rm -rf "$RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR"

if [[ "$FRAMEWORK" == "ddp" ]]; then
  export DDP_REPLACEMENT_DELAY_SECONDS="$REPLACEMENT_DELAY"
  "$PY" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="$WORLD_SIZE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29520 \
    --max_restarts=5 \
    --monitor_interval=3 \
    train.py \
      --framework ddp \
      --failure crash \
      --seed "$SEED" \
      --config config/ddp.yaml \
      --runtime-dir "$RUNTIME_DIR" \
      --smoke \
      --smoke-target-loss 0.1 \
      --smoke-max-wall-clock 60 &
  TORCHRUN_PID=$!

  "$PY" sidecar_crash_controller.py \
    --framework ddp \
    --schedule "$SCHEDULE" \
    --seed "$SEED" \
    --runtime-dir "$RUNTIME_DIR" \
    --world-size "$WORLD_SIZE" \
    --replacement-delay-seconds "$REPLACEMENT_DELAY" \
    --log "$RUNTIME_DIR/sidecar.jsonl" &
  SIDECAR_PID=$!

  cleanup() {
    kill "$SIDECAR_PID" 2>/dev/null || true
    kill "$TORCHRUN_PID" 2>/dev/null || true
    wait 2>/dev/null || true
  }
  trap cleanup EXIT
  # Give it ~60s max to fire both crashes and recover.
  timeout_pid=""
  ( sleep 90; kill -TERM "$TORCHRUN_PID" 2>/dev/null || true ) &
  timeout_pid=$!
  wait "$TORCHRUN_PID" || true
  kill "$timeout_pid" 2>/dev/null || true
  kill "$SIDECAR_PID" 2>/dev/null || true

elif [[ "$FRAMEWORK" == "diloco" ]]; then
  export MASTER_ADDR=localhost
  export MASTER_PORT=29530
  export WORLD_SIZE
  CONTROL_FILESTORE="/tmp/diloco_smoke_crash"
  rm -f "$CONTROL_FILESTORE"

  # Spawn helper for the sidecar to launch replacements.
  cat > "$RUNTIME_DIR/spawn_replacement.sh" <<SPAWN_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export CUDA_VISIBLE_DEVICES=
export WANDB_MODE=disabled
exec "$PY" train.py \
  --framework diloco \
  --failure crash \
  --seed $SEED \
  --H 4 \
  --config config/diloco.yaml \
  --runtime-dir "$RUNTIME_DIR" \
  --control-filestore "$CONTROL_FILESTORE" \
  --smoke \
  --smoke-target-loss 0.1 \
  --smoke-max-wall-clock 60 \
  --rejoin \
  --crash-epoch "\$CRASH_EPOCH"
SPAWN_EOF
  chmod +x "$RUNTIME_DIR/spawn_replacement.sh"

  WORKERS=()
  for ((r=0; r<WORLD_SIZE; r++)); do
    RANK="$r" LOCAL_RANK="$r" \
    "$PY" train.py \
      --framework diloco \
      --failure crash \
      --seed "$SEED" \
      --H 4 \
      --config config/diloco.yaml \
      --runtime-dir "$RUNTIME_DIR" \
      --control-filestore "$CONTROL_FILESTORE" \
      --smoke \
      --smoke-target-loss 0.1 \
      --smoke-max-wall-clock 60 &
    WORKERS+=($!)
  done

  "$PY" sidecar_crash_controller.py \
    --framework diloco \
    --schedule "$SCHEDULE" \
    --seed "$SEED" \
    --runtime-dir "$RUNTIME_DIR" \
    --control-filestore "$CONTROL_FILESTORE" \
    --world-size "$WORLD_SIZE" \
    --replacement-delay-seconds "$REPLACEMENT_DELAY" \
    --diloco-replacement-cmd "bash $RUNTIME_DIR/spawn_replacement.sh" \
    --log "$RUNTIME_DIR/sidecar.jsonl" &
  SIDECAR_PID=$!

  cleanup() {
    kill "$SIDECAR_PID" 2>/dev/null || true
    for pid in "${WORKERS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  }
  trap cleanup EXIT

  # Bound wall-clock — test doesn't need to run forever.
  ( sleep 90; for pid in "${WORKERS[@]}"; do kill "$pid" 2>/dev/null || true; done ) &
  for pid in "${WORKERS[@]}"; do
    wait "$pid" || true
  done
  kill "$SIDECAR_PID" 2>/dev/null || true

else
  echo "unknown framework: $FRAMEWORK" >&2
  exit 2
fi

echo ""
echo "=== sidecar log ==="
cat "$RUNTIME_DIR/sidecar.jsonl" || echo "(empty)"
echo ""
echo "=== progress.json ==="
cat "$RUNTIME_DIR/progress.json" || echo "(missing)"
