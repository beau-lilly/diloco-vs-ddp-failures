#!/usr/bin/env bash
# DiLoCo crash run — direct shell launcher, NOT torchrun.
#
# torchrun's agent tears down survivors on any worker exit regardless of
# --max_restarts, which would destroy DiLoCo's "survivors keep running"
# property (pitfall #10). This launcher spawns N workers directly; each
# calls init_process_group(init_method="env://") itself, and a side-band
# FileStore at $CONTROL_FILESTORE coordinates the rejoin dance.
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
: "${REPLACEMENT_DELAY_SECONDS:=30}"

export MASTER_ADDR="localhost"
export MASTER_PORT
export WORLD_SIZE="$NPROC"
export CONTROL_FILESTORE="/tmp/diloco_control_s${SEED}_H${H}"
rm -f "$CONTROL_FILESTORE"

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# The sidecar spawns replacement workers using this helper.
cat > "$RUNTIME_DIR/spawn_replacement.sh" <<'SPAWN_EOF'
#!/usr/bin/env bash
# Spawn a single DiLoCo replacement worker. RANK and CRASH_EPOCH are passed
# in via the sidecar's environment.
set -euo pipefail
cd "$PROJECT_ROOT"
exec python train.py \
  --framework diloco \
  --failure crash \
  --seed "$SEED" \
  --H "$H" \
  --config config/diloco.yaml \
  --runtime-dir "$RUNTIME_DIR" \
  --control-filestore "$CONTROL_FILESTORE" \
  --rejoin \
  --crash-epoch "$CRASH_EPOCH"
SPAWN_EOF
chmod +x "$RUNTIME_DIR/spawn_replacement.sh"

# Environment for both initial workers and the spawn helper
export PROJECT_ROOT SEED H RUNTIME_DIR CONTROL_FILESTORE

WORKER_PIDS=()
for ((r=0; r<NPROC; r++)); do
  RANK="$r" LOCAL_RANK="$r" \
  python train.py \
    --framework diloco \
    --failure crash \
    --seed "$SEED" \
    --H "$H" \
    --config config/diloco.yaml \
    --runtime-dir "$RUNTIME_DIR" \
    --control-filestore "$CONTROL_FILESTORE" &
  WORKER_PIDS+=($!)
done

python sidecar_crash_controller.py \
    --framework diloco \
    --schedule "$SCHEDULE" \
    --seed "$SEED" \
    --runtime-dir "$RUNTIME_DIR" \
    --control-filestore "$CONTROL_FILESTORE" \
    --world-size "$NPROC" \
    --replacement-delay-seconds "$REPLACEMENT_DELAY_SECONDS" \
    --diloco-replacement-cmd "bash $RUNTIME_DIR/spawn_replacement.sh" \
    --log "$RUNTIME_DIR/sidecar.jsonl" &
SIDECAR_PID=$!

cleanup() {
  kill "$SIDECAR_PID" 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

# Wait for at least one worker to exit. Note: after a crash + rejoin, the
# original PID for the victim rank is dead, but a replacement PID is alive.
# We wait for all initially-launched PIDs; once they all exit, the run is done.
FAIL=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    # Expected for the victim worker(s) — they were SIGKILLed. Don't error.
    :
  fi
done

kill "$SIDECAR_PID" 2>/dev/null || true
