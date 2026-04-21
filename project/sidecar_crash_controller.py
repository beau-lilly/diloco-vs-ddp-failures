"""Out-of-process crash scheduler.

Contract (see measurement-plan.md + implementation-notes.md):

  * Reads `runtime/progress.json` at ~10 Hz.
  * When `tokens_raw` crosses `schedule[next_idx]` AND no recovery is in
    flight, picks a victim rank (deterministic per run via seeded RNG),
    sends SIGKILL, and records the event.
  * Suppresses later thresholds until recovery is complete.
  * Recovery-complete detection differs by framework:
      - DDP: wait for torchrun to restart the workers and tokens_raw to
        exceed the at-crash high-water mark, plus the env-var-injected
        30 s replacement sleep in the respawn path.
      - DiLoCo: sidecar owns the spawn; it sleeps 30 s, spawns a
        replacement with --rejoin --crash-epoch=<N>, and then watches the
        control FileStore for `recovery_complete_<N>`.

Inputs:
  --framework ddp|diloco
  --schedule  comma-separated token thresholds (e.g. "7500000,15000000,22500000")
  --seed      int (deterministic victim selection)
  --runtime-dir  default "runtime"
  --control-filestore   required for DiLoCo
  --world-size  int
  --replacement-delay-seconds   default 30
  --diloco-replacement-cmd   command template to spawn a DiLoCo replacement
    (receives RANK, CRASH_EPOCH as env vars; the script in
    scripts/run_diloco_crash.sh writes a helper launcher at runtime)

Usage:
    python sidecar_crash_controller.py \
        --framework diloco \
        --schedule 7500000,15000000,22500000 \
        --seed 0 \
        --runtime-dir runtime \
        --control-filestore /tmp/diloco_control_0 \
        --world-size 4 \
        --diloco-replacement-cmd "bash scripts/spawn_replacement.sh"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from control_plane import DiLoCoControlStore, read_all_worker_pids, read_progress, read_rank_tokens


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["ddp", "diloco"], required=True)
    ap.add_argument("--schedule", required=True,
                    help="comma-separated token thresholds, e.g. '7500000,15000000,22500000'")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--runtime-dir", default="runtime")
    ap.add_argument("--control-filestore", default=None)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--replacement-delay-seconds", type=float, default=30.0)
    ap.add_argument("--diloco-replacement-cmd", default=None,
                    help="shell command to spawn a DiLoCo replacement; receives "
                         "RANK and CRASH_EPOCH in the environment")
    ap.add_argument("--poll-hz", type=float, default=10.0)
    ap.add_argument("--log", default=None, help="optional JSONL event log path")
    ap.add_argument("--end-when", default="progress-gone",
                    choices=["progress-gone", "forever"],
                    help="'progress-gone' exits when progress.json hasn't updated "
                         "for ~30s (run completed); 'forever' for manual teardown")
    return ap.parse_args()


def _log(log_path: Path | None, payload: dict) -> None:
    print(f"[sidecar] {payload}", flush=True)
    if log_path is None:
        return
    with open(log_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def main():
    args = parse_args()
    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    progress_path = runtime_dir / "progress.json"
    log_path = Path(args.log) if args.log else None
    schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
    if not schedule:
        print("[sidecar] no thresholds — exiting")
        return
    rng = random.Random(args.seed * 99991 + 13)

    control_store: DiLoCoControlStore | None = None
    if args.framework == "diloco":
        if args.control_filestore is None:
            print("[sidecar] DiLoCo requires --control-filestore", file=sys.stderr)
            sys.exit(2)
        control_store = DiLoCoControlStore(Path(args.control_filestore),
                                           world_size=args.world_size)

    # Sidecar-owned state
    next_idx = 0
    recovery_in_flight = False
    current_crash_epoch = 0
    tokens_at_crash = 0
    victim_rank = -1
    # Snapshot of each rank's cumulative tokens at the last successful outer
    # commit (DiLoCo only). Updated on every outer step by observing the
    # tokens_committed advance in progress.json.
    tokens_rank_at_last_commit = {r: 0 for r in range(args.world_size)}
    last_tokens_committed = 0

    # For DDP: keep track of the 30s replacement delay we want to inject on
    # restart. We set an env var in a file that the respawned worker reads;
    # torchrun preserves the parent's env though, so exporting at spawn time
    # of the run script is enough.
    os.environ.setdefault("DDP_REPLACEMENT_DELAY_SECONDS", str(args.replacement_delay_seconds))

    _log(log_path, {"event": "sidecar_start", "schedule": schedule, "seed": args.seed})

    interval = 1.0 / args.poll_hz
    last_progress_update = time.time()
    while True:
        time.sleep(interval)
        prog = read_progress(progress_path)
        if prog is None:
            # Run may not have started yet — be patient.
            if time.time() - last_progress_update > 120.0:
                _log(log_path, {"event": "no_progress_too_long", "path": str(progress_path)})
                break
            continue
        last_progress_update = time.time()

        tokens_raw = int(prog.get("tokens_raw", 0))
        tokens_committed = int(prog.get("tokens_committed", 0))

        # Opportunistically refresh the per-rank committed snapshot for
        # DiLoCo lost_tokens bookkeeping. Any time tokens_committed advances,
        # each rank's current count becomes its new "at last commit" value.
        if args.framework == "diloco" and tokens_committed > last_tokens_committed:
            for r in range(args.world_size):
                tokens_rank_at_last_commit[r] = read_rank_tokens(runtime_dir, r)
            last_tokens_committed = tokens_committed

        # Recovery completion detection
        if recovery_in_flight:
            if args.framework == "ddp":
                # torchrun has restarted + workers reloaded + tokens_raw
                # advanced past the pre-crash high-water mark.
                if tokens_raw > tokens_at_crash:
                    _log(log_path, {"event": "ddp_recovery_complete",
                                    "tokens_raw_now": tokens_raw,
                                    "tokens_at_crash": tokens_at_crash})
                    recovery_in_flight = False
            else:
                assert control_store is not None
                complete_key = f"recovery_complete_{current_crash_epoch}"
                if control_store.get(complete_key) is not None:
                    _log(log_path, {"event": "diloco_recovery_complete",
                                    "crash_epoch": current_crash_epoch})
                    recovery_in_flight = False
            if recovery_in_flight:
                continue

        # Is it time to fire the next crash?
        if next_idx >= len(schedule):
            # All scheduled crashes fired. Exit when progress stops advancing.
            if args.end_when == "progress-gone":
                last_mtime = progress_path.stat().st_mtime if progress_path.exists() else 0
                if time.time() - last_mtime > 30.0:
                    _log(log_path, {"event": "run_complete_exit"})
                    break
            continue

        threshold = schedule[next_idx]
        if tokens_raw < threshold:
            continue

        # Fire a crash.
        victim_rank = rng.randrange(args.world_size)
        pids = read_all_worker_pids(runtime_dir, args.world_size)
        victim_pid = pids.get(victim_rank)
        if victim_pid is None:
            _log(log_path, {"event": "no_pid_for_victim", "rank": victim_rank,
                            "known": list(pids.keys())})
            # Skip this threshold — can't do anything useful. Advance so we
            # don't loop forever.
            next_idx += 1
            continue

        current_crash_epoch = next_idx + 1

        # Compute DiLoCo lost_tokens prediction (for the log) — victim's
        # tokens since the last committed outer sync.
        lost_tokens_diloco = None
        if args.framework == "diloco":
            victim_now = read_rank_tokens(runtime_dir, victim_rank)
            lost_tokens_diloco = max(0, victim_now - tokens_rank_at_last_commit[victim_rank])

        if args.framework == "diloco":
            assert control_store is not None
            control_store.mark_rejoin_pending(current_crash_epoch)
        _log(log_path, {
            "event": "crash",
            "framework": args.framework,
            "crash_epoch": current_crash_epoch,
            "threshold_tokens": threshold,
            "tokens_raw_at_crash": tokens_raw,
            "tokens_committed_at_crash": tokens_committed,
            "victim_rank": victim_rank,
            "victim_pid": victim_pid,
            "lost_tokens_diloco_estimate": lost_tokens_diloco,
        })

        try:
            os.kill(victim_pid, signal.SIGKILL)
        except ProcessLookupError:
            _log(log_path, {"event": "victim_already_gone", "rank": victim_rank, "pid": victim_pid})
        tokens_at_crash = tokens_raw
        recovery_in_flight = True
        next_idx += 1

        if args.framework == "diloco":
            # After the delay, spawn a replacement.
            def _spawn_replacement():
                if args.diloco_replacement_cmd is None:
                    _log(log_path, {"event": "no_replacement_cmd_set",
                                    "note": "set --diloco-replacement-cmd to spawn replacements"})
                    return
                env = {**os.environ,
                       "RANK": str(victim_rank),
                       "LOCAL_RANK": str(victim_rank),
                       "CRASH_EPOCH": str(current_crash_epoch)}
                _log(log_path, {"event": "spawning_replacement",
                                "rank": victim_rank, "crash_epoch": current_crash_epoch,
                                "cmd": args.diloco_replacement_cmd})
                subprocess.Popen(args.diloco_replacement_cmd, shell=True, env=env)

            # Sleep-then-spawn. We block the sidecar's main loop for the
            # delay period — that's fine because no other threshold can fire
            # during recovery anyway.
            time.sleep(args.replacement_delay_seconds)
            _spawn_replacement()

    _log(log_path, {"event": "sidecar_exit"})


if __name__ == "__main__":
    main()
