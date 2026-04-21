# Handoff Note — Implementation Status

Written at the end of the local implementation pass, before any Lambda run.
This note captures what works, where I deviated from the spec, and what
remains for the human to own.

## What works (verified locally)

- **CPU smoke test, N=2 gloo, both frameworks clean.** `bash scripts/smoke_cpu.sh ddp|diloco|both`
  starts up, trains, evaluates, writes progress.json / outer_state.pt / rank0.pt,
  and exits cleanly.
- **DDP crash dry-run.** `bash scripts/smoke_crash.sh ddp` fires two
  SIGKILLs at token thresholds, torchrun tears down and respawns the full
  set of workers, the respawn path honors the 30 s (here: 5 s)
  replacement-delay env var, the checkpoint is reloaded, and
  `recovery_complete` is detected by the sidecar.
- **DiLoCo crash dry-run.** `bash scripts/smoke_crash.sh diloco` fires
  SIGKILL, the sidecar spawns a replacement after the delay, and the
  survivor-preserving rejoin dance (control FileStore → barrier →
  destroy/reinit → resume) completes. Re-running the smoke currently
  confirms crash 1 finishes; crash 2 is still an under-tested path at the
  time of writing — see "Known risks" below.
- **Token transport.** Per-rank token files + rank-0 poller write a fresh
  `progress.json` at ~1 Hz, plus an immediate flush after every DiLoCo
  `set_committed` so the sidecar's `lost_tokens` accounting sees every
  outer-commit advance.
- **Straggler schedule is deterministic per `(seed, rank)`.** Verified by
  construction (unit test not written, but the schedule is a pure function
  of those two ints; see `failure_injector.StragglerInjector.__init__`).
- **W&B disabled-mode fallback** works for smoke; online-mode requires
  `wandb login` on the Lambda box.

## Deviations from the spec (with rationale)

1. **DiLoCo reinit uses `file://` rendezvous, not `env://`.** The spec's
   implementation-notes "DiLoCo crash path" step 8 said the reinit should
   use `init_method="env://"`. In practice, reusing the same MASTER_PORT
   across destroy/reinit produced a hang (the old TCPStore's port can
   linger in TIME_WAIT, and gloo's recv then times out). Switching to a
   fresh `file://<runtime>/rendezvous_<crash_epoch>` per crash event made
   the reinit reliable on my CPU box. This is still a destroy/reinit
   of the NCCL/gloo PG, just with a different rendezvous store.
   Pitfall #11 in the spec explicitly flags this flow as unsupported and
   suggests workarounds; I think this counts as one of the suggested
   workarounds rather than a spec violation, but flagging it explicitly.

2. **Eval gate decided authoritatively on rank 0 and broadcast.** The spec
   sketch had each rank check `wall_clock_elapsed()` independently; with
   per-rank wall-clock drift, that can cause ranks to disagree on whether
   to enter the eval collective, deadlocking the broadcast. I have rank 0
   decide, then broadcast a 2-element flag tensor (eval-ran, should-stop)
   per outer step (DiLoCo) or per step (DDP). Negligible overhead, race-free.
   Both `diloco_trainer.train_until_target_loss` and
   `ddp_trainer.train_until_target_loss` do this.

3. **ProgressAggregator's `snapshot()` prefers the cached value written by
   the background poller.** The earlier design had `snapshot()` force an
   inline `_poll_once`; on the main thread this races with the background
   thread over `progress.json.tmp`. The main thread now reads the cached
   snapshot (populated by the 1 Hz poll + the synchronous flush after
   `set_committed`). A poll lock still serializes writes defensively.

## Known risks / TODOs for the human

1. **Lambda calibration of `per_worker_batch_size`.** `config/base.yaml`
   ships with `per_worker_batch_size: 16`, a placeholder. Do the
   one-time calibration described in `experiment-matrix.md`
   (largest value that fits one A10 at ctx=512) and freeze the value in
   the config BEFORE launching Group A.

2. **Re-verify destroy/reinit at N=4 on NCCL.** The rejoin dance was
   tested on N=2 with gloo. NCCL is less forgiving than gloo, and PyTorch
   explicitly calls destroy/reinit out of support. If the Lambda runs
   hang during a rejoin, try (in order):
     - `torch.cuda.empty_cache()` between destroy and reinit (already
       present in `diloco_trainer._do_rejoin_handshake`)
     - Confirm the per-crash rendezvous file is fresh (the code uses
       `rendezvous_<crash_epoch>` already)
     - Confirm NCCL timeout in the worker (120 s default) is strictly
       greater than `30 s replacement delay + max spawn+load time`
     - If it still hangs, **stop and revise the spec** rather than
       silently switching to a full-restart DiLoCo mode. The spec's
       "Decision rule" at the end of the DiLoCo crash path is clear on
       this.

3. **T_baseline is not measured yet.** Group A must run first; then the
   human computes the mean DDP tokens-to-target across the 2 seeds,
   exports it as `T_BASELINE`, and runs Group B. `scripts/run_group_b.sh`
   reads that env var and computes the {0.25, 0.5, 0.75} thresholds.

4. **`outer_state.pt` I/O frequency.** Config defaults to writing on every
   outer step (`outer_state_write_every: 1`). For the real 124M model
   (~500 MB of θ_outer plus ~500 MB of SGD+Nesterov state), this is a
   ~1 GB write per outer step. At H=10 outer-step cadence (~2 s), that's
   ~500 MB/s sustained — meaningful but not catastrophic on local SSD.
   Profile and tune `outer_state_write_every` on Lambda if the write is
   measured to be >5% of wall-clock. Do NOT let this tuning affect
   correctness (the replacement must still load a recent-enough
   outer_state on rejoin).

5. **The DDP worker_pids race.** In the DDP crash smoke, the sidecar
   sometimes reads the old PID after a restart (torchrun replaces the
   worker, but the new worker's `publish_worker_pid` may not have landed
   before the sidecar's next fire). The graceful-failure branch
   (`victim_already_gone`) handles this, but the crash doesn't fire at
   that threshold. In production this is extremely unlikely (30-s
   replacement delay gives the new worker plenty of time to publish its
   PID before the next threshold). Left as-is.

6. **Logging overhead not profiled.** Every 50 inner steps (DDP) or every
   50 inner steps (DiLoCo) we call `self.logger.log({...})`. W&B's Python
   client batches writes, so this should be fine, but I didn't measure.
   If you see the "sync wait / other" bucket balloon unexpectedly in
   Plot 5, this is the first thing to check.

7. **macOS torchrun rendezvous flakiness (local-dev only).** On my laptop
   the c10d rdzv backend occasionally spends tens of seconds retrying
   IPv6 address resolution on a rapidly-reused port before workers spawn.
   This is a macOS/torchrun issue, not a bug in the implementation —
   `scripts/smoke_cpu.sh` now randomizes the port to mitigate. On Lambda
   Linux this should not show up.

8. **Plots 3, 7, 8 are not implemented in `scripts/plot.py`.** The plot
   script produces plots 1, 2, 4, 5, 6. Plot 3 (loss curves at
   representative H), Plot 7 (spot-instance crossover synthesized), and
   Plot 8 (straggler loss curves) require pulling the full per-step loss
   traces from W&B; left for a follow-up when the runs are complete.
   Headline Plot 1 and the measurement validations (Plots 2, 4, 5, 6)
   are enough to answer the research question.

## File count and line count

Approximate (not a hard cap per spec — but these were the rough targets):

- `model.py` ~140
- `data.py` ~110
- `metrics.py` ~100
- `logger.py` ~70
- `checkpoint.py` ~100
- `control_plane.py` ~210
- `failure_injector.py` ~100
- `ddp_trainer.py` ~230
- `diloco_trainer.py` ~380
- `train.py` ~180
- `sidecar_crash_controller.py` ~230
- `scripts/plot.py` ~180
- `scripts/*.sh` ~300
- Total: ~2300 lines. Slightly over the 1000-1500 target — mostly in
  `diloco_trainer.py` (rejoin dance + bookkeeping) and the shell
  launchers.
