# DiLoCo vs DDP Under Failure Injection

Minimal DiLoCo implementation and DDP-elastic baseline on top of nanoGPT,
sweeping DiLoCo's H parameter under clean, crash, and straggler regimes.
The spec lives in `../spec/`; this README documents the implementation and
how to reproduce.

## Code layout

```
project/
  train.py                 # single entrypoint; dispatches by --framework
  model.py                 # nanoGPT (adapted from Karpathy)
  data.py                  # TinyStories sharded dataloader
  ddp_trainer.py           # DDP-elastic training loop
  diloco_trainer.py        # DiLoCo inner/outer loop + rejoin handshake
  metrics.py               # wall-clock / tokens / comm / event bookkeeping
  logger.py                # W&B wrapper (degrades cleanly when disabled)
  checkpoint.py            # save/load for DDP + DiLoCo outer_state
  control_plane.py         # token transport + side-band FileStore
  failure_injector.py      # in-process straggler injector
  sidecar_crash_controller.py  # out-of-process crash scheduler
  prepare_tinystories.py   # one-time token-ization → .bin
  config/{base,ddp,diloco}.yaml
  scripts/
    smoke_cpu.sh           # N=2 gloo CPU smoke, both frameworks
    smoke_crash.sh         # crash dry-run on CPU
    run_one_clean.sh       # torchrun launcher for one clean / straggler cell
    run_ddp_crash.sh       # torchrun + sidecar for a DDP crash run
    run_diloco_crash.sh    # direct-shell + sidecar for a DiLoCo crash run
    run_group_a.sh         # Group A — clean H sweep
    run_group_b.sh         # Group B — crash H sweep (requires T_BASELINE)
    run_group_c.sh         # Group C — straggler secondary cut
    plot.py                # pulls W&B runs and produces plots
```

## Prerequisites

- Python 3.11+ (3.13 tested locally for the CPU smoke test)
- PyTorch 2.1+ with NCCL for multi-GPU or gloo for CPU smoke
- `tiktoken`, `numpy`, `pyyaml`, `wandb`
- `datasets` (one-time, for `prepare_tinystories.py`)
- For DiLoCo crash runs: a shared filesystem visible to all workers (the
  FileStore and rendezvous files live there). Single-node only.

On Lambda 4×A10:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install torch numpy pyyaml tiktoken wandb datasets
wandb login
```

## One-time data prep

```bash
python prepare_tinystories.py --out data_cache/tinystories_train.bin --split train
python prepare_tinystories.py --out data_cache/tinystories_val.bin   --split validation
```

## Reproduction order (Lambda)

Per `experiment-matrix.md` "Bootstrap / ordering constraint":

```bash
# 1. Clean H sweep first — establishes T_baseline from DDP runs
bash scripts/run_group_a.sh
# Inspect W&B and compute mean tokens-to-target for DDP cells. Export:
export T_BASELINE=30000000   # example

# 2. Crash sweep (thresholds = {0.25, 0.5, 0.75} × T_BASELINE)
bash scripts/run_group_b.sh

# 3. Straggler secondary cut
bash scripts/run_group_c.sh

# 4. Plots (reads from W&B)
python scripts/plot.py --entity YOUR_WANDB_ENTITY --out plots/
```

Outputs land in:

- `runtime/<cell>_s<seed>/` — progress.json, per-rank token files, sidecar.jsonl,
  worker PIDs, DiLoCo per-rank cursors
- `checkpoints/ddp/rank<r>.pt` — DDP full checkpoints (5-min cadence)
- `checkpoints/diloco/outer_state.pt` — DiLoCo θ_outer + outer optimizer state
- W&B project `diloco-vs-ddp-failures` — all metrics per `measurement-plan.md`

## CPU smoke test (before any paid run)

```bash
PY=python bash scripts/smoke_cpu.sh ddp       # ~2s
PY=python bash scripts/smoke_cpu.sh diloco    # ~2s
PY=python bash scripts/smoke_crash.sh ddp     # ~60s; 2 crashes via torchrun
PY=python bash scripts/smoke_crash.sh diloco  # ~60s; 2 crashes via destroy/reinit
```

Each exercises the full control path (token transport, sidecar, logger,
crash rejoin). The smoke tests use a 2-layer/128-hidden model against a
synthetic 200k-token corpus so they run on a laptop CPU.

## Launcher asymmetry (important)

- **DDP crash runs** use `torchrun --max_restarts=10`. When a worker dies,
  torchrun tears down the whole set of N and respawns them; the sidecar
  injects a 30 s sleep via `DDP_REPLACEMENT_DELAY_SECONDS` in the respawn
  path.
- **DiLoCo crash runs** bypass torchrun entirely
  (`scripts/run_diloco_crash.sh`). Workers are spawned directly and each
  calls `init_process_group(init_method="env://")` itself. The side-band
  `FileStore` at `$CONTROL_FILESTORE` coordinates the survivor-preserving
  rejoin dance: survivors wait for `rejoin_pending`, everyone rendezvouses
  on a per-crash-epoch barrier, the PG is destroyed and re-initialized via
  a fresh `file://` rendezvous (not env://, to avoid MASTER_PORT
  TIME_WAIT issues), and the replacement loads both `θ_outer` **and** the
  outer optimizer state from `outer_state.pt`, skips its inner loop, and
  contributes Δ=0 on the first post-rejoin outer step.

## Hyperparameter knobs

Edit `config/base.yaml` (shared), `config/ddp.yaml`, or
`config/diloco.yaml`. The `per_worker_batch_size` in `base.yaml` must be
calibrated once on the final Lambda SKU — see `experiment-matrix.md`.

## Key invariants (please don't break without reading the spec)

1. `model` in `diloco_trainer.py` is a bare `nn.Module`; never wrap it in
   `DistributedDataParallel`.
2. Pseudo-gradient sign is `θ_outer − θ_local`. Flipping it diverges.
3. Fresh `AdamW` per outer step; persistent `SGD(nesterov=True)` across the
   whole run.
4. Before `outer_optimizer.step()`, live params are restored to θ_outer.
5. On DiLoCo rejoin, load **both** θ_outer and the outer optimizer's momentum
   buffer; loading only θ_outer silently breaks correctness.
6. DDP is not a point on the H axis — it is a separate synchronous
   reference baseline. Plots render it as a horizontal band.

## Troubleshooting

- **DiLoCo crash run hangs on reinit**: confirm `/tmp/diloco_control_*` is
  writable and not stale. A rendezvous file from a previous crash event
  remaining at `runtime/<cell>/rendezvous_<N>` is fine; each crash uses a
  new file.
- **Workers stuck after a crash on NCCL**: check
  `NCCL_ASYNC_ERROR_HANDLING=1` and `TORCH_NCCL_BLOCKING_WAIT=1` are set
  (done by the launcher scripts). The NCCL collective timeout is 120 s
  for DiLoCo crash runs to cover the 30 s replacement delay plus spawn.
- **`tokens_raw` lags a bit behind truth**: the aggregator polls at 1 Hz
  by design, so `progress.json` trails reality by up to a second. That is
  the granularity the sidecar uses to fire crashes — fine for thresholds
  at millions of tokens.
