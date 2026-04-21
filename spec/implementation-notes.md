# Implementation Notes

This document specifies the code layout, key functions, failure injection mechanics, and operational details. Written to be read by Claude Code (or a human engineer) before writing any code.

## Code layout

```
project/
  train.py                  # single entrypoint; dispatches to DDP or DiLoCo based on --framework
  model.py                  # nanoGPT model definition (copied/adapted from Karpathy)
  data.py                   # TinyStories loading, tokenizer, sharded dataloader
  ddp_trainer.py            # DDP-elastic training loop + checkpointing
  diloco_trainer.py         # DiLoCo inner/outer training loop
  failure_injector.py       # In-process straggler injector only
  sidecar_crash_controller.py  # Authoritative crash scheduler / SIGKILL sidecar for crash runs
  metrics.py                # Wall-clock, tokens, comm bytes, breakdown counters
  logger.py                 # W&B wrapper with the schema from measurement-plan.md
  checkpoint.py             # save/load helpers (used by ddp_trainer only, but DiLoCo uses it for final snapshot)
  config/
    base.yaml               # shared hyperparams
    ddp.yaml                # DDP-specific overrides
    diloco.yaml             # DiLoCo-specific overrides
  scripts/
    run_group_a.sh          # Group A — clean H sweep (DDP + DiLoCo H ∈ {10,50,100,500}, 2 seeds each)
    run_group_b.sh          # Group B — crash H sweep (same axes)
    run_group_c.sh          # Group C — straggler secondary cut (DDP + DiLoCo H=50, 2 seeds each)
    run_diloco_crash.sh     # direct-launch helper for DiLoCo crash runs (no torchrun)
    plot.py                 # downloads W&B data and produces the H-axis plots
  README.md                 # how to reproduce
```

**Target code size:** ~1000-1500 lines total. The DiLoCo trainer should be ~200-300 lines on top of a shared skeleton.

## Launch command

The launcher depends on the framework × failure combination.

**Case 1 — any clean run, any straggler run, and DDP crash runs.** Use `torchrun`, which handles rendezvous and (for DDP crash runs) the kill-and-respawn cascade.

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    --max_restarts={10 if framework=="ddp" and failure=="crash" else 0} \
    train.py \
    --framework {ddp|diloco} \
    --failure {none|crash|straggler} \
    --seed {0|1} \
    --H {10|50|100|500}           # DiLoCo only; ignored for DDP
    --config config/{ddp,diloco}.yaml
```

**Case 2 — DiLoCo crash runs (the one regime where torchrun cannot be used).** PyTorch's torchrun / TorchElastic agent documents worker failure as "all workers are stopped and restarted up to `max_restarts`," and on any worker exit it tears down the surviving processes regardless of what `--max_restarts` is set to. `--max_restarts=0` means "do not restart," **not** "leave surviving workers alone." Using torchrun for DiLoCo crash runs would therefore destroy the exact survivor continuity the experiment is trying to measure.

For DiLoCo crash runs we bypass torchrun entirely and start the N worker processes directly from a plain shell loop. Process-group formation still uses ordinary `env://` initialization (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`). The extra shared file is **not** the process-group rendezvous backend; it is a side-band control channel used only to coordinate the destroy/reinit rejoin flow during crash recovery. The initial launch looks like:

```bash
# scripts/run_diloco_crash.sh
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export CONTROL_FILESTORE=/tmp/diloco_control_${SEED}  # side-band control store for rejoin flags
rm -f "$CONTROL_FILESTORE"

for RANK in 0 1 2 3; do
    RANK=$RANK LOCAL_RANK=$RANK \
    python train.py \
        --framework diloco \
        --failure crash \
        --seed "$SEED" \
        --H "$H" \
        --control-filestore "$CONTROL_FILESTORE" \
        --config config/diloco.yaml &
done

python sidecar_crash_controller.py \
    --framework diloco \
    --schedule TOKENS_FILE \
    --seed "$SEED" \
    --control-filestore "$CONTROL_FILESTORE" &

wait
```

Each worker calls `dist.init_process_group(backend="nccl", init_method="env://")` at startup; rank/world/master come from the exported env vars. A replacement process launched later (by the sidecar, after the 30 s delay) uses the same env vars but `RANK=<victim_rank>` so it takes the dead worker's slot. There is no supervising agent, so one worker dying does not by itself tear down the others. The `CONTROL_FILESTORE` path is opened separately as `dist.FileStore(...)` (or an equivalent file-backed side-channel) and is used only for crash/rejoin coordination flags such as `rejoin_pending`, `replacement_ready`, and `recovery_complete`.

Backend: NCCL. `CUDA_VISIBLE_DEVICES=0,1,2,3` on the Lambda 4×A10 VM. Timeout on collective ops set to 60 seconds (`init_process_group(timeout=timedelta(seconds=60))`) so that NCCL doesn't hang forever when a worker dies. Note that for the DiLoCo crash path, we want this timeout to be comfortably longer than `(30 s replacement delay + replacement spawn/load/reinit time)` so that survivors don't time out before the replacement arrives at the outer barrier. Since the replacement joins the first post-rejoin outer step with **zero pseudo-gradient** and does **not** run H local steps first, the timeout no longer needs to budget for a full replacement inner loop. A 120 s timeout is still a conservative starting point for DiLoCo crash runs.

## Shared skeleton: `train.py`

```python
def main():
    args = parse_args()
    dist_ctx = init_distributed(args)
    # dist_ctx does two things:
    #   1. init_process_group(backend="nccl", init_method="env://") for the real process group
    #   2. if args.control_filestore is set (DiLoCo crash path), open a side-band
    #      FileStore used only for rejoin coordination flags.
    model = build_model(args.config).to(device)
    dataloader = build_dataloader(args.config, rank, world_size, seed=args.seed)
    logger = WandbLogger(args)
    control_plane = build_control_plane(args)
    straggler_injector = build_straggler_injector(args.failure, args.seed)

    if args.framework == "ddp":
        trainer = DDPTrainer(model, dataloader, logger, control_plane, straggler_injector, args.config)
    elif args.framework == "diloco":
        trainer = DiLoCoTrainer(model, dataloader, logger, control_plane, straggler_injector, args.config)

    trainer.train_until_target_loss(target_loss=2.0)

    if rank == 0:
        logger.log_final_metrics(trainer.metrics)
    destroy_process_group()
```

## DDP trainer: `ddp_trainer.py`

```python
class DDPTrainer:
    def __init__(self, ...):
        self.model = DistributedDataParallel(model, device_ids=[local_rank])
        self.optimizer = AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
        self.checkpoint_interval_sec = 300   # 5 minutes
        self.last_checkpoint_time = time.perf_counter()
        self.last_checkpoint_tokens = 0
        # ... metrics setup

    def train_until_target_loss(self, target_loss):
        self.maybe_restore_checkpoint()  # handles rejoin-after-crash case
        while True:
            batch = next(self.dataloader)
            loss = self.step(batch)
            self.metrics.record_step(...)
            self.control_plane.publish_progress(
                tokens_raw=self.metrics.tokens_raw,
                tokens_committed=self.metrics.tokens_committed,
            )  # no-op except crash runs; the sidecar owns crash scheduling
            if self.should_eval():
                eval_loss = self.evaluate()
                if eval_loss <= target_loss:
                    self.metrics.record_target_reached()
                    return
            if self.should_checkpoint():
                self.save_checkpoint()

    def step(self, batch):
        # See "Communication instrumentation" below for why we use a DDP comm hook
        # rather than wrapping loss.backward() with a timer. DDP's real all-reduce
        # happens inside autograd hooks during backward(), implemented in C++ —
        # a Python-level timer around backward() cannot cleanly separate compute
        # from communication, and a Python wrapper around dist.all_reduce never
        # sees DDP's traffic at all.
        with self.metrics.time("step_total"):
            logits = self.model(batch.input_ids)
            loss = cross_entropy(logits, batch.labels)
            loss.backward()   # comm hook records comm_bytes and comm_seconds internally
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        # At this point self.metrics has:
        #   step_total (wall clock of the whole step)
        #   comm_seconds and comm_bytes (from the DDP comm hook)
        # Compute bucket = step_total − comm_seconds − optimizer_seconds.
        return loss.item()
```

### DDP checkpoint format

The goal is "reasonable near-determinism on resume" — not bit-exact replay (which is effectively impossible with NCCL reductions on multi-GPU anyway). The checkpoint captures enough state that a crash-and-resume run reproduces the same batches, the same crash schedule, the same straggler schedule, and the same loss trajectory modulo CUDA non-determinism.

```python
import random, numpy as np

state = {
    # --- model + optimizer ---
    "model": self.model.module.state_dict(),  # unwrap DDP
    "optimizer": self.optimizer.state_dict(),

    # --- RNG state for every source that can affect training ---
    "torch_rng_state": torch.get_rng_state(),
    "cuda_rng_state_all": torch.cuda.get_rng_state_all(),  # per-device CUDA RNG
    "python_rng_state": random.getstate(),
    "numpy_rng_state": np.random.get_state(),

    # --- progress counters ---
    "tokens_raw": self.metrics.tokens_raw,
    "tokens_committed": self.metrics.tokens_committed,
    "step": self.metrics.step,
    "wall_clock_offset": time.perf_counter() - self.start_time,

    # --- dataloader cursor (so we don't re-read the same batches) ---
    "dataloader_sampler_epoch": self.dataloader.sampler.epoch,
    "dataloader_position": self.dataloader.position(),  # per-rank cursor into the shard

    "straggler_injector": {
        "schedule_position": self.straggler_injector.schedule_position,
        "rng_state": self.straggler_injector.rng.getstate(),
        # Note: if stragglers are precomputed per M-6, we only need schedule_position.
        # The rng_state field is there for fallback if we keep any online sampling.
    },
}
torch.save(state, f"checkpoints/rank{rank}.pt")
```

Each rank writes its own RNG, dataloader cursor, and straggler-injector state. The model state is identical across ranks under DDP, but it's cheap to write once per rank; optionally have only rank 0 write the model payload and have other ranks write a stub plus their bookkeeping. On load, rank 0 broadcasts the model state to all workers. The crash controller's `next_idx` and RNG live in the sidecar process, not inside the worker checkpoint: the sidecar stays alive across DDP worker restarts and remains the authoritative owner of which crash thresholds have already fired.

**DiLoCo checkpointing:** DiLoCo does not write wall-clock-interval checkpoints in steady state (that's a DDP concern). However, DiLoCo writes `theta_outer.pt` after every successful outer step, plus a small committed-state record for each rank containing that rank's dataloader cursor at the last committed outer boundary. On rejoin, the replacement worker restores `theta_outer.pt` and the dead rank's last committed dataloader cursor. It does NOT attempt to preserve the dead worker's uncommitted inner-loop position; that work is intentionally discarded and is exactly what `lost_tokens` measures.

## DiLoCo trainer: `diloco_trainer.py`

```python
class DiLoCoTrainer:
    def __init__(self, ...):
        self.model = model   # NOT wrapped in DDP
        self.outer_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.7,
            momentum=0.9,
            nesterov=True,
        )
        self.H = args.H  # inner steps per outer step; primary swept axis, ∈ {10, 50, 100, 500}
        # ...

    def train_until_target_loss(self, target_loss):
        while True:
            self.do_outer_step()
            if self.should_eval():
                eval_loss = self.evaluate()
                if eval_loss <= target_loss:
                    self.metrics.record_target_reached()
                    return

    def do_outer_step(self):
        # Snapshot θ_outer (the state the outer optimizer owns and will update)
        theta_outer = {name: p.detach().clone() for name, p in self.model.named_parameters()}

        # Fresh inner optimizer each outer step
        inner_opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95),
        )

        # Inner loop (no sync with other workers). This mutates self.model's
        # parameters in place; by the end of the loop the live params are θ_local.
        for inner_step in range(self.H):
            batch = next(self.dataloader)
            with self.metrics.time("compute"):
                logits = self.model(batch.input_ids)
                loss = cross_entropy(logits, batch.labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                inner_opt.step()
                inner_opt.zero_grad()
            self.metrics.record_inner_step(tokens=batch.num_tokens, loss=loss.item())
            self.control_plane.publish_progress(
                tokens_raw=self.metrics.tokens_raw,
                tokens_committed=self.metrics.tokens_committed,
            )  # no-op except crash runs; the sidecar owns crash scheduling

        # Compute pseudo-gradients from (θ_outer, θ_local) and all-reduce them.
        deltas = {}
        with self.metrics.time("communication"):
            for name, p in self.model.named_parameters():
                delta = theta_outer[name] - p.detach()   # pseudo-gradient (sign convention: see algorithm-notes)
                dist.all_reduce(delta, op=dist.ReduceOp.AVG)   # PyTorch 1.13+ supports AVG directly
                deltas[name] = delta

        # CRITICAL: restore θ_outer into the live model BEFORE calling the outer
        # optimizer, so that outer_optimizer.step() updates θ_outer (not θ_local).
        # The inner loop mutated self.model in place; without this restore, the
        # outer SGD+Nesterov step would be applied from θ_local and the algorithm
        # would not match algorithm-notes.md. See pitfall #9 below.
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.data.copy_(theta_outer[name])
                p.grad = deltas[name]

        # Now apply the Nesterov SGD update using the persistent momentum buffer.
        # Live params are θ_outer, .grad is Δ_global, so .step() computes:
        #   θ_outer ← θ_outer − outer_lr * Nesterov_update(Δ_global).
        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()
```

**Key points:**
- `self.model` is NOT wrapped in `DistributedDataParallel`. DDP's autograd hooks would all-reduce gradients every step, defeating the whole purpose.
- The inner AdamW is created fresh inside `do_outer_step`, so its running moments do not persist. This is intentional and specified in the algorithm.
- `self.outer_optimizer` is created once in `__init__` and persists for the whole run, so its Nesterov momentum buffer carries across outer steps. This is critical.
- Pseudo-gradient sign: `theta_outer - p`, consistent with algorithm-notes.md. Do NOT flip this.

## Failure injection

### Crash controller: `sidecar_crash_controller.py`

Crash injection is **entirely out-of-process**. Workers never decide when to crash and never call `maybe_inject()` for crash mode. The authoritative crash scheduler is the sidecar controller started by the run script.

Worker/sidecar contract:

1. Each worker writes its PID and rank to `worker_pids.json` at startup.
2. Rank 0 periodically publishes the authoritative `tokens_raw` and `tokens_committed` counters to `runtime/progress.json` on the shared local filesystem after each step / inner step. The sidecar reads this file; there is no alternate progress-transport mechanism in this spec.
3. The sidecar owns the deterministic crash schedule, the crash RNG, and the `next_idx` pointer saying which threshold fires next.
4. When the sidecar observes `tokens_raw >= schedule[next_idx]` and no recovery is in flight, it picks the victim rank, sends `SIGKILL`, and advances `next_idx`.
5. During DiLoCo recovery the sidecar also writes the control-store flags (`rejoin_pending`, `replacement_ready_*`, `recovery_complete_*`) that survivors and the replacement poll.

This split is the intentional control-plane contract:

- **Workers own training state and publish progress.**
- **The sidecar owns crash timing and process death.**
- **Only the straggler injector is in-process.**

**Mechanism detail.** The crash-recovery path differs between frameworks — this is a deliberate asymmetry, not a spec gap. See `algorithm-notes.md` ("Shared infrastructure") for the rationale.

#### DDP crash path (full kill-and-respawn via torchrun)

When torchrun's `--max_restarts` kicks in, it **kills all remaining worker processes and respawns the full set of N workers**. It does not spawn a single replacement and rejoin it to the existing process group — PyTorch's NCCL process groups do not support membership shrinkage, so any crash forces a total restart. This is the load-bearing property of the DDP crash measurement.

The injection and restart pipeline is:

1. The run script (`run_group_b.sh`) launches `torchrun` as a subprocess **and** a sidecar controller process.
2. The sidecar controller reads a local JSON file written by rank 0 to observe the current global token counter.
3. When a token threshold is crossed, the controller randomly picks a victim worker rank (seeded deterministically from the run seed), reads that worker's PID from a `worker_pids.json` file (written by each worker at startup), and sends `SIGKILL` to that PID.
4. The dead worker triggers the full torchrun restart cascade: surviving workers block at their next all-reduce, NCCL times out after ~60 seconds, survivor training loops exception out, torchrun detects the exits and kills everyone, then respawns N fresh workers which rendezvous and load the last checkpoint.
5. The 30-second "replacement delay" we want to simulate is enforced on top of this by inserting a 30-second sleep in the startup path of the respawned workers, keyed on an environment variable set by the sidecar, so the restart appears to take at least 30 seconds regardless of how quickly torchrun actually moves.

```
DDP launch pattern:
  torchrun --max_restarts=10 --monitor_interval=5 ... train.py --framework ddp ...
  python sidecar_crash_controller.py --framework ddp --schedule TOKENS_FILE --seed SEED
```

#### DiLoCo crash path (required)

For DiLoCo, the experiment is measuring the property "surviving workers do not roll back." The DDP kill-everyone restart model destroys that property on contact. The DiLoCo crash regime therefore requires a survivor-preserving recovery path built around process-group destroy/reinit at the next outer-sync boundary. There is no alternate DiLoCo crash mode in this spec.

Survivors stay alive, a replacement worker is started by the sidecar, and the four-rank NCCL process group is torn down and re-initialized around the replacement at the next outer-sync boundary.

The injection and rejoin pipeline is:

1. The run script (`run_group_b.sh`, Case 2 launcher above) starts the N workers directly (no torchrun) and also launches the sidecar controller.
2. The sidecar observes the global token counter as before.
3. When a token threshold is crossed, the sidecar SIGKILLs the chosen victim rank **and immediately writes** `rejoin_pending=<crash_epoch>` to the shared control `FileStore`. This "crash in flight" marker is what prevents survivors from entering a collective before the replacement exists. While `rejoin_pending` is set, the controller suppresses any later crash thresholds.
4. The SIGKILL causes that worker's process to exit. Because there is no torchrun agent, the survivors are not torn down; they keep running their own inner loops.
5. Survivors are unaffected during the inner loop because DiLoCo does not communicate there.
6. At each outer-sync boundary, every survivor first checks `rejoin_pending` on the shared control `FileStore`. If no crash is pending, the survivor proceeds to the outer all-reduce as usual. If a crash is pending, the survivor does **not** call any collective on the old process group. Instead it waits in the rejoin path below. This is the load-bearing race fix: survivors wait on the control store before they ever enter a collective the dead worker cannot participate in.
7. The sidecar, after the fixed 30-second delay, spawns the replacement process with the same env-var-based rendezvous config as the initial launch, plus a `--rejoin --crash-epoch=<crash_epoch>` flag. The replacement:
   - Reads the current `θ_outer` from `theta_outer.pt` (rank 0 writes this after every successful outer step).
   - **Skips local inner steps for this rejoin event.** Its first contribution after rejoin is intentionally `Δ = 0`.
   - Attaches to the shared control `FileStore` at `$CONTROL_FILESTORE`.
   - Writes `replacement_ready_<crash_epoch>=1` and the replacement's PID/endpoint to the control store.
   - Waits on a `barrier_before_reinit_<crash_epoch>` key that survivors will set.
8. Rendezvous dance (every process, survivor and replacement, does this exactly once per crash event):
   - Wait until `replacement_ready_<crash_epoch>` is present.
   - Set `barrier_before_reinit_<crash_epoch>` on arrival.
   - Wait on the control `FileStore` until all N participants (survivors + replacement) have set `barrier_before_reinit_<crash_epoch>`.
   - Call `dist.destroy_process_group()`.
   - Call `dist.init_process_group(backend="nccl", init_method="env://")` again (all workers still have the same `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` env vars). The replacement's `RANK` matches the dead worker's rank.
   - Clear the rejoin keys on the control `FileStore`, write `recovery_complete_<crash_epoch>=1`, and clear `rejoin_pending` so the next crash event starts from a clean slate.
9. After the reinit, each worker arrives at the outer all-reduce on the new process group. Survivors contribute their real pseudo-gradients from the just-finished local round. The replacement contributes a zero pseudo-gradient on this first rejoin step because it loaded the committed `θ_outer` and skipped local training. `θ_outer` then advances by one outer step. On subsequent rounds, the replacement behaves like any ordinary worker again and performs H local steps before outer sync.

**Why this is fragile.** PyTorch's `torch.distributed` docs explicitly note that calling `destroy_process_group()` and then reinitializing at runtime is outside the supported fault-tolerance model: NCCL process groups are not designed to be torn down and rebuilt mid-run, and PyTorch does not guarantee CUDA state remains clean across such a cycle. In practice the flow often works at small world sizes if (a) all processes synchronize the destroy/reinit at the exact same logical point (the barrier in step 8 is what enforces this), (b) no collective is in flight on the old process group, and (c) no tensor created on the old process group is used on the new one. Budget ~150 lines of code and expect iteration. See pitfall #10 below.

```
DiLoCo crash launch pattern:
  bash scripts/run_diloco_crash.sh   # directly spawns N workers, no torchrun
  python sidecar_crash_controller.py --framework diloco --schedule TOKENS_FILE --seed SEED
```

**Important:** the control `FileStore` and the `theta_outer.pt` snapshot live on the shared filesystem of the single VM. This is fine for the single-node experiment. On multi-node this would need a real distributed store; we explicitly scope to single-node to avoid that complication.

**Decision rule.** This rejoin path is the DiLoCo crash experiment. If the destroy/reinit cycle produces a hang or CUDA error that cannot be stabilized, do **not** substitute a full-restart DiLoCo path and still call it the same experiment. Stop the crash-regime implementation, revise the spec explicitly, and only then resume work.

#### What gets measured (both paths)

For each crash event, record (a) the victim rank, (b) the token counter at crash time (raw), and (c) the token counter at the last committed progress point — the last checkpoint for DDP, the last outer sync for DiLoCo. `lost_tokens` is then computed using the framework-specific rule from `measurement-plan.md`: `tokens_raw − tokens_committed` for DDP, or the victim's local `worker_tokens_since_commit[rank]` subtotal for DiLoCo.

### Straggler injector

**Design note:** the straggler schedule is **precomputed at startup from the seed**, not sampled online during training. This is critical for within-seed comparability between DDP and DiLoCo. DDP and DiLoCo have different step durations, different barriers, and different eval cadences, so an online sampler tied to `time.perf_counter()` will draw at different points and produce different schedules even under identical seeds. Precomputation makes "DDP seed-0 and DiLoCo seed-0 saw the same straggler events" a real, verifiable claim.

```python
class StragglerInjector:
    def __init__(self, p_per_minute=0.15, slow_duration_range=(30, 90),
                 slowdown_factor=3.0, seed=0, rank=0,
                 expected_run_minutes=120):
        """
        Precomputes a deterministic straggler schedule for THIS rank,
        covering a window comfortably longer than any expected run.
        The schedule is a list of (enter_at_elapsed_seconds, duration_seconds)
        episodes, generated from a per-rank RNG seeded from (global_seed, rank).
        """
        # Per-worker, deterministic. `random.Random` accepts int/str/bytes but
        # NOT tuples, so combine (seed, rank) into a single int rather than
        # passing a tuple.
        self.rng = random.Random(seed * 10_000 + rank)
        self.p = p_per_minute
        self.slowdown = slowdown_factor
        self.schedule = []  # list of (start_seconds, duration_seconds)
        for minute in range(expected_run_minutes):
            if self.rng.random() < self.p:
                duration = self.rng.uniform(*slow_duration_range)
                self.schedule.append((minute * 60.0, duration))
        self.schedule_position = 0  # index of next unseen episode (checkpointed)
        self.start_time = None      # set on first step_hook call
        self.slow_until_elapsed = -1.0

    def step_hook(self, step_compute_time):
        if self.start_time is None:
            self.start_time = time.perf_counter()
        elapsed = time.perf_counter() - self.start_time
        # Advance schedule cursor past any episodes whose start time we've passed
        while (self.schedule_position < len(self.schedule)
               and self.schedule[self.schedule_position][0] <= elapsed):
            start_s, dur_s = self.schedule[self.schedule_position]
            self.slow_until_elapsed = max(self.slow_until_elapsed, start_s + dur_s)
            self.log_event(start_s, dur_s)
            self.schedule_position += 1
        if elapsed < self.slow_until_elapsed:
            extra = step_compute_time * (self.slowdown - 1.0)
            time.sleep(extra)
```

The `step_hook` is called at the end of every training step. The schedule is fully determined by `(seed, rank)` at construction time; the step hook just reads it against the wall-clock elapsed time since the first step.

**Bug fix vs prior sketch:** the earlier sketch initialized `self.next_sample_at = 60.0` and compared against `time.perf_counter()` directly. `time.perf_counter()` returns an arbitrary monotonic value (potentially thousands of seconds into the process lifetime), so the first step already satisfies `now >= 60.0` and the sampler fires immediately. The precomputed-schedule design above avoids this class of bug by anchoring everything to `elapsed = now − start_time`, with `start_time` recorded on the first `step_hook` call.

**Per-worker independence:** each worker's schedule is generated from its own `(seed, rank)` RNG, so workers get different schedules. Multiple workers can be slow simultaneously. This matches the real-straggler model — stragglers affect workers independently.

**Cross-framework comparability:** since the schedule is a pure function of `(seed, rank)`, DDP seed-0 and DiLoCo seed-0 generate the exact same per-rank episode list. The wall-clock timestamps at which episodes fire are also the same (measured as elapsed-since-start), though the *training progress* at those timestamps differs between frameworks (which is precisely the effect we're measuring).

**Resume-after-crash:** the schedule is regenerable from the seed, so on restart, the replacement process re-runs the constructor deterministically. `schedule_position` is persisted in the checkpoint (see "DDP checkpoint format") so that episodes already-fired are not re-fired.

### Why time.sleep instead of CPU-bound busy-wait

A `time.sleep(x)` releases the GIL and yields the core. A busy-wait would consume host CPU that could interfere with other workers on the same VM (particularly NCCL's background threads). Since we're on a shared 4-GPU VM, sleep is the correct choice.

## Communication instrumentation

The wall-clock breakdown (Metric #5) and the communication-bytes plot (Plot 4) both require measuring communication time and byte volume symmetrically across frameworks. A naive implementation would wrap `dist.all_reduce` at the Python level, but that only works for DiLoCo — DDP's real gradient all-reduce happens inside autograd hooks during `loss.backward()`, implemented in C++, and is invisible to a Python wrapper. If we instrument only the Python path, DDP's Plot 4 bar is zero and Plot 5's "communication" bucket for DDP is mislabeled as "compute."

**Approach: use DDP's built-in comm-hook API, and use a matching wrapper on the DiLoCo side.**

For DDP, register a communication hook via `model.register_comm_hook(state, hook_fn)`. The hook runs inside backward when DDP is about to all-reduce a gradient bucket.

Note that PyTorch's default all-reduce hook (`default_hooks.allreduce_hook`) expects a **`ProcessGroup`** (or `None`, which means `dist.group.WORLD`) as its `state` argument — it does not accept arbitrary state objects. To pass our metrics through, we wrap the process group and the metrics in a small state class and call `default_hooks.allreduce_hook` with the process-group attribute extracted from it:

```python
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import torch.distributed as dist
import time

class CommHookState:
    def __init__(self, process_group, metrics):
        self.process_group = process_group
        self.metrics = metrics

def instrumented_allreduce_hook(state: CommHookState, bucket):
    t0 = time.perf_counter()
    tensor = bucket.buffer()
    state.metrics.comm_bytes += tensor.element_size() * tensor.numel()
    # default_hooks.allreduce_hook wants a ProcessGroup as its state, not our wrapper.
    fut = default_hooks.allreduce_hook(state.process_group, bucket)
    def _record(fut_inner):
        state.metrics.comm_seconds += time.perf_counter() - t0
        return fut_inner.value()[0]
    return fut.then(_record)

hook_state = CommHookState(process_group=dist.group.WORLD, metrics=self.metrics)
self.model.register_comm_hook(state=hook_state, hook=instrumented_allreduce_hook)
```

This gives us `comm_bytes` and `comm_seconds` that reflect the *actual* NCCL all-reduces DDP performs. Note that `comm_seconds` as measured this way charges the full barrier-wait-plus-transfer time to communication, which is what we want for the "under stragglers, DDP's comm bucket balloons" story.

For DiLoCo, the communication path is visible — the outer-step `dist.all_reduce` is written explicitly in Python. Wrap it with the same byte-counting and timing logic so the two frameworks are instrumented symmetrically:

```python
def instrumented_allreduce(tensor, op, metrics):
    t0 = time.perf_counter()
    metrics.comm_bytes += tensor.element_size() * tensor.numel()
    work = dist.all_reduce(tensor, op=op, async_op=False)
    metrics.comm_seconds += time.perf_counter() - t0
    return work
```

**Sanity check (do this once, then trust the counters):** run `torch.profiler` on a single clean DDP step and verify that the comm-hook-measured `comm_seconds` is within ~10% of the profiler-measured NCCL time. This confirms the hook is catching everything and not double-counting.

**What gets charged to which bucket (Plot 5):**

- **Compute:** forward + backward excluding the comm-hook's recorded communication time.
- **Communication:** sum of `comm_seconds` from the hook (DDP) or the wrapper (DiLoCo). Includes barrier wait, because NCCL's all-reduce blocks until all participants arrive.
- **Sync wait / other:** everything else — checkpoint I/O, rendezvous, torchrun restart overhead, the 30-second simulated replacement delay, eval time outside the training step.

This framing makes DDP and DiLoCo directly comparable on Plot 5 and on Plot 4.

## Evaluation cadence

Evaluate validation loss frequently enough that the "first eval ≤ target" measurement is not the dominant source of quantization error in the headline wall-clock-to-target metric.

**Target cadence: one eval approximately every 30 seconds of wall-clock**, subject to the constraint that **eval overhead is < 5% of total wall-clock**. At ~35 min per clean run, 30-second cadence gives ~70 evals — well above the older "10–20 evals" budget, which was too coarse: at 10–20 evals, the headline metric is quantized to ~2–3 min, and DDP vs DiLoCo wall-clock gaps under crashes are on the order of 3–5 min. A 2-min quantization blurs exactly the effect we're trying to measure.

**Cadence concretization:**

- Use a small fixed eval-batch subset (e.g., 16 batches) sized so an eval pass completes in a few seconds. Don't use the full held-out split for every eval.
- For DDP: evaluate whenever `wall_clock_since_last_eval >= 30 s`.
- For DiLoCo: evaluate only at outer-step boundaries (the model is only in a synchronized state there). Pick an eval-every-K-outer-steps value K such that K × outer_step_duration ≈ 30 s. At H=10 this may mean every outer step; at H=500 this may mean every 1 outer step anyway, because large-H outer steps are themselves ~30 s+.
- If eval overhead creeps above 5%, increase the eval-batch subset stride (more batches seen, but less often) rather than going back to a coarse wall-clock cadence.

**Exactly-once eval at target:** once the eval loss crosses the target, record `wall_clock_at_first_crossing` as the moment the eval *completed*, not the moment the training step before it finished. This is a small but consistent convention — apply it identically to both frameworks.

Eval uses a held-out validation split of TinyStories. Fix the eval batches at startup so eval numbers are comparable across runs. Eval runs on rank 0's model (under DDP, all ranks are identical; under DiLoCo, rank 0's model is θ_outer at an outer-step boundary).

## Dev / debugging workflow

1. **Local CPU smoke test first.** Run `train.py --framework diloco --failure none --seed 0` with `N=2` on CPU (set `CUDA_VISIBLE_DEVICES=""` and use `gloo` backend) to shake out obvious bugs. Use a tiny model (2 layers, 128 hidden) and a tiny dataset (first 1000 TinyStories samples) to run in under a minute.
2. **Kaggle 2×T4 dev.** Same smoke test with real GPUs. Verify NCCL comes up, DDP baseline converges on a short run, DiLoCo converges on a short run, W&B logging works, failure injection triggers at the scheduled token counts.
3. **Single paid Lambda run.** Run one clean baseline on Lambda 4×A10 end-to-end to sanity check convergence and measure `T_baseline`.
4. **Lock in crash token thresholds.** Based on `T_baseline`, compute {0.25, 0.50, 0.75} × `T_baseline` and commit them to config.
5. **Full matrix on Lambda.** Run all 24 runs (12 experimental cells × 2 seeds). Monitor W&B dashboard for divergence or stalls.
6. **Re-run any non-converging cells** with tuned hyperparameters, if needed. Do NOT report non-converging runs.

## Key pitfalls to watch for

1. **DiLoCo with DDP wrapper** — if you accidentally wrap the DiLoCo model in `DistributedDataParallel`, you'll get per-step all-reduces and the experiment is meaningless. The DiLoCo model should be bare `nn.Module`.
2. **Inner AdamW persistence** — if you accidentally create the inner optimizer once and reuse it across outer steps, the running moments persist in a way the algorithm doesn't account for. Create it fresh each outer step.
3. **Outer optimizer recreation** — the opposite mistake. If you recreate the outer optimizer each outer step, you lose the persistent Nesterov momentum buffer and DiLoCo's convergence will be much worse than it should be.
4. **Pseudo-gradient sign** — `θ_outer − θ_local`, not the other way around. Getting this wrong produces a training loop that diverges.
5. **Tokens-consumed counter bug on rollback** — after a DDP crash rollback, the tokens counter must reflect only *committed* tokens (those in the checkpoint). Tokens processed since the checkpoint are NOT committed. If you double-count, tokens-to-target is artificially inflated for DDP.
6. **NCCL timeout** — default NCCL timeout is 30 minutes. Under crash injection, a worker dying mid-all-reduce will hang survivors for that long unless the timeout is reduced. Set `NCCL_TIMEOUT=60` (seconds) or use PyTorch's `init_process_group(timeout=timedelta(seconds=60))`.
7. **Eval during inner loop for DiLoCo** — don't. Only eval at outer boundaries.
8. **Data shard overlap on respawn** — when a worker is respawned, it must resume the dead worker's shard from that rank's last **committed** dataloader cursor, not from the beginning of the shard and not from a heuristic skip based on raw tokens. The dead worker's uncommitted inner-loop batches are intentionally lost; committed batches must not be replayed.
9. **Outer step applied from θ_local instead of θ_outer (DiLoCo)** — the inner loop mutates `self.model`'s parameters in place, so by the end of the inner loop the live params are θ_local, not θ_outer. If you call `self.outer_optimizer.step()` at that point, SGD+Nesterov updates θ_local instead of θ_outer, which does not match the algorithm in `algorithm-notes.md` and produces a subtly wrong training loop. Fix: after computing and all-reducing the pseudo-gradients but **before** calling `self.outer_optimizer.step()`, copy θ_outer back into the live parameters (`p.data.copy_(theta_outer[name])`) and then assign `p.grad = delta_global`. The corrected sequence is reflected in the DiLoCo trainer sketch above.
10. **Using `torchrun` at all for DiLoCo crash runs** — torchrun's TorchElastic agent tears down surviving worker processes on ANY worker exit, regardless of `--max_restarts`. `--max_restarts=0` means "do not restart," NOT "leave surviving workers alone." If the DiLoCo crash runs are launched via torchrun, the whole "survivors keep running" property the experiment relies on is destroyed on first crash. For DiLoCo crash runs use the plain shell launcher (Case 2 in "Launch command") that spawns N workers directly with `init_process_group(init_method="env://")` and uses the control `FileStore` only as a side-band coordination channel, not as the actual process-group init backend. The DDP crash path is the only place `torchrun --max_restarts>0` is used.
11. **Assuming `destroy_process_group()` + `init_process_group()` mid-run is a supported PyTorch fault-tolerance flow** — it is not. PyTorch's distributed docs describe the destroy/reinit cycle as outside the supported model, and NCCL process groups are not designed to be torn down and rebuilt at runtime. This crash regime depends on making that path work at small world sizes. If it does not, stop and revise the experiment rather than silently substituting a full-restart DiLoCo path or papering over the hang with ad-hoc retries or CUDA resets.
12. **NCCL collective timeout shorter than the DiLoCo rejoin window** — if NCCL's collective timeout is shorter than `(30 s replacement delay + replacement spawn/load/reinit time)`, survivors will time out at the outer all-reduce before the replacement arrives, and the DiLoCo crash path collapses to the DDP failure mode. Because the replacement skips H local steps on its first rejoin and contributes `Δ=0`, the timeout budget should be sized around rejoin overhead rather than local training time. Size the timeout conservatively for DiLoCo crash runs (120 s is a reasonable starting point).

## References

- DiLoCo paper: Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models," arXiv 2311.08105, 2023.
- nanoGPT: https://github.com/karpathy/nanoGPT
- TinyStories: Eldan and Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" arXiv 2305.07759, 2023.
- PyTorch DDP docs: https://pytorch.org/docs/stable/notes/ddp.html
- TorchElastic: https://pytorch.org/docs/stable/elastic/run.html
- OpenDiLoCo (reference only, not used): https://github.com/PrimeIntellect-ai/OpenDiLoCo
