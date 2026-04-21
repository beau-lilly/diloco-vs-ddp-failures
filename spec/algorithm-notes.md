# Algorithm Notes — DiLoCo and DDP-Elastic

This document specifies the exact training algorithms for both frameworks. The goal is that a reader can implement either from this spec alone without re-reading the source papers.

## Reference: what "DDP" does (for contrast)

Standard `torch.nn.parallel.DistributedDataParallel` runs one synchronous step at a time:

1. Every worker computes a forward pass on its own local minibatch.
2. Every worker computes a backward pass, producing local gradients.
3. `all_reduce(gradients, op=SUM)` runs across workers (implicitly, via DDP's autograd hooks). Every worker ends up holding identical, averaged gradients.
4. Every worker applies the same optimizer step to its (identical) model parameters.

Every worker blocks at step 3 until the slowest worker arrives at the barrier. This is the "tail at scale" pathology that DiLoCo is designed to sidestep.

## DiLoCo: overview

**Paper:** Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models," 2023 (DeepMind).

DiLoCo replaces per-step gradient all-reduce with an **outer loop that operates on pseudo-gradients** once every H inner steps. The inner loop is local AdamW on each worker. The outer loop is SGD with Nesterov momentum applied to the pseudo-gradients (parameter deltas) all-reduced across workers.

### Two-loop structure

```
θ_outer ← initial model parameters (identical on all workers)
outer_momentum_buffer ← zeros_like(θ_outer)  # persistent across outer steps

for outer_step in range(num_outer_steps):
    # ---- INNER LOOP (each worker runs independently, no sync) ----
    θ_local ← copy of θ_outer
    inner_optimizer ← AdamW(θ_local, lr=inner_lr)  # FRESH each outer step
    for inner_step in range(H):
        batch ← next minibatch from this worker's data shard
        loss ← forward(θ_local, batch)
        grads ← backward(loss)
        inner_optimizer.step(grads)   # updates θ_local in-place
    # ---- END INNER LOOP ----

    # ---- OUTER STEP (synchronous, all workers participate) ----
    Δ_local ← θ_outer − θ_local         # PSEUDO-GRADIENT (sign-flipped param delta)
    Δ_global ← all_reduce(Δ_local, op=MEAN)
    # Apply SGD+Nesterov with persistent momentum buffer:
    outer_momentum_buffer ← 0.9 * outer_momentum_buffer + Δ_global
    update ← 0.9 * outer_momentum_buffer + Δ_global       # Nesterov lookahead
    θ_outer ← θ_outer − outer_lr * update
    # ---- END OUTER STEP ----
```

### Critical correctness details

1. **Pseudo-gradient sign.** Δ_k = θ_outer − θ_local_k, NOT θ_local_k − θ_outer. Inner AdamW already moved parameters *downhill* (in the direction that reduced loss), so θ_local is "where we want θ_outer to be." The pseudo-gradient is therefore the amount θ_outer needs to *increase* to catch up, which is θ_outer − θ_local (positive in the direction of descent). The outer optimizer then subtracts `outer_lr * update`, moving θ_outer toward θ_local.

2. **Persistent outer momentum buffer.** The SGD+Nesterov momentum buffer on the outer optimizer **persists across outer steps**, for the entire training run. It is NOT reset each outer step. This is crucial — the persistent momentum is half the reason DiLoCo works at all, because it smooths across the noise of individual outer-step pseudo-gradients.

3. **Fresh inner optimizer each outer step.** Each worker creates a new AdamW instance at the start of each inner loop, initialized from θ_outer. The AdamW state (running moments) is local and ephemeral — it does NOT persist across outer steps. This is different from normal training and is required for DiLoCo's math to work out.

4. **Data sharding.** Each worker sees a disjoint shard of the data (standard data-parallel sharding). Workers do NOT see the same batches.

5. **Only the parameters are all-reduced.** Optimizer states, gradients, activations — none of these cross worker boundaries. The only sync traffic is the pseudo-gradient all-reduce at each outer step, which is the reason DiLoCo's communication cost is 1/H of DDP's.

### PyTorch implementation pattern

The outer optimizer can be implemented directly using `torch.optim.SGD` with `momentum=0.9, nesterov=True`, by treating Δ_global as a "gradient" and manually placing it in `param.grad`. The subtlety is that the inner loop mutates the live parameters in place, so by the time we want to take the outer step the live state is θ_local. Before calling `outer_optimizer.step()`, we must restore θ_outer into the live parameters — otherwise the outer SGD+Nesterov step is applied from θ_local and the algorithm is wrong.

```python
outer_optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.7,            # outer_lr — see hyperparameters below
    momentum=0.9,
    nesterov=True,
)

# At each outer step:
# 1. Snapshot θ_outer BEFORE the inner loop begins.
theta_outer = {n: p.detach().clone() for n, p in model.named_parameters()}

# 2. Run the inner loop (mutates live params from θ_outer → θ_local).

# 3. Compute Δ_local = θ_outer − θ_local per parameter, then all_reduce(AVG).
deltas = {}
for n, p in model.named_parameters():
    d = theta_outer[n] - p.detach()
    dist.all_reduce(d, op=dist.ReduceOp.AVG)
    deltas[n] = d

# 4. CRITICAL: restore θ_outer into the live params, then assign Δ_global to .grad.
with torch.no_grad():
    for n, p in model.named_parameters():
        p.data.copy_(theta_outer[n])
        p.grad = deltas[n]

# 5. Now the outer optimizer steps from θ_outer with pseudo-gradient Δ_global.
outer_optimizer.step()        # SGD+Nesterov update using the persistent buffer
outer_optimizer.zero_grad()
```

This is cleaner than hand-rolling the Nesterov update because PyTorch's SGD already maintains the persistent momentum buffer correctly. The restore-before-step is load-bearing: without it, `outer_optimizer.step()` would update θ_local instead of θ_outer.

### Hyperparameters (DiLoCo)

| Hyperparameter | Value | Notes |
|---|---|---|
| `H` (inner steps per outer step) | **50** initially | Paper tested {10, 50, 100, 500}. H=50 is a reasonable default for small models. If communication cost is noticeably harmful at H=50, sweep upward. |
| Inner optimizer | AdamW | Standard nanoGPT settings |
| Inner learning rate | 3e-4 (cosine decay) | Match nanoGPT default |
| Inner weight decay | 0.1 | Match nanoGPT default |
| Inner betas | (0.9, 0.95) | Match nanoGPT default |
| Outer optimizer | SGD | With Nesterov momentum |
| Outer learning rate | **0.7** | From DiLoCo paper. Held constant throughout training (no decay). |
| Outer momentum | 0.9 | Nesterov-enabled |
| Outer LR schedule | Constant | No decay on outer LR |
| Grad clipping (inner) | 1.0 | Applied within inner loop only |

### What DiLoCo does on a crash

If a worker dies mid-inner-loop, its partial inner-loop work is lost, but the **surviving workers do not stop**. They continue running their own inner loops on their own data shards. When the replacement worker joins (at the next outer sync boundary), it loads the latest committed `θ_outer`, **skips local inner steps for that rejoin event**, and participates in the next outer step with a zero pseudo-gradient. There is no rollback and no re-execution of lost work beyond what the dead worker itself had done.

**Crash-step state transition (the one we implement):**

1. Worker `v` receives SIGKILL at wall-clock `t_crash`, somewhere inside its inner loop (inner step `k` out of `H`).
2. Surviving workers `{0, ..., N-1} \ {v}` continue their own inner loops unaffected. The authoritative cluster-wide `tokens_raw` counter (see `measurement-plan.md`) keeps advancing. For crash accounting, rank 0 (or the sidecar) also maintains `worker_tokens_since_commit[rank]`, reset at every successful outer step. If worker `v` dies, that worker-local subtotal becomes DiLoCo `lost_tokens`; survivor subtotals are not lost.
3. When each survivor finishes its own H inner steps, it enters the outer-sync point and **blocks at a rendezvous barrier** waiting for all N participants. The survivors do NOT proceed with a smaller world size; the outer all-reduce keeps `world_size = N` semantics. This is the simplification we take for scope.
4. A fixed 30-second replacement delay elapses (simulating cluster scheduler detection + pod spawn), then the replacement worker process is launched. It joins the process group via a file-based rendezvous (see implementation below), loads the latest committed `θ_outer`, and arrives at the outer-sync barrier **without doing local inner steps first**.
5. At that first post-rejoin outer sync, the replacement contributes `Δ_rejoin = θ_outer − θ_outer = 0`, while the survivors contribute their real pseudo-gradients. The all-reduce proceeds with all N workers participating. `θ_outer` advances by one outer step, and `tokens_committed` is updated to the current `tokens_raw`. Training continues.

This choice — "survivors finish their inner loop then freeze at the outer barrier until the replacement is up, loaded, and reattached" — means:

- DiLoCo `lost_tokens` per crash ≈ the dead worker's partial inner-loop work (≤ 1 worker × H inner steps). Survivors lose nothing.
- World size is fixed at N for every outer step that is actually executed. There is no "outer step with only N−1 participants."
- Survivor idle time at the outer barrier ≈ (30 s replacement delay) + (replacement spawn/load/reinit time). This idle time is charged to the "sync wait / other" bucket in the wall-clock breakdown (Metric #5).
- Crash thresholds are compared against the global raw token counter while survivors are running, but the controller suppresses any second crash until the first rejoin finishes. See `measurement-plan.md` for committed-progress semantics.

**Implementation mechanism:** the replacement worker reattaches to the process group via an `env://` reinitialization on the regular NCCL process group, coordinated by a side-band `torch.distributed.FileStore` control channel, NOT through `torchrun --max_restarts`. This is the key divergence from the DDP path — see `implementation-notes.md` "DiLoCo crash path" for the rendezvous wiring. We deliberately do NOT reuse torchrun's kill-and-respawn mechanism for DiLoCo, because it would destroy the "survivors keep running" property that the experiment is measuring.

### What DiLoCo does on a straggler

Fast workers finish their H inner steps quickly and then wait at the outer-sync all-reduce for the slow worker to arrive. The slow worker eventually arrives (possibly with its own H inner steps completed, or possibly fewer if the straggler window was longer than its inner-loop duration). The all-reduce proceeds and everyone continues.

The cost of a straggler episode to DiLoCo is therefore: **(slowdown amount) × (time the straggler overlaps with an outer-sync barrier)**. Between outer syncs, the slow worker's slowdown does not affect the fast workers — they are doing their own local work. This is the core communication-avoidance property.

## DDP-elastic baseline

### Structure

The DDP-elastic baseline is standard synchronous data-parallel training with three extensions:

1. **TorchElastic-style rendezvous** — the launcher (`torchrun --nnodes=1:1 --nproc_per_node=4 --rdzv_backend=c10d ...`) handles the full kill-and-respawn cascade on crash. When a worker dies, NCCL's process group is corrupted and cannot be resumed; under `--max_restarts=N`, torchrun kills any remaining worker processes, respawns a fresh set of N workers, and they rendezvous and load the last checkpoint. This is a total restart, not an in-place membership patch — see "What DDP-elastic does on a crash" below for the step-by-step sequence.
2. **Periodic full-model checkpointing** — every 5 minutes of wall-clock training time, rank 0 writes `{model.state_dict(), optimizer.state_dict(), rng_state, token_counter, step_counter}` to a local checkpoint file. Checkpoint I/O is synchronous and blocks the training loop (acceptable at 5-minute cadence because it's a small fraction of wall-clock).
3. **On crash, roll back to last checkpoint.** All workers (including the replacement) load the last checkpoint and resume from there. Work done since the last checkpoint is lost.

### Hyperparameters (DDP-elastic)

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | Same as DiLoCo inner |
| Learning rate | 3e-4 (cosine decay) | Same as DiLoCo inner |
| Weight decay | 0.1 | Same as DiLoCo inner |
| Betas | (0.9, 0.95) | Same as DiLoCo inner |
| Grad clipping | 1.0 | Same as DiLoCo inner |
| Checkpoint interval | **5 minutes wall-clock** | Tunable. 5 minutes is a realistic production cadence — short enough that crash recovery isn't catastrophic, long enough that checkpoint I/O isn't dominant. |
| Backend | NCCL | Standard for multi-GPU single-node |
| Rendezvous | c10d | TorchElastic built-in |

### What DDP-elastic does on a crash

Standard PyTorch DDP has no graceful recovery for mid-run worker loss — NCCL process groups cannot tolerate peer failures, and `torch.distributed` has no "shrink membership" primitive. Recovery is therefore always a full restart:

1. A worker dies (SIGKILL). The time of death is arbitrary — it could be during forward, backward, optimizer step, or all-reduce; the survivors don't know immediately because NCCL has no heartbeat.
2. Surviving workers continue their own current step until they reach the next collective operation (the all-reduce at the end of their backward pass). At that point they block, waiting for the dead worker's contribution that will never arrive.
3. NCCL eventually times out (configured to ~60 seconds) and raises an error. The survivors' training loops exception out.
4. `torchrun` notices that its child worker processes have exited with errors. Under `--max_restarts=N`, it **kills any remaining worker processes** (because the corrupted process group can't be cleanly resumed) and respawns a fresh set of N workers.
5. The fresh workers initialize a new process group and load from the last checkpoint on disk.
6. Training resumes from the checkpoint. **All work done since the last checkpoint, on every worker, is lost.**

The dominant cost per crash is therefore the **rolled-back work** — roughly `0.5 × checkpoint_interval` worth of tokens on average, across all N workers. At a 5-minute checkpoint interval, each crash throws away ~2.5 minutes of cluster-wide progress. On top of that, there is a roughly fixed ~60-90 second restart overhead (NCCL timeout + relaunch + checkpoint load), which is smaller than the rollback cost and largely determined by NCCL timeout configuration rather than any experimental variable.

**What is NOT happening (common misconception):** the surviving workers do NOT "pause, wait for the replacement, and continue where they left off." That would require custom machinery that standard PyTorch DDP does not provide. The restart is total.

### What DDP-elastic does on a straggler

The slow worker arrives late at the backward-pass all-reduce barrier on every step it's slow on. All other workers block on the barrier, waiting for it. Effective throughput for the entire cluster during the straggler window ≈ 1 / slowdown_factor (so at 3× slowdown, effective throughput drops to 33% of clean).

This is the classic "tail at scale" phenomenon and is the reason DiLoCo is expected to show a large advantage under straggler conditions.

## Shared infrastructure

Both frameworks should share the following so that only the sync logic and the crash-rejoin logic differ:

- Model definition (nanoGPT)
- Dataset loading and sharding
- Forward pass implementation
- Inner optimizer (AdamW) configuration
- Logging (W&B)
- Checkpointing format (same file layout; DiLoCo writes at outer-step boundaries, DDP writes every 5 minutes)

The launch mechanism, the sync logic, the outer optimizer (DiLoCo only), and the crash-recovery path differ:

- **Initial launch.** Clean runs, straggler runs, and DDP crash runs use `torchrun`. DiLoCo crash runs bypass torchrun — they are started from a plain shell loop that spawns N workers directly and each worker calls `init_process_group(init_method="env://")` itself. A side-band `FileStore` is opened only for recovery flags (`rejoin_pending`, `replacement_ready`, `recovery_complete`), not as the process-group rendezvous backend. This is because torchrun's agent tears down surviving workers on any worker exit, regardless of `--max_restarts`, which would destroy DiLoCo's survivor-continuity property on a crash. See `implementation-notes.md` "Launch command" for the two launch cases.
- **DDP-elastic on crash:** torchrun's `--max_restarts` kills all remaining workers, respawns the full set of N, and loads the last checkpoint. Standard PyTorch behavior. Described below.
- **DiLoCo on crash:** survivors stay alive; a replacement worker is spawned by a sidecar controller and the process group is destroyed-and-reinitialized around it at the next outer-sync boundary, coordinated through a `FileStore`. This rendezvous-plus-reinit path is not a supported PyTorch fault-tolerance flow and is therefore the main engineering risk in the crash regime. Described in `implementation-notes.md` under "DiLoCo crash path." This is the deliberate asymmetry that makes the experiment measurable.

This asymmetry is intentional — the whole point of the crash regime is to observe that DiLoCo's recovery model is different from DDP's. Everything *else* (model, data, inner optimizer, logging) is held fixed so the comparison is clean.
