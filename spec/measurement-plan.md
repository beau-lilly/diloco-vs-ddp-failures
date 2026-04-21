# Measurement Plan

## Token counters and committed-progress semantics

Multiple metrics and failure-injection decisions depend on a precise notion of "how much progress has been made." The spec uses one authoritative raw counter and one authoritative notion of "committed" progress per framework. These are NOT interchangeable with worker-local step counters or per-worker token counts.

**Global raw token counter (`tokens_raw`).** A single counter maintained on rank 0. Each worker emits a "tokens processed by this worker" contribution (`per_worker_batch_size × context_length`) whenever it finishes a local minibatch. Rank 0 sums those contributions into the authoritative cluster-wide raw counter. Under ordinary 4-worker operation, this is equivalent to saying that each synchronized DDP step advances `tokens_raw` by `global_batch_size × context_length`, where `global_batch_size = N × per_worker_batch_size`. For DiLoCo, the same cluster-wide amount is accumulated over a full "round" of N workers each finishing one local inner step. During DiLoCo crash recovery, however, the counter may temporarily advance from fewer than N workers while the replacement is outstanding, so the authoritative definition is "sum of completed worker contributions reported to rank 0," not "exactly one global-batch quantum per logical step."

`tokens_raw` is the counter the crash controller compares against its thresholds. Under the DiLoCo crash path, `tokens_raw` is not rewound on a crash — survivors never roll back, so the cluster-wide count of "tokens the cluster has chewed through" does not go backwards. Under the DDP crash path, `tokens_raw` IS rewound on restart: after all workers reload the last checkpoint, the authoritative `tokens_raw` is reset to the checkpointed value (which equals `tokens_committed` at that moment), and advances again from there. This is a consequence of the kill-everyone restart model.

In both frameworks, "already-fired" crash thresholds are tracked by the sidecar itself — not inferred from `tokens_raw` — so a rewind of `tokens_raw` never causes a threshold to re-fire. The sidecar persists for the whole lifetime of the run, including DDP worker restarts, so its `next_idx` stays authoritative without living inside the worker checkpoint.

**Committed tokens per framework (`tokens_committed`).**

- **DDP:** `tokens_committed` = value of `tokens_raw` at the last successful checkpoint write. Updated exactly when `save_checkpoint()` returns successfully.
- **DiLoCo:** `tokens_committed` = value of `tokens_raw` at the last successful outer-step all-reduce + outer optimizer step (both must complete).

**Use of each counter:**

- **Crash thresholds** fire against `tokens_raw`. Threshold bookkeeping lives in the sidecar's own `next_idx`, not in any worker-local counter. Only one crash may be "in flight" at a time: once a crash fires, the sidecar/controller suppresses later thresholds until that recovery finishes. This prevents overlapping rejoin windows in DiLoCo.
- **Crash event logging** records (a) the victim rank, (b) `tokens_raw` at crash time, (c) `tokens_committed` at crash time, and (d) framework-specific `lost_tokens`:
  - **DDP:** `lost_tokens = tokens_raw − tokens_committed` at crash time, because the whole cluster rolls back.
  - **DiLoCo:** `lost_tokens = victim_tokens_since_last_outer_sync`, where rank 0 (or the sidecar) tracks a per-worker `worker_tokens_since_commit[rank]` counter that resets to zero after every successful outer step. Survivor work since the last outer sync is NOT lost and must not be charged to `lost_tokens`. The replacement does not create an additional pre-sync loss term, because on its first rejoin event it skips local steps and contributes `Δ=0`.
- **Wall-clock to target** does NOT rewind on crash. It is a pure wall-clock measurement from the first optimizer step to the first eval that crosses the target.
- **tokens_to_target (headline secondary metric)** — see definition below.

**tokens_to_target definition.** The correct formula depends on whether the raw counter rewinds during recovery:

- **Clean runs:** `tokens_to_target = tokens_raw_at_first_crossing`
- **DDP crash runs:** `tokens_to_target = tokens_raw_at_first_crossing`
- **DiLoCo crash runs:** `tokens_to_target = tokens_raw_at_first_crossing − sum(lost_tokens over all crashes in this run)`

Why the split: in DDP, `tokens_raw` is rewound on restart, so the final counter already excludes rolled-back work and subtracting `lost_tokens` again would double-count the rollback penalty. In DiLoCo, the global raw counter is monotonic across the crash because survivors keep running, so the victim's discarded local work remains inside `tokens_raw` unless we subtract it explicitly. The replacement does not add extra uncommitted local work before its first rejoin, because it skips local steps and contributes `Δ=0` on that first post-rejoin outer sync. This is the accurate "effective tokens consumed to target" reading. It is still NOT `tokens_committed` at first crossing, because `tokens_committed` for DDP only advances at 5-minute checkpoint boundaries and would under-report the true progress by up to one checkpoint interval.

## Primary metrics

### 1. Wall-clock seconds to target loss

The headline metric. Time from training start (process launch → first optimizer step) until the first evaluation run reports validation loss ≤ 2.0. Measured on rank 0 using `time.perf_counter()` with start recorded at the first training step (not at `dist.init_process_group` — we don't want to include rendezvous time in the baseline).

Reported per run, then aggregated across seeds within each cell as mean ± std.

### 2. Tokens consumed to target loss

Reported using the framework-aware definition from "Token counters and committed-progress semantics" above:

- Clean runs: `tokens_raw_at_first_crossing`
- DDP crash runs: `tokens_raw_at_first_crossing`
- DiLoCo crash runs: `tokens_raw_at_first_crossing − sum(lost_tokens)`

Under clean conditions there are no crashes, so this is just "total tokens processed." Under DDP crash recovery, `tokens_raw` has already been rewound to the last committed point on restart, so the final counter already excludes rolled-back work. Under DiLoCo crash recovery, only the dead worker's uncommitted local progress is removed via `sum(lost_tokens)`.

Wall-clock, in contrast, continues to tick during and after rollback — that's the point.

Aggregated across seeds as mean ± std.

## Secondary metrics

### 3. Communication volume (bytes all-reduced)

Counted per run, reported as total bytes moved through all-reduce across the full run. Collected symmetrically across frameworks:

- **DDP:** via a DDP communication hook registered on the model (`model.register_comm_hook`). The hook intercepts every gradient-bucket all-reduce that DDP performs inside backward, records `tensor.element_size() * tensor.numel()` for the bucket, and times the hook's future so comm_seconds is accurate. A plain Python wrapper around `dist.all_reduce` does NOT work here — DDP's all-reduce is invoked from C++ autograd hooks and is invisible to Python wrappers.
- **DiLoCo:** via a thin Python wrapper around the explicit outer-step `dist.all_reduce` that records the same two quantities.

See `implementation-notes.md` "Communication instrumentation" for the exact hooks.

**Expected result:** DiLoCo should be roughly 1/H the communication of DDP (with H=50, ~50× reduction). This is a sanity check on the implementation, not a novel finding — if we don't see the expected ratio, something is wrong with the DiLoCo sync logic, or the DDP comm hook is missing buckets.

### 4. Lost tokens per crash

The explanatory variable that decomposes crash overhead into "rollback cost" and "everything else." The correct definition is framework-specific:

- **DDP:** `lost_tokens = tokens_raw − tokens_committed` at the moment of SIGKILL, where `tokens_committed` is the last checkpoint. The whole cluster rolls back, so all uncommitted work is lost.
- **DiLoCo:** `lost_tokens = victim_tokens_since_last_outer_sync`. Rank 0 (or the sidecar controller) tracks a per-worker `worker_tokens_since_commit[rank]` counter that resets to zero after every successful outer step. Only the victim's uncommitted local work is discarded; survivor work since the last outer sync is retained.

For **DDP**, this is the dominant cost of a crash. When torchrun kills all workers and restarts them from the last checkpoint, every token processed between the last checkpoint write and the crash is thrown away. Expected value is roughly `0.5 × checkpoint_interval`'s worth of tokens (crashes are uniformly distributed within the checkpoint cycle). At a 5-minute checkpoint interval with ~4 workers, rough estimate is ~2M tokens lost per crash. Reported per crash, and averaged across the 3 crashes per run and the 2 seeds per cell.

For **DiLoCo**, this is measured as a **symmetry check**, expected to be much smaller — roughly (1 worker × tokens done by the victim since the last outer sync), bounded by H inner steps on a single worker. Surviving workers lose nothing (they keep running their own inner loops per the DiLoCo crash-step state transition in `algorithm-notes.md`). Concretely, if the victim was at inner step `k` out of `H` when SIGKILLed, `lost_tokens ≈ k × per_worker_batch × context_length`, and this is the only contribution to the crash's rolled-back work. The replacement does not add a second loss term before the next sync, because it rejoins from committed `θ_outer` and contributes `Δ=0` on its first post-rejoin outer step. If this number comes out near-zero as expected, that empirically validates the "DiLoCo doesn't pay rollback cost" claim. If it comes out surprisingly non-zero, that's a real finding worth investigating.

**Not measured separately (and why):** A fixed per-crash recovery overhead exists in DDP (roughly NCCL-timeout + torchrun restart + checkpoint load, ~60-90 seconds total). This is dominated by the NCCL timeout, which is a configuration knob rather than an experimental outcome, so it's not interesting to decompose further. It will still show up automatically in the wall-clock-to-target headline and the wall-clock breakdown (Metric #5).

### 5. Wall-clock breakdown

For each run, break wall-clock into three buckets:

- **Compute:** forward + backward time *excluding* the communication time recorded by the comm hook (DDP) or comm wrapper (DiLoCo).
- **Communication:** sum of `comm_seconds` from the comm hook (DDP) or comm wrapper (DiLoCo). For DDP this includes barrier wait, because the comm hook measures the full duration of the bucket's all-reduce future. For DiLoCo this is just the outer-step all-reduce time.
- **Sync wait / other:** everything else — checkpoint I/O, rendezvous, torchrun restart overhead, the 30-second simulated replacement delay, eval time outside the training step.

See `implementation-notes.md` "Communication instrumentation" for how each bucket is measured in code. Reported as stacked bar chart per cell.

**Expected result:** under stragglers, DDP's "communication" bucket should balloon (workers waiting at all-reduce barrier), while DiLoCo's should stay small. Under crashes, DDP's "sync wait / other" bucket should balloon (restart overhead + re-doing the rolled-back work after reload), while DiLoCo's should stay small.

## Logging

**Weights and Biases (W&B)** as the backend. One W&B run per `(cell, seed)` pair — 12 cells × 2 seeds = 24 W&B runs total. Tags: `framework`, `group`, `seed`, `N`, `H` (DiLoCo only; left null for DDP since DDP is not a point on the H axis), `checkpoint_interval` (DDP only), `failure_config`.

Logged per step (or per ~50 steps to reduce logging overhead):
- `train/loss`
- `train/lr` (inner LR for DiLoCo, LR for DDP)
- `train/tokens_raw` (the authoritative rank-0 counter defined in "Token counters and committed-progress semantics")
- `train/tokens_committed`
- `train/wall_clock_seconds`
- `train/comm_bytes_cumulative`
- `train/grad_norm`

Logged per eval:
- `eval/loss`
- `eval/wall_clock_at_eval`
- `eval/tokens_at_eval`

Logged once at end of run:
- `final/wall_clock_to_target`
- `final/tokens_to_target`
- `final/total_comm_bytes`
- `final/mean_lost_tokens_per_crash` (if crashes were injected)
- `final/num_crashes` (should match the scheduled count)
- `final/num_straggler_events` (counted from the straggler sampling log)

Logged per failure event:
- `failure/event_type` ("crash" or "straggler")
- `failure/event_time` (wall-clock)
- `failure/event_tokens` (tokens at event)
- `failure/victim_rank` (which worker)
- `failure/lost_tokens` (crashes only — tokens since last checkpoint or last outer sync)

## Plots for the final writeup

**Plotting convention for the H sweep.** The X-axis on sweep plots is DiLoCo's H ∈ {10, 50, 100, 500}. DDP-elastic is *not* a point on the H axis — it's a different algorithm (raw-gradient all-reduce + single AdamW), and "DiLoCo at H=1" is a degenerate configuration that would not behave like DDP. Instead, DDP is rendered as a **horizontal reference line** across the full H range on every sweep figure, with a shaded ±1 std band from the DDP seeds. That makes the comparison visually crisp: the region of the DiLoCo curve *below* the DDP reference line is the range of H for which DiLoCo beats DDP.

### Plot 1: wall-clock-to-target vs H (HEADLINE)

One figure. X-axis: H (log scale: 10, 50, 100, 500). Y-axis: wall-clock seconds to target loss.

- Two DiLoCo curves (clean regime, crash regime) with ±1 std error bars across 2 seeds per cell.
- Two DDP horizontal reference bands (clean regime, crash regime), ±1 std across DDP seeds.

**This is the headline figure of the writeup.** Under clean, the DiLoCo curve is expected to be U-shaped; whether it sits below or above the DDP-clean reference at its best H is the clean-regime question. Under crash, the DDP reference band rises (rollback cost) while the DiLoCo curve rises more gently. The *range of H for which the crash-DiLoCo curve is below the crash-DDP reference band* is the answer to "which H values make DiLoCo strictly preferable to DDP under failures."

### Plot 2: tokens-to-target vs H

One figure. X-axis: H (DiLoCo values only). Y-axis: tokens consumed at target loss.

- Two DiLoCo curves (clean, crash).
- Two DDP horizontal reference bands (clean, crash).

Shows the "slack tax" — how much extra data DiLoCo needs to reach the same loss as H grows. Expected to be monotonically increasing in H. Flip side of Plot 1: makes clear that if DiLoCo wins on wall-clock, it's because *communication savings outpace convergence degradation*, not because DiLoCo is tokens-efficient.

### Plot 3: loss curves at representative H values

One figure per regime (2 figures: clean, crash). Each has two subplots:

- **Left subplot:** loss vs wall-clock seconds. Lines for DDP and for DiLoCo H ∈ {10, 50, 500} (low/mid/high exemplars) per seed. DDP drawn in a distinct color/style to mark it as the synchronous reference.
- **Right subplot:** loss vs tokens consumed. Same layout.

Gives the reader the underlying trajectories, not just the endpoints.

### Plot 4: communication bytes vs H

One figure. X-axis: H (DiLoCo). Y-axis: total bytes all-reduced (log scale). One DiLoCo line per regime; DDP as horizontal reference band.

DiLoCo's line is expected to drop as ~1/H. DDP's reference should sit far above the DiLoCo curve at every H we run. Serves as an implementation sanity check (wrong slope means DiLoCo's sync logic has a bug) and quantifies the communication advantage that's paying for DiLoCo's tokens-inefficiency in Plot 2.

### Plot 5: wall-clock breakdown by H (stacked bars)

One figure per regime (2 figures). X-axis: {DDP, DiLoCo H=10, DiLoCo H=50, DiLoCo H=100, DiLoCo H=500}, with DDP visually separated by a gap/divider from the DiLoCo sweep. Y-axis: seconds. Stack per bar: compute / communication / sync wait.

Shows *where* the wall-clock cost lives in each configuration. Under crashes, DDP's bar should be dominated by sync-wait (the rollback and restart). Under clean, DiLoCo's small-H bars should be dominated by communication, large-H bars by compute (since comm is ~1/H).

### Plot 6: lost tokens per crash vs H

One figure. X-axis: H (DiLoCo). Y-axis: mean lost tokens per crash event. Crash regime only. DiLoCo line plus DDP horizontal reference band.

Operationalizes the "DiLoCo loses one worker's H inner steps; DDP loses everyone's work since last checkpoint" story quantitatively. DDP's reference should show ~0.5 × checkpoint_interval × global_throughput tokens lost per crash. DiLoCo's line should be a small, H-linear number bounded by (1 worker × H × per-worker-batch × context).

### Plot 7: spot-instance crossover (synthesized)

One figure. X-axis: assumed preemption rate (crashes per wall-clock-hour, swept analytically from measured per-crash cost). Y-axis: effective cost-to-target in dollars, assuming $3/hr on-demand for DDP and $0.50/hr spot for DiLoCo. One line per H. The intersection with the DDP horizontal shows the "breakeven preemption rate" for each H.

This is the "so what" plot that ties the experiment back to spot-instance economics. It's synthesized (not directly measured) because we can't actually run at arbitrary preemption rates on the budget, but it's directly grounded in the measured per-crash cost from Plot 6 and the clean-regime throughput from Plot 1.

### Plot 8 (secondary cut): straggler loss curves

One figure. Loss vs wall-clock, DDP and DiLoCo H=50 under straggler regime, with clean baselines as dashed lines for reference.

The classic "DDP eats the tail, DiLoCo absorbs it" demonstration. Not part of the main H sweep because stragglers aren't the headline story.

## Statistical analysis

For the primary comparison (wall-clock-to-target as a function of H under each regime), report:

- Mean and std across 2 seeds per cell.
- Paired difference per seed across adjacent H values and across regimes (e.g., "on seed 0, DiLoCo H=50 was X seconds faster than the DDP reference baseline under crash regime"), because the same seed controls the same failure schedule across all cells.
- If the seed-paired differences are consistent in sign across both seeds and the gap is large compared to within-seed noise, note this as a strong (though not formally significance-tested) signal. With only 2 seeds per cell, formal hypothesis testing is underpowered and should not be claimed.

If any cell shows high within-cell variance (the two seeds disagree substantially), flag it in the writeup and add a third seed for that cell specifically. Budget permits this for up to ~4 additional runs.

## Data preservation

All W&B runs stay on W&B indefinitely (project is public or team-scoped for course review).

Raw checkpoints are NOT preserved long-term due to disk cost — only the final model from one canonical run per framework is archived (for qualitative "did the model actually learn English?" sanity check).

Plots are generated from W&B export data (downloaded to a local CSV / parquet), with a plotting script committed to the repo so figures are reproducible from logged data alone.
