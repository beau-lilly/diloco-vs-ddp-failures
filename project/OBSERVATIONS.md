# Observations — running notes during the experiment

Running log of things noticed during the actual Lambda runs. Meant to be
a scratchpad for "fix after the runs finish" items and for anything
surprising we want to revisit when writing up the results.

## Bug #1 — `sync_wait_other_seconds` is ~97% of wall-clock in DDP

**When/where:** 2026-04-21, Group A on Lambda 8×A100 (40 GB SXM4), during
the DDP clean runs (`ddp_none_s0` / `ddp_none_s1` at `target_loss: 2.5`).
Observed in the final metrics payload logged at run completion.

**What we saw:** for a DDP clean run that actually trained correctly
(26.7M tokens, loss 10 → 2.5 in 356 s wall-clock), the breakdown metrics
are:

```
final/total_wall_clock:        359.27 s
final/total_compute_seconds:     8.67 s
final/total_optimizer_seconds:   1.31 s
final/total_comm_seconds:        0.69 s
final/sync_wait_other_seconds: 348.59 s   # derived
```

Compute + optimizer + comm = ~10.7 s out of 359 s; everything else is
falling into the `sync_wait / other` bucket.

**Headline metric is unaffected.** `final/wall_clock_to_target: 355.91 s`
is measured directly from `time.perf_counter()` between the first
training step and the first eval that crosses the target — it does not
depend on the bucketed timers. All the other headline metrics
(`tokens_to_target`, `total_comm_bytes`, `num_crashes`, etc.) are also
unaffected.

**What is affected:** Plot 5 (stacked-bar wall-clock breakdown by cell)
will show an implausibly large `sync_wait / other` bucket dominating the
bars, obscuring the compute-vs-comm contrast the plot is supposed to
highlight.

**Probable root cause** (to verify when we get back to the code):

1. `DDPTrainer.step` times forward+backward inside `metrics.time("compute")`,
   but the training loop has a bunch of additional work each iteration
   that is NOT inside any timing bucket — most notably `self._evaluate()`
   on rank 0 and the `dist.broadcast(flags)` that happens every step.
   On ranks 1–3, the broadcast blocks waiting for rank 0's eval to
   finish, so tens of seconds of eval-wait time accumulate in
   `wall_clock_elapsed` without being bucketed.
2. Less likely but also possible: the DDP comm hook is only counting the
   very fast portion of the all-reduce (not barrier-wait), so `comm_seconds`
   is under-counted. If so, the measured `comm_seconds` of 0.69 s over
   400 steps is too low even for NVLink NCCL.

**Fix (post-run, low priority):**

1. Wrap `self._evaluate()` in `with self.metrics.time("eval")` and add an
   eval bucket to the breakdown.
2. Wrap `dist.broadcast(flags, src=0)` in a new `barrier/sync` bucket so
   the per-step wait-for-rank-0 shows up correctly.
3. Re-verify the DDP comm hook catches the whole future duration with a
   `torch.profiler` sanity check (the spec's "Communication
   instrumentation" note mentions doing this once).

Tracking this as a follow-up — not worth killing Group A to fix now.
All headline metrics needed for Plots 1, 2, 4, 6 are correct; Plot 5
will need a re-compute once the fix lands.
