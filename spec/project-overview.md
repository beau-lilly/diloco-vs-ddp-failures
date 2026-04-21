# Project Overview — DiLoCo vs DDP Under Failure Injection

## One-line summary

Build a minimal DiLoCo implementation and a DDP-elastic baseline on top of nanoGPT, sweep DiLoCo's H parameter (inner steps per outer sync) under both clean and crash-injected conditions, and characterize the slack-vs-convergence tradeoff that determines whether spot-priced distributed training is worth the preemption risk.

## Course context

This project is for **CPS 390-03 (Distributed Systems)**. The course allows project "flavors" including "make the next lab," "deploy and measure," "explore a transition," and "frame a problem." This project blends **"implement a paper"** (DiLoCo, Douillard et al. 2023) with **"deploy and measure"** (cloud multi-GPU experiments with rigorous fault-tolerance measurement).

Although the substrate is ML training, the **research questions are distributed systems questions**: how much staleness can workers tolerate before eventual consistency costs more than it saves? How does that tradeoff shift when workers start failing? What infrastructure do you need to support dynamic membership without tearing down the process group? ML is the vehicle, not the subject.

## Real-world motivation: spot instance economics

The "so what" behind the experiment is the price gap between on-demand and spot (preemptible) GPUs. Spot instances are 60–90% cheaper than on-demand — an A100 that costs ~$3/hr on-demand can be had for ~$0.50/hr spot — but they can be reclaimed by the cloud provider with as little as 30 seconds of notice, at any time. Any distributed training setup that runs on them has to treat mid-run worker loss as a **baseline operating condition**, not an exceptional event.

Synchronous DDP pays a full checkpoint rollback on every preemption, which at spot-level preemption rates can wipe out the price advantage entirely. DiLoCo's H inner steps are precisely the slack budget that lets surviving workers keep making progress while a dead worker is respawned. The central question the experiment answers is: **how large does H need to be before spot-priced training beats on-demand DDP**, and where does that crossover sit on the H-vs-convergence curve?

## Research question

Given a fixed convergence target (TinyStories eval loss ≤ 2.0), how does wall-clock cost to reach that target vary as a function of DiLoCo's slack parameter H, under both clean and crash-injected conditions, and for which H values does DiLoCo beat synchronous DDP?

- **DDP-elastic** — standard synchronous data-parallel training with TorchElastic + periodic checkpointing. This is the **synchronous reference baseline** — the production-realistic incumbent against which the DiLoCo H sweep is compared. It is *not* on the H axis; DDP is a different algorithm (raw-gradient all-reduce + single AdamW optimizer), not "DiLoCo with H=1." On the sweep plots, DDP shows up as a horizontal reference line with a seed-variance band, one line per regime.
- **DiLoCo** — Distributed Low-Communication training with an inner AdamW loop and an outer SGD+Nesterov loop on pseudo-gradients. Swept across H ∈ {10, 50, 100, 500}.

Secondary question (smaller cut, not swept): how do DDP and DiLoCo compare under probabilistic stragglers at one representative H?

## Hypothesis

**Clean regime.** DiLoCo's tokens-to-target will degrade monotonically as H grows, because longer inner loops accumulate more per-worker drift. DiLoCo's wall-clock-to-target is expected to be non-monotonic in H: at small H, communication cost dominates; at large H, convergence inefficiency dominates; there should be a sweet spot in the middle. DDP sits at a fixed reference point — no H dependence since it synchronizes every step — and its wall-clock position is set by per-step communication cost. Under clean conditions DDP is expected to be competitive with or faster than the best DiLoCo H because its per-step update is algorithmically tighter.

**Crash regime.** DDP's rollback cost grows in proportion to its checkpoint interval and is paid on every crash. Under the specified DiLoCo crash path, rollback cost is bounded by (1 worker × H inner steps) and does not propagate to surviving workers. The DiLoCo H-vs-wall-clock curve is therefore expected to shift upward relatively less than the DDP reference line when crashes are introduced. This claim depends on the survivor-preserving rejoin path actually working; there is no alternate full-restart DiLoCo crash mode in this spec.

**Headline expected finding.** Under crash conditions, there exists some DiLoCo H at which DiLoCo strictly dominates DDP in wall-clock-to-target. The range of dominating H values widens as failure rate increases. In spot-instance economic terms: for any realistic preemption rate, there is an H that makes spot-priced DiLoCo cheaper than on-demand DDP.

## Infrastructure framing (for the writeup background)

The experimental implementation deliberately splits the crash path by framework. DDP uses the simplest machinery PyTorch actually supports (`torchrun` + full kill-and-restart on crash, with a fixed delay to simulate replacement time). DiLoCo's crash path is more ambitious: a single-node sidecar/controller plus a destroy/reinit cycle at the next outer-sync boundary so that survivors stay alive while a replacement rejoins. That rejoin flow is explicitly labeled unsupported/fragile in the implementation notes, but it is the required crash path for the DiLoCo side of the experiment. The writeup will still include a background section that explains **Prime Intellect's ElasticDeviceMesh** (the infrastructure layer behind OpenDiLoCo and INTELLECT-1) in the vocabulary of the course: heartbeats for liveness detection, dynamic membership, "deathrattle" notifications, and staleness-tolerant consistency. Translating what production systems actually do into the distributed-systems primitives we studied is a first-class deliverable, distinct from the implementation.

## Key design decisions (locked in)

- **N = 4 workers, fixed.** Not varying worker count. Rationale: large enough for synchronization and failure effects to be visible, small enough to keep the implementation single-node and the matrix affordable. Tradeoff accepted: no scaling story across N.
- **H is the primary swept axis** for DiLoCo: H ∈ {10, 50, 100, 500}. DDP is a separate synchronous reference baseline, not a point on the H axis — it is a different algorithm (raw-gradient all-reduce + single AdamW) and plotting it at "H=1" would be misleading. On every sweep figure, DDP is shown as a horizontal reference line with its own seed-variance band. Everything else (failure regime, seed) is either single-value or small.
- **Methodology: train-to-target-loss, not fixed wall-clock.** Every run trains until TinyStories eval loss ≤ 2.0. The headline metric is wall-clock-seconds-to-target; tokens-to-target is the secondary metric.
- **Max-wall-clock is a hyperparameter indicator, not a cutoff.** If a run fails to converge within 3× baseline duration, that indicates hyperparameters or system parameters need adjustment — not a result to report. **All reported runs must converge.** Re-tune and re-run if necessary.
- **Build minimal DiLoCo from scratch over nanoGPT.** ~200-300 lines on top of Karpathy's nanoGPT. Do NOT use OpenDiLoCo / Hivemind / FSDP — fighting those dependencies is not the point, and the symmetry of having DDP and DiLoCo share the same training loop (differing only in sync logic) is essential for fair comparison.
- **No full-restart fallback for the DiLoCo crash regime.** The DiLoCo crash experiment is defined by survivor continuity plus replacement-at-next-outer-sync. If that path cannot be stabilized, stop and revise the spec rather than silently substituting a different crash semantics.
- **No HuggingFace Trainer or Accelerate.** Use `torch.distributed` directly. `torchrun` is the launcher for clean runs, straggler runs, and DDP crash runs; the DiLoCo crash path uses a direct shell launcher plus a sidecar controller because `torchrun` tears down survivors on any worker exit. Rationale: Trainer's abstractions don't fit DiLoCo's two-loop structure, and explicit checkpoint / rejoin control is required to measure recovery time cleanly.
- **Single multi-GPU VM, not multi-node.** Straggler experiments measure synchronization wait, not wire bandwidth, so a single box is fine. Multi-node would add real network variance that is out of scope.
- **Hardware: Lambda Labs 4×A10 for final runs; Kaggle 2×T4 for dev.** Colab is single-GPU and therefore useless for distributed experiments.

## Deliverables

1. **Code repository** containing:
   - Minimal DiLoCo implementation on nanoGPT (~200-300 lines delta)
   - DDP-elastic baseline training script
   - Failure injection module (crash and straggler injectors)
   - Metrics collection and W&B logging integration
   - Plotting scripts for the final figures
2. **Writeup** (~8-12 pages) containing:
   - Background on DDP, DiLoCo, and fault tolerance in distributed training
   - Implementation notes and architectural decisions
   - Experiment methodology
   - Results with all primary and secondary metric plots
   - Discussion of findings, limitations, and future work
3. **Presentation** (15-20 minutes) covering the research question, method, headline results, and lessons learned.

## Scope non-goals (explicitly out of scope)

- Tensor parallelism, pipeline parallelism, expert parallelism — data parallelism only.
- Multi-node (cross-machine) training — single multi-GPU VM only.
- Varying model size, dataset, or tokenizer — one fixed configuration.
- Varying the outer optimizer or inner optimizer beyond the values specified in `algorithm-notes.md`.
- Varying worker count N — fixed at 4.
- Building production tooling (deployment dashboards, CI, etc.).
- Scaling to LLM-sized models. The fixed experimental model is GPT-2-small scale (~124M parameters), not a larger LLM regime.

## Success criteria

The project is successful if all of the following hold:

1. All cells in the H sweep converge to target loss on at least the clean (Group A) runs.
2. All 24 experimental runs complete and are logged to W&B.
3. The final writeup contains quantitative wall-clock-to-target and tokens-to-target curves as a function of H, with error bars across seeds, under both clean and crash regimes.
4. The writeup includes a background section explaining Prime Intellect's ElasticDeviceMesh infrastructure in course vocabulary (heartbeats / membership / consensus / staleness).
5. The writeup ties the H-vs-convergence tradeoff back to spot instance economics: "at preemption rate X, H = Y is the break-even point."
6. The implementation and measurement methodology are reproducible from the spec files alone by a reader with PyTorch distributed experience.

## Files in this spec

- `project-overview.md` — this file
- `algorithm-notes.md` — DiLoCo mechanics, DDP-elastic baseline details
- `experiment-matrix.md` — run matrix, failure injection schedules, hyperparameters
- `measurement-plan.md` — metrics, logging, plots, statistical analysis
- `implementation-notes.md` — code layout, key functions, torchrun invocation, failure injection mechanics
