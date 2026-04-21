# CPS 390-03 Final Project Spec — DiLoCo vs DDP Under Failure Injection

This directory contains the complete specification for Beau's CPS 390-03 final project. It is written to be handed directly to Claude Code (or a human implementer with PyTorch distributed experience) as a self-contained work order.

## Read order

1. **`project-overview.md`** — start here. Research question, hypothesis, locked-in design decisions, deliverables, scope.
2. **`algorithm-notes.md`** — exact DiLoCo algorithm (pseudo-gradients, inner/outer loops, Nesterov momentum), DDP-elastic baseline, shared infrastructure requirements.
3. **`experiment-matrix.md`** — the 24-run matrix (12 cells × 2 seeds), failure injection schedules, model/dataset/hardware, target loss methodology.
4. **`measurement-plan.md`** — metrics, W&B logging schema, plots for the writeup, statistical analysis approach.
5. **`implementation-notes.md`** — code layout, key functions, torchrun invocation, failure injection mechanics, pitfalls.

## Single-paragraph summary

Build a minimal DiLoCo implementation and a DDP-elastic baseline on top of nanoGPT, both sharing the same training loop skeleton. Train nanoGPT on TinyStories to target eval loss 2.0 on a Lambda Labs 4×A10 VM, with N=4 workers fixed. Sweep DiLoCo's H (inner steps per outer sync) across {10, 50, 100, 500}, with DDP-elastic as a separate synchronous reference baseline (not a point on the H axis — DDP is algorithmically distinct). Run the DiLoCo sweep plus DDP reference under both clean and crash-injected conditions (20 runs, 2 seeds each), plus a smaller straggler secondary cut (4 runs), for 24 runs total. Measure wall-clock-to-target and tokens-to-target as primary metrics, plus communication bytes, lost tokens per crash, and wall-clock breakdown. Log everything to W&B. Produce a headline figure showing DiLoCo's wall-clock-to-target vs H under clean and crash regimes, with DDP as horizontal reference bands — the range of H where the crash-DiLoCo curve drops below the crash-DDP band answers "for what H is spot-priced DiLoCo strictly cheaper than on-demand DDP?" Every reported run must converge to target loss; the 3× baseline max-wall-clock is a tuning indicator, not a cutoff.

## The two most important design decisions

**1. H is the primary swept axis for DiLoCo, with DDP as a separate reference baseline.** DiLoCo sweeps over H ∈ {10, 50, 100, 500}. DDP is *not* on that axis (it's a different algorithm — raw-gradient all-reduce + single AdamW — and "DiLoCo at H=1" is a broken degenerate config). DDP appears on every sweep plot as a horizontal reference band. The comparison question becomes: "for which H values does DiLoCo's curve drop below DDP's band, under each regime?"

**2. Train-to-target-loss methodology with token-count-based failure scheduling.** Run duration is an outcome, not an input. Failure schedules cannot be expressed as percentages of wall-clock. Every reported run must actually converge, and non-converging runs indicate a tuning problem to fix before reporting.

## The single most important implementation pitfall

**Do not wrap the DiLoCo model in DistributedDataParallel.** DDP's autograd hooks will all-reduce gradients on every backward pass, completely defeating DiLoCo's communication-avoidance and making the experiment meaningless. The DiLoCo model is a bare `nn.Module`; only the pseudo-gradient all-reduce at outer-step boundaries is intentional. See `implementation-notes.md` for the other pitfalls.

## Not included in this spec (deliberate omissions)

- Week-by-week schedule or timeline. The spec is "what" and "how," not "when."
- Writeup draft text. The writeup is a downstream deliverable, written after runs complete.
- Slide deck for the presentation. Same.
- Comparison to other fault-tolerance papers (Bamboo, Oobleck). Useful for writeup background, not for implementation.
- Checkpoint-interval sweep. A stretch sensitivity analysis, not part of the main matrix.
- A working reimplementation of Prime Intellect's ElasticDeviceMesh. The writeup *explains* that infrastructure in course vocabulary; the experiment uses standard `torchrun` restart behavior for DDP and a much smaller single-node DiLoCo rejoin attempt rather than production-grade elastic membership machinery.
