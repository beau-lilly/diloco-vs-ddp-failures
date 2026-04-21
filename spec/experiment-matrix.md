# Experiment Matrix

## Model and dataset (fixed across all runs)

| Parameter | Value | Notes |
|---|---|---|
| Model | nanoGPT (GPT-2 small variant) | Karpathy reference implementation |
| Parameter count | ~124M | 12 layers, hidden dim 768, 12 heads (GPT-2 small scale) |
| Layers | 12 | Match GPT-2 small |
| Hidden dim | 768 | Match GPT-2 small |
| Heads | 12 | Match GPT-2 small |
| Context length | 512 | Small enough to fit on A10, large enough to be non-trivial |
| Tokenizer | GPT-2 BPE | Standard nanoGPT tokenizer |
| Dataset | TinyStories | Cleaner, smaller, faster to converge than OpenWebText/C4 |
| Per-worker batch size | One-time calibration on final Lambda SKU, then frozen | Set to the largest value that fits on one GPU at context length 512; record the chosen integer in `config/base.yaml` before Group A and do not sweep it |
| Global batch size | 4 × per-worker | N=4 workers |

**Rationale for TinyStories:** TinyStories is a small synthetic dataset of simple English stories. It converges quickly (within tens of millions of tokens) to usable loss levels on small GPT models, making it feasible to run dozens of training runs within the project budget. It's also a published benchmark with known loss targets.

## Target loss

**TinyStories eval loss ≤ 2.0** on a held-out validation set, evaluated approximately every 30 seconds of wall-clock (or the nearest outer-step cadence for DiLoCo), per `implementation-notes.md`.

Rationale for 2.0: published Karpathy benchmarks and community reproductions place small-model TinyStories loss at roughly 1.5-1.8 after meaningful training; 2.0 is a round, defensible target that represents clear convergence without being so tight that runs take prohibitively long. Both DDP-elastic and DiLoCo should reach it under clean conditions within reasonable wall-clock.

## Worker count

**N = 4, fixed.** See `project-overview.md` for rationale.

## Hardware

**Final runs:** Lambda Labs 4×A10 (target budget ~$3/hr). Single VM. NCCL over PCIe (the A10 is a PCIe card — it does not have NVLink; NCCL still works fine over PCIe, it just doesn't get NVLink bandwidth).
**Development:** Kaggle 2×T4 free tier. Used only for debugging and initial tuning; final numbers come from Lambda.

**Hardware verification step (do before booking):** before committing to Lambda 4×A10, verify on the Lambda console (or equivalent provider) that (a) the 4×A10 SKU is actually available, (b) the per-hour price matches the ~$3/hr budget assumption, and (c) whatever is being used as the "spot / preemptible" option is real and has the preemption characteristics the Plot 7 synthesis assumes. If the 4×A10 SKU is unavailable or substantially more expensive, swap to the cheapest available 4-GPU PCIe-class SKU (4×A6000, 4×L4, etc.) and update the hardware line above. The algorithmic question is insensitive to the specific GPU; only the wall-clock baseline and budget shift. Do NOT assume A10-specific behavior (e.g., memory size) without checking the exact SKU — per-worker batch size will need to be re-tuned for whatever card is actually used.

## Methodology: train-to-target-loss

Every run trains until the validation loss crosses the 2.0 threshold. The primary measurement is **wall-clock seconds elapsed from training start until the threshold crossing**. Secondary measurement is **tokens consumed from start until threshold crossing**.

**Max-wall-clock is an indicator, not a cutoff.** If any run fails to converge within 3× the clean baseline duration, that is a signal that something needs to be re-tuned (hyperparameters, failure rate, DDP checkpoint interval, etc.) — it is NOT a result to report. Tune until convergence, then re-run. **Every reported run must converge.**

## Run matrix

**Primary experimental axis: the DiLoCo H sweep, compared against a synchronous DDP reference.** H is the number of inner steps per outer step in DiLoCo — the "slack budget" each worker gets before it must synchronize. DiLoCo is swept across H ∈ {10, 50, 100, 500}. **DDP is a separate synchronous reference baseline, not a point on the H axis.** DDP is algorithmically distinct from DiLoCo: it all-reduces raw gradients rather than pseudo-gradients, runs a single optimizer rather than an inner/outer pair, and has no outer Nesterov momentum buffer. Running "DiLoCo at H=1" would be a degenerate configuration (inner AdamW moments never accumulate) that would perform worse than DDP for reasons unrelated to slack, so it is not included. DDP appears on every sweep plot as a horizontal reference line (one per regime) with its own seed-variance band.

The crash regime sharpens the tradeoff because DDP pays a full rollback-to-last-checkpoint on every crash, while DiLoCo's specified crash path loses only the victim worker's uncommitted local work and does not roll back survivors.

24 total runs across 3 groups, 2 seeds per cell.

| Group | Framework / H | Seeds | Failure condition | Count |
|---|---|---|---|---|
| A (clean) | DDP-elastic (sync reference) | {0, 1} | None | 2 |
| A (clean, H sweep) | DiLoCo H=10 | {0, 1} | None | 2 |
| A (clean, H sweep) | DiLoCo H=50 | {0, 1} | None | 2 |
| A (clean, H sweep) | DiLoCo H=100 | {0, 1} | None | 2 |
| A (clean, H sweep) | DiLoCo H=500 | {0, 1} | None | 2 |
| B (crash) | DDP-elastic (sync reference) | {0, 1} | 3 transient crashes, token-scheduled | 2 |
| B (crash, H sweep) | DiLoCo H=10 | {0, 1} | 3 transient crashes, token-scheduled | 2 |
| B (crash, H sweep) | DiLoCo H=50 | {0, 1} | 3 transient crashes, token-scheduled | 2 |
| B (crash, H sweep) | DiLoCo H=100 | {0, 1} | 3 transient crashes, token-scheduled | 2 |
| B (crash, H sweep) | DiLoCo H=500 | {0, 1} | 3 transient crashes, token-scheduled | 2 |
| C (straggler, secondary cut) | DDP-elastic | {0, 1} | Probabilistic straggler process | 2 |
| C (straggler, secondary cut) | DiLoCo H=50 | {0, 1} | Probabilistic straggler process | 2 |
| **Total** | | | | **24** |

### Group A — clean H sweep

Purpose: establish the clean convergence curve as a function of H for DiLoCo, with DDP as the synchronous-baseline reference. Two stories live in this group:

1. **Tokens-to-target vs H (DiLoCo) plus DDP reference.** DiLoCo convergence is expected to degrade monotonically as H grows, because longer inner loops accumulate more per-worker drift before the outer sync corrects it. DDP's tokens-to-target is a single reference value (it has no H). This establishes the "how much does slack cost you in data efficiency?" curve under zero-failure conditions.
2. **Wall-clock-to-target vs H (DiLoCo) plus DDP reference.** Wall-clock tells a different story than tokens because DiLoCo's communication cost drops as 1/H. DDP pays per-step all-reduce cost with no amortization. DiLoCo at small H pays communication + small convergence penalty; at large H pays convergence penalty but near-free communication. DDP and DiLoCo at the best H may be competitive under clean conditions.

No failures injected. Both frameworks run normally to target loss.

### Group B — crash H sweep

Purpose: measure how the H-vs-convergence tradeoff shifts when workers die during training. This is the headline experimental result.

**Injection schedule (token-count-based):**

After Group A runs complete, compute the average tokens-to-target for DDP-elastic at baseline. Call this `T_baseline`. Crashes are scheduled at:

- Crash 1 at `0.25 × T_baseline` tokens consumed (25% into the expected run)
- Crash 2 at `0.50 × T_baseline` tokens consumed (50%)
- Crash 3 at `0.75 × T_baseline` tokens consumed (75%)

The same absolute token thresholds are used for all Group B runs (DDP anchor and every H cell), so all cells see failures at the same points in their token progression. Because token progression is decoupled from wall-clock, this remains deterministic even when runs take different amounts of wall-clock time to complete.

**Injection mechanism:**
- A controller process (running on rank 0 or as a sidecar) monitors the global token counter.
- When the counter crosses a threshold, the controller picks one worker rank at random (seeded deterministically per run), sends SIGKILL to that worker's process, and marks that crash as "in flight." No later crash threshold may fire until the current recovery finishes.
- **DDP:** the dead worker eventually causes a full torchrun kill-and-respawn cascade. The fixed 30-second replacement delay is injected into the restart path.
- **DiLoCo:** the sidecar immediately marks `rejoin_pending`, survivors continue their local inner loops, and after the fixed 30-second delay the sidecar spawns a replacement worker which loads the latest committed `θ_outer`, **skips local inner steps for that rejoin event**, and joins the next outer-sync boundary via the side-band control store plus a destroy/reinit cycle on the NCCL process group. On that first post-rejoin outer step the replacement contributes a zero pseudo-gradient. This survivor-preserving rejoin path is the DiLoCo crash experiment; do not substitute a full-restart DiLoCo path and still treat it as the same cell.

Each run sees **exactly 3 crashes at deterministic token thresholds**.

### Group C — probabilistic stragglers (secondary cut)

Purpose: confirm the classic DDP-vs-DiLoCo straggler story at one representative H. This is a secondary cut, not part of the main H sweep, because the straggler story is qualitatively well-understood (DDP waits at the per-step barrier; DiLoCo absorbs straggler time into H) and doesn't need a full sweep to demonstrate. Running just DDP and DiLoCo H=50 under stragglers is enough to show the effect; more H values would buy narrative width without buying clarity on the headline question (which is about crashes, not stragglers).

**Injection mechanism:** at process startup, each worker precomputes a deterministic schedule of straggler episodes over a window comfortably longer than any expected run. The schedule is generated by walking one-minute boundaries, independently drawing Bernoulli(p) per minute per worker, and for each success drawing a duration from `Uniform(30, 90)` seconds. During training the worker consults the schedule against its elapsed wall-clock and applies a 3× slowdown when it is inside an episode's window.

- **Per-worker-per-minute slowdown probability:** `p = 0.15`
- **Slowdown duration:** `Uniform(30, 90)` seconds
- **Slowdown factor:** 3× (i.e., each training step takes 3× as long during the slow window)
- **Slowdown implementation:** `time.sleep(step_time * 2)` inserted at the end of each forward+backward pass while in the slow state, effectively tripling wall-clock per step. This is preferable to CPU-bound busy-waiting because it doesn't consume host CPU that could affect other workers on a shared VM.

**Expected events per run:** with p=0.15/minute/worker and N=4, expected ~0.6 events/minute total. For a ~35-minute clean run, that's roughly ~20 expected straggler episodes in the cluster over the run, well above the ~8-10 I quoted earlier. If this turns out to be too aggressive in practice (DDP can't converge within 3× baseline), tune `p` down to 0.08-0.10 and re-run.

The random seed controls both the sampling of which minutes trigger straggler episodes and the draw of each episode's duration. The straggler schedule is **precomputed at process startup** from `(seed, rank)`, NOT sampled online — see `implementation-notes.md` "Straggler injector" for the construction and the rationale. Precomputation is what makes "DDP seed-0 and DiLoCo seed-0 see the same straggler schedule" a real claim; an online sampler tied to wall-clock would diverge between frameworks because step timing differs.

**Why not combine crashes and stragglers in the same run:** attribution. If a DDP run sees both a crash and a straggler and its wall-clock blows up, you can't cleanly say how much came from the rollback cost vs the barrier wait. Separate runs give clean causal attribution for each failure class. If time permits as a stretch, a single "combined chaos" run can be added for narrative, but not as the main evidence.

## Seeds

Seeds control:
1. Model initialization
2. Data shuffling order
3. Failure injection RNG (which worker crashes, which minutes straggle, how long each straggler lasts)

Seeds used: `{0, 1}`. The same seed value means the same data order and the same failure schedule across all cells sharing that seed, which allows paired (within-seed) comparisons across H values and between DDP and DiLoCo. 2 seeds per cell is tight; flag any cell where the within-seed difference is large compared to the between-H effect and consider adding a third seed for that cell specifically.

## Hyperparameter notes for the experimental runs

The DiLoCo `H` (inner steps per outer step) is **the primary swept axis** of the matrix: H ∈ {10, 50, 100, 500}. DDP-elastic is NOT a point on the H axis — it is a separate synchronous reference baseline and is rendered on every sweep plot as a horizontal reference band (one per regime), with its own seed-variance shading. See `measurement-plan.md` "Plots for the final writeup" for the plotting convention.

The DDP checkpoint interval is held fixed at **5 minutes wall-clock** across all DDP runs in the matrix. A checkpoint-interval sensitivity sweep is a nice-to-have but not part of the main matrix.

All other hyperparameters (learning rates, batch size, model config, context length, dataset, tokenizer) are held fixed as specified above and in `algorithm-notes.md`.

## Bootstrap / ordering constraint

The crash injection schedule depends on `T_baseline`, which is computed from Group A DDP-elastic runs. Therefore the runs must be executed in this order:

1. **Group A first** (both frameworks, 10 runs: DDP × 2 seeds + DiLoCo × 4 H values × 2 seeds). Record `T_baseline` as mean tokens-to-target for DDP-elastic Group A runs.
2. **Groups B and C** (14 runs: B = 10, C = 4) can then run in any order, using the computed thresholds.

If dev-time probing suggests `T_baseline` will be (say) ~30M tokens, preliminary crash thresholds of {7.5M, 15M, 22.5M} tokens can be set for dry-run testing. Final thresholds are locked in after Group A completes.

## Budget estimate

At ~35 minutes per clean run on 4×A10 ($3/hr), clean runs are ~$1.75 each. Failure runs will take longer (maybe 50-60 minutes each for DDP under crashes, less for DiLoCo at large H), so call it ~$2.50-3.00 each on average. Very large H cells (H=500) may take longer to converge in tokens — budget an extra buffer for those.

- Group A (clean H sweep): 10 × $1.75 ≈ $18
- Group B (crash H sweep): 10 × $3.00 ≈ $30
- Group C (straggler secondary cut): 4 × $3.00 ≈ $12
- **Total runs budget: ~$60**
- Buffer for re-runs, debugging, dev iteration, H=500 slow convergence: ~$40
- **Total project budget estimate: ~$100**

This assumes all runs converge on the first try, which they will not. Re-tuning buffer is essential.
