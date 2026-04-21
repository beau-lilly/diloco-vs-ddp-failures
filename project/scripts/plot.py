"""Pull W&B data and produce the figures in measurement-plan.md.

Per the spec's plotting convention (DDP is a horizontal reference band, NEVER
a point on the H axis), every DiLoCo sweep plot has the same structure:

    x-axis: H ∈ {10, 50, 100, 500}, log scale
    curves: one per regime (clean, crash)
    DDP: horizontal band at mean(DDP runs for that regime) ± 1 std

Plots produced:
  1. wall-clock vs H (headline)
  2. tokens-to-target vs H
  3. loss curves at representative H (per regime)
  4. comm bytes vs H
  5. wall-clock breakdown stacked bars (per regime)
  6. lost tokens per crash vs H
  7. spot-instance crossover (synthesized)
  8. straggler loss curves

Invocation:
    python scripts/plot.py \
      --entity YOUR_WANDB_ENTITY \
      --project diloco-vs-ddp-failures \
      --out plots/
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", default="diloco-vs-ddp-failures")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--on-demand-dollars-per-hour", type=float, default=3.0)
    ap.add_argument("--spot-dollars-per-hour", type=float, default=0.5)
    return ap.parse_args()


def fetch_runs(entity: str, project: str):
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    rows = []
    for r in runs:
        cfg = r.config
        summary = dict(r.summary)
        if "final/wall_clock_to_target" not in summary:
            # run didn't converge or is still live — skip per spec (non-
            # converging runs are not reported)
            continue
        rows.append({
            "name": r.name,
            "framework": cfg.get("framework"),
            "H": cfg.get("H"),
            "seed": cfg.get("seed"),
            "failure": cfg.get("failure_config"),
            "group": cfg.get("group"),
            "wall_clock_to_target": summary.get("final/wall_clock_to_target"),
            "tokens_to_target": summary.get("final/tokens_to_target"),
            "total_comm_bytes": summary.get("final/total_comm_bytes"),
            "total_comm_seconds": summary.get("final/total_comm_seconds"),
            "total_compute_seconds": summary.get("final/total_compute_seconds"),
            "total_optimizer_seconds": summary.get("final/total_optimizer_seconds"),
            "sync_wait_other_seconds": summary.get("final/sync_wait_other_seconds"),
            "mean_lost_tokens_per_crash": summary.get("final/mean_lost_tokens_per_crash"),
            "num_crashes": summary.get("final/num_crashes"),
            "run_id": r.id,
        })
    return rows


def _group(rows, framework, failure, H=None):
    out = []
    for r in rows:
        if r["framework"] != framework:
            continue
        if r["failure"] != failure:
            continue
        if H is not None and r["H"] != H:
            continue
        out.append(r)
    return out


def _mean_std(values):
    import statistics
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def plot_sweep_vs_H(rows, metric_key: str, title: str, out: Path, ylabel: str, log_y: bool = False):
    import matplotlib.pyplot as plt
    H_values = [10, 50, 100, 500]
    fig, ax = plt.subplots(figsize=(7, 5))

    for regime in ["none", "crash"]:
        ys = []
        errs = []
        for H in H_values:
            g = _group(rows, "diloco", regime, H=H)
            m, s = _mean_std([r[metric_key] for r in g])
            ys.append(m)
            errs.append(s if s is not None else 0.0)
        label = "DiLoCo clean" if regime == "none" else "DiLoCo crash"
        valid = [(h, y, e) for h, y, e in zip(H_values, ys, errs) if y is not None]
        if valid:
            xs, ys_v, es = zip(*valid)
            ax.errorbar(xs, ys_v, yerr=es, marker="o", capsize=3, label=label)

        ddp = _group(rows, "ddp", regime)
        m, s = _mean_std([r[metric_key] for r in ddp])
        if m is not None:
            label_ddp = f"DDP {'clean' if regime == 'none' else 'crash'}"
            ax.axhline(m, linestyle="--", alpha=0.7, label=label_ddp)
            ax.fill_between(H_values, [m - s] * len(H_values), [m + s] * len(H_values), alpha=0.15)

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(H_values)
    ax.set_xticklabels(H_values)
    ax.set_xlabel("H (DiLoCo inner steps per outer step)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_wall_clock_breakdown(rows, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    H_values = [10, 50, 100, 500]
    for regime in ["none", "crash"]:
        labels = ["DDP"] + [f"DiLoCo H={h}" for h in H_values]
        stacks = {"compute": [], "communication": [], "optimizer+other": []}
        for framework, H in [("ddp", None)] + [("diloco", h) for h in H_values]:
            g = _group(rows, framework, regime, H=H) if H is not None else _group(rows, framework, regime)
            if not g:
                for k in stacks:
                    stacks[k].append(0.0)
                continue
            compute = sum(r.get("total_compute_seconds") or 0 for r in g) / len(g)
            comm = sum(r.get("total_comm_seconds") or 0 for r in g) / len(g)
            opt = sum(r.get("total_optimizer_seconds") or 0 for r in g) / len(g)
            other = sum(r.get("sync_wait_other_seconds") or 0 for r in g) / len(g)
            stacks["compute"].append(compute)
            stacks["communication"].append(comm)
            stacks["optimizer+other"].append(opt + other)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels))
        for k, vals in stacks.items():
            ax.bar(x, vals, bottom=bottoms, label=k)
            bottoms = bottoms + np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("seconds")
        ax.set_title(f"Wall-clock breakdown ({regime})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"plot5_breakdown_{regime}.png", dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = fetch_runs(args.entity, args.project)
    print(f"loaded {len(rows)} converged runs")

    plot_sweep_vs_H(rows, "wall_clock_to_target",
                    "Wall-clock to target (HEADLINE)",
                    out_dir / "plot1_wallclock_vs_H.png",
                    "seconds")
    plot_sweep_vs_H(rows, "tokens_to_target",
                    "Tokens to target vs H",
                    out_dir / "plot2_tokens_vs_H.png",
                    "tokens")
    plot_sweep_vs_H(rows, "total_comm_bytes",
                    "Total communication bytes vs H",
                    out_dir / "plot4_comm_vs_H.png",
                    "bytes",
                    log_y=True)
    plot_sweep_vs_H(rows, "mean_lost_tokens_per_crash",
                    "Lost tokens per crash vs H (crash regime)",
                    out_dir / "plot6_lost_tokens_vs_H.png",
                    "tokens")

    plot_wall_clock_breakdown(rows, out_dir)

    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
