"""Generate the headline plots from hardcoded final-metric numbers.

Bypasses W&B entirely — useful when wandb API is slow or the project
has a lot of ghost/failed runs to paginate through. Values come
straight from the fill_missing_b.log + group_a.log tables we already
pulled. Edit the dicts below if numbers change.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --- Measured wall-clock-to-target (seconds) per cell, per seed ---
DATA_WALL = {
    "ddp": {
        "clean": [356.15, 355.92],
        "crash": [674.19, 675.48],
    },
    10: {
        "clean": [615.10, 641.85],
        "crash": [709.37, 705.13],
    },
    50: {
        "clean": [572.08, 570.03],
        "crash": [657.38, 659.03],
    },
    100: {
        "clean": [570.20, 570.23],
        "crash": [659.35, 660.53],
    },
    500: {
        "clean": [764.48, 765.00],
        "crash": [1202.02, 1201.54],
    },
}

# --- Measured mean lost_tokens per crash (avg across the 3 crashes in a run) ---
DATA_LOST = {
    "ddp": {"crash": [2490368, 2555904]},
    10: {"crash": [163840, 163840]},
    50: {"crash": [819200, 819200]},
    100: {"crash": [1638400, 1638400]},
    500: {"crash": [8192000, 8192000]},
}


def _mean_std(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def plot_wallclock_vs_H(out_path: Path):
    H_vals = [10, 50, 100, 500]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for regime, color in [("clean", "tab:blue"), ("crash", "tab:red")]:
        means, stds = [], []
        for H in H_vals:
            m, s = _mean_std(DATA_WALL[H][regime])
            means.append(m)
            stds.append(s)
        ax.errorbar(
            H_vals, means, yerr=stds, marker="o", capsize=4,
            linewidth=2, color=color,
            label=f"DiLoCo {regime}",
        )

        ddp_m, ddp_s = _mean_std(DATA_WALL["ddp"][regime])
        ax.axhline(ddp_m, linestyle="--", linewidth=1.5, color=color, alpha=0.8,
                   label=f"DDP {regime} ({ddp_m:.0f} s)")
        ax.fill_between(H_vals, [ddp_m - ddp_s] * 4, [ddp_m + ddp_s] * 4,
                        color=color, alpha=0.12)

    ax.set_xscale("log")
    ax.set_xticks(H_vals)
    ax.set_xticklabels(H_vals)
    ax.set_xlabel("H  (DiLoCo inner steps per outer step)", fontsize=12)
    ax.set_ylabel("Wall-clock seconds to target loss", fontsize=12)
    ax.set_title("Wall-clock-to-target vs H  (target loss ≤ 2.5)", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_lost_tokens_vs_H(out_path: Path):
    H_vals = [10, 50, 100, 500]
    labels = ["DDP"] + [f"H={H}" for H in H_vals]
    means = [_mean_std(DATA_LOST["ddp"]["crash"])[0]]
    stds = [_mean_std(DATA_LOST["ddp"]["crash"])[1]]
    for H in H_vals:
        m, s = _mean_std(DATA_LOST[H]["crash"])
        means.append(m)
        stds.append(s)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["tab:orange"] + ["tab:green"] * len(H_vals)
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=stds, capsize=4, color=colors, edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Mean lost tokens per crash (log scale)", fontsize=12)
    ax.set_title("Per-crash cost: DDP vs DiLoCo at each H", fontsize=13)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    for x, m in zip(xs, means):
        ax.annotate(f"{m/1e6:.2f}M", xy=(x, m), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots", help="output directory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_wallclock_vs_H(out / "plot1_wallclock_vs_H.png")
    plot_lost_tokens_vs_H(out / "plot6_lost_tokens_vs_H.png")


if __name__ == "__main__":
    main()
