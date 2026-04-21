"""W&B wrapper implementing the schema from measurement-plan.md.

Degrades to a silent stub when wandb isn't importable or `mode='disabled'` is
set (CPU smoke tests run with WANDB_MODE=disabled so they don't phone home).
Only rank 0 ever gets a live W&B client; other ranks hold a stub.
"""
from __future__ import annotations

import os
from typing import Any

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover — smoke tests don't need wandb
    wandb = None


class WandbLogger:
    def __init__(self, cfg: dict, run_name: str, tags: dict[str, Any], rank: int):
        self.rank = rank
        self.active = False
        self._run = None
        if rank != 0:
            return
        if wandb is None:
            return
        mode = cfg.get("wandb", {}).get("mode", "online")
        if os.environ.get("WANDB_MODE") == "disabled" or mode == "disabled":
            return
        try:
            self._run = wandb.init(
                project=cfg["wandb"]["project"],
                entity=cfg["wandb"].get("entity"),
                name=run_name,
                tags=[f"{k}={v}" for k, v in tags.items() if v is not None],
                config={**cfg, **tags},
                mode=mode,
                reinit=True,
            )
            self.active = True
        except Exception as exc:  # network issues, auth, etc. — don't kill the run
            print(f"[wandb] init failed: {exc!r}; continuing without W&B")

    # --- logging API (all calls are no-ops on non-rank-0) ---
    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.active:
            return
        self._run.log(payload, step=step)

    def log_failure(self, event_type: str, **kwargs) -> None:
        if not self.active:
            return
        self._run.log({f"failure/{k}": v for k, v in {"event_type": event_type, **kwargs}.items()})

    def finish(self) -> None:
        if self.active and self._run is not None:
            self._run.finish()
            self.active = False


def build_run_name(framework: str, H: int | None, failure: str, seed: int) -> str:
    if framework == "diloco":
        return f"diloco_H{H}_{failure}_s{seed}"
    return f"ddp_{failure}_s{seed}"
