"""Checkpoint save/load for DDP and outer-state persistence for DiLoCo.

DDP: full per-rank state (model, optimizer, RNGs, progress, dataloader cursor,
straggler schedule_position).

DiLoCo: outer_state.pt holds θ_outer plus the outer SGD+Nesterov state_dict —
both are required on rejoin (loading only θ_outer silently breaks
correctness per algorithm-notes.md). Plus a per-rank dataloader-cursor
record at the last committed outer boundary.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# DDP full checkpoint (per-rank file)
# ---------------------------------------------------------------------------
def save_ddp_checkpoint(
    path: Path,
    *,
    model_state: dict,
    optimizer_state: dict,
    tokens_raw: int,
    tokens_committed: int,
    step: int,
    wall_clock_offset: float,
    dataloader_position: int,
    straggler_state: dict,
) -> None:
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "tokens_raw": tokens_raw,
        "tokens_committed": tokens_committed,
        "step": step,
        "wall_clock_offset": wall_clock_offset,
        "dataloader_position": dataloader_position,
        "straggler_state": straggler_state,
    }
    _atomic_torch_save(state, path)


def load_ddp_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch_rng_state"])
    if state.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])


# ---------------------------------------------------------------------------
# DiLoCo outer-state snapshot (rank 0 writes; all ranks can read on rejoin)
# ---------------------------------------------------------------------------
def save_outer_state(
    path: Path,
    *,
    theta_outer_state_dict: dict,
    outer_optimizer_state_dict: dict,
    tokens_raw: int,
    tokens_committed: int,
    outer_step: int,
) -> None:
    state = {
        "theta_outer": theta_outer_state_dict,
        "outer_optimizer": outer_optimizer_state_dict,
        "tokens_raw": tokens_raw,
        "tokens_committed": tokens_committed,
        "outer_step": outer_step,
    }
    _atomic_torch_save(state, path)


def load_outer_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_rank_cursor(path: Path, *, dataloader_position: int, tokens_this_rank: int) -> None:
    state = {
        "dataloader_position": dataloader_position,
        "tokens_this_rank": tokens_this_rank,
    }
    _atomic_torch_save(state, path)


def load_rank_cursor(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
