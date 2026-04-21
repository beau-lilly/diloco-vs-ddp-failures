"""TinyStories loading, tokenization, and per-rank sharded streaming.

Data is stored as a flat `uint16` memmap'd token file (the nanoGPT convention).
`prepare_tinystories.py` (shipped alongside) does the one-time tokenization.
The smoke-test mode falls back to a synthetic corpus so the code can run
offline on a laptop.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Batch:
    input_ids: torch.Tensor   # [B, T]
    labels: torch.Tensor      # [B, T]  (next-token targets)
    num_tokens: int           # input_ids.numel()


class ShardedBinDataloader:
    """Streams contiguous chunks of a pre-tokenized `.bin` file.

    Shard assignment: worker `rank` gets offsets `rank, rank+N, rank+2N, ...`
    into a virtual "chunk stream" of size `context_length + 1`. This is
    lightweight and deterministic, and the cursor (chunk index) is what we
    persist across crashes.
    """

    def __init__(
        self,
        bin_path: str,
        rank: int,
        world_size: int,
        per_worker_batch_size: int,
        context_length: int,
        seed: int,
        device: torch.device,
    ):
        self.path = bin_path
        self.rank = rank
        self.world_size = world_size
        self.B = per_worker_batch_size
        self.T = context_length
        self.seed = seed
        self.device = device

        self._data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        # Number of distinct (T+1)-sized chunks we could pull
        self._num_chunks = (len(self._data) - 1) // self.T
        # Deterministic per-rank shuffle of chunk indices, so rank r consumes
        # a reproducible stream. Cursor is the position in this stream.
        self._order = self._make_order()
        self._cursor = 0   # index into self._order

    def _make_order(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed * 1_000_003 + 17)
        order = rng.permutation(self._num_chunks)
        # Assign chunks round-robin across ranks: rank r gets order[r::N]
        return order[self.rank :: self.world_size]

    # --- cursor management (checkpointable) ---
    def position(self) -> int:
        return self._cursor

    def set_position(self, pos: int) -> None:
        self._cursor = pos % len(self._order)

    # --- iteration ---
    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        xs = np.empty((self.B, self.T), dtype=np.int64)
        ys = np.empty((self.B, self.T), dtype=np.int64)
        for b in range(self.B):
            chunk_idx = int(self._order[self._cursor])
            self._cursor = (self._cursor + 1) % len(self._order)
            start = chunk_idx * self.T
            xs[b] = self._data[start : start + self.T].astype(np.int64)
            ys[b] = self._data[start + 1 : start + self.T + 1].astype(np.int64)
        x = torch.from_numpy(xs).to(self.device, non_blocking=True)
        y = torch.from_numpy(ys).to(self.device, non_blocking=True)
        return Batch(input_ids=x, labels=y, num_tokens=self.B * self.T)


def _ensure_smoke_data(path: Path, vocab_size: int, n_tokens: int, seed: int) -> None:
    """Synthesize a tiny deterministic token stream for the CPU smoke test.

    Not intended for convergence — just enough to exercise the pipeline.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tokens = rng.integers(low=0, high=min(vocab_size, 1024), size=n_tokens, dtype=np.uint16)
    tokens.tofile(path)


def build_dataloader(
    cfg: dict,
    rank: int,
    world_size: int,
    seed: int,
    device: torch.device,
    split: str = "train",
    smoke: bool = False,
) -> ShardedBinDataloader:
    data_cfg = cfg["data"]
    bin_rel = data_cfg["tinystories_train"] if split == "train" else data_cfg["tinystories_val"]
    bin_path = Path(bin_rel)
    if smoke:
        # Synthesize a small corpus if one isn't present.
        n_tokens = 200_000 if split == "train" else 20_000
        smoke_path = bin_path.with_name(f"smoke_{split}.bin")
        _ensure_smoke_data(smoke_path, cfg["model"]["vocab_size"], n_tokens, seed=42 if split == "train" else 43)
        bin_path = smoke_path
    if not bin_path.exists():
        raise FileNotFoundError(
            f"TinyStories bin not found at {bin_path}. "
            f"Run prepare_tinystories.py, or pass --smoke for a synthetic corpus."
        )
    return ShardedBinDataloader(
        bin_path=str(bin_path),
        rank=rank,
        world_size=world_size,
        per_worker_batch_size=data_cfg["per_worker_batch_size"],
        context_length=data_cfg["context_length"],
        seed=seed,
        device=device,
    )


def build_eval_batches(cfg: dict, device: torch.device, smoke: bool = False):
    """Fixed eval batches, identical across ranks & runs, evaluated on rank 0 only."""
    data_cfg = cfg["data"]
    loader = build_dataloader(cfg, rank=0, world_size=1, seed=7, device=device, split="val", smoke=smoke)
    # Override batch size for eval so each eval pass is cheap
    loader.B = min(loader.B, 4) if smoke else loader.B
    eval_batches = []
    for _ in range(data_cfg["eval_batches"]):
        eval_batches.append(next(loader))
    return eval_batches
