"""Token transport + side-band control channel.

Per measurement-plan.md the ONLY authoritative transport for progress is a set
of per-rank token files plus a rank-0-written aggregate:

    runtime/tokens_rank_{r}.txt   — cumulative tokens processed by rank r
    runtime/progress.json          — { tokens_raw, tokens_committed, last_update }
    runtime/worker_pids.{r}.json   — per-rank PID file (for the crash sidecar)

All writes use write-tmp-then-rename so readers never see torn state.

Rank 0 spins up a background poller that reads the N token files at ~1 Hz and
(re)writes progress.json. `tokens_committed` is published separately by the
framework when it hits a commit boundary (5-min checkpoint for DDP, successful
outer step for DiLoCo).

The side-band FileStore (opened only for DiLoCo crash runs) lives in
`DiLoCoControlStore`. For clean runs and all DDP runs, the control store is
not needed — the sidecar uses progress.json + signals + torchrun restart.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch.distributed as dist


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------
def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    # Rename is atomic w.r.t. readers on POSIX; we deliberately skip fsync to
    # keep per-inner-step token publication cheap. A power failure could lose
    # the most recent write, which is fine for this experiment — tokens will
    # be re-published on restart from the checkpoint/outer-state.
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Worker PID advertisement
# ---------------------------------------------------------------------------
def publish_worker_pid(runtime_dir: Path, rank: int) -> None:
    path = runtime_dir / f"worker_pids.{rank}.json"
    _atomic_write_text(path, json.dumps({"rank": rank, "pid": os.getpid()}))


def read_all_worker_pids(runtime_dir: Path, world_size: int) -> dict[int, int]:
    pids = {}
    for r in range(world_size):
        p = runtime_dir / f"worker_pids.{r}.json"
        if p.exists():
            try:
                pids[r] = json.loads(p.read_text())["pid"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass
    return pids


# ---------------------------------------------------------------------------
# Per-rank token-count writer (ALL ranks)
# ---------------------------------------------------------------------------
class RankTokenFile:
    """Rank-local token counter writer.

    `publish(tokens)` is called once per minibatch. Writes are cheap and
    atomic (<32-byte rename on a single node). Rank 0's poller reads these.
    """

    def __init__(self, runtime_dir: Path, rank: int):
        self.path = runtime_dir / f"tokens_rank_{rank}.txt"
        self.runtime_dir = runtime_dir
        self.rank = rank
        runtime_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, "0")

    def publish(self, tokens_cumulative: int) -> None:
        _atomic_write_text(self.path, str(tokens_cumulative))


# ---------------------------------------------------------------------------
# Rank-0 aggregator — polls N rank files, writes progress.json
# ---------------------------------------------------------------------------
@dataclass
class ProgressSnapshot:
    tokens_raw: int
    tokens_committed: int
    per_rank_tokens: dict[int, int]
    last_update: float


class ProgressAggregator:
    """Runs a background thread on rank 0; polls per-rank token files at ~1 Hz."""

    def __init__(self, runtime_dir: Path, world_size: int, hz: float = 1.0):
        self.runtime_dir = runtime_dir
        self.world_size = world_size
        self.interval = 1.0 / hz
        self.progress_path = runtime_dir / "progress.json"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()  # serializes _poll_once calls
        self._tokens_committed = 0
        self._per_rank_last = {r: 0 for r in range(world_size)}
        self._last_snapshot: ProgressSnapshot | None = None

    def set_committed(self, tokens_committed: int) -> None:
        with self._lock:
            self._tokens_committed = tokens_committed
        # Publish immediately. The sidecar's lost_tokens bookkeeping depends
        # on observing every committed advance; without a synchronous flush,
        # the background thread's ~1 Hz poll can let the sidecar misattribute
        # survivor work to lost_tokens when a crash fires shortly after an
        # outer step.
        try:
            self._poll_once()
        except Exception:
            pass

    def _read_rank(self, rank: int) -> int:
        p = self.runtime_dir / f"tokens_rank_{rank}.txt"
        try:
            return int(p.read_text().strip() or "0")
        except (OSError, ValueError):
            return self._per_rank_last[rank]

    def _poll_once(self) -> None:
        # Serialize poll calls so the background thread and main-thread
        # snapshot() don't race on the shared progress.json.tmp path.
        with self._poll_lock:
            total = 0
            per_rank: dict[int, int] = {}
            for r in range(self.world_size):
                v = self._read_rank(r)
                # Token counters only go up within a run. If an individual rank
                # file happens to be missing or mid-rename, fall back to the
                # last observed value for that rank.
                v = max(v, self._per_rank_last[r])
                self._per_rank_last[r] = v
                per_rank[r] = v
                total += v
            with self._lock:
                committed = self._tokens_committed
            payload = {
                "tokens_raw": total,
                "tokens_committed": committed,
                "per_rank_tokens": per_rank,
                "last_update": time.time(),
            }
            _atomic_write_text(self.progress_path, json.dumps(payload))
            self._last_snapshot = ProgressSnapshot(
                tokens_raw=total,
                tokens_committed=committed,
                per_rank_tokens=per_rank,
                last_update=payload["last_update"],
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # pragma: no cover — don't kill training
                print(f"[progress-aggregator] poll failed: {exc!r}")
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="progress-agg", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Final poll so progress.json reflects the last state before we exit.
        try:
            self._poll_once()
        except Exception:
            pass

    def snapshot(self) -> ProgressSnapshot:
        # Prefer the cached snapshot written by the background poller (the
        # common case). Only fall back to an in-line poll if the aggregator
        # hasn't produced one yet (before the first tick after startup).
        if self._last_snapshot is not None:
            return self._last_snapshot
        self._poll_once()
        assert self._last_snapshot is not None
        return self._last_snapshot


def read_progress(progress_path: Path) -> dict | None:
    """Read progress.json (sidecar-side). Returns None if file missing / malformed."""
    try:
        return json.loads(progress_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_rank_tokens(runtime_dir: Path, rank: int) -> int:
    """Sidecar helper — direct per-rank read (for DiLoCo lost_tokens bookkeeping)."""
    p = runtime_dir / f"tokens_rank_{rank}.txt"
    try:
        return int(p.read_text().strip() or "0")
    except (OSError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# DiLoCo-only control FileStore wrapper
# ---------------------------------------------------------------------------
class DiLoCoControlStore:
    """Thin wrapper over torch.distributed.FileStore for rejoin-flag coordination.

    Opens a FileStore rooted at `path`. All flag names are string keys; values
    are short byte strings (typically b"1").

    Used by:
      * workers: read `rejoin_pending`, arrive at `barrier_before_reinit_<epoch>`
      * replacement: write `replacement_ready_<epoch>`, wait on the barrier
      * sidecar: write `rejoin_pending`, read `recovery_complete_<epoch>`
    """

    def __init__(self, path: Path, world_size: int, is_master: bool = False, timeout_seconds: float = 3600.0):
        from datetime import timedelta
        path.parent.mkdir(parents=True, exist_ok=True)
        # FileStore wants a filename; it is created if absent. The `numWorkers`
        # argument controls internal FileStore bookkeeping — we use
        # world_size + 1 (N workers + sidecar). Passing too low a number
        # just limits the automatic cleanup, not correctness.
        self.store = dist.FileStore(str(path), world_size + 1)
        self.store.set_timeout(timedelta(seconds=timeout_seconds))
        self.world_size = world_size

    # --- primitives ---
    def set(self, key: str, value: bytes | str = b"1") -> None:
        if isinstance(value, str):
            value = value.encode()
        self.store.set(key, value)

    def has(self, key: str) -> bool:
        """Non-blocking existence check (FileStore.get blocks until key exists)."""
        try:
            return bool(self.store.check([key]))
        except Exception:
            return False

    def get(self, key: str) -> bytes | None:
        """Non-blocking read. Returns None if the key is absent."""
        if not self.has(key):
            return None
        try:
            return self.store.get(key)
        except Exception:
            return None

    def delete(self, key: str) -> bool:
        try:
            return bool(self.store.delete_key(key))
        except Exception:
            return False

    def wait(self, keys: list[str], timeout_seconds: float = 300.0) -> None:
        from datetime import timedelta
        self.store.wait(keys, timedelta(seconds=timeout_seconds))

    # --- high-level protocol helpers ---
    def rejoin_pending(self) -> int:
        """Return crash epoch if a rejoin is pending, else 0."""
        v = self.get("rejoin_pending")
        if v is None:
            return 0
        try:
            return int(v.decode())
        except ValueError:
            return 0

    def mark_rejoin_pending(self, epoch: int) -> None:
        self.set("rejoin_pending", str(epoch).encode())

    def clear_rejoin_pending(self) -> None:
        self.delete("rejoin_pending")
