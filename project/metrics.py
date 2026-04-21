"""Metrics bookkeeping shared by both trainers.

Holds everything that flows into the W&B schema and the per-crash event log:
  * wall-clock start and target-reached timestamp
  * tokens_raw (monotonic, advances per minibatch) and tokens_committed
    (advances at checkpoint / outer-step boundaries)
  * comm_bytes and comm_seconds accumulated by the framework's instrumentation
    hook (DDP) or wrapper (DiLoCo)
  * per-step compute / optimizer timings (bucket breakdown)
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field


@dataclass
class CrashEventRecord:
    event_tokens_raw: int
    event_tokens_committed: int
    wall_clock_seconds: float
    victim_rank: int
    lost_tokens: int


@dataclass
class StragglerEventRecord:
    start_elapsed_seconds: float
    duration_seconds: float


@dataclass
class Metrics:
    # --- wall-clock ---
    start_time: float | None = None
    target_reached_wall_clock: float | None = None
    target_reached_tokens: int | None = None

    # --- progress counters ---
    tokens_raw: int = 0
    tokens_committed: int = 0
    step: int = 0          # number of local minibatch steps this rank has run
    outer_step: int = 0    # DiLoCo only

    # --- comm instrumentation (filled by the hook/wrapper) ---
    comm_bytes: int = 0
    comm_seconds: float = 0.0

    # --- wall-clock breakdown buckets (seconds) ---
    compute_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    # "sync wait / other" is derived: wall_clock_elapsed - compute - optimizer - comm

    # --- event logs ---
    crashes: list[CrashEventRecord] = field(default_factory=list)
    stragglers: list[StragglerEventRecord] = field(default_factory=list)

    # ------------------------------------------------------------
    def mark_start(self) -> None:
        if self.start_time is None:
            self.start_time = time.perf_counter()

    def wall_clock_elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    def record_target_reached(self) -> None:
        if self.target_reached_wall_clock is None:
            self.target_reached_wall_clock = self.wall_clock_elapsed()
            self.target_reached_tokens = self.tokens_raw

    @contextlib.contextmanager
    def time(self, bucket: str):
        """Accumulate elapsed time into a named bucket."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            if bucket == "compute":
                self.compute_seconds += dt
            elif bucket == "optimizer":
                self.optimizer_seconds += dt
            elif bucket == "communication":
                # DiLoCo's explicit all_reduce wrapper path calls this.
                # DDP's comm_seconds is set by the comm hook directly.
                self.comm_seconds += dt
            else:
                raise ValueError(f"unknown metrics bucket: {bucket!r}")

    def sync_wait_other_seconds(self) -> float:
        return max(
            0.0,
            self.wall_clock_elapsed()
            - self.compute_seconds
            - self.optimizer_seconds
            - self.comm_seconds,
        )

    def summary_final(self) -> dict:
        return {
            "final/wall_clock_to_target": self.target_reached_wall_clock,
            "final/tokens_to_target": self.target_reached_tokens,
            "final/total_comm_bytes": self.comm_bytes,
            "final/total_comm_seconds": self.comm_seconds,
            "final/total_compute_seconds": self.compute_seconds,
            "final/total_optimizer_seconds": self.optimizer_seconds,
            "final/total_wall_clock": self.wall_clock_elapsed(),
            "final/sync_wait_other_seconds": self.sync_wait_other_seconds(),
            "final/num_crashes": len(self.crashes),
            "final/num_straggler_events": len(self.stragglers),
            "final/mean_lost_tokens_per_crash": (
                sum(c.lost_tokens for c in self.crashes) / len(self.crashes)
                if self.crashes
                else 0
            ),
        }
