"""In-process straggler injector.

The schedule is fully determined by (seed, rank) at construction time; the
step hook reads it against elapsed wall-clock, never samples online. This is
what makes "DDP seed-0 and DiLoCo seed-0 saw the same schedule" a real claim
(see implementation-notes.md "Straggler injector" for the rationale).

Crash injection lives in sidecar_crash_controller.py, out of process.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class StragglerConfig:
    p_per_minute: float = 0.15
    slow_duration_min: float = 30.0
    slow_duration_max: float = 90.0
    slowdown_factor: float = 3.0
    expected_run_minutes: int = 180


class StragglerInjector:
    """Deterministic straggler schedule per (seed, rank).

    - `schedule` is a list of (start_elapsed_seconds, duration_seconds).
    - `step_hook(step_compute_time)` is called at the end of every training
      step; if the current wall-clock is inside an active episode, sleeps
      enough to achieve `slowdown_factor x` total step duration.
    """

    def __init__(self, cfg: StragglerConfig, seed: int, rank: int):
        # random.Random doesn't accept tuples, so we mix (seed, rank) to a single int.
        self.rng = random.Random(seed * 10_000 + rank)
        self.cfg = cfg
        self.schedule: list[tuple[float, float]] = []
        for minute in range(cfg.expected_run_minutes):
            if self.rng.random() < cfg.p_per_minute:
                dur = self.rng.uniform(cfg.slow_duration_min, cfg.slow_duration_max)
                self.schedule.append((minute * 60.0, dur))
        self.schedule_position = 0
        self.start_time: float | None = None
        self.slow_until_elapsed = -1.0
        self._events_fired: list[tuple[float, float]] = []

    def step_hook(self, step_compute_time: float) -> None:
        if self.start_time is None:
            self.start_time = time.perf_counter()
        elapsed = time.perf_counter() - self.start_time
        while (
            self.schedule_position < len(self.schedule)
            and self.schedule[self.schedule_position][0] <= elapsed
        ):
            start_s, dur_s = self.schedule[self.schedule_position]
            self.slow_until_elapsed = max(self.slow_until_elapsed, start_s + dur_s)
            self._events_fired.append((start_s, dur_s))
            self.schedule_position += 1
        if elapsed < self.slow_until_elapsed:
            extra = step_compute_time * (self.cfg.slowdown_factor - 1.0)
            if extra > 0:
                time.sleep(extra)

    # --- checkpoint integration ---
    def state_for_checkpoint(self) -> dict:
        return {
            "schedule_position": self.schedule_position,
            "rng_state": self.rng.getstate(),
        }

    def load_state(self, state: dict) -> None:
        self.schedule_position = state["schedule_position"]
        # rng_state restored for completeness; with the precomputed schedule
        # the sampler never draws at runtime, but restoring keeps the object
        # exactly equivalent to a fresh instance at the same resume point.
        self.rng.setstate(state["rng_state"])

    def events_since_last_checkpoint(self) -> list[tuple[float, float]]:
        return list(self._events_fired)


def build_straggler_injector(
    failure_mode: str,
    cfg: dict,
    seed: int,
    rank: int,
) -> StragglerInjector | None:
    if failure_mode != "straggler":
        return None
    s_cfg = cfg.get("straggler", {})
    return StragglerInjector(
        StragglerConfig(
            p_per_minute=s_cfg.get("p_per_minute", 0.15),
            slow_duration_min=s_cfg.get("slow_duration_min_seconds", 30.0),
            slow_duration_max=s_cfg.get("slow_duration_max_seconds", 90.0),
            slowdown_factor=s_cfg.get("slowdown_factor", 3.0),
            expected_run_minutes=s_cfg.get("expected_run_minutes", 180),
        ),
        seed=seed,
        rank=rank,
    )
