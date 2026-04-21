"""DDP-elastic synchronous trainer.

Reference algorithm: standard data-parallel all-reduce + AdamW, with 5-minute
wall-clock checkpoints. Communication is instrumented with a DDP comm hook
(see implementation-notes.md "Communication instrumentation" — a Python
wrapper around dist.all_reduce does NOT catch DDP's real traffic, which
happens inside C++ autograd hooks).

The crash path for DDP is ENTIRELY external: the sidecar sends SIGKILL, NCCL
times out, torchrun restarts the whole set of N workers, and each fresh
worker calls `maybe_restore_checkpoint()` at startup. There is no in-process
"shrink membership" logic.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP

from checkpoint import load_ddp_checkpoint, restore_rng, save_ddp_checkpoint
from control_plane import ProgressAggregator, RankTokenFile
from metrics import Metrics


# ---------------------------------------------------------------------------
# Communication hook
# ---------------------------------------------------------------------------
class CommHookState:
    """Wrapper passed as `state` to DDP's comm hook.

    default_hooks.allreduce_hook expects a ProcessGroup (or None = WORLD) as
    its state — so our wrapper carries the PG plus our metrics container, and
    the hook delegates to the default hook using the PG attribute.
    """

    def __init__(self, process_group, metrics: Metrics):
        self.process_group = process_group
        self.metrics = metrics


def _instrumented_allreduce_hook(state: CommHookState, bucket):
    import torch.futures as _futures  # noqa: F401 — ensures futures API is importable
    t0 = time.perf_counter()
    tensor = bucket.buffer()
    state.metrics.comm_bytes += tensor.element_size() * tensor.numel()
    fut = default_hooks.allreduce_hook(state.process_group, bucket)

    def _record(fut_inner):
        state.metrics.comm_seconds += time.perf_counter() - t0
        return fut_inner.value()[0]

    return fut.then(_record)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def _cosine_lr(step: int, base_lr: float, warmup: int, total_steps_hint: int, min_lr_mult: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total_steps_hint:
        return base_lr * min_lr_mult
    progress = (step - warmup) / max(1, total_steps_hint - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_mult + (1.0 - min_lr_mult) * coeff)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class DDPTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        dataloader,
        eval_batches,
        logger,
        cfg: dict,
        rank: int,
        world_size: int,
        local_rank: int,
        device: torch.device,
        runtime_dir: Path,
        straggler_injector=None,
        checkpoint_path: Path | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device
        self.cfg = cfg
        self.dataloader = dataloader
        self.eval_batches = eval_batches
        self.logger = logger
        self.runtime_dir = runtime_dir
        self.straggler = straggler_injector

        # Wrap in DDP. device_ids must be [local_rank] for NCCL, None for gloo (CPU).
        ddp_kwargs = {}
        if dist.get_backend() == "nccl":
            ddp_kwargs["device_ids"] = [local_rank]
        self.model = DDP(model, **ddp_kwargs)

        opt_cfg = cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
        )

        self.metrics = Metrics()
        self.model.register_comm_hook(
            state=CommHookState(dist.group.WORLD, self.metrics),
            hook=_instrumented_allreduce_hook,
        )

        self.base_lr = opt_cfg["lr"]
        sched = cfg.get("schedule", {})
        self.warmup_steps = sched.get("warmup_steps", 200)
        self.min_lr_mult = sched.get("min_lr_mult", 0.1)
        # Rough total-steps hint for cosine; real training ends at target loss.
        self.total_steps_hint = sched.get("total_steps_hint", 20_000)

        self.checkpoint_interval = cfg["ddp"]["checkpoint_interval_seconds"]
        # Anchor checkpoints under runtime_dir so concurrent cells don't
        # collide on the same rank0.pt. Config value is honored as an
        # explicit override if set to an absolute path.
        cfg_dir = Path(cfg["ddp"].get("checkpoint_dir", "checkpoints/ddp"))
        if cfg_dir.is_absolute():
            self.checkpoint_dir = cfg_dir
        else:
            self.checkpoint_dir = runtime_dir / "checkpoints" / "ddp"
        self.checkpoint_path = checkpoint_path or (self.checkpoint_dir / f"rank{rank}.pt")
        self._last_checkpoint_time: float | None = None
        self._tokens_this_rank = 0

        # Token transport
        self.rank_tokens = RankTokenFile(runtime_dir, rank)
        self.progress_agg: ProgressAggregator | None = None
        if rank == 0:
            self.progress_agg = ProgressAggregator(runtime_dir, world_size)
            self.progress_agg.start()

        # DDP crash simulated replacement delay (honored by RESPAWNED workers only)
        delay_env = os.environ.get("DDP_REPLACEMENT_DELAY_SECONDS")
        if delay_env and self._is_respawn():
            delay = float(delay_env)
            if delay > 0 and rank == 0:
                print(f"[ddp] respawn detected; sleeping {delay:.1f}s to simulate replacement delay")
            time.sleep(delay)

    def _is_respawn(self) -> bool:
        # torchrun exposes TORCHELASTIC_RESTART_COUNT (>=1 means a restart).
        return int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0") or "0") >= 1

    # ------------------------------------------------------------------
    def maybe_restore_checkpoint(self) -> None:
        """Load the last checkpoint if one exists — called at start of every run."""
        if not self.checkpoint_path.exists():
            return
        state = load_ddp_checkpoint(self.checkpoint_path)
        self.model.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        restore_rng(state)
        self.metrics.tokens_raw = state["tokens_raw"]
        self.metrics.tokens_committed = state["tokens_committed"]
        self.metrics.step = state["step"]
        self._tokens_this_rank = state["tokens_raw"] // self.world_size
        self.dataloader.set_position(state["dataloader_position"])
        if self.straggler is not None and state.get("straggler_state") is not None:
            self.straggler.load_state(state["straggler_state"])
        # Re-publish committed token floor immediately so the sidecar sees the
        # post-rollback value ASAP.
        self.rank_tokens.publish(self._tokens_this_rank)
        if self.progress_agg is not None:
            self.progress_agg.set_committed(state["tokens_committed"])
        if self.rank == 0:
            print(
                f"[ddp] restored checkpoint @ tokens_raw={state['tokens_raw']:,}, "
                f"tokens_committed={state['tokens_committed']:,}, step={state['step']}"
            )

    def save_checkpoint(self) -> None:
        save_ddp_checkpoint(
            self.checkpoint_path,
            model_state=self.model.module.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            tokens_raw=self.metrics.tokens_raw,
            tokens_committed=self.metrics.tokens_committed,
            step=self.metrics.step,
            wall_clock_offset=self.metrics.wall_clock_elapsed(),
            dataloader_position=self.dataloader.position(),
            straggler_state=self.straggler.state_for_checkpoint() if self.straggler else {},
        )
        # Commit = last successfully written checkpoint.
        self.metrics.tokens_committed = self.metrics.tokens_raw
        if self.progress_agg is not None:
            self.progress_agg.set_committed(self.metrics.tokens_committed)
        self._last_checkpoint_time = time.perf_counter()

    # ------------------------------------------------------------------
    def _set_lr(self, step: int) -> float:
        lr = _cosine_lr(step, self.base_lr, self.warmup_steps, self.total_steps_hint, self.min_lr_mult)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def _should_checkpoint(self) -> bool:
        if self._last_checkpoint_time is None:
            return self.metrics.wall_clock_elapsed() >= self.checkpoint_interval
        return (time.perf_counter() - self._last_checkpoint_time) >= self.checkpoint_interval

    def _should_eval(self, last_eval_at: float) -> bool:
        return (self.metrics.wall_clock_elapsed() - last_eval_at) >= self.cfg["train"]["eval_every_seconds"]

    def _evaluate(self) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.eval_batches:
                x = batch.input_ids.to(self.device)
                y = batch.labels.to(self.device)
                _, loss = self.model(x, y)
                losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(1, len(losses))

    # ------------------------------------------------------------------
    def train_until_target_loss(self) -> None:
        self.maybe_restore_checkpoint()
        self.metrics.mark_start()
        if self._last_checkpoint_time is None:
            self._last_checkpoint_time = time.perf_counter()
        target = self.cfg["train"]["target_loss"]
        max_wc = self.cfg["train"]["max_wall_clock_seconds"]
        last_eval_at = -1e9

        self.model.train()
        while True:
            self._set_lr(self.metrics.step)
            step_t0 = time.perf_counter()
            with self.metrics.time("compute"):
                batch = next(self.dataloader)
                _, loss = self.model(batch.input_ids, batch.labels)
                loss.backward()
            with self.metrics.time("optimizer"):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            step_compute = time.perf_counter() - step_t0

            # DDP is lockstep — every rank has processed the same number of tokens.
            # Cluster-wide raw token count = per-rank local count × world_size.
            self.metrics.step += 1
            self._tokens_this_rank = self.metrics.step * batch.num_tokens
            self.metrics.tokens_raw = self._tokens_this_rank * self.world_size
            # Each rank publishes its own cumulative contribution; rank 0's
            # aggregator sums them (and tolerates minor ordering skew).
            self.rank_tokens.publish(self._tokens_this_rank)

            if self.straggler is not None:
                self.straggler.step_hook(step_compute)

            if self.rank == 0 and self.metrics.step % 50 == 0:
                self.logger.log({
                    "train/loss": loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/tokens_raw": self.metrics.tokens_raw,
                    "train/tokens_committed": self.metrics.tokens_committed,
                    "train/wall_clock_seconds": self.metrics.wall_clock_elapsed(),
                    "train/comm_bytes_cumulative": self.metrics.comm_bytes,
                    "train/step": self.metrics.step,
                })

            # Rank 0 authoritatively decides whether to eval this step; one
            # int broadcast per step keeps all ranks agreeing on whether the
            # eval collective is live (per-rank wall-clock drift is tiny but
            # non-zero, and a disagreement here would deadlock).
            flags = torch.zeros(2, device=self.device)
            if self.rank == 0 and self._should_eval(last_eval_at):
                flags[0] = 1.0
                eval_loss = self._evaluate()
                self.logger.log({
                    "eval/loss": eval_loss,
                    "eval/wall_clock_at_eval": self.metrics.wall_clock_elapsed(),
                    "eval/tokens_at_eval": self.metrics.tokens_raw,
                })
                if eval_loss <= target:
                    self.metrics.record_target_reached()
                    flags[1] = 1.0
                last_eval_at = self.metrics.wall_clock_elapsed()
            dist.broadcast(flags, src=0)
            if flags[1].item() > 0:
                # Everyone commits progress so the final checkpoint reflects target-reach.
                if self.rank == 0:
                    self.save_checkpoint()
                dist.barrier()
                return

            if self._should_checkpoint() and self.rank == 0:
                self.save_checkpoint()
            # Non-zero ranks also re-fsync their token file so tokens_raw keeps
            # advancing on the aggregator — no explicit action needed beyond
            # publish() above.

            if self.metrics.wall_clock_elapsed() > max_wc:
                if self.rank == 0:
                    print(
                        f"[ddp] exceeded max_wall_clock={max_wc}s; stopping for retune "
                        f"per measurement plan"
                    )
                dist.barrier()
                return

    def shutdown(self) -> None:
        if self.progress_agg is not None:
            self.progress_agg.stop()
