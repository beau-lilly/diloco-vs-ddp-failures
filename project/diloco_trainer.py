"""DiLoCo trainer — two-loop training with low-frequency pseudo-gradient all-reduce.

Key properties enforced by this module (see algorithm-notes.md + pitfalls
#1-12 in implementation-notes.md):

  * self.model is a bare nn.Module — NOT wrapped in DistributedDataParallel.
    Wrapping it would cause DDP's autograd hooks to all-reduce every step and
    defeat communication avoidance.
  * A fresh AdamW is constructed inside each outer step (its running moments
    are local and ephemeral). The outer SGD+Nesterov optimizer persists for
    the entire run; its momentum buffer is half the reason DiLoCo works.
  * Pseudo-gradient sign is θ_outer − θ_local (never reversed).
  * Before outer_optimizer.step(), the live parameters are restored to
    θ_outer; otherwise SGD+Nesterov would update θ_local instead.
  * Only the pseudo-gradient all-reduce crosses worker boundaries. The
    comm wrapper around dist.all_reduce records bytes + time symmetrically
    with DDP's comm hook so Plot 4/5 are apples-to-apples.
  * Crash path: survivors wait for rejoin_pending on the control FileStore
    before entering any collective. The N workers rendezvous via a barrier
    key, tear down the NCCL PG, reinit, and proceed. The replacement loads
    θ_outer AND the outer optimizer's momentum buffer from outer_state.pt
    (loading only θ_outer silently breaks correctness), skips its inner
    loop for that first post-rejoin outer step, and contributes Δ = 0.
"""
from __future__ import annotations

import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from checkpoint import load_outer_state, load_rank_cursor, save_outer_state, save_rank_cursor
from control_plane import DiLoCoControlStore, ProgressAggregator, RankTokenFile
from metrics import Metrics


def _cosine_inner_lr(step: int, base_lr: float, warmup: int, total_steps_hint: int, min_lr_mult: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total_steps_hint:
        return base_lr * min_lr_mult
    progress = (step - warmup) / max(1, total_steps_hint - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_mult + (1.0 - min_lr_mult) * coeff)


class DiLoCoTrainer:
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
        control_store: DiLoCoControlStore | None = None,
        is_rejoining: bool = False,
        rejoin_crash_epoch: int = 0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device
        self.cfg = cfg
        self.model = model  # bare nn.Module — do NOT wrap in DDP
        self.dataloader = dataloader
        self.eval_batches = eval_batches
        self.logger = logger
        self.runtime_dir = runtime_dir
        self.straggler = straggler_injector
        self.control_store = control_store

        self.H = int(cfg["diloco"]["H"])
        d_cfg = cfg["diloco"]
        self.outer_state_path = Path(d_cfg["outer_state_path"])
        self.outer_state_dir = Path(d_cfg["outer_state_dir"])
        self.outer_state_write_every = int(d_cfg.get("outer_state_write_every", 1))

        # Persistent outer optimizer — its Nesterov buffer must survive the whole run.
        o_cfg = cfg["outer_optimizer"]
        self.outer_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=o_cfg["lr"],
            momentum=o_cfg["momentum"],
            nesterov=True,
        )

        i_cfg = cfg["inner_optimizer"]
        self._inner_lr = i_cfg["lr"]
        self._inner_wd = i_cfg["weight_decay"]
        self._inner_beta1 = i_cfg["beta1"]
        self._inner_beta2 = i_cfg["beta2"]
        self._inner_warmup = i_cfg.get("warmup_steps", 200)
        self._inner_min_lr_mult = i_cfg.get("min_lr_mult", 0.1)
        self._inner_total_hint = i_cfg.get("total_steps_hint", 20_000)

        self.metrics = Metrics()

        # Token transport (per-rank file + rank-0 aggregator)
        self.rank_tokens = RankTokenFile(runtime_dir, rank)
        self.progress_agg: ProgressAggregator | None = None
        if rank == 0:
            self.progress_agg = ProgressAggregator(runtime_dir, world_size)
            self.progress_agg.start()
        self._tokens_this_rank = 0

        # Outer-step commit cursor: per-rank dataloader position at the last
        # successful outer step. Used by the replacement on rejoin.
        self.rank_cursor_path = runtime_dir / f"diloco_cursor.{rank}.pt"

        # Rejoin bookkeeping
        self._rejoining_this_outer_step = is_rejoining
        self._pending_rejoin_epoch = rejoin_crash_epoch
        self._handled_crash_epochs: set[int] = set()

        # If we're a replacement, load outer_state and this rank's committed cursor
        if is_rejoining:
            self._restore_from_outer_state()
            self._restore_rank_cursor_if_present()

    # ------------------------------------------------------------------
    def _restore_from_outer_state(self) -> None:
        if not self.outer_state_path.exists():
            raise FileNotFoundError(
                f"replacement at rank {self.rank} cannot rejoin: {self.outer_state_path} does not exist. "
                f"This should not happen in Group B (at least one outer step must have committed)."
            )
        st = load_outer_state(self.outer_state_path)
        with torch.no_grad():
            self.model.load_state_dict(st["theta_outer"])
        self.outer_optimizer.load_state_dict(st["outer_optimizer"])
        # Note: tokens_committed and outer_step are global values from rank-0's
        # write; we adopt them so our logs match the cluster-wide view.
        self.metrics.tokens_committed = st.get("tokens_committed", 0)
        self.metrics.outer_step = st.get("outer_step", 0)
        if self.rank == self.local_rank == 0 or True:
            print(f"[diloco rank={self.rank}] loaded outer_state @ outer_step={self.metrics.outer_step}, "
                  f"tokens_committed={self.metrics.tokens_committed:,}")

    def _restore_rank_cursor_if_present(self) -> None:
        if self.rank_cursor_path.exists():
            st = load_rank_cursor(self.rank_cursor_path)
            self.dataloader.set_position(int(st["dataloader_position"]))
            self._tokens_this_rank = int(st["tokens_this_rank"])
            self.rank_tokens.publish(self._tokens_this_rank)

    # ------------------------------------------------------------------
    # Instrumented all-reduce wrapper (symmetric with DDP's comm hook)
    # ------------------------------------------------------------------
    def _instrumented_allreduce(self, tensor: torch.Tensor, op) -> None:
        t0 = time.perf_counter()
        self.metrics.comm_bytes += tensor.element_size() * tensor.numel()
        dist.all_reduce(tensor, op=op)
        self.metrics.comm_seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Rejoin handshake — called at outer-sync boundary when rejoin_pending
    # is set on the control store. Every process in the new world calls this
    # exactly once per crash event.
    # ------------------------------------------------------------------
    def _do_rejoin_handshake(self, crash_epoch: int, is_replacement: bool) -> None:
        if self.control_store is None:
            raise RuntimeError("rejoin handshake invoked with no control store")
        store = self.control_store
        ready_key = f"replacement_ready_{crash_epoch}"
        my_barrier_key = f"barrier_before_reinit_{crash_epoch}_{self.rank}"
        complete_key = f"recovery_complete_{crash_epoch}"

        if is_replacement:
            # Advertise that we're alive and holding θ_outer.
            store.set(ready_key, b"1")
        else:
            # Survivors wait for the replacement to show up.
            try:
                store.wait([ready_key], timeout_seconds=600.0)
            except Exception as exc:
                raise RuntimeError(f"timed out waiting for {ready_key}: {exc!r}") from exc

        # Arrive at the barrier.
        store.set(my_barrier_key, b"1")
        all_barrier_keys = [f"barrier_before_reinit_{crash_epoch}_{r}" for r in range(self.world_size)]
        try:
            store.wait(all_barrier_keys, timeout_seconds=600.0)
        except Exception as exc:
            raise RuntimeError(f"timed out waiting for all barrier keys {all_barrier_keys}: {exc!r}") from exc

        # Tear down the old PG (survivors) or skip (replacement never init'd one).
        if not is_replacement:
            try:
                dist.destroy_process_group()
            except Exception as exc:
                # destroy_process_group is not officially supported mid-run.
                # Log and continue; init_process_group below is the load-bearing
                # action. (Pitfall #11.)
                print(f"[diloco rank={self.rank}] destroy_process_group raised {exc!r}; continuing to reinit")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # All N processes call init_process_group together.
        # We use a FRESH file:// rendezvous for each reinit rather than env://.
        # Reusing env:// means reusing MASTER_PORT; on macOS/Linux the old
        # TCPStore's port can stay in TIME_WAIT for tens of seconds after
        # destroy_process_group tears it down, which makes the reinit flaky.
        # A per-epoch file-based rendezvous sidesteps that cleanly.
        timeout = timedelta(seconds=self.cfg.get("crash", {}).get("nccl_timeout_seconds", 120))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        rendezvous_file = self.runtime_dir / f"rendezvous_{crash_epoch}"
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{rendezvous_file.resolve()}",
            world_size=self.world_size,
            rank=self.rank,
            timeout=timeout,
        )

        if self.rank == 0:
            # Clear rejoin keys so the sidecar's next_idx can advance.
            store.set(complete_key, b"1")
            store.clear_rejoin_pending()

        self._handled_crash_epochs.add(crash_epoch)

    # ------------------------------------------------------------------
    def _maybe_handle_pending_rejoin(self) -> bool:
        """If rejoin_pending is set, run the handshake. Returns True if handled."""
        if self.control_store is None:
            return False
        epoch = self.control_store.rejoin_pending()
        if not epoch or epoch in self._handled_crash_epochs:
            return False
        is_replacement = bool(self._rejoining_this_outer_step) and epoch == self._pending_rejoin_epoch
        self._do_rejoin_handshake(epoch, is_replacement=is_replacement)
        return True

    # ------------------------------------------------------------------
    # Inner loop — runs H local steps mutating live params θ_outer → θ_local
    # ------------------------------------------------------------------
    def _run_inner_loop(self, theta_outer: dict[str, torch.Tensor]) -> float:
        """Run H inner AdamW steps. Returns last loss (for logging)."""
        inner_opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._inner_lr,
            weight_decay=self._inner_wd,
            betas=(self._inner_beta1, self._inner_beta2),
        )
        last_loss = float("nan")
        clip = self.cfg["train"]["grad_clip"]
        for inner_step in range(self.H):
            # Apply inner LR schedule based on global inner-step count
            lr = _cosine_inner_lr(
                self.metrics.step,
                self._inner_lr,
                self._inner_warmup,
                self._inner_total_hint,
                self._inner_min_lr_mult,
            )
            for g in inner_opt.param_groups:
                g["lr"] = lr

            step_t0 = time.perf_counter()
            with self.metrics.time("compute"):
                batch = next(self.dataloader)
                _, loss = self.model(batch.input_ids, batch.labels)
                loss.backward()
            with self.metrics.time("optimizer"):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                inner_opt.step()
                inner_opt.zero_grad(set_to_none=True)
            step_compute = time.perf_counter() - step_t0

            self.metrics.step += 1
            self._tokens_this_rank += batch.num_tokens
            self.rank_tokens.publish(self._tokens_this_rank)

            if self.straggler is not None:
                self.straggler.step_hook(step_compute)

            last_loss = loss.item()

            if self.rank == 0 and self.metrics.step % 50 == 0:
                self.logger.log({
                    "train/loss": last_loss,
                    "train/lr": lr,
                    "train/tokens_raw": self._get_tokens_raw(),
                    "train/tokens_committed": self.metrics.tokens_committed,
                    "train/wall_clock_seconds": self.metrics.wall_clock_elapsed(),
                    "train/comm_bytes_cumulative": self.metrics.comm_bytes,
                    "train/inner_step": self.metrics.step,
                    "train/outer_step": self.metrics.outer_step,
                })
        return last_loss

    def _get_tokens_raw(self) -> int:
        if self.progress_agg is not None:
            snap = self.progress_agg.snapshot()
            return snap.tokens_raw
        return self._tokens_this_rank * self.world_size  # fallback (non-rank-0 approximation)

    # ------------------------------------------------------------------
    # One outer step — inner loop + pseudo-gradient all-reduce + SGD+Nesterov
    # ------------------------------------------------------------------
    def _do_outer_step(self) -> float:
        # Snapshot θ_outer — the state the outer optimizer owns.
        theta_outer = {name: p.detach().clone() for name, p in self.model.named_parameters()}

        if self._rejoining_this_outer_step:
            # Replacement's first post-rejoin outer step: skip inner loop.
            # Live params remain θ_outer, so Δ_local = 0 below.
            last_loss = float("nan")
        else:
            last_loss = self._run_inner_loop(theta_outer)

        # Gate any collective on rejoin_pending — this is the load-bearing
        # race guard (spec step 6 / pitfall #10). Survivors check the control
        # store BEFORE entering all_reduce.
        if self._maybe_handle_pending_rejoin():
            # The rejoin handshake reinit'd the process group. Proceed to the
            # outer all-reduce on the fresh PG; the replacement (now attached)
            # contributes Δ=0, survivors contribute their real delta.
            pass

        # Compute Δ_local = θ_outer − θ_local (sign is load-bearing; pitfall #4).
        deltas: dict[str, torch.Tensor] = {}
        with self.metrics.time("communication"):
            for name, p in self.model.named_parameters():
                delta = theta_outer[name] - p.detach()   # Δ_local
                self._instrumented_allreduce(delta, op=dist.ReduceOp.AVG)
                deltas[name] = delta

        # CRITICAL (pitfall #9): restore θ_outer into the live parameters before
        # calling outer_optimizer.step(), so the SGD+Nesterov step is applied
        # from θ_outer, not θ_local.
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.data.copy_(theta_outer[name])
                p.grad = deltas[name]

        with self.metrics.time("optimizer"):
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad(set_to_none=True)

        self.metrics.outer_step += 1
        # Commit: the outer step completed successfully, so the cluster-wide
        # progress is now committed.
        new_tokens_raw = self._get_tokens_raw()
        self.metrics.tokens_committed = new_tokens_raw
        self.metrics.tokens_raw = new_tokens_raw
        if self.progress_agg is not None:
            self.progress_agg.set_committed(self.metrics.tokens_committed)

        # Persist: per-rank cursor always; outer_state only on rank 0 (and only
        # every K outer steps if configured for I/O throttling).
        save_rank_cursor(
            self.rank_cursor_path,
            dataloader_position=self.dataloader.position(),
            tokens_this_rank=self._tokens_this_rank,
        )
        if self.rank == 0 and self.metrics.outer_step % self.outer_state_write_every == 0:
            save_outer_state(
                self.outer_state_path,
                theta_outer_state_dict=self.model.state_dict(),
                outer_optimizer_state_dict=self.outer_optimizer.state_dict(),
                tokens_raw=new_tokens_raw,
                tokens_committed=new_tokens_raw,
                outer_step=self.metrics.outer_step,
            )

        # After the first post-rejoin outer step, the replacement behaves like
        # any other worker.
        self._rejoining_this_outer_step = False
        return last_loss

    # ------------------------------------------------------------------
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
        self.metrics.mark_start()
        target = self.cfg["train"]["target_loss"]
        max_wc = self.cfg["train"]["max_wall_clock_seconds"]
        last_eval_at = -1e9

        self.model.train()

        # Pre-loop handshake for the replacement — it has not yet called
        # init_process_group. Do the rejoin dance first, then enter the
        # outer-sync loop normally.
        if self._rejoining_this_outer_step and self._pending_rejoin_epoch:
            # The replacement has not called init_process_group. Run the
            # handshake (which ends with init_process_group). From then on
            # the replacement is a full participant.
            self._do_rejoin_handshake(self._pending_rejoin_epoch, is_replacement=True)
            # Note: we still set _rejoining_this_outer_step so the NEXT
            # _do_outer_step skips the inner loop and contributes Δ=0.

        while True:
            self._do_outer_step()

            if self.metrics.wall_clock_elapsed() > max_wc:
                if self.rank == 0:
                    print(f"[diloco] exceeded max_wall_clock={max_wc}s; stopping for retune")
                break

            # Rank 0 is authoritative for the eval cadence decision so that
            # per-rank wall-clock drift (which is tiny but non-zero) can't
            # cause a deadlock where one rank enters the eval collective and
            # another doesn't. One int broadcast per outer step is cheap.
            flags = torch.zeros(2, device=self.device)
            if self.rank == 0:
                should_eval = self.metrics.wall_clock_elapsed() - last_eval_at >= self.cfg["train"]["eval_every_seconds"]
                if should_eval:
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
                break

        dist.barrier()

    def shutdown(self) -> None:
        if self.progress_agg is not None:
            self.progress_agg.stop()
