"""Single entrypoint — dispatches to DDP or DiLoCo trainer.

Invocation for Case 1 (clean, straggler, DDP crash) is via `torchrun ...`.
Invocation for Case 2 (DiLoCo crash) is via `scripts/run_diloco_crash.sh`
which exports RANK / WORLD_SIZE / MASTER_* and spawns N processes directly.
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from control_plane import DiLoCoControlStore, publish_worker_pid
from data import build_dataloader, build_eval_batches
from failure_injector import build_straggler_injector
from logger import WandbLogger, build_run_name
from model import build_model


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(framework_cfg_path: Path) -> dict:
    base = yaml.safe_load((Path("config") / "base.yaml").read_text())
    override = yaml.safe_load(framework_cfg_path.read_text())
    return _deep_merge(base, override)


def seed_everything(seed: int, rank: int) -> None:
    # Each rank gets a rank-differentiated seed for things like dataloader
    # shuffle (though we seed the dataloader explicitly). Model init uses the
    # base seed so every rank starts with identical parameters — that's
    # critical for DDP (broadcast skipped) and DiLoCo (Δ=0 at step 0).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed * 101 + rank)
    np.random.seed(seed * 101 + rank)


def init_distributed(timeout_seconds: int) -> tuple[int, int, int, torch.device, str]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device, backend


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["ddp", "diloco"], required=True)
    ap.add_argument("--failure", choices=["none", "crash", "straggler"], default="none")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--H", type=int, default=None, help="DiLoCo inner steps per outer step")
    ap.add_argument("--config", required=True, help="path to framework YAML")
    ap.add_argument("--runtime-dir", default="runtime", help="shared sidecar/worker dir")
    ap.add_argument("--control-filestore", default=None,
                    help="DiLoCo crash path: path to side-band FileStore")
    ap.add_argument("--rejoin", action="store_true",
                    help="this process is a replacement worker (DiLoCo crash)")
    ap.add_argument("--crash-epoch", type=int, default=0,
                    help="replacement-side identifier matching the sidecar's crash event")
    ap.add_argument("--smoke", action="store_true",
                    help="use a synthetic tiny corpus and tiny model (CPU test)")
    ap.add_argument("--smoke-target-loss", type=float, default=None,
                    help="override target loss in smoke mode (e.g. 0.1 to keep training)")
    ap.add_argument("--smoke-max-wall-clock", type=float, default=None,
                    help="override max wall-clock seconds in smoke mode")
    ap.add_argument("--crash-schedule", default=None,
                    help="comma-separated token thresholds for in-process virtual "
                         "crashes (e.g. '6684672,13369344,20054016'). When set, each "
                         "trainer fires a virtual crash at each threshold instead of "
                         "waiting for an out-of-process sidecar to SIGKILL a worker.")
    return ap.parse_args()


def apply_smoke_overrides(cfg: dict) -> dict:
    # Tiny model so the CPU smoke test finishes in under a minute.
    cfg["model"] = {
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 64,
        "vocab_size": 1024,
        "dropout": 0.0,
        "bias": False,
    }
    cfg["data"]["context_length"] = 64
    cfg["data"]["per_worker_batch_size"] = 4
    cfg["data"]["eval_batches"] = 4
    cfg["train"]["target_loss"] = 7.0  # not meant to converge — smoke test
    cfg["train"]["max_wall_clock_seconds"] = 60
    cfg["train"]["eval_every_seconds"] = 5
    if cfg.get("framework") == "ddp":
        cfg["ddp"]["checkpoint_interval_seconds"] = 20
    if cfg.get("framework") == "diloco":
        cfg["diloco"]["H"] = min(cfg["diloco"]["H"], 4)
    # Don't phone home in smoke
    cfg["wandb"]["mode"] = "disabled"
    os.environ.setdefault("WANDB_MODE", "disabled")
    return cfg


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    if args.H is not None and cfg.get("framework") == "diloco":
        cfg["diloco"]["H"] = args.H
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
        if args.smoke_target_loss is not None:
            cfg["train"]["target_loss"] = args.smoke_target_loss
        if args.smoke_max_wall_clock is not None:
            cfg["train"]["max_wall_clock_seconds"] = args.smoke_max_wall_clock

    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # Initialize distributed. For DiLoCo replacement processes, we DEFER
    # init_process_group until the rejoin handshake runs (the handshake calls
    # init_process_group itself after all N processes hit the FileStore
    # barrier — see diloco_trainer._do_rejoin_handshake).
    is_diloco_replacement = args.framework == "diloco" and args.rejoin
    if not is_diloco_replacement:
        timeout = cfg.get("crash", {}).get("nccl_timeout_seconds", 120)
        rank, world_size, local_rank, device, backend = init_distributed(timeout)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    seed_everything(args.seed, rank)

    publish_worker_pid(runtime_dir, rank)

    # Build model + data
    model = build_model(cfg["model"]).to(device)
    dataloader = build_dataloader(cfg, rank=rank, world_size=world_size, seed=args.seed, device=device, smoke=args.smoke)
    eval_batches = build_eval_batches(cfg, device=device, smoke=args.smoke)

    # Logger (rank 0 only does real init)
    H = cfg.get("diloco", {}).get("H") if args.framework == "diloco" else None
    run_name = build_run_name(args.framework, H, args.failure, args.seed)
    tags = {
        "framework": args.framework,
        "group": {"none": "A", "crash": "B", "straggler": "C"}[args.failure],
        "seed": args.seed,
        "N": world_size,
        "H": H,
        "failure_config": args.failure,
    }
    if args.framework == "ddp":
        tags["checkpoint_interval"] = cfg["ddp"]["checkpoint_interval_seconds"]
    logger = WandbLogger(cfg, run_name=run_name, tags=tags, rank=rank)

    # Straggler injector (only active for --failure straggler)
    straggler = build_straggler_injector(args.failure, cfg, args.seed, rank)

    # Control store (DiLoCo crash path only)
    control_store = None
    if args.control_filestore is not None:
        control_store = DiLoCoControlStore(Path(args.control_filestore), world_size=world_size)

    # Parse virtual-crash schedule
    crash_schedule = []
    if args.crash_schedule:
        crash_schedule = [int(t) for t in args.crash_schedule.split(",") if t.strip()]

    if args.framework == "ddp":
        from ddp_trainer import DDPTrainer
        trainer = DDPTrainer(
            model=model,
            dataloader=dataloader,
            eval_batches=eval_batches,
            logger=logger,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            runtime_dir=runtime_dir,
            straggler_injector=straggler,
            crash_schedule=crash_schedule,
            crash_seed=args.seed,
        )
    else:
        from diloco_trainer import DiLoCoTrainer
        trainer = DiLoCoTrainer(
            model=model,
            dataloader=dataloader,
            eval_batches=eval_batches,
            logger=logger,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            runtime_dir=runtime_dir,
            straggler_injector=straggler,
            control_store=control_store,
            is_rejoining=args.rejoin,
            rejoin_crash_epoch=args.crash_epoch,
            crash_schedule=crash_schedule,
            crash_seed=args.seed,
        )

    try:
        trainer.train_until_target_loss()
        if rank == 0:
            final = trainer.metrics.summary_final()
            logger.log(final)
            print(f"[train] final metrics: {final}")
    finally:
        try:
            trainer.shutdown()
        except Exception as exc:
            print(f"[train] trainer.shutdown raised {exc!r}")
        if rank == 0:
            logger.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
