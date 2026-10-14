"""``lstmabar train <step>``: P6 training stages, each a module with ``run(cfg, args)``.

Stage 1 trains on cached embeddings (CPU); stage 2 adds ``L_audio`` through the pedalboard
(Colab GPU via the same CLI, ``colab/bootstrap.ipynb``).
"""

from __future__ import annotations

import argparse
import importlib

from lstmabar.config import load_config
from lstmabar.seed import seed_everything

DEFAULT_CONFIG = "configs/train/stage1.yaml"

STEPS: dict[str, tuple[str, str, str]] = {
    # step: (module under lstmabar.train, function, help)
    "stage1": ("stage1", "run", "L_param on cached embeddings (6.2)"),
    "stage2": ("stage2", "run", "add L_audio through the pedalboard (6.3, Colab)"),
}


def cmd_train(args: argparse.Namespace) -> int:
    module, func, _ = STEPS[args.step]
    cfg = load_config(args.config, args.overrides)
    seed_everything(int(cfg.seed))
    fn = getattr(importlib.import_module(f"lstmabar.train.{module}"), func)
    return int(fn(cfg, args) or 0)


def add_train_parser(sub) -> None:
    p = sub.add_parser("train", help="P6 training: " + ", ".join(STEPS))
    p.add_argument("step", choices=list(STEPS))
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--limit", type=int, default=None, help="cap items per split (smoke runs)")
    p.add_argument("--fixtures", action="store_true", help="run on the synthetic eval fixture")
    p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. seed=1")
    p.set_defaults(func=cmd_train)


__all__ = ["DEFAULT_CONFIG", "STEPS", "add_train_parser", "cmd_train"]
