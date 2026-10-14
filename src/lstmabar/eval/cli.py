"""``lstmabar eval <step>``: the P5 steps, each a module with ``run(cfg, args)``.

Steps import lazily so optional extras (transformers, librosa, anthropic) load only when used.
"""

from __future__ import annotations

import argparse
import importlib

from lstmabar.config import load_config
from lstmabar.seed import seed_everything

DEFAULT_CONFIG = "configs/eval/p5.yaml"

STEPS: dict[str, tuple[str, str, str]] = {
    # step: (module under lstmabar.eval, function, help)
    "embed": ("embed", "run", "build/refresh the CLAP + bge embedding caches (5.5)"),
    "proxy": ("proxy", "run", "draft + render the val gold-proxy distractors (D8)"),
    "sim-real": ("sim_real", "run", "test_real sim-to-real gap (D11)"),
    "run": ("run", "run", "score every method on val/test/gold and write the report (5.4)"),
    "report": ("run", "report", "re-render the report from a run dir's scores (no scoring)"),
}


def cmd_eval(args: argparse.Namespace) -> int:
    module, func, _ = STEPS[args.step]
    cfg = load_config(args.config, args.overrides)
    seed_everything(int(cfg.eval.seed))
    fn = getattr(importlib.import_module(f"lstmabar.eval.{module}"), func)
    return int(fn(cfg, args) or 0)


def add_eval_parser(sub) -> None:
    p = sub.add_parser("eval", help="P5 evaluation: " + ", ".join(STEPS))
    p.add_argument("step", choices=list(STEPS))
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--limit", type=int, default=None, help="cap items per split (smoke runs)")
    p.add_argument("--methods", nargs="+", default=None, help="run: subset of eval.methods")
    p.add_argument("--splits", nargs="+", default=None, help="embed/run: subset of splits")
    p.add_argument("--models", nargs="+", default=None, help="embed: subset of models")
    p.add_argument("--workers", type=int, default=None, help="override eval.workers")
    p.add_argument("--run-dir", default=None, help="report: the run dir to re-render")
    p.add_argument("--fixtures", action="store_true", help="run on the synthetic eval fixture")
    p.add_argument("--estimate", action="store_true", help="run: price the LLM baseline, no spend")
    p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. eval.workers=1")
    p.set_defaults(func=cmd_eval)


__all__ = ["DEFAULT_CONFIG", "STEPS", "add_eval_parser", "cmd_eval"]
