"""``lstmabar data <step>``: the P4 pipeline steps, each a module with ``run(cfg, args)``.

Steps import lazily so optional extras (librosa, numba, anthropic) load only when used.
"""

from __future__ import annotations

import argparse
import importlib

from lstmabar.config import load_config
from lstmabar.seed import seed_everything

DEFAULT_CONFIG = "configs/data/v1.yaml"

STEPS: dict[str, tuple[str, str, str]] = {
    # step: (module, function, help)
    "ingest": ("ingest", "run", "download Guitar-TECHS DI and cut dry clips (4.1)"),
    "calibrate": ("calibrate", "run", "fit white-box calibration tables (R4)"),
    "render": ("render", "run", "render wet shards (4.2)"),
    "bench": ("render", "bench", "render + analysis throughput benchmark (R8)"),
    "captions": ("captions", "run", "template + paraphrase captions (4.3)"),
    "test-real": ("test_real", "run", "real-pedal test set (4.4)"),
    "freeze": ("freeze", "run", "split checks, hashes, index, data card (4.6-4.7)"),
}


def cmd_data(args: argparse.Namespace) -> int:
    module, func, _ = STEPS[args.step]
    cfg = load_config(args.config, args.overrides)
    seed_everything(int(cfg.seed))
    fn = getattr(importlib.import_module(f"lstmabar.data.{module}"), func)
    return int(fn(cfg, args) or 0)


def add_data_parser(sub) -> None:
    p = sub.add_parser("data", help="P4 data pipeline: " + ", ".join(STEPS))
    p.add_argument("step", choices=list(STEPS))
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--fixtures", action="store_true", help="run on the synthetic fixture set")
    p.add_argument("--estimate", action="store_true", help="captions: price the run, no spend")
    p.add_argument("--pilot", action="store_true", help="captions: pilot batch only")
    p.add_argument(
        "--list", action="store_true", help="ingest/test-real: print the download plan; no download"
    )
    p.add_argument("--limit", type=int, default=None, help="cap the number of clips/items")
    p.add_argument("--workers", type=int, default=None, help="override render.workers")
    p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. render.workers=2")
    p.set_defaults(func=cmd_data)


__all__ = ["STEPS", "add_data_parser", "cmd_data"]
