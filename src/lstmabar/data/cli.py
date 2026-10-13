"""``lstmabar data <step>``: the P4 pipeline steps, each a module with ``run(cfg, args)``.

Steps import lazily so optional extras (librosa, numba, anthropic) load only when used.
"""

from __future__ import annotations

import argparse
import importlib

from lstmabar.config import load_config
from lstmabar.seed import seed_everything

DEFAULT_CONFIG = "configs/data/v1.yaml"
GOLD_CONFIG = "configs/data/gold_v1.yaml"

# Default ``--config`` per step-name prefix (first match wins); other steps use DEFAULT_CONFIG.
STEP_CONFIG_DEFAULTS: dict[str, str] = {"gold-": GOLD_CONFIG}

STEPS: dict[str, tuple[str, str, str]] = {
    # step: (module, function, help)
    "ingest": ("ingest", "run", "download Guitar-TECHS DI and cut dry clips (4.1)"),
    "calibrate": ("calibrate", "run", "fit white-box calibration tables (R4)"),
    "render": ("render", "run", "render wet shards (4.2)"),
    "bench": ("render", "bench", "render + analysis throughput benchmark (R8)"),
    "captions": ("captions", "run", "template + paraphrase captions (4.3)"),
    "test-real": ("test_real", "run", "real-pedal test set (4.4)"),
    "freeze": ("freeze", "run", "split checks, hashes, index, data card (4.6-4.7)"),
    "vocab-audit": ("vocab_audit", "run", "one-shot held-out vs caption vocab overlap (4.7)"),
    "gold-targets": ("gold", "targets_step", "validate/draft gold target regions (4.5)"),
    "gold-pair": ("gold", "run", "render gold targets + distractors on test clips (4.5)"),
    "gold-import": ("gold_import", "run", "append friend-written gold instructions (4.5)"),
}


def default_config_for(step: str) -> str:
    """The config a step uses when ``--config`` is not given."""
    for prefix, path in STEP_CONFIG_DEFAULTS.items():
        if step.startswith(prefix):
            return path
    return DEFAULT_CONFIG


def cmd_data(args: argparse.Namespace) -> int:
    module, func, _ = STEPS[args.step]
    config = args.config if args.config is not None else default_config_for(args.step)
    cfg = load_config(config, args.overrides)
    seed_everything(int(cfg.seed))
    fn = getattr(importlib.import_module(f"lstmabar.data.{module}"), func)
    return int(fn(cfg, args) or 0)


def add_data_parser(sub) -> None:
    p = sub.add_parser("data", help="P4 data pipeline: " + ", ".join(STEPS))
    p.add_argument("step", choices=list(STEPS))
    p.add_argument(
        "--config",
        default=None,
        help=f"default: {GOLD_CONFIG} for gold-* steps, else {DEFAULT_CONFIG}",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="freeze: skip unfinished shards (index marked partial; not a release)",
    )
    p.add_argument("--fixtures", action="store_true", help="run on the synthetic fixture set")
    p.add_argument("--estimate", action="store_true", help="captions: price the run, no spend")
    p.add_argument("--pilot", action="store_true", help="captions: pilot batch only")
    p.add_argument(
        "--list", action="store_true", help="ingest/test-real: print the download plan; no download"
    )
    p.add_argument("--limit", type=int, default=None, help="cap the number of clips/items")
    p.add_argument("--workers", type=int, default=None, help="override render.workers")
    p.add_argument("--quick", action="store_true", help="calibrate: tiny smoke-test grid")
    p.add_argument("--pedals", nargs="+", default=None, help="calibrate: subset of pedal ids")
    p.add_argument(
        "--report-only",
        action="store_true",
        help="calibrate: rebuild the report from tables (adds missing derived boards; no refit)",
    )
    p.add_argument(
        "--offgrid",
        action="store_true",
        help="calibrate --report-only: also run the off-grid held-out check (white-box)",
    )
    p.add_argument(
        "--allow-derived-labels",
        action="store_true",
        help="render: label white-box items with derived params if a calibration table is missing",
    )
    p.add_argument(
        "--draft-missing",
        action="store_true",
        help="gold-targets: append draft rows for labelled gold items without one",
    )
    p.add_argument(
        "--allow-draft",
        action="store_true",
        help="gold-pair: render from an unreviewed (status: draft) targets file (smoke runs)",
    )
    p.add_argument("--input", default=None, help="gold-import: friends' CSV (Form export)")
    p.add_argument(
        "--dry-run", action="store_true", help="gold-import: print the plan; write nothing"
    )
    p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. render.workers=2")
    p.set_defaults(func=cmd_data)


__all__ = [
    "DEFAULT_CONFIG",
    "GOLD_CONFIG",
    "STEPS",
    "STEP_CONFIG_DEFAULTS",
    "add_data_parser",
    "cmd_data",
    "default_config_for",
]
