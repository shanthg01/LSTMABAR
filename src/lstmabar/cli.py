"""Command-line entry point: ``lstmabar <command>``.

Local and Colab runs use the same commands; Colab notebooks only install the repo and call
this CLI.
"""

import argparse
import json
import platform
import sys

import torch

import lstmabar
from lstmabar.config import load_config
from lstmabar.runs import create_run_dir, device_name, git_info
from lstmabar.seed import seed_everything


def cmd_info(_: argparse.Namespace) -> int:
    info = {
        "lstmabar": lstmabar.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device_name(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git": git_info(),
    }
    print(json.dumps(info, indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """End-to-end check of config -> seed -> run dir -> a tiny tensor op on the active device."""
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    run_dir = create_run_dir(cfg, root=cfg.paths.runs, name="smoke")
    x = torch.randn(4, cfg.audio.sample_rate // 100, device=device_name())
    rms = x.pow(2).mean().sqrt().item()
    print(f"smoke ok: run_dir={run_dir} device={x.device} rms={rms:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lstmabar")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print versions, device and git state")
    p_info.set_defaults(func=cmd_info)

    p_smoke = sub.add_parser("smoke", help="check config, seeding and run-dir creation")
    p_smoke.add_argument("--config", default="configs/base.yaml")
    p_smoke.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. seed=1")
    p_smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
