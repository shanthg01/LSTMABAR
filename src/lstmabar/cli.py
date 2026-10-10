"""Command-line entry point: ``lstmabar <command>``.

Local and Colab runs use the same commands; Colab notebooks only install the repo and call
this CLI.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

import lstmabar
from lstmabar.config import load_config
from lstmabar.dsp.recovery import RecoveryConfig, run_recovery, write_report
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


SHARE_WARNING = (
    "warning: --share creates a public URL; anyone with the link can upload audio and run it "
    "on this machine's compute. Consider --auth user:pass."
)


def _auth_pair(value: str) -> tuple[str, str]:
    user, sep, password = value.partition(":")
    if not sep or not user or not password:
        raise argparse.ArgumentTypeError("expected user:pass")
    return user, password


def demo_launch_kwargs(args: argparse.Namespace) -> dict:
    """Keyword arguments for ``gradio.Blocks.launch`` (prints the --share warning)."""
    from lstmabar.demo.app import MAX_FILE_SIZE

    if args.share:
        print(SHARE_WARNING, file=sys.stderr)
    return {
        "server_port": args.port,
        "share": args.share,
        "auth": args.auth,
        "max_file_size": MAX_FILE_SIZE,
    }


def cmd_demo(args: argparse.Namespace) -> int:
    """Launch the Gradio demo (needs the ``demo`` extra)."""
    from lstmabar.demo.app import build_app

    try:
        app = build_app()
    except ImportError as e:
        if (e.name or "").split(".")[0] != "gradio":
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
    app.launch(**demo_launch_kwargs(args))
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    """P1.7 parameter recovery on Drive + EQ; writes ``<out>/param_recovery.{md,json}``."""

    cfg = RecoveryConfig(trials=args.trials, steps=args.steps, seed=args.seed)
    result = run_recovery(cfg)
    md, js = write_report(result, args.out)
    verdict = "PASS" if result.passed else "FAIL"
    print(
        f"recovery {verdict}: {sum(result.success)}/{cfg.trials} trials "
        f"({100 * result.success_rate:.0f}%) in {result.seconds_elapsed:.0f} s -> {md}, {js}"
    )
    return 0 if result.passed else 1


def cmd_fidelity(args: argparse.Namespace) -> int:
    """P2.5 grey-box fidelity check; writes ``<out>/greybox_fidelity.{md,json}`` and a copy
    in a run dir."""
    from lstmabar.physics.fidelity import (
        FidelityConfig,
        gate_passed,
        load_result,
        run_fidelity,
        write_report,
    )

    discussion_path = Path(args.discussion or Path(args.out) / "greybox_fidelity.discussion.md")
    discussion = ""
    if discussion_path.exists():
        discussion = discussion_path.read_text(encoding="utf-8")
    elif args.discussion:
        raise FileNotFoundError(discussion_path)

    if args.from_json:  # re-render the Markdown only (e.g. after editing the discussion)
        result = load_result(args.from_json)
    else:
        cfg = load_config(args.config, [f"seed={args.seed}", *args.overrides])
        seed_everything(cfg.seed)
        run_dir = create_run_dir(cfg, root=cfg.paths.runs, name="fidelity")
        kw = {"seed": int(cfg.seed), "volts_per_fs": float(cfg.physics.volts_per_full_scale)}
        if args.pedals:
            kw["pedals"] = tuple(args.pedals)
        fcfg = FidelityConfig.quick_config(**kw) if args.quick else FidelityConfig(**kw)
        result = run_fidelity(fcfg)
        write_report(result, run_dir, discussion)
        print(f"run dir {run_dir}")
    md, js = write_report(result, args.out, discussion)
    verdict = "PASS" if gate_passed(result) else "FAIL"
    gates = ", ".join(f"{p} {g['best_mean_db']:.2f} dB" for p, g in result.gate.items())
    print(f"fidelity {verdict}: {gates} in {result.seconds_elapsed:.0f} s -> {md}, {js}")
    return 0 if gate_passed(result) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lstmabar")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print versions, device and git state")
    p_info.set_defaults(func=cmd_info)

    p_smoke = sub.add_parser("smoke", help="check config, seeding and run-dir creation")
    p_smoke.add_argument("--config", default="configs/base.yaml")
    p_smoke.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. seed=1")
    p_smoke.set_defaults(func=cmd_smoke)

    p_demo = sub.add_parser("demo", help="launch the Gradio pedalboard demo")
    p_demo.add_argument("--port", type=int, default=7860)
    p_demo.add_argument(
        "--share", action="store_true", help="create a public gradio link (see warning)"
    )
    p_demo.add_argument(
        "--auth", type=_auth_pair, default=None, metavar="USER:PASS", help="require a login"
    )
    p_demo.set_defaults(func=cmd_demo)

    p_rec = sub.add_parser("recover", help="run the DSP parameter-recovery check (P1 exit gate)")
    p_rec.add_argument("--trials", type=int, default=RecoveryConfig.trials)
    p_rec.add_argument("--steps", type=int, default=RecoveryConfig.steps)
    p_rec.add_argument("--seed", type=int, default=0)
    p_rec.add_argument("--out", default="reports", help="directory for the report files")
    p_rec.set_defaults(func=cmd_recover)

    p_fid = sub.add_parser("fidelity", help="grey-box vs white-box fidelity check (P2 exit gate)")
    p_fid.add_argument("--quick", action="store_true", help="seconds-long smoke run")
    p_fid.add_argument("--seed", type=int, default=0)
    p_fid.add_argument("--out", default="reports", help="directory for the report files")
    p_fid.add_argument("--pedals", nargs="*", help="pedal ids (default: every white-box pedal)")
    p_fid.add_argument("--config", default="configs/base.yaml")
    p_fid.add_argument(
        "--discussion",
        default=None,
        help="Markdown appended as the report's discussion "
        "(default: <out>/greybox_fidelity.discussion.md if it exists)",
    )
    p_fid.add_argument(
        "--from-json", default=None, help="re-render the report from a saved JSON (no run)"
    )
    p_fid.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. paths.runs=...")
    p_fid.set_defaults(func=cmd_fidelity)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
