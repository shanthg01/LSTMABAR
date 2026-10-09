"""Run directories: one folder per experiment run holding config, metadata and outputs.

Layout: ``<root>/<name>/<YYYYmmdd-HHMMSS>/`` containing ``config.yaml`` (fully resolved) and
``meta.json`` (git SHA, dirty flag, versions, device).
"""

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

import lstmabar


def git_info(cwd: str | Path | None = None) -> dict:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_run_dir(cfg: DictConfig, root: str | Path = "runs", name: str | None = None) -> Path:
    name = name or cfg.get("name", "run")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    meta = {
        "git": git_info(),
        "lstmabar_version": lstmabar.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": device_name(),
        "created": stamp,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir
