import json

import numpy as np
import torch
from omegaconf import OmegaConf

from lstmabar.cli import main
from lstmabar.config import load_config
from lstmabar.runs import create_run_dir
from lstmabar.seed import seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = (torch.randn(5), np.random.rand(5))
    seed_everything(123)
    b = (torch.randn(5), np.random.rand(5))
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_load_config_composes_defaults_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "seed: 0\naudio:\n  sample_rate: 44100\n  clip_seconds: 3.0\n"
    )
    (tmp_path / "exp.yaml").write_text(
        "defaults: [base.yaml]\nname: exp\naudio:\n  clip_seconds: 2.0\n"
    )

    cfg = load_config(tmp_path / "exp.yaml", ["seed=7"])

    assert cfg.name == "exp"
    assert cfg.seed == 7
    assert cfg.audio.sample_rate == 44100
    assert cfg.audio.clip_seconds == 2.0
    assert "defaults" not in cfg


def test_create_run_dir_writes_config_and_meta(tmp_path):
    cfg = OmegaConf.create({"name": "unit", "seed": 1})
    run_dir = create_run_dir(cfg, root=tmp_path)

    assert run_dir.parent.name == "unit"
    assert OmegaConf.load(run_dir / "config.yaml").seed == 1
    meta = json.loads((run_dir / "meta.json").read_text())
    assert {"git", "torch_version", "device"} <= meta.keys()


def test_cli_smoke(tmp_path, capsys):
    assert main(["smoke", "--config", "configs/base.yaml", f"paths.runs={tmp_path}"]) == 0
    assert "smoke ok" in capsys.readouterr().out
