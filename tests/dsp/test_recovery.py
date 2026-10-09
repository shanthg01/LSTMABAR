"""P1.7 parameter recovery — the P1 exit gate (>= 90% of trials recovered on Drive + EQ).

The gate itself is slow (~7-8 min on a 10-thread CPU): excluded from default runs, run with
``uv run pytest -m slow``. The same procedure backs ``lstmabar recover``, which writes
``reports/param_recovery.md``.
"""

import json
import math

import pytest
import torch

from lstmabar.cli import main
from lstmabar.dsp.losses import multi_resolution_stft_loss
from lstmabar.dsp.recovery import (
    RecoveryConfig,
    _crop,
    format_report,
    per_example_loss,
    run_recovery,
)


def test_per_example_loss_matches_batched_loss():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(3, 4096, generator=g), torch.randn(3, 4096, generator=g)
    per = per_example_loss(a, b, crop=0)
    assert per.shape == (3,)
    torch.testing.assert_close(per.mean(), multi_resolution_stft_loss(a, b))


def test_per_example_loss_crop_zero_keeps_signal():
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(2, 4096, generator=g), torch.randn(2, 4096, generator=g)
    assert torch.isfinite(per_example_loss(a, b, crop=0)).all()
    assert not torch.allclose(per_example_loss(a, b, crop=0), per_example_loss(a, b, crop=512))
    assert _crop(a, 0).shape == a.shape and _crop(a, 512).shape == (2, 4096 - 1024)
    with pytest.raises(ValueError):
        _crop(a, -1)


def test_recovery_runs_with_crop_zero():
    cfg = RecoveryConfig(trials=2, steps=1, crop=0)
    r = run_recovery(cfg, log=lambda _: None)
    assert all(math.isfinite(v) for v in r.final_loss)
    assert 0.0 <= r.param_space["all_within_tol"] <= 1.0
    assert set(r.param_space["per_knob_within_tol"]) == set(r.knobs)
    assert "Parameter space" in format_report(r)


def test_recover_cli_writes_report(tmp_path):
    code = main(["recover", "--trials", "2", "--steps", "2", "--out", str(tmp_path)])
    assert code in (0, 1)  # 2 steps cannot recover; this only checks the plumbing
    md = (tmp_path / "param_recovery.md").read_text(encoding="utf-8")
    assert "Parameter space" in md and "drive.gain_db" in md
    data = json.loads((tmp_path / "param_recovery.json").read_text(encoding="utf-8"))
    assert len(data["final_loss"]) == 2 and data["config"]["trials"] == 2


@pytest.mark.slow
def test_parameter_recovery_exit_gate():
    cfg = RecoveryConfig()
    assert cfg.trials >= 20
    result = run_recovery(cfg, log=lambda _: None)
    print(format_report(result))
    assert result.passed, format_report(result)
