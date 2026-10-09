"""P1.7 parameter recovery — the P1 exit gate (>= 90% of trials recovered on Drive + EQ).

The gate itself is slow (~7-8 min on a 10-thread CPU): excluded from default runs, run with
``uv run pytest -m slow``. The same procedure backs ``lstmabar recover``, which writes
``reports/param_recovery.md``.
"""

import json

import pytest
import torch

from lstmabar.cli import main
from lstmabar.dsp.losses import multi_resolution_stft_loss
from lstmabar.dsp.recovery import RecoveryConfig, format_report, per_example_loss, run_recovery


def test_per_example_loss_matches_batched_loss():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(3, 4096, generator=g), torch.randn(3, 4096, generator=g)
    per = per_example_loss(a, b, crop=0)
    assert per.shape == (3,)
    torch.testing.assert_close(per.mean(), multi_resolution_stft_loss(a, b))


def test_recover_cli_writes_report(tmp_path):
    code = main(["recover", "--trials", "2", "--steps", "2", "--out", str(tmp_path)])
    assert code in (0, 1)  # 2 steps cannot recover; this only checks the plumbing
    md = (tmp_path / "param_recovery.md").read_text(encoding="utf-8")
    assert "Per-parameter" in md and "drive.gain_db" in md
    data = json.loads((tmp_path / "param_recovery.json").read_text(encoding="utf-8"))
    assert len(data["final_loss"]) == 2 and data["config"]["trials"] == 2


@pytest.mark.slow
def test_parameter_recovery_exit_gate():
    cfg = RecoveryConfig()
    assert cfg.trials >= 20
    result = run_recovery(cfg, log=lambda _: None)
    print(format_report(result))
    assert result.passed, format_report(result)
