"""P2.5 grey-box fidelity: metric, differentiable harmonic projection, --quick smoke run."""

import json

import numpy as np
import pytest
import torch

from lstmabar.analysis.harmonics import harmonic_amplitudes
from lstmabar.physics.fidelity import (
    FidelityConfig,
    HarmonicProjector,
    compute_gate,
    counted_terms,
    harmonic_errors,
    harmonic_loss,
    knob_settings,
    make_signals,
    summarize,
)
from lstmabar.physics.kb import load_kb
from lstmabar.physics.whitebox.diode_clipper import numba_available

SR = 22050


def _tone(f0, amps, n=SR // 2, phase=0.3):
    t = np.arange(n) / SR
    return sum(a * np.cos(2 * np.pi * (k + 1) * f0 * t + phase * k) for k, a in enumerate(amps))


def test_identical_signals_give_zero_error():
    x = np.stack([_tone(110.0, [0.5, 0.1, 0.05]), _tone(220.0, [0.2, 0.0, 0.01])])
    a = np.stack([harmonic_amplitudes(x[i], SR, f) for i, f in enumerate((110.0, 220.0))])
    err = harmonic_errors(a, a)
    assert err.shape == (2, 10) and np.all(err == 0)
    s = summarize(a, a, np.array([-10.0, 0.0]))
    assert s["mean_db"] == 0 and s["max_db"] == 0 and set(s["per_level"]) == {"-10", "0"}


def test_floor_and_h1_terms():
    white = np.array([[1.0, 1e-4, 1e-5, 0.1]])  # H2, H3 below -60 dBc
    grey = np.array([[0.5, 1e-6, 0.0, 0.1]])  # H2/H3 also below the floor, H4 equal (dBc)
    err = harmonic_errors(grey, white, floor_dbc=-60.0)
    # H1: 6.02 dB (absolute, no floor); H2, H3 floored on both sides → 0; H4: 0.1/0.5 vs 0.1/1.0
    np.testing.assert_allclose(err[0], [20 * np.log10(2), 0.0, 0.0, 20 * np.log10(2)], atol=1e-9)
    # One side above the floor: error is measured to the floor, not to -inf.
    err2 = harmonic_errors(np.array([[1.0, 1e-2]]), np.array([[1.0, 0.0]]), floor_dbc=-60.0)
    assert err2[0, 1] == pytest.approx(20.0)
    # Silent grey render: H1 error is inf (no floor), so it can't pass for a good match.
    assert np.isinf(harmonic_errors(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))[0, 0])


def test_masked_mean_excludes_terms_floored_on_both_sides():
    white = np.array([[1.0, 1e-4, 1e-5, 0.1]])  # H2, H3 below -60 dBc
    grey = np.array([[0.5, 1e-6, 0.01, 0.1]])  # H2 floored on both sides; H3 only on white
    mask = counted_terms(grey, white)
    np.testing.assert_array_equal(mask, [[True, False, True, True]])
    s = summarize(grey, white, np.array([0.0]), f0s=np.array([110.0]))
    err = harmonic_errors(grey, white)[0]
    assert s["mean_db"] == pytest.approx(err[[0, 2, 3]].mean())
    assert s["all_terms_mean_db"] == pytest.approx(err.mean())
    assert s["counted_terms"] == 3 and s["total_terms"] == 4
    assert s["per_level"]["0"]["h1_max_abs_db"] == pytest.approx(20 * np.log10(2))
    worst = s["worst_terms"][0]
    assert worst["harmonic"] == 3 and worst["error_db"] == pytest.approx(err.max())
    assert s["terms"]["counted"] == [[1, 0, 1, 1]]


def test_gate_verdicts():
    cfg = FidelityConfig()

    def pedal(mean):
        row = {"label": "x", "best": {"mean_db": mean, "all_terms_mean_db": mean / 2}}
        return {"has_deriver": False, "settings": [row]}

    g = compute_gate({"ds1": pedal(1.0), "ts808": pedal(3.5), "rat": pedal(9.0)}, cfg)
    assert g["ds1"]["verdict"] == "PASS" and g["ds1"]["deviation"] is None
    assert g["ts808"]["verdict"] == "PASS (documented deviation)" and not g["ts808"]["passed"]
    assert g["rat"]["verdict"] == "diagnostic"
    assert compute_gate({"ds1": pedal(3.5)}, cfg)["ds1"]["verdict"] == "FAIL"


def test_torch_projector_matches_numpy_and_is_differentiable():
    f0s = [82.41, 329.63]
    n = 4410
    x = np.stack([_tone(f, [0.7, 0.2, 0.0, 0.03, 0.001], n) for f in f0s]) + 1e-3
    proj = HarmonicProjector(f0s, n, SR, 10)
    got = proj(torch.from_numpy(x)).numpy()
    want = np.stack([harmonic_amplitudes(x[i], SR, f) for i, f in enumerate(f0s)])
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)
    xt = torch.from_numpy(x).requires_grad_(True)
    loss = harmonic_loss(proj(xt), torch.from_numpy(want))
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-3)
    harmonic_loss(proj(xt * 0.5), torch.from_numpy(want)).backward()
    assert torch.isfinite(xt.grad).all() and xt.grad.abs().sum() > 0


def test_harmonic_loss_matches_metric():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(3, 10, generator=g, dtype=torch.float64) + 1e-3
    b = torch.rand(3, 10, generator=g, dtype=torch.float64) + 1e-3
    want = harmonic_errors(a.numpy(), b.numpy()).mean()
    assert float(harmonic_loss(a, b)) == pytest.approx(want, rel=1e-9)


def test_signals_and_settings():
    cfg = FidelityConfig()
    sig = make_signals(cfg)
    assert sig.sines.shape[0] == 16 and set(sig.levels) == {-30.0, -20.0, -10.0, 0.0}
    peak = np.abs(sig.sines).max(axis=1)
    np.testing.assert_allclose(peak, 10 ** (sig.levels / 20), rtol=1e-3)
    assert sig.start / cfg.sample_rate == pytest.approx(cfg.settle, abs=1 / cfg.sample_rate)
    kb = load_kb()
    for pid in ("ts808", "ds1"):
        labels = [lb for lb, _ in knob_settings(kb[pid])]
        assert labels == ["gain min", "gain mid", "gain max", "tone min", "tone max"]


@pytest.mark.slow  # ~25 s on a laptop CPU; run with `pytest -m slow`
@pytest.mark.skipif(not numba_available(), reason="numba (whitebox extra) missing")
def test_cli_quick_smoke(tmp_path):
    from lstmabar.cli import main

    code = main(
        [
            "fidelity",
            "--quick",
            "--out",
            str(tmp_path),
            f"paths.runs={tmp_path / 'runs'}",
        ]
    )
    assert code in (0, 1)  # the gate verdict is not the point of a smoke run
    data = json.loads((tmp_path / "greybox_fidelity.json").read_text(encoding="utf-8"))
    assert set(data["pedals"]) >= {"ts808", "ds1"}
    for p in data["pedals"].values():
        row = p["settings"][0]
        assert np.isfinite(row["best"]["mean_db"]) and np.isfinite(row["best"]["mrstft_riff"])
        assert len(row["starts"]) == 2
    assert data["pedals"]["ts808"]["settings"][0]["derived"] is not None
    md = (tmp_path / "greybox_fidelity.md").read_text(encoding="utf-8")
    assert "## Gate" in md and "ts808" in md
    assert list((tmp_path / "runs" / "fidelity").glob("*/greybox_fidelity.md"))
