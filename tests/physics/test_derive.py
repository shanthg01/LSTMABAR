import math

import numpy as np
import pytest
import torch

from lstmabar.dsp.pedalboard import default_pedalboard
from lstmabar.physics.analog import analog_response_db, fit_tone, greybox_tone_db, log_grid
from lstmabar.physics.calibration import clip_volts, drive_gain_db, output_level_db
from lstmabar.physics.derive import DERIVERS, derive, preset_params, rc_hz, shaper_prior
from lstmabar.physics.kb import ClipStage, load_kb


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def test_calibration_formulas():
    assert math.isclose(drive_gain_db(1.0, 1.0), 0.0)
    assert math.isclose(drive_gain_db(10.0, 0.6), 20.0 + 20 * math.log10(1 / 0.6))
    assert math.isclose(
        drive_gain_db(10.0, 0.6, volts_per_fs=0.5), 20.0 + 20 * math.log10(0.5 / 0.6)
    )
    assert math.isclose(output_level_db(1.0), 0.0)
    assert math.isclose(clip_volts("si", 2), 1.2)


def test_ts808_textbook_numbers(kb):
    ts = kb["ts808"]
    assert math.isclose(rc_hz(ts.c("R_gain"), ts.c("C_gain")), 720.0, rel_tol=0.01)
    lo = derive(ts, {"drive": 0.0})
    hi = derive(ts, {"drive": 1.0})
    # Stage gain 1 + (51k + R_drive)/4.7k: 21.5 dB at min, 41.4 dB at max.
    assert math.isclose(lo.info["stage_gain_db"], 20 * math.log10(1 + 51e3 / 4.7e3), abs_tol=1e-6)
    assert math.isclose(lo.info["stage_gain_db"], 21.5, abs_tol=0.1)
    assert math.isclose(hi.info["stage_gain_db"], 41.4, abs_tol=0.1)
    # Calibrated Drive gain adds 20 log10(1 V / 0.6 V) = +4.4 dB.
    assert math.isclose(hi.board["drive"]["gain_db"], 41.4 + 4.44, abs_tol=0.1)
    assert math.isclose(hi.info["feedback_lp_hz"], 5660.0, rel_tol=0.01)
    assert math.isclose(hi.board["drive"]["pre_hpf_hz"], 720.0, rel_tol=0.01)


def test_drive_knob_monotonic(kb):
    gains = [
        derive(kb["ts808"], {"drive": d}).board["drive"]["gain_db"] for d in np.linspace(0, 1, 11)
    ]
    assert gains == sorted(gains)


def test_derive_validates_knobs(kb):
    with pytest.raises(KeyError, match="unknown knobs"):
        derive(kb["ts808"], {"fuzz": 0.5})
    with pytest.raises(ValueError, match="outside"):
        derive(kb["ts808"], {"drive": 1.5})


def test_level_zero_is_clamped_and_noted(kb):
    p = derive(kb["ts808"], {"level": 0.0})
    assert p.board["drive"]["level_db"] == -36.0
    assert any("level_db" in n for n in p.notes)


def test_preset_params_render(kb):
    preset = derive(kb["ts808"])
    params = preset_params(preset)
    board = default_pedalboard(22050)
    for block, p in params.items():
        for k, v in p.items():
            assert v.shape == (1,) and 0.0 <= float(v) <= 1.0, (block, k)
    assert float(params["compressor"]["enabled"]) == 0.0
    x = torch.sin(2 * math.pi * 196 * torch.arange(4410) / 22050).unsqueeze(0) * 0.3
    with torch.inference_mode():
        y = board(x, params)
    assert torch.isfinite(y).all() and y.abs().max() > 0


def test_every_kb_pedal_with_deriver_derives(kb):
    for pid, pedal in kb.items():
        if pid in DERIVERS:
            preset = derive(pedal)
            assert set(preset.board) <= {"compressor", "drive", "eq"}


def test_shaper_prior():
    sym = shaper_prior(ClipStage("c", "si", "shunt"))
    assert sym["asymmetry"] == 0.0
    asym = shaper_prior(ClipStage("c", "si", "shunt", n_pos=1, n_neg=2))
    assert math.isclose(asym["asymmetry"], 1.0)
    assert shaper_prior(ClipStage("c", "si", "feedback"))["softness"] < sym["softness"]


def test_greybox_tone_flat_and_fit_recovers():
    f = log_grid()
    assert np.allclose(greybox_tone_db(f, 44100), 0.0, atol=1e-6)
    target = (
        greybox_tone_db(f, 44100, tone_db=-4.0, low_db=2.0, mid_db=-6.0, mid_hz=1000.0, high_db=3.0)
        + 1.5
    )
    fit = fit_tone(f, target)
    assert fit.rms_error_db < 0.05
    assert math.isclose(fit.gain_db, 1.5, abs_tol=0.3)


def test_fit_first_order_lowpass():
    f = log_grid()
    w = 2 * math.pi * 723.0
    target = analog_response_db([w], [1.0, w], f)
    assert math.isclose(target[0], 0.0, abs_tol=0.05)
    fit = fit_tone(f, target)
    assert fit.rms_error_db < 2.0
