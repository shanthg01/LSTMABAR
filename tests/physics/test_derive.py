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


def test_stage_clip_volts_override_and_transistor():
    from lstmabar.physics.derive import stage_clip_volts

    assert math.isclose(stage_clip_volts(ClipStage("c", "si", "shunt", n_pos=2, n_neg=2)), 1.2)
    assert stage_clip_volts(ClipStage("c", "si", "shunt", v_clip=0.7)) == 0.7
    with pytest.raises(KeyError, match="v_clip"):
        stage_clip_volts(ClipStage("q", "si_transistor", "transistor"))


def test_nan_from_deriver_raises():
    from lstmabar.physics.derive import _clamped

    with pytest.raises(ValueError, match="NaN"):
        _clamped("drive", {"gain_db": float("nan")}, [])


def test_context_from_config():
    from lstmabar.config import load_config
    from lstmabar.physics.derive import DeriveContext
    from lstmabar.physics.kb import default_pedals_dir

    base = default_pedals_dir().parent / "configs" / "base.yaml"
    ctx = DeriveContext.from_config(load_config(base, ["physics.volts_per_full_scale=0.5"]))
    assert ctx.volts_per_fs == 0.5 and ctx.sample_rate == 44100


def test_fit_reports_bounds():
    f = log_grid()
    fit = fit_tone(f, -40.0 * np.log2(f / 40.0))  # far steeper than any setting
    assert fit.at_bounds


# --- All pedals (P2 wave 2a) --------------------------------------------------------------------

GAIN_KNOB = {
    "ts808": "drive",
    "rat": "distortion",
    "ds1": "dist",
    "big_muff": "sustain",
    "fuzz_face_si": "fuzz",
    "fuzz_face_ge": "fuzz",
}
# Tone knob and the direction in which turning it up moves brightness (RAT filter darkens).
TONE_KNOB = {
    "ts808": ("tone", 1),
    "rat": ("filter", -1),
    "ds1": ("tone", 1),
    "big_muff": ("tone", 1),
}
LEVEL_KNOB = {
    "ts808": "level",
    "rat": "volume",
    "ds1": "level",
    "big_muff": "volume",
    "fuzz_face_si": "volume",
    "fuzz_face_ge": "volume",
}
SWEEP = (0.0, 1 / 3, 2 / 3, 1.0)
TONE_RMS_BOUND_DB = 3.0  # worst case: Big Muff at tone 0 (~2.5 dB), steeper than tilt + EQ3
_CACHE: dict = {}


def _d(kb, pid, **knobs):
    """Memoized derive (each call fits a tone response)."""
    key = (pid, tuple(sorted(knobs.items())))
    if key not in _CACHE:
        _CACHE[key] = derive(kb[pid], knobs)
    return _CACHE[key]


def _brightness(preset) -> float:
    """Grey-box tone response: mean dB above 2 kHz minus mean dB below 300 Hz."""
    f = log_grid()
    r = greybox_tone_db(f, 44100, preset.board["drive"]["tone_db"], **preset.board["eq"])
    return float(r[f > 2000].mean() - r[f < 300].mean())


def test_every_kb_pedal_has_deriver(kb):
    assert set(kb) <= set(DERIVERS), sorted(set(kb) - set(DERIVERS))
    assert set(GAIN_KNOB) == set(kb)


def test_rat_textbook_numbers(kb):
    rat = kb["rat"]
    assert math.isclose(rc_hz(rat.c("R_gain_hf"), rat.c("C_gain_hf")), 1539, rel_tol=0.01)
    assert math.isclose(rc_hz(rat.c("R_gain_lf"), rat.c("C_gain_lf")), 60.5, rel_tol=0.01)
    hi = _d(kb, "rat", distortion=1.0)
    assert math.isclose(hi.info["gain_hf_ceiling_db"], 67.3, abs_tol=0.1)  # 1 + 100k/(47||560)
    assert math.isclose(hi.info["gain_lf_leg_db"], 45.1, abs_tol=0.1)  # 1 + 100k/560
    # Not 67 dB flat: read at 1 kHz, between the LF leg and the HF ceiling.
    stage = hi.info["drive_stage_gain_db"]
    assert hi.info["gain_lf_leg_db"] + 5 < stage < hi.info["gain_hf_ceiling_db"] - 3
    assert hi.info["pre_gain_db_100hz"] < stage - 10
    # Calibrated gain exceeds 60 dB: clamped, with a meaningful note.
    assert hi.board["drive"]["gain_db"] == 60.0
    note = next(n for n in hi.notes if "gain_db" in n)
    assert "clamped to 60" in note and "Drive's range" in note
    # Filter sweep 32 kHz (R7 only) .. 475 Hz (R7 + 100k).
    assert math.isclose(_d(kb, "rat", filter=0.0).info["filter_lp_hz"], 32150, rel_tol=0.01)
    assert math.isclose(_d(kb, "rat", filter=1.0).info["filter_lp_hz"], 475, rel_tol=0.01)


def test_ds1_textbook_numbers(kb):
    p = _d(kb, "ds1", dist=1.0)
    assert math.isclose(p.info["gain_corner_hz"], 72.0, rel_tol=0.01)
    assert math.isclose(p.info["clip_lp_hz"], 7234, rel_tol=0.01)
    assert math.isclose(p.info["tone_lp_hz"], 234, rel_tol=0.01)
    assert math.isclose(p.info["tone_hp_hz"], 1064, rel_tol=0.01)
    assert math.isclose(p.info["opamp_gain_db"], 20 * math.log10(1 + 100e3 / 4.7e3), abs_tol=1e-6)
    # Booster: ~35 dB per ElectroSmash, below its open-loop R8/(R9 + r_e) gain.
    assert 30.0 < p.info["booster_gain_db"] < p.info["booster_open_loop_db"]
    # Booster + op-amp (~61.5 dB textbook) exceeds Drive's range: clamped.
    assert p.board["drive"]["gain_db"] == 60.0
    assert any("gain_db" in n and "clamped" in n for n in p.notes)


def test_big_muff_textbook_numbers(kb):
    p = _d(kb, "big_muff")
    assert math.isclose(p.info["tone_lp_hz"], 408, rel_tol=0.01)
    assert math.isclose(p.info["tone_hp_hz"], 1809, rel_tol=0.01)
    # Collector-feedback bias puts Q4's collector near 7 V (ElectroSmash).
    assert math.isclose(9.0 - p.info["booster_ic_ma"] * 15.0, 7.0, abs_tol=0.5)
    assert math.isclose(p.info["output_stage_gain_db"], 20 * math.log10(15 / 3.3), abs_tol=1e-6)
    # Mid scoop near 1 kHz with Tone centred.
    tone = {k: v for k, v in p.info.items() if k.startswith("tone_db_")}
    assert min(tone, key=tone.get) == "tone_db_1000hz"
    assert tone["tone_db_100hz"] - tone["tone_db_1000hz"] > 4.0
    assert tone["tone_db_10000hz"] - tone["tone_db_1000hz"] > 4.0
    # ... and in the fitted EQ as a mid cut near 1 kHz.
    eq = p.board["eq"]
    assert eq["mid_db"] < -4.0 and 600 < eq["mid_hz"] < 1600


def test_big_muff_scoop_location(kb):
    from lstmabar.physics.derive import lp_hp_blend_tone_stack
    from lstmabar.physics.networks import cap, res

    bm = kb["big_muff"]
    load = cap(0.1e-6) + res(81e3)
    h = lp_hp_blend_tone_stack(bm, 0.5, "R_tone_lp", "C_tone_lp", "C_tone_hp", "R_tone_hp", load)
    f = np.geomspace(100, 10000, 400)
    assert 700 < f[np.argmin(h.db(f))] < 1400


@pytest.mark.parametrize("pid", ["fuzz_face_ge", "fuzz_face_si"])
def test_fuzz_face_textbook_numbers(kb, pid):
    lo, hi = _d(kb, pid, fuzz=0.0), _d(kb, pid, fuzz=1.0)
    assert math.isclose(lo.info["output_hp_hz"], 31.8, rel_tol=0.01)  # C3 into 500k
    # Emitter bypass at full fuzz: C2 against 1k ∥ r_e2 (ElectroSmash's 7.9 Hz uses 1k alone).
    r_e2 = 0.02585 / (hi.info["q2_ic_ma"] * 1e-3)
    bypass = rc_hz(1e3 * r_e2 / (1e3 + r_e2), 20e-6)
    assert math.isclose(hi.info["bypass_hp_hz"], bypass, rel_tol=0.03)  # + 1 ohm pot end
    assert 140 < bypass < 200
    r2 = 470 if pid.endswith("ge") else 330
    div = 20 * math.log10(r2 / (r2 + 8200))
    assert math.isclose(lo.info["output_divider_db"], div, abs_tol=1e-6)
    # Q2 gain ~(R2 + R3)/R_unbypassed = 8.2x at minimum fuzz (ElectroSmash).
    assert math.isclose(lo.info["q2_gain_db"], 20 * math.log10(8.2), abs_tol=0.5)
    # Fuzz up bypasses the emitter: the C2 bypass corner moves up into the audio band.
    assert hi.board["drive"]["pre_hpf_hz"] > 100.0 > lo.board["drive"]["pre_hpf_hz"]
    # Asymmetric clipping (Q2 cutoff vs saturation swing).
    assert 0.0 < hi.board["drive"]["asymmetry"] < 1.0


def test_fuzz_face_source_impedance(kb):
    from lstmabar.physics.derive import DeriveContext

    stiff = derive(kb["fuzz_face_ge"], {"fuzz": 0.5}, DeriveContext(source_ohms=100.0))
    soft = derive(kb["fuzz_face_ge"], {"fuzz": 0.5}, DeriveContext(source_ohms=50e3))
    assert stiff.info["drive_stage_gain_db"] > soft.info["drive_stage_gain_db"] + 6


def test_ts808_tone_stage(kb):
    p = _d(kb, "ts808")
    assert math.isclose(p.info["tone_shunt_hz"], 3200, rel_tol=0.05)  # R8/C6 wiper corner
    assert not any("tone knob" in n for n in p.notes)
    dark, bright = _d(kb, "ts808", tone=0.0), _d(kb, "ts808", tone=1.0)
    assert bright.info["tone_hf_db"] > dark.info["tone_hf_db"] + 6.0


def test_ts808_tone_stage_limits(kb):
    """Tone at 0: R8/C6 shunts P, cutting below the 723 Hz R7/C5 pole alone; tone at 1: the
    treble shelf levels off the 723 Hz roll-off."""
    from lstmabar.physics.derive import ts808_tone_stage
    from lstmabar.physics.networks import rc_lowpass

    ts = kb["ts808"]
    f = np.array([100.0, 3000.0, 10000.0])
    dark, bright = ts808_tone_stage(ts, 0.0).db(f), ts808_tone_stage(ts, 1.0).db(f)
    lf = 20 * math.log10(10e3 / 11e3)  # R7 into the 10k bias resistor
    assert math.isclose(dark[0], lf, abs_tol=0.3) and math.isclose(bright[0], lf, abs_tol=0.3)
    one_pole = rc_lowpass(1e3, 0.22e-6).db(f) + lf
    assert dark[1] < one_pole[1] - 3  # extra treble cut
    assert bright[1] > one_pole[1] + 6  # treble restored
    assert bright[1] - bright[0] > -6  # nearly flat to 3 kHz
    assert bright[1] - dark[1] > 10


@pytest.mark.parametrize("pid", sorted(GAIN_KNOB))
def test_gain_knob_monotonic(kb, pid):
    presets = [_d(kb, pid, **{GAIN_KNOB[pid]: v}) for v in SWEEP]
    stage = [p.info["drive_stage_gain_db"] for p in presets]
    gain = [p.board["drive"]["gain_db"] for p in presets]
    assert all(b > a for a, b in zip(stage, stage[1:], strict=False)), stage
    assert gain == sorted(gain) and gain[-1] > gain[0], gain


@pytest.mark.parametrize("pid", sorted(TONE_KNOB))
def test_tone_knob_moves_brightness(kb, pid):
    knob, sign = TONE_KNOB[pid]
    b = [sign * _brightness(_d(kb, pid, **{knob: v})) for v in SWEEP]
    assert all(y > x for x, y in zip(b, b[1:], strict=False)), b
    assert b[-1] - b[0] > 6.0, b


@pytest.mark.parametrize("pid", sorted(LEVEL_KNOB))
def test_level_knob_monotonic(kb, pid):
    levels = [_d(kb, pid, **{LEVEL_KNOB[pid]: v}).board["drive"]["level_db"] for v in SWEEP]
    assert levels == sorted(levels), levels
    assert levels[-1] > levels[-2] > levels[0], levels


def _grid(pedal):
    """Default, all-min, all-max and each knob alone at min/max."""
    names = list(pedal.pots)
    yield {}
    yield dict.fromkeys(names, 0.0)
    yield dict.fromkeys(names, 1.0)
    for n in names:
        yield {n: 0.0}
        yield {n: 1.0}


@pytest.mark.parametrize("pid", sorted(GAIN_KNOB))
def test_grid_finite_in_range_and_tone_fit(kb, pid):
    from lstmabar.physics.derive import BLOCK_SPECS

    worst = 0.0
    for knobs in _grid(kb[pid]):
        p = _d(kb, pid, **knobs)
        for block, vals in p.board.items():
            specs = {s.name: s for s in BLOCK_SPECS[block]}
            for k, v in vals.items():
                assert math.isfinite(v) and specs[k].min <= v <= specs[k].max, (knobs, k, v)
        # TS808 has no pre-HPF fit (textbook corner); post gain is -inf with volume at 0.
        skip = {"pre_hpf_fit_rms_db", "post_gain_db"}
        finite = {k: v for k, v in p.info.items() if k not in skip}
        assert all(math.isfinite(v) for v in finite.values()), (knobs, finite)
        worst = max(worst, p.info["tone_fit_rms_db"])
    assert worst < TONE_RMS_BOUND_DB, (pid, worst)


@pytest.mark.parametrize("pid", sorted(GAIN_KNOB))
def test_presets_render(kb, pid):
    params = preset_params(_d(kb, pid))
    board = default_pedalboard(22050)
    x = torch.sin(2 * math.pi * 110 * torch.arange(2205) / 22050).unsqueeze(0) * 0.5
    with torch.inference_mode():
        y = board(x, params)
    assert torch.isfinite(y).all() and y.abs().max() > 1e-4


def test_numpy_biquad_matches_torch():
    from lstmabar.physics.analog import _biquad_db, _biquad_db_torch

    f = log_grid()
    for kind, fc, q, g in [
        ("lowshelf", 1000.0, 0.707, -5.0),
        ("highshelf", 3000.0, 0.707, 7.0),
        ("peak", 640.0, 0.9, -9.0),
        ("lowshelf", 120.0, 0.707, 12.0),
    ]:
        ref = _biquad_db_torch(kind, fc, q, g, f, 44100)
        np.testing.assert_allclose(_biquad_db(kind, fc, q, g, f, 44100), ref, atol=1e-6)
