"""M2 pedal presets in the demo: dials -> derive -> board sliders, EQ lock, white-box A/B."""

import numpy as np
import pytest

from lstmabar.audio import rms, synth_riff
from lstmabar.demo.presets import (
    DIAL_MAX,
    LOCKED_BLOCKS,
    MANUAL,
    MANUAL_LABEL,
    WHITEBOX_ERROR,
    derive_preset,
    dial_config,
    dials_to_knobs,
    is_locked,
    pedal_choices,
    preset_board_values,
    preset_controls,
    preset_matches,
    preset_readout,
    whitebox_available,
    whitebox_render,
)
from lstmabar.demo.render import iter_specs, knobs_to_params, physical_from_slider
from lstmabar.dsp.pedalboard import default_pedalboard
from lstmabar.physics.derive import Preset, preset_params
from lstmabar.physics.whitebox.base import WHITEBOX, WhiteBoxModel
from tests.demo.fakes import FakeBoard

SR = 44100


@pytest.fixture(scope="module")
def kb():
    from lstmabar.physics.kb import load_kb

    return load_kb()


@pytest.fixture(scope="module")
def board():
    return default_pedalboard(SR).eval()


def test_pedal_choices_lists_manual_then_every_pedal(kb):
    choices = pedal_choices(kb)
    assert choices[0] == (MANUAL_LABEL, MANUAL)
    assert choices[1:] == [(p.name, pid) for pid, p in kb.items()]


def test_dials_use_pot_labels_defaults_and_0_10_scale(kb):
    for pedal in kb.values():
        for pot in pedal.pots.values():
            cfg = dial_config(pot)
            assert cfg["label"] == pot.label
            assert (cfg["minimum"], cfg["maximum"]) == (0.0, DIAL_MAX)
            assert cfg["value"] == pytest.approx(pot.default * 10, abs=0.05)
        defaults = [dial_config(p)["value"] for p in pedal.pots.values()]
        assert dials_to_knobs(pedal, defaults) == pytest.approx(pedal.default_knobs(), abs=0.005)


def test_dials_to_knobs_scales_clamps_and_fills_defaults(kb):
    ts = kb["ts808"]
    knobs = dials_to_knobs(ts, [10.0, None, 12.0])
    assert knobs == {"drive": 1.0, "tone": ts.pots["tone"].default, "level": 1.0}
    with pytest.raises(ValueError, match="dial values"):
        dials_to_knobs(ts, [5.0])


def test_preset_controls_match_preset_params_on_real_board(kb, board):
    """Sliders written from a preset render exactly what ``preset_params`` gives."""
    for pedal in kb.values():
        preset = derive_preset(pedal, [7.0] * len(pedal.pots), SR)
        enabled, values = preset_board_values(board, preset)
        params = knobs_to_params(board, values, enabled)
        ref = preset_params(preset)
        for block in board.block_names:
            for k, v in ref[block].items():
                assert params[block][k].item() == pytest.approx(v.item(), abs=1e-5), (block, k)
        checks, sliders = preset_controls(board, preset)
        assert checks == [preset.enabled.get(n, True) for n in board.block_names]
        for (name, spec), s in zip(iter_specs(board), sliders, strict=True):
            want = preset.board.get(name, {}).get(spec.name, spec.default)
            assert physical_from_slider(spec, s) == pytest.approx(want, rel=1e-6, abs=1e-9)
        assert preset_matches(board, preset, checks, sliders)


def test_dial_change_moves_derived_sliders(kb, board):
    ts = kb["ts808"]
    lo = derive_preset(ts, [1.0, 5.0, 5.0], SR).board["drive"]["gain_db"]
    hi = derive_preset(ts, [9.0, 5.0, 5.0], SR).board["drive"]["gain_db"]
    assert hi > lo


def test_preset_matches_detects_tweaks_and_bypass(kb, board):
    preset = derive_preset(kb["rat"], [5.0, 5.0, 5.0], SR)
    checks, sliders = preset_controls(board, preset)
    names = [(n, s.name) for n, s in iter_specs(board)]
    i = names.index(("drive", "gain_db"))
    tweaked = list(sliders)
    tweaked[i] += 0.1  # within one slider step (0.3 dB): still the preset
    assert preset_matches(board, preset, checks, tweaked)
    tweaked[i] += 5.0
    assert not preset_matches(board, preset, checks, tweaked)
    assert not preset_matches(board, preset, [not c for c in checks], sliders)


def test_eq_is_locked_only_while_a_pedal_is_active():
    assert LOCKED_BLOCKS == ("eq",)
    assert is_locked("eq", True)
    assert not is_locked("eq", False)
    assert not is_locked("drive", True)


def test_preset_mapping_on_a_board_without_all_blocks(kb):
    """Generic boards: only the knobs the board has are mapped (clamped to its ranges)."""
    fake = FakeBoard()
    preset = derive_preset(kb["ts808"], [10.0, 5.0, 5.0], fake.sample_rate)
    enabled, values = preset_board_values(fake, preset)
    assert enabled == {"drive": True, "eq": True, "level": True}
    assert values["drive"]["gain_db"] == min(preset.board["drive"]["gain_db"], 24.0)
    assert values["eq"] == {"cutoff_hz": 1000.0, "mix": 0.5}  # not in the preset: defaults


def test_readout_surfaces_notes_and_circuit_quantities(kb):
    pedal = kb["ts808"]
    preset = derive_preset(pedal, [5.0, 5.0, 5.0], SR)
    md = preset_readout(pedal, preset, whitebox=True)
    assert pedal.name in md
    assert "Circuit gain into the clipper" in md and "Pre-clip high-pass corner" in md
    assert "EQ is locked" in md and "white-box" in md
    for note in preset.notes:
        assert f"- {note}" in md
    edited = preset_readout(pedal, preset, whitebox=False, edited=True)
    assert f"custom (edited from {pedal.name})" in edited
    assert "white-box" not in edited


def test_readout_lists_clamp_notes():
    pedal_like = type("P", (), {"name": "Toy", "pots": {}})()
    preset = Preset(
        pedal_id="toy",
        knobs={},
        board={"drive": {"gain_db": 60.0}},
        notes=("drive.gain_db 72 clamped to 60 dB: circuit gain exceeds Drive's range",),
    )
    md = preset_readout(pedal_like, preset, whitebox=False)
    assert "**Approximations / clamps:**" in md and "clamped to 60 dB" in md


# --- White-box ------------------------------------------------------------------------------


class _Halve(WhiteBoxModel):
    def process_volts(self, v):
        return 0.5 * v


class _Broken(WhiteBoxModel):
    def process_volts(self, v):
        raise RuntimeError(r"solver diverged in C:\secret\path\model.py")


def test_whitebox_availability_is_discovered_at_runtime(kb, monkeypatch):
    assert whitebox_available("ts808")  # registered by physics.whitebox on import
    assert not whitebox_available("rat") or "rat" in WHITEBOX
    monkeypatch.setitem(WHITEBOX, "toy_pedal", lambda p, k, sr: _Halve(sr))
    assert whitebox_available("toy_pedal")
    monkeypatch.delitem(WHITEBOX, "toy_pedal")
    assert not whitebox_available("toy_pedal")


def test_whitebox_render_trims_and_loudness_matches(kb, monkeypatch):
    pedal = kb["rat"]
    monkeypatch.setitem(WHITEBOX, "rat", lambda p, k, sr: _Halve(sr))
    clip = synth_riff("single_notes", seconds=2.0, sample_rate=SR)
    res = whitebox_render(pedal, pedal.default_knobs(), clip, SR, True, max_seconds=1.0)
    assert res.audio is not None and res.audio.shape == (SR,)
    assert res.audio.dtype == np.float32
    assert rms(res.audio) == pytest.approx(rms(clip[:SR]), rel=1e-3)
    assert "first 1 s" in res.message
    raw = whitebox_render(pedal, pedal.default_knobs(), clip, SR, False, max_seconds=5.0)
    np.testing.assert_allclose(raw.audio, 0.5 * clip, atol=1e-6)
    assert "first" not in raw.message


def test_whitebox_failure_gives_generic_message(kb, monkeypatch, caplog):
    pedal = kb["rat"]
    monkeypatch.setitem(WHITEBOX, "rat", lambda p, k, sr: _Broken(sr))
    clip = synth_riff("single_notes", seconds=0.2, sample_rate=SR)
    res = whitebox_render(pedal, pedal.default_knobs(), clip, SR, True, max_seconds=1.0)
    assert res.audio is None
    assert res.message == WHITEBOX_ERROR
    assert "secret" not in res.message and "Traceback" not in res.message
    assert "white-box render failed" in caplog.text  # details go to the server log only


def test_whitebox_non_finite_output_is_a_failure(kb):
    pedal = kb["rat"]
    clip = synth_riff("single_notes", seconds=0.2, sample_rate=SR)

    def nan_sim(p, x, sr, knobs):
        return np.full_like(x, np.nan)

    res = whitebox_render(pedal, {}, clip, SR, simulate_fn=nan_sim, max_seconds=1.0)
    assert res.audio is None and res.message == WHITEBOX_ERROR


def test_whitebox_render_real_ts808_short_clip(kb):
    clip = synth_riff("single_notes", seconds=0.25, sample_rate=SR)
    res = whitebox_render(kb["ts808"], kb["ts808"].default_knobs(), clip, SR, True, 1.0)
    assert res.audio is not None and np.all(np.isfinite(res.audio))
    assert np.max(np.abs(res.audio)) <= 0.99 + 1e-6
