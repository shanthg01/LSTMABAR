import numpy as np
import pytest
import torch

from lstmabar.audio import rms, synth_riff
from lstmabar.demo.render import (
    default_knobs,
    describe_text,
    knobs_to_params,
    prepare_input,
    random_knobs,
    render,
    to_int16,
)
from tests.demo.fakes import FakeBoard


@pytest.fixture
def board():
    return FakeBoard()


def test_knobs_to_params_defaults_match_board_defaults(board):
    params = knobs_to_params(board, {}, {})
    ref = board.default_params(1)
    assert list(params) == board.block_names
    for name in board.block_names:
        assert set(params[name]) == set(ref[name])
        for k, v in params[name].items():
            assert v.shape == (1,)
            assert torch.allclose(v, ref[name][k], atol=1e-6)


def test_knobs_to_params_normalizes_physical_values(board):
    params = knobs_to_params(
        board,
        {"drive": {"gain_db": 12.0}, "eq": {"cutoff_hz": 10000.0, "mix": 0.25}},
        {"eq": False},
    )
    assert params["drive"]["gain_db"].item() == pytest.approx(0.75)
    assert params["eq"]["cutoff_hz"].item() == pytest.approx(1.0)
    assert params["eq"]["mix"].item() == pytest.approx(0.25)
    assert params["eq"]["enabled"].item() == 0.0
    assert params["drive"]["enabled"].item() == 1.0
    # Out-of-range values are clamped by ParamSpec.normalize.
    clamped = knobs_to_params(board, {"drive": {"gain_db": 99.0}}, {})
    assert clamped["drive"]["gain_db"].item() == 1.0


def test_default_and_random_knobs_cover_all_specs(board):
    d = default_knobs(board)
    assert d["eq"] == {"cutoff_hz": 1000.0, "mix": 0.5}
    r = random_knobs(board, np.random.default_rng(0))
    for name in board.block_names:
        for spec in board.blocks[name].param_specs:
            assert spec.min <= r[name][spec.name] <= spec.max


def test_render_shapes_dtype_and_gain(board):
    x = synth_riff("single_notes", seconds=0.5)
    params = knobs_to_params(board, {"drive": {"gain_db": -6.0}}, {"eq": False})
    dry, wet = render(board, x, params, match_loudness=False)
    assert dry.dtype == wet.dtype == np.float32
    assert dry.shape == wet.shape == x.shape
    assert rms(wet) / rms(dry) == pytest.approx(10 ** (-6 / 20), rel=1e-3)


def test_render_is_peak_safe_and_loudness_matches(board):
    x = synth_riff("power_chords", seconds=0.5)
    params = knobs_to_params(board, {"drive": {"gain_db": 24.0}}, {})
    _, wet = render(board, x, params, match_loudness=False)
    assert np.max(np.abs(wet)) <= 0.99 + 1e-6
    dry, wet = render(board, x, params, match_loudness=True)
    assert np.max(np.abs(wet)) <= 0.99 + 1e-6
    assert rms(wet) == pytest.approx(rms(dry), rel=1e-3)


def test_render_disabled_board_is_identity(board):
    x = synth_riff("clean_arpeggio", seconds=0.3)
    params = knobs_to_params(
        board, {"drive": {"gain_db": 20.0}}, dict.fromkeys(board.block_names, False)
    )
    dry, wet = render(board, x, params, match_loudness=False)
    np.testing.assert_allclose(wet, dry, atol=1e-6)


def test_describe_text_lists_physical_values_and_units(board):
    params = knobs_to_params(board, {"drive": {"gain_db": 6.0}}, {"level": False})
    lines = describe_text(board, params).splitlines()
    assert lines[0] == "| Block | Knob | Value |"
    assert "| drive | gain_db | 6 dB |" in lines
    assert "| eq | cutoff_hz | 1000 Hz |" in lines
    assert "| level | enabled | off |" in lines
    assert "| drive | enabled | on |" in lines


def test_prepare_input_uses_upload_or_falls_back_to_riff():
    sr = 22050
    riff = prepare_input(None, "single_notes", sr, max_seconds=15.0, riff_seconds=1.0)
    assert riff.shape == (sr,)
    stereo = (np.random.default_rng(0).uniform(-1, 1, (sr * 3, 2)) * 32767).astype(np.int16)
    clip = prepare_input((sr * 2, stereo), "single_notes", sr, max_seconds=1.0)
    assert clip.dtype == np.float32 and clip.shape == (sr,)
    # An empty upload also falls back to the riff.
    empty = prepare_input((sr, np.zeros(0, np.int16)), "power_chords", sr, 15.0, riff_seconds=0.5)
    assert empty.shape == (sr // 2,)


def test_to_int16_clips():
    out = to_int16(np.array([-2.0, 0.0, 0.5, 2.0], np.float32))
    assert out.dtype == np.int16
    assert out.tolist() == [-32767, 0, 16383, 32767]
