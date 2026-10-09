"""The demo's render path against the real ``default_pedalboard()`` (no gradio needed)."""

import numpy as np
import pytest

from lstmabar.audio import synth_riff
from lstmabar.demo.render import (
    default_knobs,
    describe_text,
    iter_specs,
    knobs_to_params,
    physical_from_slider,
    random_knobs,
    render,
    slider_config,
    slider_from_physical,
)
from lstmabar.dsp.pedalboard import default_pedalboard

SR = 44100


@pytest.fixture(scope="module")
def board():
    return default_pedalboard(SR).eval()


@pytest.fixture(scope="module")
def clip():
    return synth_riff("power_chords", seconds=1.0, sample_rate=SR)


def _check(board, clip, values, enabled, match):
    params = knobs_to_params(board, values, enabled)
    dry, wet = render(board, clip, params, match_loudness=match)
    assert dry.shape == wet.shape == clip.shape
    assert wet.dtype == np.float32
    assert np.all(np.isfinite(wet))
    assert np.max(np.abs(wet)) <= 0.99 + 1e-6
    text = describe_text(board, params)
    for name, spec in iter_specs(board):
        assert f"| {name} | {spec.name} |" in text
    for name in board.block_names:
        assert f"| {name} | enabled |" in text
    return wet, text


@pytest.mark.parametrize("match", [False, True])
def test_real_board_defaults(board, clip, match):
    wet, text = _check(board, clip, default_knobs(board), {}, match)
    assert np.std(wet) > 0
    assert "| drive | gain_db | 20 dB |" in text
    assert "| eq | mid_hz | 800 Hz |" in text


@pytest.mark.parametrize("seed", range(3))
def test_real_board_random_knobs(board, clip, seed):
    rng = np.random.default_rng(seed)
    enabled = {name: bool(rng.uniform() < 0.7) for name in board.block_names}
    _check(board, clip, random_knobs(board, rng), enabled, match=bool(seed % 2))


def test_real_board_extreme_drive_is_peak_safe(board, clip):
    values = default_knobs(board)
    values["drive"].update(gain_db=60.0, level_db=6.0, asymmetry=1.0, bias=0.25)
    values["eq"].update(low_db=12.0, mid_db=12.0, high_db=12.0)
    values["compressor"].update(makeup_db=24.0)
    _check(board, clip, values, {}, match=False)


def test_real_board_sliders_are_bounded(board):
    for _, spec in iter_specs(board):
        cfg = slider_config(spec)
        assert (cfg["maximum"] - cfg["minimum"]) / cfg["step"] <= 1000, spec.name
        for v in (spec.min, spec.default, spec.max):
            assert physical_from_slider(spec, slider_from_physical(spec, v)) == pytest.approx(
                v, rel=1e-6
            )
