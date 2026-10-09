import numpy as np
import pytest

from tests.demo.fakes import FakeBoard

gr = pytest.importorskip("gradio")
if not hasattr(gr, "Blocks"):  # stale namespace dir after uninstalling gradio
    pytest.skip("gradio is not installed", allow_module_level=True)

from lstmabar.demo.app import build_app  # noqa: E402


def test_build_app_with_fake_board():
    app = build_app(FakeBoard)
    assert isinstance(app, gr.Blocks)
    labels = {getattr(b, "label", None) for b in app.blocks.values()}
    # One slider per ParamSpec, generated from the board (not hard-coded).
    assert {"gain_db (dB)", "cutoff_hz (Hz), log taper", "mix", "A: Dry", "B: Wet"} <= labels
    for b in app.blocks.values():
        if isinstance(b, gr.Slider):
            assert (b.maximum - b.minimum) / b.step <= 1000, b.label


def test_render_callback_end_to_end():
    app = build_app(FakeBoard)
    fns = [f.fn for f in app.fns.values() if getattr(f.fn, "__name__", "") == "on_render"]
    assert len(fns) == 1
    board = FakeBoard()
    n_sliders = sum(len(board.blocks[n].param_specs) for n in board.block_names)
    controls = [True] * len(board.block_names) + [None] * n_sliders
    (sr_d, dry), (sr_w, wet), md = fns[0](None, "single_notes", True, *controls)
    assert sr_d == sr_w == board.sample_rate
    assert dry.shape == wet.shape and dry.dtype.name == "int16"
    assert "| drive | gain_db |" in md


def _render_fn(app):
    return next(f.fn for f in app.fns.values() if getattr(f.fn, "__name__", "") == "on_render")


def test_render_callback_log_slider_uses_position():
    app = build_app(FakeBoard)
    board = FakeBoard()
    # drive on, eq on, level on; drive gain, eq cutoff (position), eq mix, level gain
    controls = [True, True, True, 0.0, 1.0, 0.5, 0.0]
    *_, md = _render_fn(app)(None, "single_notes", False, *controls)
    assert board.blocks["eq"].param_specs[0].max == 10000.0
    assert "| eq | cutoff_hz | 10000 Hz |" in md


def test_render_callback_rejects_bad_upload():
    app = build_app(FakeBoard)
    controls = [True, True, True, None, None, None, None]
    with pytest.raises(gr.Error, match="empty"):
        _render_fn(app)((44100, np.zeros(0, np.int16)), "single_notes", True, *controls)
