import pytest

from tests.demo.fakes import FakeBoard

gr = pytest.importorskip("gradio")

from lstmabar.demo.app import _slider_step, build_app  # noqa: E402


def test_build_app_with_fake_board():
    app = build_app(FakeBoard)
    assert isinstance(app, gr.Blocks)
    labels = {getattr(b, "label", None) for b in app.blocks.values()}
    # One slider per ParamSpec, generated from the board (not hard-coded).
    assert {"gain_db (dB)", "cutoff_hz (Hz)", "mix", "A: Dry", "B: Wet"} <= labels


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


def test_slider_step_is_fine_for_log_knobs():
    board = FakeBoard()
    cutoff = board.blocks["eq"].param_specs[0]
    assert 0 < _slider_step(cutoff) <= cutoff.min / 100
