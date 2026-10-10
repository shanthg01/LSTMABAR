import numpy as np
import pytest

from lstmabar.demo.presets import MANUAL, WHITEBOX_ERROR
from lstmabar.physics.kb import load_kb
from lstmabar.physics.whitebox.base import WHITEBOX, WhiteBoxModel
from tests.demo.fakes import FakeBoard

gr = pytest.importorskip("gradio")
if not hasattr(gr, "Blocks"):  # stale namespace dir after uninstalling gradio
    pytest.skip("gradio is not installed", allow_module_level=True)

from lstmabar.demo.app import build_app  # noqa: E402

N_BLOCKS = 3
N_SLIDERS = 4  # FakeBoard: drive gain, eq cutoff + mix, level gain


@pytest.fixture(scope="module")
def kb():
    full = load_kb()
    return {pid: full[pid] for pid in ("ts808", "rat")}


def _fn(app, name):
    fns = [f.fn for f in app.fns.values() if getattr(f.fn, "__name__", "") == name]
    assert len(fns) == 1, name
    return fns[0]


def _render_fn(app):
    return _fn(app, "on_render")


def _manual_controls(controls, n_dials=0):
    return [MANUAL, False, *([None] * n_dials), *controls]


def test_build_app_with_fake_board():
    app = build_app(FakeBoard, kb={})
    assert isinstance(app, gr.Blocks)
    labels = {getattr(b, "label", None) for b in app.blocks.values()}
    # One slider per ParamSpec, generated from the board (not hard-coded).
    assert {"gain_db (dB)", "cutoff_hz (Hz), log taper", "mix", "A: Dry", "B: Wet"} <= labels
    for b in app.blocks.values():
        if isinstance(b, gr.Slider):
            assert (b.maximum - b.minimum) / b.step <= 1000, b.label


def test_build_app_has_pedal_picker_dials_whitebox_and_panel(kb):
    app = build_app(FakeBoard, kb=kb)
    blocks = list(app.blocks.values())
    labels = {getattr(b, "label", None) for b in blocks}
    assert {"Drive", "Tone", "Level", "Distortion", "Filter", "Volume"} <= labels
    picker = next(b for b in blocks if getattr(b, "label", "") == "Pedal (circuit preset)")
    assert [c[1] for c in picker.choices] == [MANUAL, "ts808", "rat"]
    wb = next(b for b in blocks if str(getattr(b, "label", "")).startswith("C: White-box"))
    assert isinstance(wb, gr.Audio) and wb.visible is False
    assert any(isinstance(b, gr.BarPlot) for b in blocks)
    assert any(isinstance(b, gr.LinePlot) for b in blocks)
    for name in ("on_pedal", "on_dial_ts808", "on_dial_rat", "on_reset", "on_randomize"):
        _fn(app, name)


def test_render_callback_end_to_end():
    app = build_app(FakeBoard, kb={})
    board = FakeBoard()
    controls = [True] * len(board.block_names) + [None] * N_SLIDERS
    out = _render_fn(app)(None, "single_notes", True, *_manual_controls(controls))
    (sr_d, dry), (sr_w, wet), wb, wb_msg, md, preset_md, arche_md, bars, harms = out
    assert sr_d == sr_w == board.sample_rate
    assert dry.shape == wet.shape and dry.dtype.name == "int16"
    assert "| drive | gain_db |" in md
    assert wb["visible"] is False and wb_msg == "" and preset_md == ""
    assert "off" in arche_md and bars.empty and harms.empty


def test_render_callback_log_slider_uses_position():
    app = build_app(FakeBoard, kb={})
    board = FakeBoard()
    # drive on, eq on, level on; drive gain, eq cutoff (position), eq mix, level gain
    controls = [True, True, True, 0.0, 1.0, 0.5, 0.0]
    out = _render_fn(app)(None, "single_notes", False, *_manual_controls(controls))
    assert board.blocks["eq"].param_specs[0].max == 10000.0
    assert "| eq | cutoff_hz | 10000 Hz |" in out[4]


def test_render_callback_rejects_bad_upload():
    app = build_app(FakeBoard, kb={})
    controls = [True, True, True, None, None, None, None]
    with pytest.raises(gr.Error, match="empty"):
        _render_fn(app)(
            (44100, np.zeros(0, np.int16)), "single_notes", True, *_manual_controls(controls)
        )


# --- Pedal presets --------------------------------------------------------------------------


def _pedal_outputs(app, kb, pedal_id, dials=None):
    dials = dials if dials is not None else [None] * sum(len(p.pots) for p in kb.values())
    out = _fn(app, "on_pedal")(pedal_id, *dials)
    n = len(kb)
    groups, preset_md, eq_note, wb = out[:n], out[n], out[n + 1], out[n + 2]
    rest = out[n + 3 :]
    return groups, preset_md, eq_note, wb, rest[:N_BLOCKS], rest[N_BLOCKS:]


def test_picking_a_pedal_fills_sliders_and_locks_eq(kb):
    app = build_app(FakeBoard, kb=kb)
    groups, preset_md, eq_note, wb, checks, sliders = _pedal_outputs(app, kb, "rat")
    assert [g["visible"] for g in groups] == [False, True]
    assert "ProCo RAT" in preset_md and eq_note["visible"] is True
    assert [c["value"] for c in checks] == [True, True, True]
    # FakeBoard order: drive.gain_db, eq.cutoff_hz, eq.mix, level.gain_db
    assert [s["interactive"] for s in sliders] == [True, False, False, True]
    assert sliders[0]["value"] == 24.0  # RAT gain clamped to the toy Gain's +24 dB
    assert wb["visible"] is ("rat" in WHITEBOX)

    groups, preset_md, eq_note, wb, checks, sliders = _pedal_outputs(app, kb, MANUAL)
    assert [g["visible"] for g in groups] == [False, False]
    assert preset_md == "" and eq_note["visible"] is False and wb["visible"] is False
    assert all(s["interactive"] is True and "value" not in s for s in sliders)


def test_dial_release_rederives(kb):
    app = build_app(FakeBoard, kb=kb)
    md, *updates = _fn(app, "on_dial_ts808")(2.0, 5.0, 5.0)
    assert "Drive 2.0" in md
    assert len(updates) == N_BLOCKS + N_SLIDERS


def test_reset_with_pedal_restores_its_dials(kb):
    app = build_app(FakeBoard, kb=kb)
    out = _fn(app, "on_reset")("ts808")
    n_dials = sum(len(p.pots) for p in kb.values())
    assert out[:3] == [5.0, 5.0, 5.0]  # ts808 dials back to the YAML defaults
    assert all(isinstance(u, dict) for u in out[3:n_dials])  # other pedals untouched
    assert "Ibanez TS808" in out[-1]
    rand = _fn(app, "on_randomize")("rat")
    assert all(0.0 <= v <= 10.0 for v in rand[3:6])


class _Halve(WhiteBoxModel):
    def process_volts(self, v):
        return 0.5 * v


class _Broken(WhiteBoxModel):
    def process_volts(self, v):
        raise RuntimeError(r"boom at C:\secret\path")


def _pedal_render(app, kb, pedal_id, analyse=False):
    dials = [None] * sum(len(p.pots) for p in kb.values())
    _, _, _, _, checks, sliders = _pedal_outputs(app, kb, pedal_id, dials)
    controls = [*dials, *(c["value"] for c in checks), *(s["value"] for s in sliders)]
    return _render_fn(app)(None, "single_notes", True, pedal_id, analyse, *controls)


@pytest.mark.parametrize("registered", [False, True])
def test_whitebox_availability_toggles_third_output(kb, monkeypatch, registered):
    monkeypatch.delitem(WHITEBOX, "rat", raising=False)
    if registered:
        monkeypatch.setitem(WHITEBOX, "rat", lambda p, k, sr: _Halve(sr))
    app = build_app(FakeBoard, kb=kb)
    _, wet, wb, wb_msg, md, preset_md, *_ = _pedal_render(app, kb, "rat")
    assert wb["visible"] is registered
    assert (wb["value"] is not None) is registered
    if registered:
        sr, audio = wb["value"]
        assert audio.dtype.name == "int16" and len(audio) == len(wet[1])
        assert "White-box circuit simulation of ProCo RAT" in wb_msg
    assert "**Grey-box:** ProCo RAT" in preset_md  # untouched sliders: exact preset


def test_whitebox_error_is_generic_in_the_app(kb, monkeypatch):
    monkeypatch.setitem(WHITEBOX, "rat", lambda p, k, sr: _Broken(sr))
    app = build_app(FakeBoard, kb=kb)
    (_, dry), (_, wet), wb, wb_msg, *_ = _pedal_render(app, kb, "rat")
    assert wb["visible"] is True and wb["value"] is None
    assert wb_msg == WHITEBOX_ERROR and "secret" not in wb_msg
    assert np.std(wet) > 0  # the grey-box render still works


def test_tweaked_sliders_render_as_custom(kb, monkeypatch):
    monkeypatch.delitem(WHITEBOX, "rat", raising=False)
    app = build_app(FakeBoard, kb=kb)
    dials = [None] * sum(len(p.pots) for p in kb.values())
    _, _, _, _, checks, sliders = _pedal_outputs(app, kb, "rat", dials)
    values = [s["value"] for s in sliders]
    values[-1] = -12.0  # turn the level block down
    controls = [*dials, *(c["value"] for c in checks), *values]
    out = _render_fn(app)(None, "single_notes", True, "rat", False, *controls)
    assert "custom (edited from ProCo RAT)" in out[5]
    assert "| level | gain_db | -12 dB |" in out[4]


def test_render_with_archetype_panel(kb, monkeypatch):
    import lstmabar.analysis.harmonics as harmonics

    real_profile = harmonics.harmonic_profile

    def fake_track(x, sr):
        return None

    def fixed_f0_profile(seg, sr, n_harmonics, track, max_frames):
        return real_profile(seg, sr, n_harmonics=n_harmonics, f0=110.0, max_frames=max_frames)

    monkeypatch.setattr(harmonics, "track_f0", fake_track)
    monkeypatch.setattr(harmonics, "harmonic_profile", fixed_f0_profile)
    monkeypatch.setitem(WHITEBOX, "rat", lambda p, k, sr: _Halve(sr))
    app = build_app(FakeBoard, kb=kb)
    *_, arche_md, bars, harms = _pedal_render(app, kb, "rat", analyse=True)
    assert "| | dry | grey-box | white-box |" in arche_md
    assert set(bars["signal"]) == {"dry", "grey-box", "white-box"}
    assert list(harms.columns) == ["signal", "harmonic", "dbc"] and len(harms) > 0
