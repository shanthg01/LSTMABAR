import numpy as np
import pytest

import lstmabar.demo.app as app_module
from lstmabar.demo.presets import MANUAL, WHITEBOX_ERROR, WHITEBOX_NEEDS_NUMBA
from lstmabar.physics.kb import load_kb
from lstmabar.physics.whitebox.base import WHITEBOX, WhiteBoxModel
from tests.demo.fakes import FakeBoard

gr = pytest.importorskip("gradio")
if not hasattr(gr, "Blocks"):  # stale namespace dir after uninstalling gradio
    pytest.skip("gradio is not installed", allow_module_level=True)

from lstmabar.demo.app import build_app  # noqa: E402

N_BLOCKS = 3
N_SLIDERS = 4  # FakeBoard: drive gain, eq cutoff + mix, level gain
N_STALE = 6  # outputs cleared on a pedal switch (B, C message, readout, panel x3)


@pytest.fixture(scope="module")
def kb():
    full = load_kb()
    return {pid: full[pid] for pid in ("ts808", "rat")}


@pytest.fixture
def numba_ok(monkeypatch):
    """White-box output C is only offered with the compiled solver; fake it present."""
    monkeypatch.setattr(app_module, "whitebox_ready", lambda: True)


def _deps(app, name):
    return [f for f in app.fns.values() if getattr(f.fn, "__name__", "") == name]


def _fn(app, name):
    deps = _deps(app, name)
    assert len(deps) == 1, name
    return deps[0].fn


def _render_fn(app):
    return _fn(app, "on_render")


def _manual_controls(controls, n_dials=0):
    return [MANUAL, False, None, *([None] * n_dials), *controls]


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
    analyse = next(b for b in blocks if str(getattr(b, "label", "")).startswith("Archetype ana"))
    assert analyse.value is False  # off by default: M1 renders don't pay for pYIN
    for name in ("on_pedal", "on_dial_ts808", "on_dial_rat", "on_reset", "on_randomize"):
        _fn(app, name)


def test_dial_events_queue_and_cover_keyboard_edits(kb):
    """B1: dial moves during a running derive are queued (always_last), not dropped, and
    keyboard / number-box edits (change, no release) re-derive too. Helpers are private."""
    app = build_app(FakeBoard, kb=kb)
    for pid in kb:
        dep = _deps(app, f"on_dial_{pid}")[0]
        assert dep.trigger_mode == "always_last"
        events = {event for _, event in dep.targets}
        assert events == {"release", "change"}
        assert len(dep.targets) == 2 * len(kb[pid].pots)
    for name in ("on_pedal", "on_dial_ts808", "on_reset", "on_randomize"):
        assert _deps(app, name)[0].api_visibility == "private"
    assert _deps(app, "on_render")[0].api_visibility == "public"


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


def _n_dials(kb):
    return sum(len(p.pots) for p in kb.values())


def _dials(kb, **per_pedal):
    """Flat dial list (kb order); ``per_pedal`` gives a pedal's values, others are None."""
    out = []
    for pid, pedal in kb.items():
        out.extend(per_pedal.get(pid, [None] * len(pedal.pots)))
    return out


def _pedal_outputs(app, kb, pedal_id, dials=None):
    dials = dials if dials is not None else _dials(kb)
    out = _fn(app, "on_pedal")(pedal_id, *dials)
    n = len(kb)
    groups, preset_md, eq_note, wb = out[:n], out[n], out[n + 1], out[n + 2]
    stale = out[n + 3 : n + 3 + N_STALE]
    state = out[n + 3 + N_STALE]
    rest = out[n + 4 + N_STALE :]
    return {
        "groups": groups,
        "preset_md": preset_md,
        "eq_note": eq_note,
        "wb": wb,
        "stale": stale,
        "state": state,
        "checks": rest[:N_BLOCKS],
        "sliders": rest[N_BLOCKS:],
    }


def _values(updates):
    return [u["value"] for u in updates]


def test_picking_a_pedal_fills_sliders_and_locks_eq(kb, numba_ok):
    app = build_app(FakeBoard, kb=kb)
    o = _pedal_outputs(app, kb, "rat")
    assert [g["visible"] for g in o["groups"]] == [False, True]
    assert "ProCo RAT" in o["preset_md"] and o["eq_note"]["visible"] is True
    assert _values(o["checks"]) == [True, True, True]
    # EQ on/off box locked too (A5); FakeBoard blocks: drive, eq, level
    assert [c["interactive"] for c in o["checks"]] == [True, False, True]
    # FakeBoard order: drive.gain_db, eq.cutoff_hz, eq.mix, level.gain_db
    assert [s["interactive"] for s in o["sliders"]] == [True, False, False, True]
    assert o["sliders"][0]["value"] == 24.0  # RAT gain clamped to the toy Gain's +24 dB
    assert o["wb"]["visible"] is ("rat" in WHITEBOX)
    assert o["state"] == {"pedal": "rat", "dials": [None, None, None]}

    o = _pedal_outputs(app, kb, MANUAL)
    assert [g["visible"] for g in o["groups"]] == [False, False]
    assert o["preset_md"] == "" and o["eq_note"]["visible"] is False
    assert o["wb"]["visible"] is False and o["state"] is None
    assert all(u["interactive"] is True and "value" not in u for u in o["checks"] + o["sliders"])


def test_switching_pedal_clears_stale_outputs(kb):
    """A6: B, the C message, the readout and the panel from the previous render go away."""
    app = build_app(FakeBoard, kb=kb)
    for pid in (MANUAL, "ts808"):
        wet, wb_msg, readout, arche_md, bars, harms = _pedal_outputs(app, kb, pid)["stale"]
        assert wet is None and wb_msg == "" and readout == ""
        assert "Render" in arche_md and bars.empty and harms.empty


def test_dial_rederives_and_records_state(kb):
    app = build_app(FakeBoard, kb=kb)
    md, state, *updates = _fn(app, "on_dial_ts808")("ts808", 2.0, 5.0, 5.0)
    assert "Drive 2.0" in md
    assert state == {"pedal": "ts808", "dials": [2.0, 5.0, 5.0]}
    assert len(updates) == N_BLOCKS + N_SLIDERS
    # Queued derive arriving after the user switched pedal: no-op.
    out = _fn(app, "on_dial_ts808")("rat", 2.0, 5.0, 5.0)
    assert all(u == gr.update() for u in out)


def test_reset_with_pedal_restores_its_dials(kb):
    app = build_app(FakeBoard, kb=kb)
    out = _fn(app, "on_reset")("ts808")
    n_dials = _n_dials(kb)
    assert out[:3] == [5.0, 5.0, 5.0]  # ts808 dials back to the YAML defaults
    assert all(isinstance(u, dict) for u in out[3:n_dials])  # other pedals untouched
    assert "Ibanez TS808" in out[-2]
    assert out[-1] == {"pedal": "ts808", "dials": [5.0, 5.0, 5.0]}
    rand = _fn(app, "on_randomize")("rat")
    assert all(0.0 <= v <= 10.0 for v in rand[3:6])
    assert rand[-1]["dials"] == rand[3:6]


class _Halve(WhiteBoxModel):
    def process_volts(self, v):
        return 0.5 * v


class _Broken(WhiteBoxModel):
    def process_volts(self, v):
        raise RuntimeError(r"boom at C:\secret\path")


def _pedal_render(app, kb, pedal_id, analyse=False):
    dials = _dials(kb)
    o = _pedal_outputs(app, kb, pedal_id, dials)
    controls = [*dials, *_values(o["checks"]), *_values(o["sliders"])]
    return _render_fn(app)(None, "single_notes", True, pedal_id, analyse, o["state"], *controls)


@pytest.mark.parametrize("registered", [False, True])
def test_whitebox_availability_toggles_third_output(kb, monkeypatch, numba_ok, registered):
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


def test_whitebox_without_numba_is_disabled_with_a_message(kb, monkeypatch):
    """A2: no 30 s NumPy-solver renders; C stays hidden and says how to enable it."""
    calls = []
    monkeypatch.setitem(WHITEBOX, "rat", lambda p, k, sr: calls.append(1) or _Halve(sr))
    monkeypatch.setattr(app_module, "whitebox_ready", lambda: False)
    app = build_app(FakeBoard, kb=kb)
    assert _pedal_outputs(app, kb, "rat")["wb"]["visible"] is False
    _, _, wb, wb_msg, *_ = _pedal_render(app, kb, "rat")
    assert wb["visible"] is False and wb["value"] is None
    assert wb_msg == WHITEBOX_NEEDS_NUMBA and calls == []


def test_whitebox_error_is_generic_in_the_app(kb, monkeypatch, numba_ok):
    monkeypatch.setitem(WHITEBOX, "rat", lambda p, k, sr: _Broken(sr))
    app = build_app(FakeBoard, kb=kb)
    (_, dry), (_, wet), wb, wb_msg, *_ = _pedal_render(app, kb, "rat")
    assert wb["visible"] is True and wb["value"] is None
    assert wb_msg == WHITEBOX_ERROR and "secret" not in wb_msg
    assert np.std(wet) > 0  # the grey-box render still works


# --- B1: render vs dial/slider sync -----------------------------------------------------------


def _render_with(app, kb, dials_now, derived_dials, slider_edit=None):
    """Board derived at ``derived_dials``; the dials now read ``dials_now`` (TS808)."""
    o = _pedal_outputs(app, kb, "ts808", _dials(kb, ts808=derived_dials))
    sliders = _values(o["sliders"])
    if slider_edit is not None:
        i, v = slider_edit
        sliders[i] = v
    controls = [*_dials(kb, ts808=dials_now), *_values(o["checks"]), *sliders]
    return _render_fn(app)(None, "single_notes", False, "ts808", False, o["state"], *controls)


def _drive_gain(md):
    line = next(x for x in md.splitlines() if x.startswith("| drive | gain_db |"))
    return float(line.split("|")[3].split()[0])


def test_render_uses_fresh_preset_when_dials_moved_without_rederive(kb, monkeypatch, numba_ok):
    """A dial move whose derive hasn't reached the sliders is rendered, not called custom."""
    seen = []

    def record(p, knobs, sr):
        seen.append(dict(knobs))
        return _Halve(sr)

    monkeypatch.setitem(WHITEBOX, "ts808", record)
    app = build_app(kb=kb)  # real board: the toy Gain would clamp both drive settings
    out = _render_with(app, kb, dials_now=[1.0, 5.0, 5.0], derived_dials=[9.0, 5.0, 5.0])
    preset_md, board_md = out[5], out[4]
    assert "custom" not in preset_md and "Drive 1.0" in preset_md
    fresh = _render_with(app, kb, dials_now=[1.0, 5.0, 5.0], derived_dials=[1.0, 5.0, 5.0])
    assert _drive_gain(board_md) == pytest.approx(_drive_gain(fresh[4]))
    stale = _render_with(app, kb, dials_now=[9.0, 5.0, 5.0], derived_dials=[9.0, 5.0, 5.0])
    assert _drive_gain(board_md) != pytest.approx(_drive_gain(stale[4]))
    # A/B use the same dial state: white-box got the dials the grey-box readout shows.
    assert seen[0]["drive"] == pytest.approx(0.1)


def test_render_after_a_hand_edit_is_custom(kb, numba_ok, monkeypatch):
    seen = []
    monkeypatch.setitem(WHITEBOX, "ts808", lambda p, k, sr: seen.append(dict(k)) or _Halve(sr))
    app = build_app(FakeBoard, kb=kb)
    # level.gain_db (last slider) turned down by hand after deriving at the current dials.
    out = _render_with(app, kb, [7.0, 5.0, 5.0], [7.0, 5.0, 5.0], slider_edit=(3, -12.0))
    assert "custom (edited from Ibanez TS808 Tube Screamer)" in out[5]
    assert "| level | gain_db | -12 dB |" in out[4]
    assert seen[0]["drive"] == pytest.approx(0.7)  # white-box still follows the dials


def test_render_before_first_derive_is_not_custom(kb, monkeypatch):
    """No derived state for this pedal yet (pick still running): render the dials' preset."""
    monkeypatch.delitem(WHITEBOX, "rat", raising=False)
    app = build_app(FakeBoard, kb=kb)
    controls = [*_dials(kb), True, True, True, None, None, None, None]
    out = _render_fn(app)(None, "single_notes", False, "rat", False, None, *controls)
    assert "custom" not in out[5] and "ProCo RAT" in out[5]


def test_render_with_archetype_panel(kb, monkeypatch, numba_ok):
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
