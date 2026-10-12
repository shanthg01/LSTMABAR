"""P4 R4: white-box calibration tables (residual lookup, level offset, persistence, run)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest
import torch

from lstmabar.config import load_config
from lstmabar.data import calibrate
from lstmabar.data.calibrate import CalibrationTable, Label, board_jump, grid_values
from lstmabar.data.layout import DataLayout
from lstmabar.physics.derive import BLOCK_SPECS

GRID = [0.0, 0.5, 1.0]
COMMITTED = Path(__file__).resolve().parents[2] / "data" / "calibration"


def _spec(block: str, knob: str):
    return next(s for s in BLOCK_SPECS[block] if s.name == knob)


def _norm(block: str, knob: str, v: float) -> float:
    return float(_spec(block, knob).normalize(torch.tensor(v, dtype=torch.float64)))


def _phys(block: str, knob: str, u: float) -> float:
    return float(_spec(block, knob).denormalize(torch.tensor(u, dtype=torch.float64)))


def _derived(i: float, j: float) -> dict:
    """Synthetic derived label: ``eq.mid_hz`` switches branch along tone (jump 0.3 > 0.25)."""
    b = {blk: {s.name: s.default_normalized for s in specs} for blk, specs in BLOCK_SPECS.items()}
    for blk in b:
        b[blk]["enabled"] = 0.0 if blk == "compressor" else 1.0
    b["drive"]["gain_db"] = 0.1 + 0.1 * i
    b["eq"]["low_db"] = 0.4
    b["eq"]["mid_hz"] = 0.1 + 0.3 * j
    b["drive"]["level_db"] = _norm("drive", "level_db", -20.0)
    return b


def _board(i: int, j: int) -> dict:
    """Synthetic fit = derived + a smooth residual (gain_db +0.1, low_db +0.05 j)."""
    b = _derived(i, j)
    b["drive"]["gain_db"] += 0.1
    b["eq"]["low_db"] += 0.05 * j
    return b


def _table(boards=None, derived_boards=None, **kw) -> CalibrationTable:
    boards = boards or [[_board(i, j) for j in range(3)] for i in range(3)]
    derived_boards = derived_boards or [[_derived(i, j) for j in range(3)] for i in range(3)]
    args = dict(
        pedal_id="ts808",
        gain_knob="drive",
        tone_knob="tone",
        level_knob="level",
        gain_values=list(GRID),
        tone_values=list(GRID),
        boards=boards,
        fit_error_db=[[1.0] * 3 for _ in range(3)],
        max_jump=0.25,
        meta={"note": "synthetic"},
        level_default=0.5,
        derived_error_db=[[2.0] * 3 for _ in range(3)],
        derived_boards=derived_boards,
    )
    args.update(kw)
    return CalibrationTable(**args)


def _approx_board(a: dict, b: dict, abs_tol: float = 1e-12) -> None:
    assert set(a) == set(b)
    for blk in a:
        assert set(a[blk]) == set(b[blk])
        for k in a[blk]:
            assert a[blk][k] == pytest.approx(b[blk][k], abs=abs_tol), f"{blk}.{k}"


def test_grid_values():
    assert grid_values(5) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert grid_values(1) == [0.5]


def test_lookup_exact_at_grid_points():
    """At a grid point, derived + residual = the fit."""
    t = _table()
    for i, g in enumerate(GRID):
        for j, tone in enumerate(GRID):
            lab = t.lookup({"drive": g, "tone": tone, "level": 0.5}, derived=_derived(i, j))
            assert isinstance(lab, Label)
            assert lab.source == "calibrated"
            _approx_board(lab.board, t.boards[i][j])
            assert lab.clamp_notes == []


def test_lookup_midpoint_is_derived_plus_mean_residual():
    t = _table()
    # The fitted boards jump 0.3 along tone (the derived branch switch): the pre-residual rule
    # would fall back; the residual is smooth, so the cell interpolates.
    assert t.board_cell_jump(0, 1, 1, 2) > t.max_jump
    assert t.cell_jump(0, 1, 1, 2) <= t.max_jump
    d = _derived(0.5, 1.5)  # the derived label at the off-grid setting
    d["eq"]["high_db"] = 0.123  # arbitrary: the lookup must use the passed board
    lab = t.lookup({"drive": 0.25, "tone": 0.75}, derived=d)
    assert lab.source == "calibrated"
    corners = [(i, j) for i in (0, 1) for j in (1, 2)]
    for blk, kv in lab.board.items():
        for k, v in kv.items():
            if k == "enabled":
                continue
            mean_r = sum(t.residual(i, j)[blk][k] for i, j in corners) / 4
            assert v == pytest.approx(d[blk][k] + mean_r, abs=1e-12)
    assert lab.board["drive"]["gain_db"] == pytest.approx(0.15 + 0.1)
    assert lab.board["eq"]["mid_hz"] == pytest.approx(0.1 + 0.3 * 1.5)  # derived's, not a blend
    assert lab.board["eq"]["low_db"] == pytest.approx(0.4 + 0.05 * 1.5)
    assert lab.board["eq"]["high_db"] == pytest.approx(0.123)


def test_lookup_falls_back_to_nearest_residual_on_large_jump():
    boards = [[_board(i, j) for j in range(3)] for i in range(3)]
    boards[2][2]["eq"]["mid_db"] = 0.95  # residual jump > 0.25 inside the upper-right cell only
    t = _table(boards)
    assert board_jump(t.residual(1, 1), t.residual(2, 2)) > t.max_jump
    d = _derived(1.2, 1.8)
    lab = t.lookup({"drive": 0.6, "tone": 0.9}, derived=d)  # cell [0.5,1]², nearest (1, 2)
    assert lab.source == "calibrated_nearest"
    r = t.residual(1, 2)
    for blk, kv in lab.board.items():
        for k, v in kv.items():
            if k != "enabled":
                assert v == pytest.approx(d[blk][k] + r[blk][k], abs=1e-12)
    # Other cells still interpolate.
    assert t.lookup({"drive": 0.2, "tone": 0.2}, derived=_derived(0, 0)).source == "calibrated"


def test_lookup_clips_to_unit_range():
    boards = [[_board(i, j) for j in range(3)] for i in range(3)]
    for row in boards:
        for b in row:
            b["eq"]["mid_db"] = 0.95
            b["eq"]["high_db"] = 0.05
    dboards = [[_derived(i, j) for j in range(3)] for i in range(3)]
    for row in dboards:
        for b in row:
            b["eq"]["mid_db"] = 0.75  # residual +0.2
            b["eq"]["high_db"] = 0.25  # residual -0.2
    t = _table(boards, dboards)
    d = _derived(0.5, 0.5)
    d["eq"]["mid_db"], d["eq"]["high_db"] = 0.9, 0.1
    lab = t.lookup({"drive": 0.25, "tone": 0.25}, derived=d)
    assert lab.board["eq"]["mid_db"] == 1.0
    assert lab.board["eq"]["high_db"] == 0.0


def test_gates_are_copied_not_interpolated():
    boards = [[_board(i, j) for j in range(3)] for i in range(3)]
    boards[0][0]["eq"]["enabled"] = 0.0
    t = _table(boards)
    near = t.lookup({"drive": 0.1, "tone": 0.1}, derived=_derived(0.2, 0.2))  # nearest (0, 0)
    far = t.lookup({"drive": 0.4, "tone": 0.4}, derived=_derived(0.8, 0.8))  # nearest (1, 1)
    assert near.source == far.source == "calibrated"
    assert near.board["eq"]["enabled"] == 0.0
    assert far.board["eq"]["enabled"] == 1.0
    assert near.board["compressor"]["enabled"] == 0.0


def test_level_offset_and_clamp_note():
    t = _table()
    # ts808 level pot is audio taper through (0.5, 0.1): full up = +20 dB vs default.
    assert calibrate.level_offset_db("ts808", "level", 1.0, 0.5) == pytest.approx(20.0)
    # The renderer passes the derived board at the actual level (here +20 dB, unclamped);
    # the lookup moves it back to the reference level, adds the residual, then the offset.
    d = _derived(0, 0)
    d["drive"]["level_db"] = _norm("drive", "level_db", -20.0 + 20.0)
    up = t.lookup({"drive": 0.0, "tone": 0.0, "level": 1.0}, derived=d)
    assert _phys("drive", "level_db", up.board["drive"]["level_db"]) == pytest.approx(0.0)
    assert up.clamp_notes == []
    # Only drive.level_db changes relative to the fit.
    base = t.boards[0][0]
    for k, v in base["drive"].items():
        if k != "level_db":
            assert up.board["drive"][k] == pytest.approx(v, abs=1e-12)

    boards = [[_board(i, j) for j in range(3)] for i in range(3)]
    for row in boards:
        for b in row:
            b["drive"]["level_db"] = _norm("drive", "level_db", -10.0)
    d = _derived(0, 0)
    d["drive"]["level_db"] = _norm("drive", "level_db", 0.0)
    loud = _table(boards).lookup({"drive": 0.0, "tone": 0.0, "level": 1.0}, derived=d)
    assert loud.board["drive"]["level_db"] == pytest.approx(1.0)
    assert len(loud.clamp_notes) == 1
    assert loud.clamp_notes[0].startswith("drive.level_db 10 clamped to 6 dB: ")


def test_level_clamped_derived_uses_reference(monkeypatch):
    """A passed board whose level the derivation clamped is recomputed at the reference."""
    t = _table()
    calls = []

    def fake_derived_board(pid, knobs, ctx):
        calls.append(dict(knobs))
        return _derived(0, 0)

    monkeypatch.setattr(calibrate, "derived_board", fake_derived_board)
    d = _derived(0, 0)
    d["drive"]["level_db"] = 0.0  # clamped at Drive's floor (level knob off)
    off = t.lookup({"drive": 0.0, "tone": 0.0, "level": 0.0}, derived=d)
    assert calls == [{"drive": 0.0, "tone": 0.0, "level": 0.5}]
    assert off.board["drive"]["level_db"] == 0.0
    assert off.clamp_notes[0].startswith("drive.level_db -inf clamped to -36 dB: ")
    # Without a passed board the same (slow) path is taken.
    t.lookup({"drive": 0.0, "tone": 0.0, "level": 0.0})
    assert calls[-1] == {"drive": 0.0, "tone": 0.0, "level": 0.5}


@pytest.mark.parametrize(("pid", "level"), [("ts808", 0.8), ("ts808", 0.2), ("ds1", 0.9)])
def test_level_offset_matches_derivation(pid, level):
    """The analytic offset equals the derivation's post-clip gain difference."""
    from lstmabar.physics.derive import derive
    from lstmabar.physics.kb import load_kb

    pedal = load_kb()[pid]
    a = derive(pedal, {"level": level}).info["post_gain_db"]
    b = derive(pedal, {"level": 0.5}).info["post_gain_db"]
    assert calibrate.level_offset_db(pid, "level", level, 0.5) == pytest.approx(a - b, abs=1e-9)


@pytest.mark.parametrize(
    "knobs",
    [
        {"drive": 0.3, "tone": 0.7, "level": 0.5},  # level at the fitted default
        {"drive": 0.6, "tone": 0.2, "level": 0.8},  # analytic level move
        {"drive": 0.9, "tone": 0.4, "level": 0.01},  # level clamped by derive: recomputed
    ],
)
def test_passed_and_computed_derived_agree(knobs):
    """On the committed ts808 table: ``lookup(knobs)`` == ``lookup(knobs, derived=...)``."""
    t = CalibrationTable.load(COMMITTED / "ts808.json")
    d = calibrate.derived_board("ts808", knobs, t.derive_context())
    a = t.lookup(knobs)
    b = t.lookup(knobs, derived=d)
    assert a.source == b.source
    assert a.clamp_notes == b.clamp_notes
    _approx_board(a.board, b.board, abs_tol=1e-6)


def test_committed_tables_exact_at_grid_points():
    for pid in ("ts808", "ds1"):
        t = CalibrationTable.load(COMMITTED / f"{pid}.json")
        assert t.derived_boards, f"{pid}: run `lstmabar data calibrate --report-only`"
        for i, g in enumerate(t.gain_values):
            for j, tone in enumerate(t.tone_values):
                knobs = {t.gain_knob: g, t.tone_knob: tone, t.level_knob: t.level_default}
                lab = t.lookup(knobs, derived=t.derived_boards[i][j])
                _approx_board(lab.board, t.boards[i][j], abs_tol=1e-9)


def test_old_table_without_derived_boards():
    data = _table().to_dict()
    del data["derived_boards"]  # a table written before residual lookup
    t = CalibrationTable.from_dict(data)
    assert t.derived_boards == []
    with pytest.raises(ValueError, match="--report-only"):
        t.lookup({"drive": 0.5, "tone": 0.5}, derived=_derived(1, 1))


def test_lookup_rejects_bad_knobs():
    t = _table()
    with pytest.raises(KeyError):
        t.lookup({"tone": 0.5}, derived=_derived(0, 0))
    with pytest.raises(ValueError):
        t.lookup({"drive": 1.5, "tone": 0.5}, derived=_derived(0, 0))


def test_save_load_round_trip(tmp_path):
    t = _table(meta={"git": {"sha": "abc", "dirty": False}, "f0s_hz": [82.41, 146.83]})
    t.boards[1][1]["drive"]["gain_db"] = 0.1 + 0.2  # non-representable decimal survives
    path = tmp_path / "cal" / "ts808.json"
    t.save(path)
    back = CalibrationTable.load(path)
    assert back == t
    assert json.loads(path.read_text(encoding="utf-8"))["pedal_id"] == "ts808"


def test_cell_centres():
    t = _table(gain_values=grid_values(5))
    assert calibrate.cell_centres(t) == [
        (g, tone) for g in (0.125, 0.375, 0.625, 0.875) for tone in (0.25, 0.75)
    ]


def _tmp_cfg(tmp_path):
    return load_config(
        "configs/data/v1.yaml",
        [f"data.calibration_dir={tmp_path / 'cal'}", f"calibration.report_dir={tmp_path / 'rep'}"],
    )


def test_run_writes_tables_and_report(tmp_path, monkeypatch):
    """``run`` plumbing (tables, report numbers = table numbers) with the fit stubbed out."""
    boards = [[_board(i, j) for j in range(3)] for i in range(3)]
    boards[2][2]["eq"]["mid_db"] = 0.95

    def fake(pid, gains, tones, fcfg, ctx, max_jump=0.25, log=print):
        assert (gains, tones) == (grid_values(5), grid_values(3))
        assert fcfg.sample_rate == 44100 and fcfg.mrstft_weight == 0.0
        t = _table(
            boards,
            pedal_id=pid,
            gain_values=GRID,
            tone_values=GRID,
            fit_error_db=[[1.0, 2.0, 3.0]] * 3,
        )
        cells = [
            [{"best_start": "derived", "starts": {"derived": 1.0}} for _ in GRID] for _ in GRID
        ]
        return t, calibrate.pedal_report(t, cells, 1.5)

    monkeypatch.setattr(calibrate, "calibrate_pedal", fake)
    cfg = _tmp_cfg(tmp_path)
    assert calibrate.run(cfg, argparse.Namespace(quick=False, pedals=None)) == 0
    for pid in ("ts808", "ds1"):
        t = CalibrationTable.load(DataLayout.from_config(cfg).calibration_table(pid))
        assert t.meta["config"]["gain_points"] == 5
        assert t.derived_boards
    rep = json.loads((tmp_path / "rep" / "calibration.json").read_text(encoding="utf-8"))
    p = rep["pedals"]["ts808"]
    assert p["fit_mean_db"] == pytest.approx(2.0)
    assert p["derived_mean_db"] == pytest.approx(2.0)
    assert p["fit_worse_than_derived"] == 3
    assert p["fallback_cells"] == 1  # residual rule: only the mid_db outlier cell
    assert p["board_fallback_cells"] == 4  # fit-board rule: every cell (tone branch switch)
    assert p["residual_tone_jumps"][0][0] == pytest.approx(0.05)
    md = (tmp_path / "rep" / "calibration.md").read_text(encoding="utf-8")
    assert "| ts808 | **2.00** | 2.00 | 3/9 |" in md
    assert "Fallback cells of the residual lookup (nearest grid point's residual): 1 of 4" in md


def test_report_only_fills_derived_boards(tmp_path, monkeypatch):
    def fake(pid, gains, tones, fcfg, ctx, max_jump=0.25, log=None):
        t = _table(pedal_id=pid, gain_values=GRID, tone_values=GRID)
        t = CalibrationTable.from_dict({**t.to_dict(), "derived_boards": []})  # old table
        cells = [[{"best_start": "derived", "starts": {"derived": 1.0}} for _ in GRID]] * 3
        return t, calibrate.pedal_report(t, cells, 1.5)

    monkeypatch.setattr(calibrate, "calibrate_pedal", fake)
    cfg = _tmp_cfg(tmp_path)
    calibrate.run(cfg, argparse.Namespace(quick=False, pedals=["ts808"]))
    rep = json.loads((tmp_path / "rep" / "calibration.json").read_text(encoding="utf-8"))
    assert rep["pedals"]["ts808"]["fallback_cells"] is None  # no residuals without derived
    md_before = (tmp_path / "rep" / "calibration.md").read_text(encoding="utf-8")
    path = DataLayout.from_config(cfg).calibration_table("ts808")
    assert CalibrationTable.load(path).derived_boards == []

    monkeypatch.setattr(
        calibrate,
        "table_derived_boards",
        lambda table, ctx: [[_derived(i, j) for j in range(3)] for i in range(3)],
    )
    assert calibrate.run(cfg, argparse.Namespace(report_only=True)) == 0
    t = CalibrationTable.load(path)
    assert t.derived_boards == [[_derived(i, j) for j in range(3)] for i in range(3)]
    assert t.boards == _table().boards  # fits untouched
    rep = json.loads((tmp_path / "rep" / "calibration.json").read_text(encoding="utf-8"))
    p = rep["pedals"]["ts808"]
    assert rep["report_rebuilt"] is True
    assert len(p["residual_tone_jumps"]) == 3 and len(p["residual_tone_jumps"][0]) == 2
    assert p["max_derived_tone_jump"] == pytest.approx(0.3)
    assert p["fallback_cells"] == 0
    assert p["fit_error_db"] == [[1.0] * 3] * 3  # stored numbers kept
    md = (tmp_path / "rep" / "calibration.md").read_text(encoding="utf-8")
    assert md != md_before and "--report-only" in md


def test_offgrid_check_stubbed(monkeypatch):
    """``offgrid_check`` plumbing: three labels per point, the residual one from lookup."""
    import numpy as np

    from lstmabar.physics import fidelity

    t = _table()
    monkeypatch.setattr(calibrate, "derived_board", lambda pid, knobs, ctx: _derived(0.5, 0.5))
    monkeypatch.setattr(fidelity, "_white", lambda *a: (np.zeros(1), np.zeros(1)))
    monkeypatch.setattr(fidelity, "_amps", lambda *a: None)
    monkeypatch.setattr(fidelity, "make_signals", lambda cfg: None)
    seen = []

    def fake_eval(board, fk, u, sig, white_amps, white_riff, fcfg):
        seen.append(u.clone())
        return [{"mean_db": float(k)} for k in range(u.shape[0])]

    monkeypatch.setattr(fidelity, "evaluate", fake_eval)
    fcfg = calibrate.fidelity_config(load_config("configs/data/v1.yaml"), quick=True)
    out = calibrate.offgrid_check(t, fcfg, None, points=[(0.25, 0.25)], log=lambda m: None)
    (row,) = out["points"]
    assert (row["derived_db"], row["nearest_fit_db"], row["residual_db"]) == (0.0, 1.0, 2.0)
    assert row["nearest_fit_corner_mean_db"] == pytest.approx((3 + 4 + 5 + 6) / 4)
    assert out["residual_beats_nearest"] == 0 and out["residual_beats_derived"] == 0
    fk = fidelity.fit_knobs(fidelity._board(fcfg))
    lab = t.lookup({"drive": 0.25, "tone": 0.25, "level": 0.5}, derived=_derived(0.5, 0.5))
    assert seen[0][2].tolist() == pytest.approx([lab.board[b][k] for b, k in fk], abs=1e-6)


@pytest.mark.slow
def test_run_quick_one_pedal(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    args = argparse.Namespace(quick=True, pedals=["ds1"])
    assert calibrate.run(cfg, args) == 0
    path = DataLayout.from_config(cfg).calibration_table("ds1")
    t = CalibrationTable.load(path)
    assert (t.gain_knob, t.tone_knob, t.level_knob) == ("dist", "tone", "level")
    assert t.gain_values == [0.0, 1.0] and t.tone_values == [0.0, 1.0]
    assert all(math.isfinite(v) for row in t.fit_error_db for v in row)
    assert all(math.isfinite(v) for row in t.derived_error_db for v in row)
    b = t.boards[0][0]
    assert set(b) == set(BLOCK_SPECS)
    for blk, specs in BLOCK_SPECS.items():
        assert set(b[blk]) == {s.name for s in specs} | {"enabled"}
    assert b["compressor"]["enabled"] == 0.0
    assert b["drive"]["enabled"] == b["eq"]["enabled"] == 1.0
    assert len(t.derived_boards) == 2 and len(t.derived_boards[0]) == 2
    lab = t.lookup({"dist": 0.3, "tone": 0.7, "level": 0.5})
    assert lab.source in ("calibrated", "calibrated_nearest")
    rep = json.loads((tmp_path / "rep" / "calibration.json").read_text(encoding="utf-8"))
    p = rep["pedals"]["ds1"]
    assert p["fit_error_db"] == t.fit_error_db
    assert p["derived_error_db"] == t.derived_error_db
    md = (tmp_path / "rep" / "calibration.md").read_text(encoding="utf-8")
    assert "## `ds1`" in md
    assert str(tmp_path) not in md
