"""P4 R4: white-box calibration tables (lookup, level offset, persistence, quick run)."""

from __future__ import annotations

import argparse
import json
import math

import pytest
import torch

from lstmabar.config import load_config
from lstmabar.data import calibrate
from lstmabar.data.calibrate import CalibrationTable, Label, board_jump, grid_values
from lstmabar.data.layout import DataLayout
from lstmabar.physics.derive import BLOCK_SPECS

GRID = [0.0, 0.5, 1.0]


def _spec(block: str, knob: str):
    return next(s for s in BLOCK_SPECS[block] if s.name == knob)


def _norm(block: str, knob: str, v: float) -> float:
    return float(_spec(block, knob).normalize(torch.tensor(v, dtype=torch.float64)))


def _phys(block: str, knob: str, u: float) -> float:
    return float(_spec(block, knob).denormalize(torch.tensor(u, dtype=torch.float64)))


def _board(i: int, j: int) -> dict:
    """Full normalized layout; smooth in (i, j) so neighbours stay within max_jump."""
    b = {blk: {s.name: s.default_normalized for s in specs} for blk, specs in BLOCK_SPECS.items()}
    for blk in b:
        b[blk]["enabled"] = 0.0 if blk == "compressor" else 1.0
    b["drive"]["gain_db"] = 0.2 + 0.1 * i
    b["eq"]["low_db"] = 0.4 + 0.05 * j
    b["drive"]["level_db"] = _norm("drive", "level_db", -20.0)
    return b


def _table(boards=None, **kw) -> CalibrationTable:
    boards = boards or [[_board(i, j) for j in range(3)] for i in range(3)]
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
    )
    args.update(kw)
    return CalibrationTable(**args)


def test_grid_values():
    assert grid_values(5) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert grid_values(1) == [0.5]


def test_lookup_exact_at_grid_points():
    t = _table()
    for i, g in enumerate(GRID):
        for j, tone in enumerate(GRID):
            lab = t.lookup({"drive": g, "tone": tone, "level": 0.5})
            assert isinstance(lab, Label)
            assert lab.source == "calibrated"
            assert lab.board == t.boards[i][j]
            assert lab.clamp_notes == []


def test_lookup_bilinear_midpoint():
    t = _table()
    lab = t.lookup({"drive": 0.25, "tone": 0.75})
    assert lab.source == "calibrated"
    corners = [t.boards[i][j] for i in (0, 1) for j in (1, 2)]
    for blk, kv in lab.board.items():
        for k, v in kv.items():
            assert v == pytest.approx(sum(c[blk][k] for c in corners) / 4, abs=1e-12)
    assert lab.board["drive"]["gain_db"] == pytest.approx(0.25)
    assert lab.board["eq"]["low_db"] == pytest.approx(0.4 + 0.05 * 1.5)


def test_lookup_falls_back_to_nearest_on_large_jump():
    boards = [[_board(i, j) for j in range(3)] for i in range(3)]
    boards[2][2]["eq"]["mid_db"] = 0.95  # jump > 0.25 inside the upper-right cell only
    t = _table(boards)
    assert board_jump(boards[1][1], boards[2][2]) > t.max_jump
    lab = t.lookup({"drive": 0.6, "tone": 0.9})  # cell [0.5, 1] x [0.5, 1], nearest (1, 2)
    assert lab.source == "calibrated_nearest"
    assert lab.board == boards[1][2]
    # Other cells still interpolate.
    assert t.lookup({"drive": 0.2, "tone": 0.2}).source == "calibrated"


def test_gates_are_copied_not_interpolated():
    boards = [[_board(i, j) for j in range(3)] for i in range(3)]
    boards[0][0]["eq"]["enabled"] = 0.0
    t = _table(boards)
    near = t.lookup({"drive": 0.1, "tone": 0.1})  # nearest corner (0, 0)
    far = t.lookup({"drive": 0.4, "tone": 0.4})  # nearest corner (1, 1)
    assert near.source == far.source == "calibrated"
    assert near.board["eq"]["enabled"] == 0.0
    assert far.board["eq"]["enabled"] == 1.0
    assert near.board["compressor"]["enabled"] == 0.0


def test_level_offset_and_clamp_note():
    t = _table()
    # ts808 level pot is audio taper through (0.5, 0.1): full up = +20 dB vs default.
    assert calibrate.level_offset_db("ts808", "level", 1.0, 0.5) == pytest.approx(20.0)
    up = t.lookup({"drive": 0.0, "tone": 0.0, "level": 1.0})
    assert _phys("drive", "level_db", up.board["drive"]["level_db"]) == pytest.approx(0.0)
    assert up.clamp_notes == []
    # Only drive.level_db changes.
    base = t.boards[0][0]
    assert {k: v for k, v in up.board["drive"].items() if k != "level_db"} == {
        k: v for k, v in base["drive"].items() if k != "level_db"
    }

    boards = [[_board(i, j) for j in range(3)] for i in range(3)]
    for row in boards:
        for b in row:
            b["drive"]["level_db"] = _norm("drive", "level_db", -10.0)
    loud = _table(boards).lookup({"drive": 0.0, "tone": 0.0, "level": 1.0})
    assert loud.board["drive"]["level_db"] == pytest.approx(1.0)
    assert len(loud.clamp_notes) == 1
    assert loud.clamp_notes[0].startswith("drive.level_db 10 clamped to 6 dB: ")

    off = t.lookup({"drive": 0.0, "tone": 0.0, "level": 0.0})
    assert off.board["drive"]["level_db"] == 0.0
    assert off.clamp_notes[0].startswith("drive.level_db -inf clamped to -36 dB: ")


@pytest.mark.parametrize(("pid", "level"), [("ts808", 0.8), ("ts808", 0.2), ("ds1", 0.9)])
def test_level_offset_matches_derivation(pid, level):
    """The analytic offset equals the derivation's post-clip gain difference."""
    from lstmabar.physics.derive import derive
    from lstmabar.physics.kb import load_kb

    pedal = load_kb()[pid]
    a = derive(pedal, {"level": level}).info["post_gain_db"]
    b = derive(pedal, {"level": 0.5}).info["post_gain_db"]
    assert calibrate.level_offset_db(pid, "level", level, 0.5) == pytest.approx(a - b, abs=1e-9)


def test_lookup_rejects_bad_knobs():
    t = _table()
    with pytest.raises(KeyError):
        t.lookup({"tone": 0.5})
    with pytest.raises(ValueError):
        t.lookup({"drive": 1.5, "tone": 0.5})


def test_save_load_round_trip(tmp_path):
    t = _table(meta={"git": {"sha": "abc", "dirty": False}, "f0s_hz": [82.41, 146.83]})
    t.boards[1][1]["drive"]["gain_db"] = 0.1 + 0.2  # non-representable decimal survives
    path = tmp_path / "cal" / "ts808.json"
    t.save(path)
    back = CalibrationTable.load(path)
    assert back == t
    assert json.loads(path.read_text(encoding="utf-8"))["pedal_id"] == "ts808"


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
    rep = json.loads((tmp_path / "rep" / "calibration.json").read_text(encoding="utf-8"))
    p = rep["pedals"]["ts808"]
    assert p["fit_mean_db"] == pytest.approx(2.0)
    assert p["derived_mean_db"] == pytest.approx(2.0)
    assert p["fit_worse_than_derived"] == 3
    assert p["fallback_cells"] == 1
    md = (tmp_path / "rep" / "calibration.md").read_text(encoding="utf-8")
    assert "| ts808 | **2.00** | 2.00 | 3/9 |" in md
    assert "Fallback cells (nearest grid point): 1 of 4" in md


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
    lab = t.lookup({"dist": 0.3, "tone": 0.7, "level": 0.5})
    assert lab.source in ("calibrated", "calibrated_nearest")
    rep = json.loads((tmp_path / "rep" / "calibration.json").read_text(encoding="utf-8"))
    p = rep["pedals"]["ds1"]
    assert p["fit_error_db"] == t.fit_error_db
    assert p["derived_error_db"] == t.derived_error_db
    md = (tmp_path / "rep" / "calibration.md").read_text(encoding="utf-8")
    assert "## `ds1`" in md
    assert str(tmp_path) not in md
