"""Nearest-pedal explanation (M4 demo, P6 6.8): gradio-free."""

import pytest

from lstmabar.demo.explain import (
    GROUPS,
    AtlasEntry,
    PedalExplainer,
    board_distance,
    group_distances,
    nearest_pedals,
    pedal_atlas,
)
from lstmabar.dsp.pedalboard import ENABLED
from lstmabar.eval.baselines.base import complete_board, default_board


def _board(**knobs):
    """Complete board (all gates on) with ``drive_gain_db=0.9``-style overrides."""
    b: dict = {}
    for key, v in knobs.items():
        blk, k = key.split("_", 1)
        b.setdefault(blk, {})[k] = v
    return complete_board(b, enabled=1.0)


def _entry(pid, name=None, **knobs):
    return AtlasEntry(pid, name or pid, {"drive": 0.5}, {"drive": "Drive"}, _board(**knobs))


def test_groups_cover_drive_and_eq_but_not_level_or_compressor():
    keys = {k for knobs in GROUPS.values() for k in knobs}
    assert ("drive", "level_db") not in keys
    assert not any(b == "compressor" for b, _ in keys)
    assert ("eq", "high_db") in keys and ("drive", "gain_db") in keys


def test_distance_is_zero_for_equal_boards_and_mean_of_groups():
    a = _board(drive_gain_db=0.8)
    assert board_distance(a, a) == 0.0
    b = _board(drive_gain_db=0.2, drive_softness=0.0)
    g = group_distances(a, b)
    assert g["gain"] == pytest.approx(0.6)
    assert g["clipping"] == pytest.approx(abs(default_board()["drive"]["softness"]) / 3)
    assert g["tone"] == 0.0
    assert board_distance(a, b) == pytest.approx(sum(g.values()) / 3)


def test_level_knob_does_not_count():
    assert board_distance(_board(drive_level_db=0.0), _board(drive_level_db=1.0)) == 0.0


def test_gated_off_eq_reads_as_flat():
    flat = _board()
    eq_off = _board(eq_high_db=0.0, eq_enabled=0.0)
    assert board_distance(eq_off, flat) == 0.0
    assert board_distance(_board(eq_high_db=0.0), flat) > 0.0


def test_nearest_pedals_ranks_by_distance_one_entry_per_pedal():
    atlas = [
        _entry("fuzz", drive_gain_db=1.0, drive_softness=0.0),
        _entry("od", "Overdrive", drive_gain_db=0.6),
        _entry("od", "Overdrive", drive_gain_db=0.75),
        _entry("clean", drive_gain_db=0.1),
    ]
    ranked = nearest_pedals(_board(drive_gain_db=0.7), atlas)
    assert [n.entry.pedal_id for n in ranked] == ["od", "fuzz", "clean"]
    assert ranked[0].entry.board["drive"]["gain_db"] == pytest.approx(0.75)
    assert ranked[0].groups["gain"] == pytest.approx(0.05)
    md = ranked[0].markdown()
    assert md.startswith("**Nearest pedal:** Overdrive at Drive 5 ")
    assert "closest on" in md and "drive gain" in md
    assert len(nearest_pedals(_board(), atlas, per_pedal=False)) == 4


def test_holdout_and_drive_off_are_labelled():
    atlas = [_entry("rat", "RAT", drive_gain_db=0.9)]
    n = nearest_pedals(_board(drive_gain_db=0.9, drive_enabled=0.0), atlas, holdout="rat")[0]
    assert n.holdout and n.drive_off
    md = n.markdown()
    assert "held-out pedal" in md and "drive off" in md
    n = nearest_pedals(_board(drive_gain_db=0.9), atlas, holdout=None)[0]
    assert not n.holdout and "held-out" not in n.markdown()


def test_explainer_with_empty_kb_or_failure_says_nothing():
    assert PedalExplainer({}).markdown(_board()) == ""
    bad = PedalExplainer({}, atlas=[_entry("x")])
    assert bad.markdown({"nope": {}}) == ""  # unknown block: logged, no explanation


def test_pedal_atlas_derives_real_pedals_and_holds_level_pots():
    from lstmabar.physics.kb import load_kb

    kb = {"fuzz_face_ge": load_kb()["fuzz_face_ge"]}
    atlas = pedal_atlas(kb, grid=(0.5,))
    assert len(atlas) == 1  # fuzz on the grid, volume held at its default
    e = atlas[0]
    assert e.dials["fuzz"] == 0.5
    assert e.dials["volume"] == pytest.approx(kb["fuzz_face_ge"].pots["volume"].default)
    assert e.board["drive"][ENABLED] == 1.0
    assert all(0.0 <= v <= 1.0 for knobs in e.board.values() for v in knobs.values())
    n = PedalExplainer(kb, atlas=atlas).nearest(e.board)
    assert n is not None and n.distance == pytest.approx(0.0, abs=1e-6)
