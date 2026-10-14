"""P5 wave 1 (f): val gold-proxy (D8): label -> region, distractors, renders, selection."""

from __future__ import annotations

import argparse
import functools
import itertools
import json

import numpy as np
import pytest

from lstmabar.data import gold, render
from lstmabar.data.gold import BRIGHTNESS, COMPRESSION, GAINS, MIDS, Region
from lstmabar.data.schema import RenderItem
from lstmabar.eval import proxy
from lstmabar.eval.data import load_split
from lstmabar.eval.fixtures import TARGETS_FILE, eval_fixture_config, make_eval_fixture
from lstmabar.eval.runner import SplitResult, UnitScores
from lstmabar.physics.derive import DeriveContext


@pytest.fixture(scope="module")
def targets():
    return _tables()


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("proxyfx")
    cfg = eval_fixture_config(root)
    make_eval_fixture(cfg, seconds=0.5)
    return cfg


@pytest.fixture(scope="module")
def phys(fx):
    return render.load_pedals(fx), DeriveContext.from_config(fx)


def _item(region: Region, targets, rng, item_id="x-r0") -> RenderItem:
    knobs, board = gold.sample(gold.expand(region, targets), rng)
    pedal = region.kind == "pedal"
    return RenderItem(
        item_id=item_id,
        clip_id="x",
        split="val",
        seed=0,
        scenario="pedal" if pedal else gold.expand(region, targets).scenario,
        engine="greybox",
        pedal_id=region.pedal,
        pedal_knobs=knobs,
        board_params=board if board is not None else _flat_board(),
        board_params_derived=None,
        label_source="derived" if pedal else "free",
        clamped=False,
        clamp_notes=[],
        input_peak_dbfs=-12.0,
        wet_gain_db=0.0,
        wet_path="v1/wet/shard-00000/x-r0.flac",
    )


@functools.cache
def _tables():
    return proxy.load_tables(TARGETS_FILE)


def _flat_board():
    _, board = gold.sample(gold.expand(Region("free", "none"), _tables()), np.random.default_rng(0))
    return board


def _regions(targets):
    for b, m, c in itertools.product(BRIGHTNESS, MIDS, COMPRESSION):
        yield Region("free", "none", b, m, c)
    for fam in gold.FAMILIES:
        for g in ["low"] if fam == "boost" else GAINS[1:]:
            yield Region("free", g, "slightly_bright", "scooped", "light", fam)
    for pid, t in targets.pedals.items():
        for g in GAINS[1:]:
            for b in BRIGHTNESS if t.tone_knob else ["neutral"]:
                yield Region("pedal", g, b, pedal=pid)


def test_label_to_region_round_trips_table_regions(targets):
    rng = np.random.default_rng(0)
    n = 0
    for region in _regions(targets):
        for _ in range(3):
            got = proxy.label_to_region(_item(region, targets, rng), targets)
            assert proxy.region_levels(got) == proxy.region_levels(region), region
            assert (got.kind, got.pedal) == (region.kind, region.pedal)
            if region.kind == "free" and region.gain != "none":
                assert got.family == region.family, region
            n += 1
    assert n > 200


def test_label_to_region_nearest_level_and_gates(targets):
    rng = np.random.default_rng(1)
    it = _item(Region("free", "none", "neutral", "neutral", "none"), targets, rng)
    b = it.board_params
    eq = dict(b["eq"])
    eq["high_db"] = gold._norm("eq", "high_db", 2.6)  # between neutral (1.5) and s.bright (3)
    eq["mid_db"] = gold._norm("eq", "mid_db", 3.5)  # between neutral and pushed
    board = {**b, "eq": eq}
    r = proxy.free_region(board, targets)
    assert (r.brightness, r.mids, r.gain, r.compression) == (
        "slightly_bright",
        "pushed",
        "none",
        "none",
    )
    off = {**board, "eq": {**eq, "enabled": 0.0}}  # EQ off = flat
    assert (proxy.free_region(off, targets).brightness, proxy.free_region(off, targets).mids) == (
        "neutral",
        "neutral",
    )
    drv = {**board, "drive": {**board["drive"], "enabled": 1.0}}
    drv["drive"]["gain_db"] = gold._norm("drive", "gain_db", 4.0)
    r = proxy.free_region(drv, targets)
    assert (r.gain, r.family) == ("low", "boost")


def test_distractor_axes_and_separation(targets, phys, fx):
    kb, ctx = phys
    rng = np.random.default_rng(2)
    cases = [
        (Region("free", "none", "bright", "scooped", "heavy"), ["brightness", "mids"]),
        (Region("free", "high", "dark", "neutral", "none", "distortion"), ["gain", "brightness"]),
        (Region("pedal", "low", "dark", pedal="ts808"), ["gain", "brightness"]),
    ]
    for region, axes in cases:
        it = _item(region, targets, rng)
        lab = proxy.label_to_region(it, targets)
        kept, dropped = proxy.item_distractors(it, lab, targets, kb, ctx, fx)
        assert [d.axis for d in kept] == axes and not dropped, region
        lc = proxy.label_concrete(it, lab)
        for d in kept:
            gap = gold.separation(lc, gold.expand(d.region, targets), d.axis, kb, ctx, targets)
            assert gap >= gold.min_gap(d.axis, fx, pedal=lc.kind == "pedal")
            levels = proxy.region_levels(d.region)
            assert levels[d.axis] != proxy.region_levels(lab).get(d.axis)


def test_distractor_failing_separation_is_dropped(targets, phys, fx):
    """A compressor at ratio 1.3 maps to 'light' (nearest), but the 'off' distractor is only
    0.3 of ratio away from the actual label: dropped."""
    kb, ctx = phys
    it = _item(
        Region("free", "none", "neutral", "neutral", "light"), targets, np.random.default_rng(3)
    )
    comp = {**it.board_params["compressor"], "ratio": gold._norm("compressor", "ratio", 1.3)}
    import dataclasses

    it = dataclasses.replace(it, board_params={**it.board_params, "compressor": comp})
    lab = proxy.label_to_region(it, targets)
    assert lab.compression == "light"
    kept, dropped = proxy.item_distractors(it, lab, targets, kb, ctx, fx)
    assert [a for a, _ in dropped] == ["compression"]
    assert "compression" not in [d.axis for d in kept]


def test_selection_score_weights(fx):
    units = [
        UnitScores("a", "val", "m", 0, {"mrstft": 1.0, "param_l1": 0.2, "afc_mrstft": 0.75}),
        UnitScores(
            "b", "val", "m", 0, {"mrstft": 3.0, "param_l1": 0.4, "afc_mrstft": float("nan")}
        ),
    ]
    res = SplitResult("val", "m", 0, units, {})
    w = {k: float(v) for k, v in fx.eval.proxy.selection_weights.items()}
    expect = w["mrstft"] * 2.0 + w["param_l1"] * 0.3 + w["afc"] * 0.25
    assert proxy.selection_score(res, fx) == pytest.approx(expect)
    assert proxy.selection_terms(res) == pytest.approx(
        {"mrstft": 2.0, "param_l1": 0.3, "afc": 0.25}
    )


def test_selection_score_without_proxy_and_failed_renders(fx):
    w = {k: float(v) for k, v in fx.eval.proxy.selection_weights.items()}
    units = [
        UnitScores("a", "val", "m", 0, {"mrstft": 1.0, "param_l1": 0.2}),
        UnitScores("b", "val", "m", 0, {"mrstft": 3.0, "param_l1": 0.4}),
    ]
    res = SplitResult("val", "m", 0, units, {}, {"proxy": False})
    with pytest.warns(UserWarning, match="no proxy 2AFC"):
        s = proxy.selection_score(res, fx)
    assert s == pytest.approx(w["mrstft"] * 2.0 + w["param_l1"] * 0.3)
    res.extra["proxy"] = True  # claims proxy scores but has none: an error
    with pytest.raises(ValueError, match="afc_mrstft"):
        proxy.selection_score(res, fx)

    # a unit whose render failed counts as the worst MR-STFT, never as missing
    failed = UnitScores(
        "c", "val", "m", 0, {"mrstft": float("nan"), "param_l1": 0.3}, {"render_failed": 1}
    )
    res = SplitResult("val", "m", 0, [*units, failed], {}, {"proxy": False})
    with pytest.warns(UserWarning):
        s = proxy.selection_score(res, fx)
    assert s == pytest.approx(w["mrstft"] * (1.0 + 3.0 + 3.0) / 3 + w["param_l1"] * 0.3)


def test_proxy_cfg_refuses_dataset_and_gold_versions(fx):
    from omegaconf import OmegaConf

    for v in ("v1", "gold_v1", "proxy"):
        with pytest.raises(ValueError):
            proxy.proxy_cfg(OmegaConf.merge(fx, {"eval": {"proxy": {"version": v}}}))


def test_run_and_load_proxy_items_end_to_end(fx, monkeypatch):
    monkeypatch.setattr(render, "librosa_available", lambda: False)  # no pYIN: fast, as CI
    args = argparse.Namespace(fixtures=True, limit=None, workers=1, overrides=[])
    assert proxy.run(fx, args) == 0
    manifest, index_path = proxy.manifest_paths(fx)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    recs = gold.read_manifest(manifest)
    val = load_split(fx, "val")
    assert index["val_items"] == len(val) and index["renders"] == len(recs) > 0
    assert all(r.gold_source == proxy.PROXY_SOURCE and r.role == "distractor" for r in recs)
    assert all(r.render["wet_path"].startswith("valproxy_v1/") for r in recs)
    by_id = {s.item_id: s for s in val}
    for r in recs:
        assert r.render["input_peak_dbfs"] == by_id[r.gold_id].item.input_peak_dbfs
        assert r.render["clip_id"] == by_id[r.gold_id].item.clip_id

    items = proxy.load_proxy_items(fx)
    assert items and {g.gold_id for g in items} == {r.gold_id for r in recs}
    for g in items:
        s = by_id[g.gold_id]
        q = g.queries()[0]
        assert (g.split, g.source) == ("val", proxy.PROXY_SOURCE)
        assert (q.unit_id, q.text_key, q.audio_key) == (
            s.item_id,
            s.captions[0].caption_id,
            s.item_id,
        )
        assert q.text == s.captions[0].text and q.input_peak_dbfs == s.item.input_peak_dbfs
        (clip,) = g.clips
        assert clip.target.render["wet_path"] == s.item.wet_path
        assert len(clip.distractors) <= int(fx.eval.proxy.distractors_per_item)
        assert [d.distractor_index for d in clip.distractors] == list(range(len(clip.distractors)))

    # resumable: a second run renders nothing new and writes the same manifest
    before = manifest.read_text(encoding="utf-8")
    assert proxy.run(fx, args) == 0
    assert manifest.read_text(encoding="utf-8") == before

    # committed manifest without its (gitignored) audio: a clear error
    from omegaconf import OmegaConf

    empty = OmegaConf.merge(fx, {"data": {"rendered_dir": str(manifest.parent / "no_audio")}})
    with pytest.raises(FileNotFoundError, match="lstmabar eval proxy"):
        proxy.load_proxy_items(empty)
