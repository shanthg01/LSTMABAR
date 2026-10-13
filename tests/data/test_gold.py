"""Gold pairing (task 4.5): targets-file schema and validation, deterministic sampling,
target vs distractor separation, test-clips-only planning, unseen-pedal flags, and that
nothing under data/manifests/v1 is touched."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from lstmabar.config import load_config
from lstmabar.data import gold as G
from lstmabar.data.render import load_pedals
from lstmabar.data.schema import DryClip, read_jsonl
from lstmabar.physics.derive import BLOCK_SPECS, DeriveContext

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "data" / "gold_v1.yaml"
TARGETS = REPO / "data" / "gold" / "targets" / "targets_v1.yaml"
V1_MANIFESTS = REPO / "data" / "manifests" / "v1"
GOLD_MANIFEST = REPO / "data" / "manifests" / "gold_v1.jsonl"
GOLD_INDEX = REPO / "data" / "manifests" / "gold_v1.index.json"


@pytest.fixture(scope="module")
def cfg():
    c = load_config(CONFIG)
    root = REPO / "data"
    return OmegaConf.merge(
        c,
        {
            "gold": {
                "gold_dir": str(root / "gold"),
                "targets_file": str(TARGETS),
                "dry_manifest": str(V1_MANIFESTS / "dry_clips.jsonl"),
            },
            "paths": {"pedals": str(REPO / "pedals")},
        },
    )


@pytest.fixture(scope="module")
def targets():
    return G.load_targets(TARGETS)


@pytest.fixture(scope="module")
def gold():
    return G.load_gold(REPO / "data" / "gold")[0]


@pytest.fixture(scope="module")
def kb(cfg):
    return load_pedals(cfg)


@pytest.fixture(scope="module")
def ctx(cfg):
    return DeriveContext.from_config(cfg)


# ------------------------------------------------------------------ targets file


def test_targets_file_validates(targets, gold, kb, ctx, cfg):
    assert G.check_rows(targets, gold, kb, ctx, cfg) == []


def test_every_labelled_gold_item_has_one_row(targets, gold):
    # Intentional tripwire: after labelling imported friend items this fails until
    # `lstmabar data gold-targets --draft-missing` has drafted (and the owner reviewed) rows.
    rows = targets.by_id()
    labelled = {g for g, it in gold.items() if G.labelled(it)}
    assert labelled == set(rows)
    assert [r.id for r in targets.items] == [g for g in gold if g in labelled]


def test_rows_copy_source_and_never_relabel(targets, gold):
    for row in targets.items:
        assert row.source == gold[row.id]["source"]
        assert row.in_scope == gold[row.id]["target"]["in_scope"]
    # `human` rows only for items the import wrote as human (friend items, h151+)
    human = {r.id for r in targets.items if r.source == "human"}
    assert all(gold[g]["source"] == "human" and int(g[1:]) > 150 for g in human)


def test_unseen_pedal_flag_matches_regions(targets, cfg):
    holdout = str(cfg.render.holdout_pedal)
    flagged = {r.id for r in targets.items if r.unseen_pedal}
    uses = {
        r.id
        for r in targets.items
        for reg in [r.target, *(d.region for d in r.distractors)]
        if reg.kind == "pedal" and reg.pedal == holdout
    }
    assert flagged == uses
    assert flagged, "the gold set names the RAT; those rows must be flagged"


def test_ambiguous_rows_say_why(targets):
    for r in targets.items:
        assert r.ambiguous == bool(r.ambiguity)
        if r.ambiguous:
            assert r.confidence == "low"


def _row(**over):
    base = {
        "id": "x001",
        "source": "llm_draft",
        "in_scope": True,
        "target": {"kind": "free", "gain": "none"},
        "distractors": [{"axis": "gain", "kind": "free", "gain": "high", "family": "fuzz"}],
        "confidence": "high",
        "ambiguous": False,
        "ambiguity": [],
        "unseen_pedal": False,
        "rationale": "r",
    }
    return base | over


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"source": "humans"}, "source"),
        ({"confidence": "sure"}, "confidence"),
        ({"ambiguous": True}, "ambiguous"),
        ({"ambiguity": ["relative"]}, "ambiguous"),
        ({"ambiguity": ["guess"], "ambiguous": True}, "ambiguity"),
        ({"distractors": []}, "at least one"),
        ({"rationale": " "}, "rationale"),
        ({"target": {"kind": "free", "gain": "high"}}, "family"),
        ({"target": {"kind": "free", "gain": "none", "family": "fuzz"}}, "family"),
        ({"target": {"kind": "pedal", "pedal": "ts808", "gain": "low", "mids": "pushed"}}, "mids"),
        ({"target": {"kind": "pedal", "gain": "low"}}, "pedal"),
        ({"target": {"kind": "free", "gain": "none", "bright": "dark"}}, "unknown keys"),
        ({"extra": 1}, "unknown keys"),
    ],
)
def test_row_validation_rejects(over, msg):
    with pytest.raises(G.GoldError, match=msg):
        G.row_from_dict(_row(**over))


def test_expand_rejects_bad_overrides(targets):
    bad = [
        G.Region("free", "none", overrides={"eq.nope": (0.0, 1.0)}),
        G.Region("free", "none", overrides={"eq.low_db": (-40.0, 0.0)}),
        G.Region("pedal", "low", pedal="ts808", overrides={"drive": (0.0, 1.5)}),
        G.Region("pedal", "low", "dark", pedal="fuzz_face_ge"),
        G.Region("pedal", "low", pedal="klon"),
    ]
    for r in bad:
        with pytest.raises(G.GoldError):
            G.expand(r, targets)


def test_tables_are_inside_the_block_ranges_and_levels_disjoint(targets):
    t = targets.tables
    levels = [t["gain_db"][g] for g in G.GAINS[1:]]
    assert all(a[1] < b[0] for a, b in zip(levels, levels[1:], strict=False))
    highs = [t["brightness"][b]["high_db"] for b in G.BRIGHTNESS]
    assert all(a[1] < b[0] for a, b in zip(highs, highs[1:], strict=False))
    mids = [t["mids"][m]["mid_db"] for m in G.MIDS]
    assert all(a[1] < b[0] for a, b in zip(mids, mids[1:], strict=False))


# ------------------------------------------------------------------ sampling


def test_sampling_is_deterministic_and_inside_the_region(targets):
    for row in targets.items[::7]:
        for reg in [row.target, *(d.region for d in row.distractors)]:
            c = G.expand(reg, targets)
            a = G.sample(c, np.random.default_rng(123))
            b = G.sample(c, np.random.default_rng(123))
            assert a == b
            knobs, board = a
            if c.kind == "pedal":
                for k, (lo, hi) in c.knobs.items():
                    assert lo - 1e-9 <= knobs[k] <= hi + 1e-9
                continue
            for block, specs in BLOCK_SPECS.items():
                assert board[block]["enabled"] == c.gates[block]
                for s in specs:
                    lo, hi = c.ranges[block][s.name]
                    import torch

                    v = float(
                        s.denormalize(torch.tensor(board[block][s.name], dtype=torch.float64))
                    )
                    assert lo - 1e-6 * max(1, abs(lo)) <= v <= hi + 1e-6 * max(1, abs(hi))


# ------------------------------------------------------------------ separation


def test_every_distractor_separates_from_its_target(targets, kb, ctx, cfg):
    assert G.check_separation(targets, kb, ctx, cfg) == []
    for row in targets.items:
        assert row.distractors[0].axis in G.AXES
        axes = [d.axis for d in row.distractors]
        assert len(axes) == len(set(axes))


def test_an_overlapping_distractor_is_caught(targets, kb, ctx, cfg):
    row = targets.items[0]
    same = replace(row, distractors=(G.Distractor("gain", row.target),))
    bad = G.check_separation(replace(targets, items=(same,)), kb, ctx, cfg)
    assert bad and bad[0][0] == row.id


def test_gain_distractor_rule(targets, kb, ctx, cfg):
    split = float(cfg.gold.gain_split_db)
    for row in targets.items:
        d = next(d for d in row.distractors if d.axis == "gain")
        c = G.expand(row.target, targets)
        hi = G.gain_range_db(c, kb, ctx, G._gain_knob(c, targets))[1]
        assert (d.region.gain == "none") == (hi >= split)


# ------------------------------------------------------------------ planning


def test_plan_uses_test_clips_only_and_is_deterministic(targets, cfg):
    clips = G.test_clips(cfg)
    assert len(clips) == int(cfg.gold.expect_test_clips)
    assert {c.split for c in clips} == {"test"}
    sub = replace(targets, items=targets.items[:12])
    a, b = G.plan(sub, clips, cfg), G.plan(sub, clips, cfg)
    assert a == b
    test_ids = {c.clip_id for c in clips}
    assert all(g.spec.clip_id in test_ids for g in a)
    assert all(g.spec.split == "gold" for g in a)
    assert all(g.spec.input_peak_dbfs == float(cfg.audio.inference_peak_dbfs) for g in a)
    n = int(cfg.gold.clips_per_item)
    k = int(cfg.gold.distractors_per_item)
    assert len(a) == sum(n * (1 + min(k, len(r.distractors))) for r in sub.items)
    # target and its distractors share the clip
    by = {}
    for g in a:
        by.setdefault((g.gold_id, g.clip_index), set()).add(g.spec.clip_id)
    assert all(len(v) == 1 for v in by.values())
    # rows are independent: one row's plan doesn't depend on the others
    one = G.plan(replace(targets, items=targets.items[5:6]), clips, cfg)
    assert [g.spec for g in one] == [g.spec for g in a if g.gold_id == targets.items[5].id]


def test_gold_version_never_targets_the_dataset(cfg):
    for bad in ("v1", "v1.1", "x_gold"):
        with pytest.raises(ValueError):
            G.gold_cfg(OmegaConf.merge(cfg, {"gold": {"version": bad}}))
    manifest, index = G.manifest_paths(cfg)
    assert V1_MANIFESTS not in manifest.parents and V1_MANIFESTS not in index.parents
    assert G.gold_cfg(cfg).data.version == "gold_v1"


# ------------------------------------------------------------------ end to end (fixtures)


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_gold_pair_end_to_end_on_fixtures(tmp_path, cfg, targets, gold):
    gold_items = [g for g, it in gold.items() if G.labelled(it)]
    from lstmabar.data.fixtures import fixture_config, make_fixture_dataset

    before = _hash_tree(V1_MANIFESTS)
    fcfg = fixture_config(tmp_path)
    clips = make_fixture_dataset(fcfg, clips_per_performer=1, seconds=0.5)
    test = [c for c in clips if c.split == "test"]
    assert test
    # tiny targets file: a free row, a white-box pedal row and an unseen (rat) row
    pick = [r for r in targets.items if r.target.kind == "free"][:1]
    pick += [r for r in targets.items if r.target.kind == "pedal" and r.target.pedal == "ts808"][:1]
    pick += [r for r in targets.items if r.unseen_pedal][:1]
    tfile = tmp_path / "targets.yaml"
    tfile.write_text(G.dump_targets("# test", replace(targets, status="reviewed"), pick), "utf-8")
    layout_manifest = tmp_path / "manifests" / "v1" / "dry_clips.jsonl"
    gcfg = OmegaConf.merge(
        fcfg,
        {"gold": OmegaConf.to_container(cfg.gold, resolve=True)},
        {
            "gold": {
                "targets_file": str(tfile),
                "gold_dir": str(REPO / "data" / "gold"),
                "dry_manifest": str(layout_manifest),
                "expect_test_clips": len(test),
                "clips_per_item": 1,
                "workers": 1,
            },
            "paths": {"pedals": str(REPO / "pedals")},
        },
    )
    args = Namespace(fixtures=True, limit=None, workers=1, allow_draft=False)
    assert G.run(gcfg, args) == 0
    # the tiny targets file lacks rows for the other labelled gold items: partial outputs
    manifest, index = G.manifest_paths(gcfg, partial=True)
    assert not G.manifest_paths(gcfg)[0].exists()
    recs = G.read_manifest(manifest)
    assert len(recs) == sum(1 + min(2, len(r.distractors)) for r in pick)
    assert {r.clip_id for r in recs} <= {c.clip_id for c in test}
    assert all(r.render["wet_path"].startswith("gold_v1/") for r in recs)
    assert any(r.unseen_pedal and r.render["pedal_id"] == "rat" for r in recs)
    ts = [r for r in recs if r.render["pedal_id"] == "ts808"]
    assert ts and all(r.render["engine"] == "whitebox" for r in ts)
    for r in recs:
        assert (tmp_path / "rendered" / r.render["wet_path"]).exists()
    idx = json.loads(index.read_text(encoding="utf-8"))
    assert idx["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert idx["renders"] == len(recs) and idx["partial"] is True
    assert idx["labelled_without_row"] == len(gold_items) - len(pick)
    # the fixture's own v1 manifests and the repo's frozen v1 manifests are untouched
    assert sorted(p.name for p in (tmp_path / "manifests" / "v1").iterdir()) == ["dry_clips.jsonl"]
    assert _hash_tree(V1_MANIFESTS) == before
    # re-running is a no-op render (all shards DONE) and reproduces the manifest
    text = manifest.read_text(encoding="utf-8")
    assert G.run(gcfg, args) == 0
    assert manifest.read_text(encoding="utf-8") == text


def test_draft_targets_file_is_refused(tmp_path, cfg, targets):
    tfile = tmp_path / "t.yaml"
    draft = replace(targets, status="draft")
    tfile.write_text(G.dump_targets("# t", draft, targets.items[:1]), "utf-8")
    c = OmegaConf.merge(cfg, {"gold": {"targets_file": str(tfile)}})
    with pytest.raises(RuntimeError, match="draft"):
        G.run(c, Namespace(allow_draft=False, limit=None, workers=1))


def test_fixtures_flag_refused_on_the_real_manifests(cfg):
    with pytest.raises(ValueError, match="fixtures"):
        G.run(cfg, Namespace(allow_draft=False, limit=None, workers=1, fixtures=True))


def test_partial_runs_never_replace_the_manifest(cfg):
    full, _ = G.manifest_paths(cfg)
    part, part_index = G.manifest_paths(cfg, partial=True)
    assert part != full and part.name == "gold_v1.partial.jsonl"
    assert part_index.name == "gold_v1.partial.index.json"


def test_status_rewrite_keeps_the_comment(targets):
    prefix = "version: 1\nstatus: reviewed   # owner approved\n"
    out = G.dump_targets("", replace(targets, status="draft"), [], prefix=prefix)
    assert "status: draft   # owner approved" in out


# ------------------------------------------------------------------ committed artefacts


@pytest.mark.skipif(not GOLD_MANIFEST.exists(), reason="gold manifest not rendered yet")
def test_committed_gold_manifest_matches_its_index(targets, cfg):
    text = GOLD_MANIFEST.read_text(encoding="utf-8")
    idx = json.loads(GOLD_INDEX.read_text(encoding="utf-8"))
    assert idx["manifest_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert idx["targets_sha256"] == hashlib.sha256(TARGETS.read_bytes()).hexdigest()
    recs = G.read_manifest(GOLD_MANIFEST)
    test_ids = {
        c.clip_id
        for c in read_jsonl(V1_MANIFESTS / "dry_clips.jsonl", DryClip)
        if c.split == "test"
    }
    assert {r.clip_id for r in recs} <= test_ids
    rows = targets.by_id()
    for r in recs:
        row = rows[r.gold_id]
        assert r.gold_source == row.source and r.unseen_pedal == row.unseen_pedal
        assert r.render["split"] == "gold"
    assert idx["renders"] == len(recs)


def test_targets_file_parses_as_plain_yaml():
    doc = yaml.safe_load(TARGETS.read_text(encoding="utf-8"))
    assert doc["status"] in ("draft", "reviewed")


# ------------------------------------------------------------------ friend examples


def test_example_regions_are_valid_and_never_the_holdout(cfg, targets):
    regions = G.example_regions(cfg, targets)
    assert regions and all(r.pedal != cfg.render.holdout_pedal for _, r in regions)
    bad = OmegaConf.merge(
        cfg,
        {"gold": {"examples": [{"id": "ex01", "kind": "pedal", "pedal": "rat", "gain": "low"}]}},
    )
    with pytest.raises(G.GoldError, match="holdout"):
        G.example_regions(bad, targets)
    bad = OmegaConf.merge(
        cfg, {"gold": {"examples": [{"id": "e1", "kind": "free", "gain": "none"}]}}
    )
    with pytest.raises(G.GoldError, match="exNN"):
        G.example_regions(bad, targets)


def test_gold_examples_on_fixtures(tmp_path, cfg):
    from lstmabar.data.fixtures import fixture_config, make_fixture_dataset

    fcfg = fixture_config(tmp_path)
    make_fixture_dataset(fcfg, clips_per_performer=1, seconds=0.5)
    ecfg = OmegaConf.merge(
        fcfg,
        {"gold": OmegaConf.to_container(cfg.gold, resolve=True)},
        {
            "gold": {
                "dry_manifest": str(tmp_path / "manifests" / "v1" / "dry_clips.jsonl"),
                "examples": [
                    {"id": "ex01", "kind": "free", "gain": "none", "brightness": "bright"},
                    {"id": "ex02", "kind": "pedal", "pedal": "big_muff", "gain": "high"},
                ],
            },
            "paths": {"pedals": str(REPO / "pedals")},
        },
    )
    assert G.examples_step(ecfg, Namespace()) == 0
    out = tmp_path / "rendered" / "gold_examples"
    assert sorted(p.name for p in out.glob("*.wav")) == [
        "ex01_dry.wav",
        "ex01_wet.wav",
        "ex02_dry.wav",
        "ex02_wet.wav",
    ]
    meta = json.loads((out / "examples.json").read_text(encoding="utf-8"))
    assert [m["id"] for m in meta] == ["ex01", "ex02"]
