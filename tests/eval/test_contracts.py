"""P5 wave 0: eval data loaders, method interface, embedding cache, CLI, fixture."""

from __future__ import annotations

import numpy as np
import pytest

from lstmabar.cli import build_parser
from lstmabar.data.gold import SOURCES
from lstmabar.eval import data as D
from lstmabar.eval.baselines.base import (
    BOARD_LAYOUT,
    REGISTRY,
    Method,
    board_vector,
    complete_board,
    default_board,
    vector_board,
)
from lstmabar.eval.cli import STEPS
from lstmabar.eval.fixtures import FIXTURE_GOLD, eval_fixture_config, make_eval_fixture
from lstmabar.models.embed_cache import (
    EmbeddingTable,
    audio_sha256,
    cache_path,
    l2_normalize,
    load_table,
    lookup_queries,
    text_sha256,
)
from lstmabar.train.cli import STEPS as TRAIN_STEPS


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("evalfx")
    cfg = eval_fixture_config(root)
    counts = make_eval_fixture(cfg, seconds=0.5)
    return cfg, counts


def test_fixture_counts(fx):
    _, counts = fx
    assert counts["train"] == 8 and counts["val"] == 4 and counts["test"] == 4
    assert counts["test_unseen_pedal"] == 2
    assert counts["gold_renders"] == 2 * sum(1 + len(d) for *_, d in FIXTURE_GOLD)
    assert counts["test_real"] == 2


def test_load_split_joins_captions_and_levels(fx):
    cfg, _ = fx
    items = D.load_split(cfg, "test")
    assert [i.item_id for i in items] == sorted(i.item_id for i in items)
    it = items[0]
    assert len(it.captions) == 2
    qs = it.queries()
    assert [q.query_id for q in qs] == [c.caption_id for c in it.captions]
    assert all(q.unit_id == it.item_id and q.audio_key == it.item_id for q in qs)
    assert all(q.split == "test" for q in qs)
    x = qs[0].load_dry()
    peak = 20 * np.log10(np.max(np.abs(x)))
    assert peak == pytest.approx(it.item.input_peak_dbfs, abs=1e-6)
    assert it.load_wet().shape == x.shape
    assert set(it.label) == {"compressor", "drive", "eq"}
    assert len(D.load_split(cfg, "train", limit=3)) == 3
    with pytest.raises(ValueError):
        D.load_split(cfg, "gold")


def test_gold_items_grouped_by_clip(fx):
    cfg, _ = fx
    items = D.load_gold_items(cfg)
    assert [g.gold_id for g in items] == [g for g, *_ in FIXTURE_GOLD]
    g = items[0]
    assert g.source in SOURCES and g.instruction.endswith("(fixture)")
    assert len(g.clips) == 2
    for c in g.clips:
        assert c.target.role == "target"
        assert [d.distractor_index for d in c.distractors] == [0, 1]
    qs = g.queries()
    assert len({q.audio_key for q in qs}) == 2 and qs[0].audio_key.endswith("@-15.0")
    assert all(q.split == "gold" for q in qs)
    x = qs[0].load_dry()
    assert 20 * np.log10(np.max(np.abs(x))) == pytest.approx(-15.0, abs=1e-6)
    assert D.load_render_audio(cfg, g.clips[0].target).shape == x.shape
    assert "headline" in g.subsets()
    assert items[2].subsets() == ["all", "ambiguous"]


def test_proxy_style_gold_item_uses_val_cache_keys(fx, tmp_path):
    """A val proxy unit (GoldItem, split val) resolves against val tables keyed like
    SynthItem.queries()."""
    import dataclasses

    cfg, _ = fx
    val = D.load_split(cfg, "val")[0]
    g = dataclasses.replace(
        D.load_gold_items(cfg)[0],
        gold_id=val.item_id,
        split="val",
        text_key=val.captions[0].caption_id,
        audio_key=val.item_id,
        clips=D.load_gold_items(cfg)[0].clips[:1],
    )
    q = g.queries()[0]
    assert (q.split, q.unit_id, q.text_key, q.audio_key) == (
        "val",
        val.item_id,
        val.captions[0].caption_id,
        val.item_id,
    )
    keys = [c.caption_id for c in val.captions]
    EmbeddingTable(keys, np.eye(len(keys), 3), ["x"] * len(keys), "bge-base-en-v1.5", "text").save(
        cache_path(tmp_path, "bge-base-en-v1.5", "val", "text")
    )
    EmbeddingTable([val.item_id], np.eye(1, 3), ["x"], "bge-base-en-v1.5", "audio").save(
        cache_path(tmp_path, "bge-base-en-v1.5", "val", "audio")
    )
    assert lookup_queries(tmp_path, "bge-base-en-v1.5", "text", [q]).shape == (1, 3)
    assert lookup_queries(tmp_path, "bge-base-en-v1.5", "audio", [q]).shape == (1, 3)


def test_test_real_loader(fx):
    cfg, _ = fx
    real = D.load_test_real(cfg)
    assert [r.item.item_id for r in real] == ["fx-real-0", "fx-real-1"]
    assert real[1].internal_only and not real[0].internal_only
    assert real[0].kb_pedal == "ts808" and real[1].kb_pedal is None
    x, sr = real[0].load_dry()
    assert sr == 48000 and x.ndim == 1


def test_board_helpers():
    b = default_board()
    assert all(v["enabled"] == 0.0 for v in b.values())
    full = complete_board({"drive": {"gain_db": 1.7, "enabled": 1.0}})
    assert full["drive"]["gain_db"] == 1.0 and full["eq"]["enabled"] == 0.0
    assert vector_board(board_vector(full)) == full
    assert len(board_vector(full)) == len(BOARD_LAYOUT)
    with pytest.raises(KeyError):
        complete_board({"reverb": {}})
    with pytest.raises(ValueError):
        complete_board({"eq": {"low_db": float("nan")}})


def test_method_registry_names():
    assert set(REGISTRY) == {
        "noop",
        "mean",
        "knn_clap",
        "knn_bge",
        "llm_zero_shot",
        "text2fx",
        "p6_model",
    }

    class Const(Method):
        name = "const"

        def predict(self, queries):
            return [default_board() for _ in queries]

    m = Const()
    assert m.select([]) == {} and m.predict([None]) == [default_board()]


def test_embedding_table_roundtrip(tmp_path):
    v = l2_normalize(np.random.default_rng(0).normal(size=(3, 4)))
    t = EmbeddingTable(
        ["a", "b", "c"], v, [text_sha256(s) for s in "abc"], "bge-base-en-v1.5", "text"
    )
    p = t.save(cache_path(tmp_path, "bge-base-en-v1.5", "val", "text"))
    assert p.parts[-3:] == ("bge", "bge-base-en-v1.5", "val_text.npz")
    t2 = load_table(tmp_path, "bge-base-en-v1.5", "val", "text")
    np.testing.assert_allclose(t2.lookup(["c", "a"]), v[[2, 0]])
    assert t2.stale({"a": text_sha256("a"), "b": "x", "z": "y"}) == ["b", "z"]
    with pytest.raises(KeyError, match="'q'"):
        t2.lookup(["q"])
    with pytest.raises(FileNotFoundError, match="lstmabar eval embed"):
        load_table(tmp_path, "bge-base-en-v1.5", "test", "text")
    assert audio_sha256(np.zeros(4)) == audio_sha256(np.zeros(4, dtype=np.float32))


def test_cli_registers_eval_and_train():
    p = build_parser()
    a = p.parse_args(["eval", "run", "eval.workers=1", "--limit", "2"])
    assert a.step == "run" and a.overrides == ["eval.workers=1"]
    assert set(STEPS) == {"embed", "proxy", "sim-real", "run", "report", "results"}
    assert set(TRAIN_STEPS) == {"stage1", "stage2"}
    assert p.parse_args(["train", "stage1"]).step == "stage1"
