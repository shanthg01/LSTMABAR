"""D4 trivial and kNN baselines on the eval fixture with fake embedding tables."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lstmabar.eval import data as D
from lstmabar.eval.baselines.base import (
    BOARD_LAYOUT,
    board_vector,
    build_method,
    complete_board,
    default_board,
)
from lstmabar.eval.baselines.knn import aggregate, top_k
from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture
from lstmabar.models.embed_cache import EmbeddingTable, cache_path, text_sha256

MODELS = {"knn_clap": "clap-htsat-unfused", "knn_bge": "bge-base-en-v1.5"}


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("evalfx")
    cfg = eval_fixture_config(root)
    make_eval_fixture(cfg, seconds=0.5)
    return cfg, D.load_split(cfg, "train"), D.load_split(cfg, "val")


def _write(cfg, model, split, ids, texts, vectors):
    EmbeddingTable(ids, vectors, [text_sha256(t) for t in texts], model, "text").save(
        cache_path(cfg.eval.embeddings.dir, model, split, "text")
    )


def _one_hot_cache(cfg, model, train, val):
    """Train captions get one-hot vectors; each val caption copies one train caption's
    vector (val caption j -> train caption j)."""
    tcaps = [c for it in train for c in it.captions]
    vcaps = [c for it in val for c in it.captions]
    eye = np.eye(len(tcaps), dtype=np.float32)
    _write(cfg, model, "train", [c.caption_id for c in tcaps], [c.text for c in tcaps], eye)
    vv = eye[: len(vcaps)]
    _write(cfg, model, "val", [c.caption_id for c in vcaps], [c.text for c in vcaps], vv)
    owner = {c.caption_id: it for it in train for c in it.captions}
    return [owner[tcaps[j].caption_id] for j in range(len(vcaps))]


def test_noop_gates_off(fx):
    cfg, train, val = fx
    m = build_method("noop", cfg)
    m.fit(train)
    qs = [q for it in val for q in it.queries()]
    out = m.predict(qs)
    assert len(out) == len(qs)
    assert all(b[blk]["enabled"] == 0.0 for b in out for blk in b)
    assert out[0] == default_board(0.0)


def test_mean_matches_masked_hand_computation(fx):
    cfg, train, val = fx
    m = build_method("mean", cfg)
    m.fit(train)
    pred = m.predict(val[0].queries())[0]
    labels = [complete_board(it.label) for it in train]
    for blk, knobs in pred.items():
        gates = [lab[blk]["enabled"] for lab in labels]
        assert pred[blk]["enabled"] == (1.0 if np.mean(gates) >= 0.5 else 0.0)
        on = [lab for lab in labels if lab[blk]["enabled"] >= 0.5]
        for knob, v in knobs.items():
            if knob == "enabled":
                continue
            want = np.mean([lab[blk][knob] for lab in on]) if on else default_board()[blk][knob]
            assert v == pytest.approx(want)
    assert m.info()["train_items"] == len(train)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_knn_k1_returns_neighbour_label(fx, name):
    cfg, train, val = fx
    owners = _one_hot_cache(cfg, MODELS[name], train, val)
    m = build_method(name, cfg)
    m.fit(train)
    m.set_hparams({"k": 1})
    qs = [q for it in val for q in it.queries()]
    out = m.predict(qs)
    for pred, owner in zip(out, owners, strict=True):
        lab = complete_board(owner.label)
        want = {b: dict(k, enabled=1.0 if k["enabled"] >= 0.5 else 0.0) for b, k in lab.items()}
        assert board_vector(pred) == pytest.approx(board_vector(want))
    info = m.info()
    assert info["model"] == MODELS[name] and info["k"] == 1
    assert info["train_captions"] == sum(len(it.captions) for it in train)
    assert m.hparam_grid() == [{"k": k} for k in cfg.eval.baselines.knn.k_grid]


def test_knn_select_picks_k1_when_val_copies_train(fx):
    cfg, train, val = fx
    owners = _one_hot_cache(cfg, MODELS["knn_bge"], train, val)
    # Make the copied neighbour also the val item's own label: then k = 1 is exact.
    # Each val item gets one caption, copying train caption j (so its owner's label).
    vitems = []
    for j, it in enumerate(val):
        cap = it.captions[0]
        item = dataclasses.replace(it.item, board_params=owners[j * len(it.captions)].label)
        vitems.append(dataclasses.replace(it, item=item, captions=(cap,)))
    m = build_method("knn_bge", cfg)
    m.fit(train)
    sel = m.select(vitems)
    assert sel["chosen"] == {"k": 1} and m.k == 1
    assert sel["scores"][0] == {"k": 1, "param_l1": 0.0, "gate_error": 0.0, "score": 0.0}
    assert m.info()["selection"]["criterion"].startswith("val masked param L1")


def test_knn_missing_cache_names_embed_command(tmp_path, fx):
    cfg, train, _ = fx
    cfg2 = eval_fixture_config(tmp_path)  # empty cache dir
    m = build_method("knn_clap", cfg2)
    with pytest.raises(FileNotFoundError, match="lstmabar eval embed"):
        m.fit(train)


def test_knn_stale_cache_rejected(fx, tmp_path):
    cfg, train, _ = fx
    cfg2 = eval_fixture_config(tmp_path)
    tcaps = [c for it in train for c in it.captions]
    EmbeddingTable(
        [c.caption_id for c in tcaps],
        np.eye(len(tcaps)),
        ["0" * 64] * len(tcaps),
        "bge-base-en-v1.5",
        "text",
    ).save(cache_path(cfg2.eval.embeddings.dir, "bge-base-en-v1.5", "train", "text"))
    with pytest.raises(ValueError, match="stale"):
        build_method("knn_bge", cfg2).fit(train)


def test_top_k_ties_and_aggregate_masking():
    table = np.array([[1, 0], [1, 0], [0, 1], [1, 0]], dtype=np.float32)
    assert top_k(np.array([[1.0, 0.0]]), table, 2).tolist() == [[0, 1]]
    assert top_k(np.array([[0.0, 1.0]]), table, 10).tolist() == [[2, 0, 1, 3]]
    # three neighbours: drive on in two with gain 0.2 / 0.4, off in one with gain 1.0
    di = BOARD_LAYOUT.index(("drive", "enabled"))
    gi = next(i for i, (b, k) in enumerate(BOARD_LAYOUT) if b == "drive" and k != "enabled")
    base = np.array(board_vector(default_board(0.0)))
    rows = np.stack([base.copy() for _ in range(3)])
    rows[:, di] = [1.0, 1.0, 0.0]
    rows[:, gi] = [0.2, 0.4, 1.0]
    out = aggregate(rows[None])[0]
    assert out[di] == 1.0 and out[gi] == pytest.approx(0.3)
    rows[:, di] = 0.0  # nobody enables drive: median over all, gate off
    out = aggregate(rows[None])[0]
    assert out[di] == 0.0 and out[gi] == pytest.approx(0.4)
