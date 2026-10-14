"""P6 stage 1 (6.1-6.2): predictor, dataset, losses, training loop, ModelMethod round-trip.

Built on the eval fixture with fake (deterministic, hash-derived) embedding tables.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lstmabar.config import load_config
from lstmabar.eval.baselines.base import BOARD_LAYOUT, board_vector, build_method
from lstmabar.eval.data import Query, load_split
from lstmabar.eval.fixtures import make_eval_fixture
from lstmabar.models.method import ModelMethod
from lstmabar.models.predictor import (
    BoardHeadLayout,
    ParamPredictor,
    PredictorConfig,
    load_checkpoint,
)
from lstmabar.train import stage1
from lstmabar.train.dataset import Stage1Dataset, item_weight, load_tables
from lstmabar.train.losses import gate_bce, masked_l1, param_loss

STAGE1 = Path(__file__).resolve().parents[2] / "configs" / "train" / "stage1.yaml"


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage1fx")
    cfg = stage1.fixture_config(load_config(STAGE1), root)
    make_eval_fixture(cfg, seconds=0.5)
    stage1.write_fixture_embeddings(cfg)
    return cfg


def _cfg(fx, **over):
    dots = [f"{k}={v}" for k, v in over.items()]
    base = {
        "train": {"epochs": 3, "batch_size": 4, "patience": 100, "num_threads": 1},
        "model": {"adapter_dim": 32, "hidden_dim": 64},
    }
    return OmegaConf.merge(fx, base, OmegaConf.from_dotlist(dots))


def _data(cfg, split="train"):
    items = load_split(cfg, split)
    text, audio = load_tables(cfg, split)
    return items, Stage1Dataset(items, text, audio, float(cfg.train.clamped_weight))


# ============================================================================ predictor


def test_board_layout_and_output_range():
    lay = BoardHeadLayout.default()
    assert lay.layout == BOARD_LAYOUT
    assert lay.n_knobs + lay.n_gates == len(BOARD_LAYOUT)
    for mapping in ("physics", "unconstrained"):
        m = ParamPredictor(PredictorConfig(8, 6, mapping=mapping))
        out = m(torch.randn(5, 8) * 10, torch.randn(5, 6) * 10)
        board = m.board(out).detach()
        assert board.shape == (5, len(BOARD_LAYOUT))
        assert float(board.min()) >= 0.0 and float(board.max()) <= 1.0


def test_input_ablations_build_only_needed_adapters():
    t = ParamPredictor(PredictorConfig(8, 6, inputs="text"))
    assert t.audio_adapter is None and t(torch.randn(2, 8), None)["knobs"].shape[0] == 2
    a = ParamPredictor(PredictorConfig(8, 6, inputs="audio"))
    assert a.text_adapter is None and a(None, torch.randn(2, 6))["knobs"].shape[0] == 2
    with pytest.raises(ValueError):
        PredictorConfig(8, 6, mapping="nope")


def test_unconstrained_mapping_round_trip():
    lay = BoardHeadLayout.default()
    u = torch.rand(7, lay.n_knobs, dtype=torch.float64)
    z = lay.normalized_to_linear(u)
    assert torch.allclose(lay.linear_to_normalized(z), u, atol=1e-9)
    # linear-taper knobs: identical in both spaces; log-taper knobs: not
    for j, s in enumerate(lay.specs):
        same = torch.allclose(z[:, j], u[:, j], atol=1e-9)
        assert same == (s.taper == "linear")
    # out-of-range values clamp into [0, 1]
    big = lay.linear_to_normalized(torch.full((1, lay.n_knobs), 3.0, dtype=torch.float64))
    assert torch.allclose(big, torch.ones_like(big))


# ============================================================================ data / losses


def test_labels_and_masks_match_items(fx):
    items, ds = _data(fx)
    lay = BoardHeadLayout.default()
    for i, it in enumerate(items):
        vec = board_vector(it.label)
        for j, (pos, g) in enumerate(zip(lay.knob_index, lay.knob_block, strict=True)):
            b = lay.blocks[g]
            on = it.label[b]["enabled"] >= 0.5
            assert float(ds.knob_mask[i, j]) == float(on)
            assert float(ds.knobs[i, j]) == pytest.approx(vec[pos])
        for g, b in enumerate(lay.blocks):
            assert float(ds.gates[i, g]) == float(it.label[b]["enabled"] >= 0.5)
    assert ds.text.shape[0] == sum(len(i.captions) for i in items)


def test_clamped_weight_applies_to_greybox_only(fx):
    items, _ = _data(fx)
    base = items[0]
    grey = dataclasses.replace(base, item=dataclasses.replace(base.item, clamped=True))
    white = dataclasses.replace(
        base, item=dataclasses.replace(base.item, clamped=True, engine="whitebox")
    )
    assert item_weight(base, 0.25) == 1.0
    assert item_weight(grey, 0.25) == 0.25
    assert item_weight(white, 0.25) == 1.0
    text, audio = load_tables(fx, "train")
    ds = Stage1Dataset([grey, white, *items[1:]], text, audio, clamped_weight=0.25)
    assert ds.weights.tolist()[:3] == [0.25, 1.0, 1.0]


def test_masked_l1_ignores_masked_entries_and_weights():
    pred = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    tgt = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    assert float(masked_l1(pred, tgt, mask)) == pytest.approx(1.0 / 3.0)
    w = torch.tensor([1.0, 0.0])
    assert float(masked_l1(pred, tgt, mask, w)) == pytest.approx(0.0)
    assert float(masked_l1(pred, tgt, torch.zeros_like(mask))) == 0.0
    logits = torch.zeros(2, 3)
    assert float(gate_bce(logits, torch.ones(2, 3))) == pytest.approx(np.log(2.0))
    total, parts = param_loss(pred, logits, tgt, torch.ones(2, 3), mask, gate_weight=2.0)
    assert parts["loss"] == pytest.approx(1.0 / 3.0 + 2.0 * np.log(2.0))
    assert float(total) == pytest.approx(parts["loss"])


def test_caption_sampling_and_audio_shuffle(fx):
    _, ds = _data(fx)
    g = torch.Generator().manual_seed(0)
    picks = torch.stack([ds.sample_captions(g) for _ in range(50)])
    assert torch.equal(ds.cap_item[picks], torch.arange(len(ds)).expand_as(picks))
    assert (picks != picks[0]).any()  # both captions get sampled
    ds.set_epoch(torch.Generator().manual_seed(1), permute_audio=True)
    assert ds._audio_perm is not None and sorted(ds._audio_perm.tolist()) == list(range(len(ds)))
    row = ds[0]
    assert torch.equal(row["audio"], ds.audio[ds._audio_perm[0]])
    n_eval = sum(len(b) for b in ds.eval_batches(3))
    assert n_eval == ds.text.shape[0]


# ============================================================================ training


def test_overfit_and_loss_decreases(fx):
    over = {"train.epochs": 150, "train.batch_size": 8, "train.lr": 1e-2, "model.dropout": 0.0}
    cfg = _cfg(fx, **over)
    _, ds = _data(cfg)
    s = stage1.train_model(cfg, ds, ds)  # val = train: overfit check
    hist_first = stage1.evaluate(
        stage1.build_model(cfg, ds.text.shape[1], ds.audio.shape[1]), ds, cfg
    )
    assert s["val"]["score"] < 0.25 * hist_first["score"]
    assert s["val"]["param_l1"] < 0.05
    assert s["val"]["gate_acc"] == 1.0


def test_determinism_with_fixed_seed(fx):
    cfg = _cfg(fx)
    _, tr = _data(cfg)
    _, va = _data(cfg, "val")
    a = stage1.train_model(cfg, tr, va)["model"].state_dict()
    b = stage1.train_model(cfg, tr, va)["model"].state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    c = stage1.train_model(_cfg(fx, seed=1), tr, va)["model"].state_dict()
    assert any(not torch.equal(a[k], c[k]) for k in a)


@pytest.mark.parametrize(
    "over", [{"model.mapping": "unconstrained"}, {"model.shuffle_audio": True}]
)
def test_ablation_configs_train(fx, over):
    cfg = _cfg(fx, **over)
    _, tr = _data(cfg)
    _, va = _data(cfg, "val")
    s = stage1.train_model(cfg, tr, va)
    assert np.isfinite(s["val"]["score"])


def test_select_split_never_gold_or_test(fx):
    with pytest.raises(ValueError):
        stage1.load_data(_cfg(fx, **{"eval.select_split": "gold"}))
    with pytest.raises(ValueError):
        stage1.load_data(_cfg(fx, **{"eval.select_split": "test"}))


# ======================================================================= checkpoint + method


def _queries(cfg, split="val"):
    return [q for it in load_split(cfg, split) for q in it.queries()]


def test_checkpoint_round_trip_through_model_method(fx, tmp_path):
    cfg = _cfg(fx)
    tr, va = stage1.load_data(cfg)
    s = stage1.train_model(cfg, tr, va, run_dir=tmp_path)
    for f in ("metrics.json", "summary.json", "checkpoint.pt"):
        assert (tmp_path / f).exists()
    model, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    assert payload["meta"]["config"]["model"]["inputs"] == "both"
    assert "sha" in payload["meta"]["git"]

    qs = _queries(cfg)
    m = build_method(
        "p6_model",
        OmegaConf.merge(cfg, {"eval": {"model": {"checkpoint": str(tmp_path / "checkpoint.pt")}}}),
    )
    assert isinstance(m, ModelMethod) and m.uses_audio and m.deterministic
    boards = m.predict(qs)
    assert len(boards) == len(qs)
    got = np.array([board_vector(b) for b in boards])
    with torch.no_grad():
        ref = torch.cat(
            [s["model"].board(s["model"](b.text, b.audio)) for b in va.eval_batches(64)]
        )
    np.testing.assert_allclose(got, ref.double().numpy(), atol=1e-6)
    info = m.info()
    assert info["checkpoint"] == "checkpoint.pt" and info["inputs"] == "both"


def test_model_method_missing_queries(fx, tmp_path):
    cfg = _cfg(fx, **{"model.inputs": "text"})
    tr, va = stage1.load_data(cfg)
    stage1.train_model(cfg, tr, va, run_dir=tmp_path)
    q = _queries(cfg)[0]
    demo = Query(
        **{
            **dataclasses.asdict(q),
            "text": "a brand new demo text",
            "text_key": "demo-0",
            "split": "demo",
        }
    )
    m = ModelMethod(cfg, checkpoint=tmp_path / "checkpoint.pt", encode_missing=False)
    assert not m.uses_audio
    with pytest.raises(KeyError):
        m.predict([demo])
    m2 = ModelMethod(cfg, checkpoint=tmp_path / "checkpoint.pt")
    try:
        from lstmabar.models import encoders

        encoders.load_text_encoder("clap-htsat-unfused")
    except NotImplementedError:
        with pytest.raises(RuntimeError, match="not implemented"):
            m2.predict([demo])
    except Exception:  # noqa: BLE001 - implemented but no model available: not tested here
        pass
    # a stale cached text (same key, different text) is not silently reused
    stale = Query(**{**dataclasses.asdict(q), "text": q.text + " changed"})
    with pytest.raises(KeyError):
        m.predict([stale])


def test_seeded_checkpoint_template(fx, tmp_path):
    cfg = _cfg(fx)
    tr, va = stage1.load_data(cfg)
    stage1.train_model(cfg, tr, va, run_dir=tmp_path / "s3")
    m = ModelMethod(cfg, seed=3, checkpoint=str(tmp_path / "s{seed}" / "checkpoint.pt"))
    assert not m.deterministic and m.checkpoint_path.parent.name == "s3"
    with pytest.raises(FileNotFoundError):
        ModelMethod(cfg, seed=4, checkpoint=str(tmp_path / "s{seed}" / "checkpoint.pt"))
    with pytest.raises(ValueError):
        ModelMethod(None).predict([])
