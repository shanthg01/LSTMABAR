"""P6 stage 2 (6.3): audio crops, L_audio through the pedalboard, training loop, checkpoint
compatibility with ``ModelMethod`` (p6_model).

Built on the eval fixture (0.5 s clips) with fake embeddings and a tiny stage-1 checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lstmabar.config import load_config
from lstmabar.data import levels
from lstmabar.dsp.pedalboard import default_pedalboard
from lstmabar.eval.baselines.base import board_vector, build_method
from lstmabar.eval.data import load_split
from lstmabar.eval.fixtures import make_eval_fixture
from lstmabar.models.method import ModelMethod
from lstmabar.models.predictor import load_checkpoint
from lstmabar.train import stage1, stage2

STAGE2 = Path(__file__).resolve().parents[2] / "configs" / "train" / "stage2.yaml"


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage2fx")
    cfg = stage2.fixture_config(load_config(STAGE2), root)
    cfg = OmegaConf.merge(
        cfg,
        {
            "model": {"adapter_dim": 32, "hidden_dim": 64},
            "train": {"num_threads": 1, "batch_size": 4, "epochs": 2, "patience": 100},
            "stage2": {"num_workers": 0, "select": {"mrstft_items": 2}},
        },
    )
    make_eval_fixture(cfg, seconds=0.5)
    stage1.write_fixture_embeddings(cfg)
    tr, va = stage1.load_data(cfg)
    stage1.train_model(cfg, tr, va, run_dir=root / "s1" / "seed0")
    cfg.stage2.init_checkpoint = str(root / "s1" / "seed{seed}" / "checkpoint.pt")
    return cfg


def _cfg(fx, **over):
    return OmegaConf.merge(fx, OmegaConf.from_dotlist([f"{k}={v}" for k, v in over.items()]))


def _setup(cfg):
    cfg = OmegaConf.merge(cfg, {})  # resolve_init edits cfg.model in place
    init, payload = stage2.resolve_init(cfg)
    tr, va = stage2.load_data(cfg)
    return cfg, init, payload, tr, va


# ============================================================================ data


def test_crops_align_with_stored_audio(fx):
    items = load_split(fx, "train")
    crops = stage2.AudioCrops(items)
    dry_full, wet_full = crops.load(0)
    stored_wet = items[0].load_wet()
    np.testing.assert_allclose(dry_full, items[0].load_dry()[: dry_full.size])
    np.testing.assert_allclose(wet_full, stored_wet[: wet_full.size])
    assert levels.peak_dbfs(dry_full) == pytest.approx(items[0].item.input_peak_dbfs, abs=1e-6)
    n = dry_full.size
    # start of the clip: the pre-roll is zero padding
    d, w = crops[(0, 0.0, 4000, 100)]
    assert d.shape == (4100,) and w.shape == (4000,)
    assert torch.all(d[:100] == 0)
    np.testing.assert_allclose(d[100:].numpy(), dry_full[:4000], atol=1e-6)
    # end of the clip: the pre-roll is real dry
    d, w = crops[(0, 1.0, 4000, 100)]
    np.testing.assert_allclose(d.numpy(), dry_full[n - 4100 :], atol=1e-6)
    np.testing.assert_allclose(w.numpy(), wet_full[n - 4000 :], atol=1e-6)
    # crops longer than the clip are cut to the clip
    d, w = crops[(0, 0.5, 10 * n, 0)]
    assert w.shape == (n,)
    dry, wet = stage2.collate_crops([(torch.ones(110), torch.ones(100)), (d, w)])
    assert dry.shape == (2, 110) and wet.shape == (2, 100)


def test_epoch_batches_are_seeded():
    a = stage2.epoch_batches(10, 4, torch.Generator().manual_seed(0), 100, [1.0, 1.5], 5)
    b = stage2.epoch_batches(10, 4, torch.Generator().manual_seed(0), 100, [1.0, 1.5], 5)
    assert a == b and [len(x) for x in a] == [4, 4, 2]
    assert sorted(k[0] for x in a for k in x) == list(range(10))
    for x in a:
        assert len({k[2] for k in x}) == 1 and 100 <= x[0][2] <= 150


# ============================================================================ losses / grads


def _batch(cfg, tr, n_pre=64):
    crops = stage2.AudioCrops(tr.items)
    idx = list(range(min(4, len(tr))))
    dry, wet = stage2.collate_crops([crops[(i, 0.3, 8192, n_pre)] for i in idx])
    b = tr._gather(torch.tensor(idx), None, None)
    return b, dry, wet


def test_gradients_reach_predictor_through_pedalboard(fx):
    cfg, init, _, tr, _ = _setup(fx)
    board = default_pedalboard(int(cfg.audio.sample_rate))
    b, dry, wet = _batch(cfg, tr)
    init.train()
    out = init(b.text, b.audio)
    y = stage2.render(board, init, out, dry)[:, 64:]
    loss = stage2.audio_loss(y, wet, cfg.eval.metrics.mrstft_fft_sizes)
    loss.backward()
    for name, p in init.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for blk, head in init.heads.items():
        g = head.weight.grad
        # every knob row and the gate row (last) of every block head receives gradient
        assert (g.abs().sum(dim=1) > 0).all(), blk
    assert init.text_adapter[0].weight.grad.abs().sum() > 0


def test_audio_loss_overfits_a_batch(fx):
    cfg, init, _, tr, _ = _setup(fx)
    torch.manual_seed(0)
    board = default_pedalboard(int(cfg.audio.sample_rate))
    b, dry, wet = _batch(cfg, tr)
    opt = torch.optim.Adam(init.parameters(), lr=3e-3)
    fft = cfg.eval.metrics.mrstft_fft_sizes
    losses = []
    for _ in range(25):
        out = init(b.text, b.audio)
        loss = stage2.audio_loss(stage2.render(board, init, out, dry)[:, 64:], wet, fft)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]


def test_peak_normalize_is_scale_invariant():
    x = torch.randn(2, 4096)
    w = torch.randn(2, 4096)
    a = stage2.audio_loss(x, w, (512, 1024))
    b = stage2.audio_loss(3.0 * x, 0.1 * w, (512, 1024))
    assert float(a) == pytest.approx(float(b), rel=1e-4)


# ============================================================================ training


def _train(cfg, run_dir=None, **over):
    cfg = _cfg(cfg, **{"stage2.max_steps": 3, **over})
    cfg, init, payload, tr, va = _setup(cfg)
    s = stage2.train_model(cfg, tr, va, run_dir, init=init, init_payload=payload)
    return cfg, s


def test_train_writes_checkpoint_scored_by_model_method(fx, tmp_path):
    cfg, s = _train(fx, run_dir=tmp_path / "seed0", **{"train.lr": 1e-3})
    for f in ("metrics.json", "summary.json", "checkpoint.pt"):
        assert (tmp_path / "seed0" / f).exists()
    assert s["steps_run"] == 3 and np.isfinite(s["val"]["mrstft"])
    assert s["val"]["score"] <= s["val_init"]["score"]
    _, payload = load_checkpoint(tmp_path / "seed0" / "checkpoint.pt")
    assert payload["meta"]["stage"] == 2 and payload["meta"]["text_model"]

    tmpl = str(tmp_path / "seed{seed}" / "checkpoint.pt")
    m = build_method("p6_model", OmegaConf.merge(cfg, {"eval": {"model": {"checkpoint": tmpl}}}))
    assert isinstance(m, ModelMethod) and not m.deterministic
    qs = [q for it in load_split(cfg, "val") for q in it.queries()]
    got = np.array([board_vector(b) for b in m.predict(qs)])
    _, va = stage2.load_data(cfg)
    with torch.no_grad():
        ref = torch.cat(
            [s["model"].board(s["model"](b.text, b.audio)) for b in va.eval_batches(64)]
        )
    np.testing.assert_allclose(got, ref.double().numpy(), atol=1e-6)


def test_determinism_with_fixed_seed(fx):
    a = _train(fx)[1]["model"].state_dict()
    b = _train(fx)[1]["model"].state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_export_checkpoint_and_guards(fx, tmp_path):
    exp = str(tmp_path / "export" / "seed{seed}" / "checkpoint.pt")
    _train(fx, run_dir=tmp_path / "run", **{"stage2.export_checkpoint": exp})
    assert (tmp_path / "export" / "seed0" / "checkpoint.pt").exists()
    with pytest.raises(NotImplementedError):
        _train(fx, **{"train.loss.clap_weight": 0.1})
    with pytest.raises(ValueError):
        stage2.load_data(_cfg(fx, **{"eval.select_split": "gold"}))
    with pytest.raises(FileNotFoundError):
        stage2.resolve_init(_cfg(fx, seed=7))


def test_resolve_init_takes_model_from_checkpoint(fx):
    cfg = _cfg(fx, **{"model.text_model": "larger_clap_general", "model.inputs": "text"})
    init, _ = stage2.resolve_init(cfg)
    assert str(cfg.model.text_model) == str(fx.model.text_model)
    assert str(cfg.model.inputs) == init.config.inputs == "both"


@pytest.mark.slow
def test_cli_fixtures_smoke(tmp_path):
    import argparse

    cfg = load_config(STAGE2, [f"paths.runs={tmp_path.as_posix()}"])
    assert stage2.run(cfg, argparse.Namespace(fixtures=True, limit=None)) == 0
    assert list((tmp_path / "train_stage2").glob("*/checkpoint.pt"))
