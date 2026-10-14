"""P5 task 5.5: embedding encoders and the ``lstmabar eval embed`` cache builder."""

from __future__ import annotations

import argparse
import copy
import hashlib

import numpy as np
import pytest

from lstmabar.eval import embed
from lstmabar.eval.data import gold_audio_key, load_gold_items, load_split
from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture
from lstmabar.models.embed_cache import (
    EmbeddingTable,
    audio_sha256,
    cache_path,
    load_table,
    lookup_queries,
    text_sha256,
)

DIM = 8


class FakeEncoder:
    """Deterministic hash embeddings; counts every input it embeds."""

    def __init__(self, name: str, dim: int = DIM):
        self.name = name
        self.dim = dim
        self.calls: list[int] = []

    def _vec(self, b: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(self.name.encode() + b).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_text(self, texts, batch_size=32):
        self.calls.append(len(texts))
        return np.stack([self._vec(t.encode()) for t in texts]).reshape(-1, self.dim)

    def encode_audio(self, audio, sample_rate, batch_size=16):
        assert sample_rate == 44100
        self.calls.append(len(audio))
        return np.stack([self._vec(np.float32(a).tobytes()) for a in audio]).reshape(-1, self.dim)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("embed_fx")
    cfg = eval_fixture_config(root, ["eval.embeddings.batch_size=4"])
    make_eval_fixture(cfg, seconds=0.5)
    return cfg


class _Made(dict):
    """name -> FakeEncoder, plus the list of loader calls."""

    def __init__(self):
        super().__init__()
        self.loads: list[str] = []


@pytest.fixture
def fakes(monkeypatch):
    made = _Made()
    loads = made.loads

    def load(name, device="cpu", lean=False):
        made.setdefault(name, FakeEncoder(name, 6 if name.startswith("bge") else DIM))
        loads.append(name)
        return made[name]

    monkeypatch.setattr(embed, "load_text_encoder", load)
    monkeypatch.setattr(embed, "load_audio_encoder", load)
    return made


def _n_embedded(made) -> int:
    return sum(sum(e.calls) for e in made.values())


def _args(**kw):
    base = dict(models=None, splits=None, limit=None, fixtures=False, overrides=[])
    base.update(kw)
    return argparse.Namespace(**base)


def test_plan_order():
    p = embed.plan(["c1"], ["b1"], embed.ALL_SPLITS)
    kinds = [k for _, _, k in p]
    assert kinds[: 2 * len(embed.ALL_SPLITS)] == ["text"] * 2 * len(embed.ALL_SPLITS)
    audio = [(s, k) for m, s, k in p if k != "text"]
    assert audio[:2] == [("gold", "audio"), ("gold", "wet")]
    assert audio[-1] == ("train", "audio")
    assert ("train", "wet") not in audio
    assert not any(m == "b1" and k != "text" for m, _, k in p)


def test_build_full_cache(fx, fakes, tmp_path):
    cfg = copy.deepcopy(fx)
    cfg.eval.embeddings.dir = str(tmp_path / "cache")
    cfg.eval.report_dir = str(tmp_path / "reports")
    assert embed.run(cfg, _args()) == 0
    cdir = tmp_path / "cache"

    # text: every caption, keyed by caption_id, sha of the text
    for split in ("train", "val", "test", "test_unseen_pedal"):
        items = load_split(cfg, split)
        caps = {c.caption_id: c.text for it in items for c in it.captions}
        for m in ("larger_clap_general", "clap-htsat-unfused", "bge-base-en-v1.5"):
            t = load_table(cdir, m, split, "text")
            assert sorted(t.ids) == sorted(caps)
            assert t.sha256[t.ids.index(next(iter(caps)))] == text_sha256(caps[next(iter(caps))])
            np.testing.assert_allclose(np.linalg.norm(t.vectors, axis=1), 1.0, rtol=1e-5)
        # audio: dry at the item's level; sha of that exact array
        t = load_table(cdir, "clap-htsat-unfused", split, "audio")
        assert t.ids == sorted(it.item_id for it in items)
        it0 = items[0]
        assert t.sha256[0] == audio_sha256(it0.queries_for_audio().load_dry())
        if split != "train":
            w = load_table(cdir, "clap-htsat-unfused", split, "wet")
            assert w.ids == t.ids
            assert w.sha256[0] == audio_sha256(it0.load_wet())
    assert not cache_path(cdir, "clap-htsat-unfused", "train", "wet").exists()
    assert not cache_path(cdir, "bge-base-en-v1.5", "val", "audio").exists()

    gold = load_gold_items(cfg)
    gt = load_table(cdir, "bge-base-en-v1.5", "gold", "text")
    assert gt.ids == sorted(g.gold_id for g in gold)
    assert gt.dim == 6
    ga = load_table(cdir, "larger_clap_general", "gold", "audio")
    keys = {gold_audio_key(c.clip_id, -15.0) for g in gold for c in g.clips}
    assert set(ga.ids) == keys
    gw = load_table(cdir, "larger_clap_general", "gold", "wet")
    renders = {r.item_id for g in gold for c in g.clips for r in (c.target, *c.distractors)}
    assert set(gw.ids) == renders

    # the cache serves the Query keys
    v = lookup_queries(cdir, "clap-htsat-unfused", "audio", gold[0].queries())
    assert v.shape == (len(gold[0].clips), DIM)

    report = (tmp_path / "reports" / embed.REPORT_NAME).read_text(encoding="utf-8")
    assert "clap-htsat-unfused | gold | wet" in report
    assert str(tmp_path) not in report and ":\\" not in report

    # second run: nothing recomputed, files untouched
    n0 = _n_embedded(fakes)
    fakes.loads.clear()
    mtime = cache_path(cdir, "clap-htsat-unfused", "val", "audio").stat().st_mtime_ns
    stats = embed.build(cfg)
    assert _n_embedded(fakes) == n0
    assert all(s.n_computed == 0 for s in stats)
    assert fakes.loads == []  # lazy: a fully cached rerun loads no encoder
    assert cache_path(cdir, "clap-htsat-unfused", "val", "audio").stat().st_mtime_ns == mtime


def test_incremental_stale_and_prune(fx, fakes, tmp_path):
    cfg = copy.deepcopy(fx)
    cfg.eval.embeddings.dir = str(tmp_path / "cache")
    cdir = tmp_path / "cache"
    embed.build(cfg, models=["bge-base-en-v1.5"], splits=["val"])
    path = cache_path(cdir, "bge-base-en-v1.5", "val", "text")
    t = EmbeddingTable.load(path)
    n = len(t)
    # corrupt one sha (as if the caption text changed), drop one row, add a dead id
    sha = list(t.sha256)
    sha[0] = "0" * 64
    ids = list(t.ids)
    ids[-1] = "dead-id"
    EmbeddingTable(ids, t.vectors, sha, t.model, t.kind).save(path)
    stats = embed.build(cfg, models=["bge-base-en-v1.5"], splits=["val"])
    (s,) = stats
    assert s.n_computed == 2 and s.n_kept == n - 2 and s.n_pruned == 1
    t2 = EmbeddingTable.load(path)
    assert t2.ids == t.ids and t2.sha256 == t.sha256
    np.testing.assert_allclose(t2.vectors, t.vectors, atol=1e-6)

    # --limit never prunes rows beyond the limit
    stats = embed.build(cfg, models=["bge-base-en-v1.5"], splits=["val"], limit=1)
    assert stats[0].n_computed == 0 and len(EmbeddingTable.load(path)) == n


def test_periodic_save_and_resume(fx, fakes, tmp_path, monkeypatch):
    cfg = copy.deepcopy(fx)
    cfg.eval.embeddings.dir = str(tmp_path / "cache")
    cfg.eval.embeddings.save_every = 2  # audio batches are batch_size // 2 = 2
    cdir = tmp_path / "cache"
    items = load_split(cfg, "train")
    assert len(items) >= 4
    real = embed.audio_rows

    def interrupted(cfg_, split, limit):
        for i, row in enumerate(real(cfg_, split, limit)):
            if i == len(items) - 1:
                raise KeyboardInterrupt
            yield row

    monkeypatch.setattr(embed, "SOURCES", {**embed.SOURCES, "audio": interrupted})
    with pytest.raises(KeyboardInterrupt):
        embed.build(cfg, models=["clap-htsat-unfused"], splits=["train"])
    path = cache_path(cdir, "clap-htsat-unfused", "train", "audio")
    partial = EmbeddingTable.load(path)
    assert 2 <= len(partial) < len(items)
    monkeypatch.setattr(embed, "SOURCES", {**embed.SOURCES, "audio": real})
    stats = embed.build(cfg, models=["clap-htsat-unfused"], splits=["train"])
    audio = next(s for s in stats if s.kind == "audio")
    assert audio.n_kept == len(partial)
    assert len(EmbeddingTable.load(path)) == len(items)


def test_wrong_model_or_dim_rebuilds(fx, fakes, tmp_path):
    cfg = copy.deepcopy(fx)
    cfg.eval.embeddings.dir = str(tmp_path / "cache")
    path = cache_path(tmp_path / "cache", "bge-base-en-v1.5", "gold", "text")
    EmbeddingTable(["fx001"], np.ones((1, 3)), ["x"], "BAAI/bge-base-en-v1.5", "text").save(path)
    (s,) = embed.build(cfg, models=["bge-base-en-v1.5"], splits=["gold"])
    assert s.n_computed == len(load_gold_items(cfg))
    assert EmbeddingTable.load(path).dim == 6


def test_bad_args(fx, fakes):
    with pytest.raises(ValueError, match="not in eval.embeddings"):
        embed.build(fx, models=["nope"])
    with pytest.raises(ValueError, match="unknown splits"):
        embed.build(fx, splits=["nope"])


def test_fixtures_flag(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(embed.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "fx"))
    from lstmabar.config import load_config
    from lstmabar.eval.fixtures import EVAL_CONFIG

    cfg = load_config(EVAL_CONFIG)
    rc = embed.run(cfg, _args(fixtures=True, models=["bge-base-en-v1.5"], splits=["gold"]))
    assert rc == 0
    assert (tmp_path / "fx" / "cache" / "bge" / "bge-base-en-v1.5" / "gold_text.npz").exists()
    assert (tmp_path / "fx" / "reports" / embed.REPORT_NAME).exists()


# ---------------------------------------------------------------------- real encoders


def _real(name):
    pytest.importorskip("transformers")
    from lstmabar.models.encoders import is_cached

    if not is_cached(name):
        pytest.skip(f"{name} not in the local HF cache")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["clap-htsat-unfused", "larger_clap_general"])
def test_clap_encoder(name, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _real(name)
    from lstmabar.models.encoders import load_audio_encoder, load_text_encoder

    enc = load_text_encoder(name)
    assert enc is not None and enc.dim == 512
    t = enc.encode_text(["bright clean guitar", "heavy distortion"], batch_size=1)
    assert t.shape == (2, 512) and t.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(t[:1], enc.encode_text(["bright clean guitar"]), atol=1e-5)
    a_enc = load_audio_encoder(name)
    rng = np.random.default_rng(0)
    x = [0.1 * rng.standard_normal(44100).astype(np.float32) for _ in range(3)]
    a = a_enc.encode_audio(x, 44100, batch_size=2)
    assert a.shape == (3, 512) and a.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(a[:1], a_enc.encode_audio(x[:1], 44100), atol=1e-4)
    del a_enc
    # lean loaders (what `eval embed` uses) give the same vectors as the full model
    lean_t = load_text_encoder(name, lean=True)
    np.testing.assert_allclose(lean_t.encode_text(["heavy distortion"]), t[1:], atol=1e-5)
    with pytest.raises(RuntimeError, match="audio tower"):
        lean_t.encode_audio(x[:1], 44100)
    del lean_t
    lean_a = load_audio_encoder(name, lean=True)
    np.testing.assert_allclose(lean_a.encode_audio(x, 44100, batch_size=2), a, atol=1e-4)


def test_bge_encoder(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _real("bge-base-en-v1.5")
    from lstmabar.models.encoders import load_audio_encoder, load_text_encoder

    enc = load_text_encoder("bge-base-en-v1.5")
    v = enc.encode_text(["make it brighter", "make it brighter", "add fuzz"])
    assert v.shape == (3, 768) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(v[0], v[1], atol=1e-5)
    with pytest.raises(KeyError, match="no audio tower"):
        load_audio_encoder("bge-base-en-v1.5")
