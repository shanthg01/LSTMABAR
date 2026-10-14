"""D4 Text2FX-style baseline (eval/baselines/text2fx.py), with a fake differentiable encoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lstmabar.dsp.pedalboard import ENABLED, default_pedalboard
from lstmabar.eval.baselines.base import BOARD_LAYOUT, build_method, complete_board
from lstmabar.eval.baselines.text2fx import LimitExceeded, Text2Fx
from lstmabar.eval.data import load_split
from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture
from lstmabar.models.embed_cache import EmbeddingTable, cache_path, text_sha256

DIM = 16


class FakeAudioEmbedder:
    """Fixed random projection of the mean log-magnitude STFT (differentiable)."""

    def __init__(self, dim: int = DIM, n_fft: int = 512):
        g = torch.Generator().manual_seed(0)
        self.n_fft = n_fft
        self.w = torch.randn(n_fft // 2 + 1, dim, generator=g)
        self.window = torch.hann_window(n_fft)

    def __call__(self, y: torch.Tensor, sr: int) -> torch.Tensor:
        s = torch.stft(y, self.n_fft, self.n_fft // 4, window=self.window, return_complex=True)
        feat = (s.abs() + 1e-5).log().mean(-1)
        feat = feat - feat.mean(-1, keepdim=True)
        return torch.nn.functional.normalize(feat @ self.w, dim=-1)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("t2fx")
    cfg = eval_fixture_config(
        root, ["eval.baselines.text2fx.steps=8", "eval.baselines.text2fx.lr=0.2"]
    )
    make_eval_fixture(cfg, seconds=0.5)
    return cfg, load_split(cfg, "test")


def _target_text(audio_embed, q) -> np.ndarray:
    """A text 'embedding' that is the fake audio embedding of a heavy-drive render."""
    board = default_pedalboard(q.sample_rate)
    x = torch.from_numpy(q.load_dry().astype(np.float32))[None]
    p = {"drive": {"gain_db": torch.tensor([0.95]), ENABLED: torch.tensor([1.0])}}
    with torch.no_grad():
        return audio_embed(board(x, p), q.sample_rate)[0].numpy()


def _method(cfg, texts: dict[str, np.ndarray] | None = None) -> tuple[Text2Fx, list[int]]:
    m = build_method("text2fx", cfg)
    m.audio_embedder = FakeAudioEmbedder()
    calls: list[int] = []

    def text_embedder(ts):
        calls.append(len(ts))
        return np.stack([texts[t] for t in ts])

    m.text_embedder = text_embedder
    return m, calls


def test_registry_and_config(fx):
    cfg, _ = fx
    m = build_method("text2fx", cfg)
    assert isinstance(m, Text2Fx)
    assert (m.steps, m.lr, m.clap_model) == (8, 0.2, "clap-htsat-unfused")
    assert m.max_items == 60 and m.uses_audio and m.deterministic
    assert Text2Fx(None).steps == 200  # defaults without a config


def test_optimization_increases_similarity(fx):
    cfg, items = fx
    q = items[0].queries()[0]
    emb = FakeAudioEmbedder()
    m, calls = _method(cfg, {q.text: _target_text(emb, q)})
    (board,) = m.predict([q])
    assert calls == [1]  # no cache in the fixture: the text tower fallback
    assert board == complete_board(board)  # complete, valid, clamped
    assert [(b, k) for b in board for k in board[b]] == list(BOARD_LAYOUT)
    assert all(board[b][ENABLED] in (0.0, 1.0) for b in board)
    info = m.info()
    assert info["n_optimized"] == 1 and info["text_embeddings"] == {"cache": 0, "encoder": 1}
    assert info["mean_sim_best"] > info["mean_sim_init"] + 1e-3
    # Same (text, audio) again: memoized, no further optimization.
    assert m.predict([q]) == [board]
    assert m.info()["n_optimized"] == 1


def test_text_from_cache(fx):
    cfg, items = fx
    q = items[0].queries()[0]
    v = _target_text(FakeAudioEmbedder(), q)
    m, calls = _method(cfg, {})
    table = EmbeddingTable([q.text_key], v[None], [text_sha256(q.text)], m.clap_model, "text")
    path = cache_path(cfg.eval.embeddings.dir, m.clap_model, q.split, "text")
    table.save(path)
    try:
        np.testing.assert_allclose(m.text_embeddings([q])[0], v, atol=1e-6)
        assert calls == []
        # A stale sha falls back to the encoder.
        EmbeddingTable([q.text_key], v[None], ["0" * 64], m.clap_model, "text").save(path)
        m2, calls2 = _method(cfg, {q.text: v})
        m2.text_embeddings([q])
        assert calls2 == [1]
    finally:
        path.unlink()


def test_max_items_guard(fx):
    cfg, items = fx
    assert len(items) >= 2
    m = build_method("text2fx", cfg)
    m.max_items = 1
    qs = [it.queries()[0] for it in items[:2]]
    with pytest.raises(LimitExceeded, match="--limit 1"):
        m.predict(qs)
    # The limit is cumulative per split over the method's lifetime.
    m.audio_embedder = FakeAudioEmbedder()
    m.text_embedder = lambda ts: np.stack([_target_text(m.audio_embedder, qs[0])] * len(ts))
    m.steps = 1
    m.predict(qs[:1])
    m.predict(qs[:1])  # same unit again: fine
    with pytest.raises(LimitExceeded):
        m.predict(qs[1:2])
    m.max_items = None
    assert len(m.predict(qs)) == 2
