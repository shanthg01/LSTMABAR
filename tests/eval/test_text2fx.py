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
        root, ["eval.baselines.text2fx.steps=1", "eval.baselines.text2fx.lr=0.2"]
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
    assert (m.steps, m.lr, m.clap_model) == (1, 0.2, "clap-htsat-unfused")
    assert m.max_items == 4 and m.uses_audio and m.deterministic
    assert Text2Fx(None).steps == 200  # defaults without a config


def _score(m: Text2Fx, q, board, text: np.ndarray) -> float:
    """Similarity of exactly ``board`` (0/1 gates) with ``text``."""
    x = torch.from_numpy(q.load_dry().astype(np.float32))[None]
    v = torch.tensor([[board[b][k] for b, k in BOARD_LAYOUT]], dtype=torch.float32)
    t = torch.from_numpy(text)[None]
    with torch.no_grad():
        return float(m._similarity(m.audio_embedder, m._board(q.sample_rate), x, t, v, 44100)[0])


def test_predict_smoke_returns_scored_board(fx):
    cfg, items = fx  # 1 step
    q = items[0].queries()[0]
    text = _target_text(FakeAudioEmbedder(), q)
    m, calls = _method(cfg, {q.text: text})
    (board,) = m.predict([q])
    assert calls == [1]  # no cache in the fixture: the text tower fallback
    assert board == complete_board(board)  # complete, valid, clamped
    assert [(b, k) for b in board for k in board[b]] == list(BOARD_LAYOUT)
    assert all(board[b][ENABLED] in (0.0, 1.0) for b in board)
    info = m.info()
    assert info["n_optimized"] == 1 and info["text_embeddings"] == {"cache": 0, "encoder": 1}
    # The reported similarity is that of the returned (discrete-gate) board ...
    assert info["mean_sim_returned"] == pytest.approx(_score(m, q, board, text), abs=1e-5)
    # ... which is the best of all gate combinations with those knobs.
    for b in board:
        flipped = {k: dict(v) for k, v in board.items()}
        flipped[b][ENABLED] = 1.0 - flipped[b][ENABLED]
        assert _score(m, q, flipped, text) <= info["mean_sim_returned"] + 1e-5
    # Same (text, audio) again: memoized, no further optimization.
    assert m.predict([q]) == [board]
    assert m.info()["n_optimized"] == 1


def test_all_nan_similarity(fx):
    cfg, items = fx
    q = items[0].queries()[0]
    m, _ = _method(cfg, {q.text: np.full(DIM, np.nan, dtype=np.float32)})
    (board,) = m.predict([q])
    assert board == complete_board(board)
    info = m.info()
    assert np.isnan(info["mean_sim_best"]) and np.isnan(info["mean_sim_returned"])


@pytest.mark.slow
def test_optimization_increases_similarity(fx):
    cfg, items = fx
    q = items[0].queries()[0]
    m, _ = _method(cfg, {q.text: _target_text(FakeAudioEmbedder(), q)})
    m.steps, m.lr = 8, 0.2
    m.predict([q])
    info = m.info()
    assert info["mean_sim_best"] > info["mean_sim_init"] + 1e-3


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
    with pytest.raises(LimitExceeded, match="--limit"):
        m.predict(qs)
    # The limit is cumulative per split over the method's lifetime.
    m.audio_embedder = FakeAudioEmbedder()
    m.text_embedder = lambda ts: np.stack([_target_text(m.audio_embedder, qs[0])] * len(ts))
    m.predict(qs[:1])
    m.predict(qs[:1])  # same (text, audio) again: memoized, free
    with pytest.raises(LimitExceeded):
        m.predict(qs[1:2])
    # Distinct (text, audio) pairs are counted, not units: a second caption of the same
    # item is another optimization.
    caps = items[0].queries()
    if len(caps) > 1:
        with pytest.raises(LimitExceeded):
            m.predict(caps[1:2])
    m.max_items = None
    assert len(m.predict(qs)) == 2


def test_limit_is_not_run_in_eval_run():
    from lstmabar.eval.run import _cannot_run_types, _reason

    e = LimitExceeded("text2fx: too many")
    assert isinstance(e, _cannot_run_types())
    assert _reason(e).startswith("limit")


def test_clap_diff_train_is_noop():
    from lstmabar.models.clap_diff import ClapAudioDiff

    m = ClapAudioDiff.__new__(ClapAudioDiff)
    torch.nn.Module.__init__(m)
    m.model = torch.nn.Dropout(0.5)
    m.train()
    assert not m.training and not m.model.training
