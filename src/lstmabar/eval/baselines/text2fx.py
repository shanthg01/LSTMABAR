"""Text2FX-style D4 baseline: per-query optimization of the board against CLAP.

For each query, the normalized knobs **and** the block gates of ``default_pedalboard`` are
optimized as logits through a sigmoid (Adam, ``eval.baselines.text2fx.steps`` steps at
``lr``) to maximize the cosine similarity between the instruction's CLAP text embedding and
the CLAP audio embedding of the render on the query's dry input
(:class:`lstmabar.models.clap_diff.ClapAudioDiff`; the render is peak-normalized to
``render.wet_peak_dbfs`` first, as stored wets are, so CLAP cannot reward loudness). Gates are
soft (sigmoid, threshold-free) during optimization and rounded to 0/1 at the end; the
returned board is the one from the step with the highest similarity.

Init: every knob at its default, every gate at 0.5 (logit 0). Nothing is random (no init
noise, deterministic CLAP forward), so the method is ``deterministic``.

Text embeddings come from the embedding cache (``lstmabar eval embed``; per split, checked
against the text's sha256) and fall back to the CLAP text tower for missing or stale ids.

**Cost limit.** Each query costs ``steps`` CLAP forward/backward passes, so the method refuses
to optimize more than ``max_items`` units (items / gold items) per split over its lifetime
(the run step must cap splits with ``--limit``); ``max_items: null`` lifts the limit (Colab).
Expected weak beyond drive amount (spike A); kept for that contrast.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from lstmabar.data.schema import FloatBoard
from lstmabar.dsp.pedalboard import ENABLED, Pedalboard, default_pedalboard
from lstmabar.eval.baselines.base import BOARD_LAYOUT, Method, complete_board, default_board
from lstmabar.eval.data import Query
from lstmabar.models import embed_cache

DEFAULTS: dict[str, Any] = {
    "clap_model": "clap-htsat-unfused",
    "steps": 200,
    "lr": 0.05,
    "max_items": 60,
}
WET_PEAK_DBFS = -1.0  # fallback when the config has no render.wet_peak_dbfs
LOGIT_EPS = 1e-3

AudioEmbedder = Callable[[torch.Tensor, int], torch.Tensor]  # (B, T), sr -> (B, D) normalized
TextEmbedder = Callable[[Sequence[str]], np.ndarray]  # texts -> (N, D) normalized


def _get(cfg: Any, path: str, default: Any) -> Any:
    node = cfg
    for part in path.split("."):
        if node is None:
            return default
        try:
            node = node.get(part) if hasattr(node, "get") else getattr(node, part, None)
        except Exception:  # noqa: BLE001 - omegaconf missing keys
            return default
    return default if node is None else node


def _logit(p: float) -> float:
    p = min(max(float(p), LOGIT_EPS), 1.0 - LOGIT_EPS)
    return math.log(p / (1.0 - p))


class LimitExceeded(RuntimeError):
    """More units than ``eval.baselines.text2fx.max_items`` were requested for one split."""


class Text2Fx(Method):
    name = "text2fx"
    uses_audio = True
    deterministic = True

    def __init__(self, cfg: Any = None, seed: int = 0) -> None:
        super().__init__(cfg, seed)
        node = _get(cfg, "eval.baselines.text2fx", None)
        p = dict(DEFAULTS)
        if node is not None:
            for k in DEFAULTS:
                if k in node:
                    p[k] = node[k]
        self.clap_model = str(p["clap_model"])
        self.steps = int(p["steps"])
        self.lr = float(p["lr"])
        self.max_items = None if p["max_items"] is None else int(p["max_items"])
        self.cache_dir = _get(cfg, "eval.embeddings.dir", None)
        self.wet_peak = float(_get(cfg, "render.wet_peak_dbfs", WET_PEAK_DBFS))
        # Injectable for tests; loaded lazily from the local HF cache otherwise.
        self.audio_embedder: AudioEmbedder | None = None
        self.text_embedder: TextEmbedder | None = None
        self._boards: dict[int, Pedalboard] = {}
        self._units: dict[str, set[str]] = {}
        self._memo: dict[tuple[str, str], FloatBoard] = {}
        self._stats: list[dict[str, float]] = []
        self._text_source = {"cache": 0, "encoder": 0}

    # ------------------------------------------------------------------ embedders

    def _load_audio_embedder(self) -> AudioEmbedder:
        if self.audio_embedder is None:
            from lstmabar.models.clap_diff import ClapAudioDiff
            from lstmabar.models.encoders import ClapEncoder

            self.audio_embedder = ClapAudioDiff(ClapEncoder(self.clap_model, towers=("audio",)))
        return self.audio_embedder

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.text_embedder is None:
            from lstmabar.models.encoders import ClapEncoder

            enc = ClapEncoder(self.clap_model, towers=("text",))
            self.text_embedder = enc.encode_text
        return np.asarray(self.text_embedder(list(texts)), dtype=np.float32)

    def text_embeddings(self, queries: Sequence[Query]) -> np.ndarray:
        """``(N, D)``: cached vectors where present and fresh, the text tower otherwise."""
        out: list[np.ndarray | None] = [None] * len(queries)
        if self.cache_dir is not None:
            tables: dict[str, embed_cache.EmbeddingTable | None] = {}
            for i, q in enumerate(queries):
                if q.split not in tables:
                    try:
                        tables[q.split] = embed_cache.load_table(
                            self.cache_dir, self.clap_model, q.split, "text"
                        )
                    except (FileNotFoundError, KeyError):
                        tables[q.split] = None
                t = tables[q.split]
                if t is None or t.stale({q.text_key: embed_cache.text_sha256(q.text)}):
                    continue
                out[i] = t.lookup([q.text_key])[0]
        miss = [i for i, v in enumerate(out) if v is None]
        if miss:
            enc = self._encode_texts([queries[i].text for i in miss])
            for i, v in zip(miss, enc, strict=True):
                out[i] = v
        self._text_source["cache"] += len(queries) - len(miss)
        self._text_source["encoder"] += len(miss)
        if not queries:
            return np.zeros((0, 0), dtype=np.float32)
        return embed_cache.l2_normalize(np.stack(out))

    # ------------------------------------------------------------------ optimization

    def _board(self, sr: int) -> Pedalboard:
        if sr not in self._boards:
            self._boards[sr] = default_pedalboard(sr)
        return self._boards[sr]

    def optimize(
        self, dry: np.ndarray, sample_rate: int, text_emb: np.ndarray
    ) -> tuple[FloatBoard, dict[str, float]]:
        """Optimize one board for one dry clip and one text embedding."""
        embed = self._load_audio_embedder()
        board = self._board(int(sample_rate))
        init = default_board(0.5)
        logits = torch.tensor(
            [_logit(init[b][k]) for b, k in BOARD_LAYOUT], dtype=torch.float32
        ).requires_grad_(True)
        x = torch.from_numpy(np.asarray(dry, dtype=np.float32))[None]
        t = torch.from_numpy(np.asarray(text_emb, dtype=np.float32))[None]
        opt = torch.optim.Adam([logits], lr=self.lr)
        peak = 10.0 ** (self.wet_peak / 20.0)
        best_sim, best = -math.inf, torch.sigmoid(logits.detach())
        first = math.nan
        t0 = time.perf_counter()
        for step in range(self.steps + 1):
            v = torch.sigmoid(logits)
            params = board.from_vector(v[None])
            y = board(x, params)
            y = y * (peak / y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8))
            sim = (embed(y, int(sample_rate)) * t).sum()
            s = float(sim.detach())
            if step == 0:
                first = s
            if math.isfinite(s) and s > best_sim:
                best_sim, best = s, v.detach().clone()
            if step == self.steps:
                break  # the last iterate is evaluated, not stepped
            opt.zero_grad()
            (-sim).backward()
            if logits.grad is not None:
                logits.grad.nan_to_num_(0.0, 0.0, 0.0)
            opt.step()
        out: FloatBoard = {}
        for (b, k), val in zip(BOARD_LAYOUT, best.tolist(), strict=True):
            out.setdefault(b, {})[k] = float(round(val)) if k == ENABLED else float(val)
        stats = {
            "sim_init": first,
            "sim_best": best_sim,
            "seconds": time.perf_counter() - t0,
        }
        return complete_board(out), stats

    # ------------------------------------------------------------------ Method API

    def _check_limit(self, queries: Sequence[Query]) -> None:
        if self.max_items is None:
            return
        for split in {q.split for q in queries}:
            units = self._units.get(split, set()) | {q.unit_id for q in queries if q.split == split}
            if len(units) > self.max_items:
                raise LimitExceeded(
                    f"text2fx: {len(units)} units requested on split {split!r}, more than "
                    f"eval.baselines.text2fx.max_items={self.max_items} (each query costs "
                    f"{self.steps} CLAP forward/backward passes); cap the run with --limit "
                    f"{self.max_items} or set eval.baselines.text2fx.max_items=null"
                )

    def predict(self, queries: Sequence[Query]) -> list[FloatBoard]:
        queries = list(queries)
        self._check_limit(queries)
        for q in queries:
            self._units.setdefault(q.split, set()).add(q.unit_id)
        todo: dict[tuple[str, str], Query] = {}
        for q in queries:
            key = (q.text, q.audio_key)
            if key not in self._memo and key not in todo:
                todo[key] = q
        if todo:
            qs = list(todo.values())
            temb = self.text_embeddings(qs)
            for key, q, te in zip(todo, qs, temb, strict=True):
                board, stats = self.optimize(q.load_dry(), q.sample_rate, te)
                self._memo[key] = board
                self._stats.append(stats)
        return [{b: dict(k) for b, k in self._memo[(q.text, q.audio_key)].items()} for q in queries]

    def info(self) -> dict[str, Any]:
        st = self._stats

        def mean(k: str) -> float:
            return float(np.mean([s[k] for s in st])) if st else math.nan

        return {
            "name": self.name,
            "clap_model": self.clap_model,
            "steps": self.steps,
            "lr": self.lr,
            "max_items": self.max_items,
            "n_optimized": len(st),
            "seconds_per_query": mean("seconds"),
            "mean_sim_init": mean("sim_init"),
            "mean_sim_best": mean("sim_best"),
            "text_embeddings": dict(self._text_source),
        }


__all__ = ["LimitExceeded", "Text2Fx"]
