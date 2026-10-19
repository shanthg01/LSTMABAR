"""Text2FX-style D4 baseline: per-query optimization of the board against CLAP.

For each query, the normalized knobs **and** the block gates of ``default_pedalboard`` are
optimized as logits through a sigmoid (Adam, ``eval.baselines.text2fx.steps`` steps at
``lr``) to maximize the cosine similarity between the instruction's CLAP text embedding and
the CLAP audio embedding of the render on the query's dry input
(:class:`lstmabar.models.clap_diff.ClapAudioDiff`; the render is peak-normalized to
``render.wet_peak_dbfs`` first, as stored wets are, so CLAP cannot reward loudness). Gates are
soft (sigmoid, threshold-free) during optimization. The knobs of the step with the highest
similarity are kept and every 0/1 gate combination is scored with them (gates are a wet/dry
crossfade, so plain rounding would change the sound); the best combination is returned and
``info()["mean_sim_returned"]`` is the similarity of exactly the returned boards.

Init: every knob at its default, every gate at 0.5 (logit 0). Nothing is random (no init
noise, deterministic CLAP forward), so the method is ``deterministic``.

Text embeddings come from the embedding cache (``lstmabar eval embed``; per split, checked
against the text's sha256) and fall back to the CLAP text tower for missing or stale ids.

**Cost limit.** Each optimization costs ``steps`` CLAP forward/backward passes (measured
locally: ~26 min for 200 steps on a 3 s clip, 2 threads, loaded machine), so the method
refuses (:class:`LimitExceeded`, recorded by ``eval run`` as "not run: limit ...") more than
``max_items`` distinct (text, audio) optimizations per split over its lifetime; cap the run
with ``--limit``, or set ``max_items: null`` (Colab).
Expected weak beyond drive amount (spike A); kept for that contrast.
"""

from __future__ import annotations

import itertools
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
    "max_items": 4,
    "device": "auto",
}
WET_PEAK_DBFS = -1.0  # fallback when the config has no render.wet_peak_dbfs
LOGIT_EPS = 1e-3
SECONDS_PER_STEP = 8.0  # measured: 1563 s / 200 steps, 3 s clip, 2 threads, loaded machine
_GATE_IDX = [i for i, (_, k) in enumerate(BOARD_LAYOUT) if k == ENABLED]

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


def _key(q: Query) -> tuple[str, str]:
    return (q.text, q.audio_key)


def _logit(p: float) -> float:
    p = min(max(float(p), LOGIT_EPS), 1.0 - LOGIT_EPS)
    return math.log(p / (1.0 - p))


def resolve_device(name: str) -> torch.device:
    """``auto`` -> CUDA if available, else CPU; otherwise the named device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
        self.device = resolve_device(str(p["device"]))
        self.cache_dir = _get(cfg, "eval.embeddings.dir", None)
        self.wet_peak = float(_get(cfg, "render.wet_peak_dbfs", WET_PEAK_DBFS))
        # Injectable for tests; loaded lazily from the local HF cache otherwise.
        self.audio_embedder: AudioEmbedder | None = None
        self.text_embedder: TextEmbedder | None = None
        self._boards: dict[int, Pedalboard] = {}
        self._keys: dict[str, set[tuple[str, str]]] = {}
        self._memo: dict[tuple[str, str], FloatBoard] = {}
        self._stats: list[dict[str, float]] = []
        self._text_source = {"cache": 0, "encoder": 0}

    # ------------------------------------------------------------------ embedders

    def _load_audio_embedder(self) -> AudioEmbedder:
        if self.audio_embedder is None:
            from lstmabar.models.clap_diff import ClapAudioDiff
            from lstmabar.models.encoders import ClapEncoder

            enc = ClapEncoder(self.clap_model, device=str(self.device), towers=("audio",))
            self.audio_embedder = ClapAudioDiff(enc).to(self.device)
        return self.audio_embedder

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.text_embedder is None:
            from lstmabar.models.encoders import ClapEncoder

            enc = ClapEncoder(self.clap_model, device=str(self.device), towers=("text",))
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

    def _similarity(self, embed, board, x, t, v: torch.Tensor, sr: int) -> torch.Tensor:
        """``(B,)`` cosine of the peak-normalized render of board vectors ``v`` ``(B, P)``
        on ``x`` ``(1, T)`` with the text embedding ``t`` ``(1, D)``."""
        y = board(x.expand(v.shape[0], -1), board.from_vector(v))
        peak = 10.0 ** (self.wet_peak / 20.0)
        y = y * (peak / y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8))
        return (embed(y, sr) * t).sum(-1)

    def optimize(
        self, dry: np.ndarray, sample_rate: int, text_emb: np.ndarray
    ) -> tuple[FloatBoard, dict[str, float]]:
        """Optimize one board for one dry clip and one text embedding.

        Gates are a wet/dry crossfade, so rounding the soft gates would change the sound (and
        round(0.5) = 0 would switch untouched blocks off): with the knobs of the best soft
        step, every 0/1 gate combination is scored and the best is returned
        (``sim_returned`` is the similarity of exactly the returned board)."""
        embed = self._load_audio_embedder()
        sr = int(sample_rate)
        board = self._board(sr)
        init = default_board(0.5)
        dev = self.device
        logits = torch.tensor(
            [_logit(init[b][k]) for b, k in BOARD_LAYOUT], dtype=torch.float32, device=dev
        ).requires_grad_(True)
        x = torch.from_numpy(np.asarray(dry, dtype=np.float32))[None].to(dev)
        t = torch.from_numpy(np.asarray(text_emb, dtype=np.float32))[None].to(dev)
        opt = torch.optim.Adam([logits], lr=self.lr)
        best_sim, best = -math.inf, torch.sigmoid(logits.detach())
        first = math.nan
        t0 = time.perf_counter()
        for step in range(self.steps + 1):
            v = torch.sigmoid(logits)
            sim = self._similarity(embed, board, x, t, v[None], sr)[0]
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
        # Discrete gates: score all 2^n combinations with the best soft step's knobs.
        combos = torch.tensor(
            list(itertools.product((0.0, 1.0), repeat=len(_GATE_IDX))), device=dev
        )
        cand = best[None].repeat(combos.shape[0], 1)
        cand[:, _GATE_IDX] = combos
        sims = []
        with torch.no_grad():
            for c in cand:  # one at a time: CLAP memory on a 16 GB machine
                sims.append(float(self._similarity(embed, board, x, t, c[None], sr)[0]))
        finite = [i for i, s in enumerate(sims) if math.isfinite(s)]
        pick = max(finite, key=lambda i: sims[i]) if finite else 0
        out: FloatBoard = {}
        for (b, k), val in zip(BOARD_LAYOUT, cand[pick].tolist(), strict=True):
            out.setdefault(b, {})[k] = float(val)
        stats = {
            "sim_init": first,
            "sim_best": best_sim if math.isfinite(best_sim) else math.nan,
            "sim_returned": sims[pick] if finite else math.nan,
            "seconds": time.perf_counter() - t0,
        }
        return complete_board(out), stats

    # ------------------------------------------------------------------ Method API

    def _check_limit(self, queries: Sequence[Query]) -> None:
        """At most ``max_items`` distinct (text, audio) optimizations per split, counted over
        the method's lifetime (memoized repeats are free)."""
        if self.max_items is None:
            return
        for split in {q.split for q in queries}:
            keys = self._keys.get(split, set()) | {_key(q) for q in queries if q.split == split}
            if len(keys) > self.max_items:
                raise LimitExceeded(
                    f"text2fx: {len(keys)} distinct (text, audio) optimizations requested on "
                    f"split {split!r}, more than eval.baselines.text2fx.max_items="
                    f"{self.max_items}. Each costs {self.steps} CLAP forward/backward passes "
                    f"(~{self.steps * SECONDS_PER_STEP / 60:.0f} min on the local CPU at the "
                    f"measured ~{SECONDS_PER_STEP:.0f} s/step); cap the run with --limit or set "
                    "eval.baselines.text2fx.max_items=null (Colab)"
                )

    def predict(self, queries: Sequence[Query]) -> list[FloatBoard]:
        queries = list(queries)
        self._check_limit(queries)
        for q in queries:
            self._keys.setdefault(q.split, set()).add(_key(q))
        todo: dict[tuple[str, str], Query] = {}
        for q in queries:
            key = _key(q)
            if key not in self._memo and key not in todo:
                todo[key] = q
        if todo:
            qs = list(todo.values())
            temb = self.text_embeddings(qs)
            for key, q, te in zip(todo, qs, temb, strict=True):
                board, stats = self.optimize(q.load_dry(), q.sample_rate, te)
                self._memo[key] = board
                self._stats.append(stats)
        return [{b: dict(k) for b, k in self._memo[_key(q)].items()} for q in queries]

    def info(self) -> dict[str, Any]:
        st = self._stats

        def mean(k: str) -> float:
            v = [s[k] for s in st if math.isfinite(s[k])]
            return float(np.mean(v)) if v else math.nan

        return {
            "name": self.name,
            "clap_model": self.clap_model,
            "steps": self.steps,
            "lr": self.lr,
            "max_items": self.max_items,
            "device": str(self.device),
            "n_optimized": len(st),
            "seconds_per_query": mean("seconds"),
            "mean_sim_init": mean("sim_init"),
            "mean_sim_best": mean("sim_best"),  # soft gates (optimization objective)
            "mean_sim_returned": mean("sim_returned"),  # the returned 0/1-gate boards
            "text_embeddings": dict(self._text_source),
        }


__all__ = ["LimitExceeded", "Text2Fx"]
