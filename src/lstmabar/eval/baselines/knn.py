"""kNN caption-retrieval baselines (D4): ``KnnClap`` (CLAP text tower) and ``KnnBge``
(``bge-base-en-v1.5``). Text only.

``fit(train)`` loads the cached train caption embeddings (``lstmabar eval embed`` builds them;
nothing is embedded here) and pairs every train caption with its item's label. ``predict``
looks up each query's cached text embedding (per-split tables, ``Query.split``), takes the k
train captions with the highest cosine similarity (ties: lower table index) and aggregates:

- knobs: per knob the median over the neighbours whose block is enabled (label gate >= 0.5);
  if none enables it, the median over all k neighbours;
- gates: on iff at least half the neighbours enable the block (output 1.0 / 0.0).

k selection (D8, val only, never gold): the runner's ``select_hparams`` walks
:meth:`KnnMethod.hparam_grid` with the val gold-proxy score; :meth:`KnnMethod.select` is a
cheap internal fallback that picks the k minimizing **mean masked param L1 + mean gate error
on val captions** (per item first, then over items; no rendering). The report must state
which of the two criteria picked the reported k (``info()["selection"]["criterion"]``).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from lstmabar.data.schema import FloatBoard
from lstmabar.dsp.pedalboard import ENABLED
from lstmabar.eval.baselines.base import (
    BOARD_LAYOUT,
    Method,
    board_vector,
    complete_board,
    default_board,
    vector_board,
)
from lstmabar.eval.data import Query, SynthItem
from lstmabar.models import embed_cache

EMBED_HINT = "run `lstmabar eval embed` first"
CHUNK = 256  # queries per similarity matmul (256 x 45k float32 ~ 46 MB)

_BLOCKS: tuple[str, ...] = tuple(dict.fromkeys(b for b, _ in BOARD_LAYOUT))
_GATE_IDX = np.array([BOARD_LAYOUT.index((b, ENABLED)) for b in _BLOCKS])
_KNOB_IDX = np.array([i for i, (_, k) in enumerate(BOARD_LAYOUT) if k != ENABLED])
_KNOB_BLOCK = np.array([_BLOCKS.index(BOARD_LAYOUT[i][0]) for i in _KNOB_IDX])


def _label_matrix(boards: Sequence[FloatBoard]) -> np.ndarray:
    """``(N, len(BOARD_LAYOUT))`` float64 of complete boards."""
    if not boards:
        return np.zeros((0, len(BOARD_LAYOUT)))
    return np.array([board_vector(complete_board(b)) for b in boards], dtype=np.float64)


def top_k(queries: np.ndarray, table: np.ndarray, k: int, chunk: int = CHUNK) -> np.ndarray:
    """``(Q, k)`` indices of the k most cosine-similar rows of ``table`` per query, most
    similar first; ties broken by the lower row index (deterministic)."""
    n = table.shape[0]
    k = min(int(k), n)
    out = np.zeros((queries.shape[0], k), dtype=np.int64)
    if k == 0:
        return out
    t = embed_cache.l2_normalize(table)
    q = embed_cache.l2_normalize(queries)
    for s in range(0, q.shape[0], chunk):
        sims = q[s : s + chunk] @ t.T  # (c, N)
        if k < n:
            cand = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        else:
            cand = np.broadcast_to(np.arange(n), (sims.shape[0], n))
        # A tie at the k-th similarity may leave equal rows outside ``cand``; widen to every
        # row at or above the k-th value so the lower-index rule is exact.
        kth = np.take_along_axis(sims, cand, axis=1).min(axis=1, keepdims=True)
        for r in range(sims.shape[0]):
            rows = np.flatnonzero(sims[r] >= kth[r])
            order = np.lexsort((rows, -sims[r, rows]))
            out[s + r] = rows[order[:k]]
    return out


def aggregate(neighbour_labels: np.ndarray) -> np.ndarray:
    """``(Q, k, L)`` neighbour label vectors -> ``(Q, L)`` boards (median of enabled
    neighbours per knob, falling back to the median of all; majority gate)."""
    q, k, _ = neighbour_labels.shape
    out = np.zeros((q, len(BOARD_LAYOUT)))
    if k == 0:
        out[:] = board_vector(default_board(0.0))
        return out
    gates = neighbour_labels[:, :, _GATE_IDX] >= 0.5  # (Q, k, B)
    knobs = neighbour_labels[:, :, _KNOB_IDX]  # (Q, k, K)
    mask = gates[:, :, _KNOB_BLOCK]  # (Q, k, K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> fallback
        med = np.nanmedian(np.where(mask, knobs, np.nan), axis=1)
    med = np.where(np.isnan(med), np.median(knobs, axis=1), med)
    out[:, _KNOB_IDX] = med
    out[:, _GATE_IDX] = (gates.mean(axis=1) >= 0.5).astype(np.float64)
    return out


def masked_errors(pred: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: mean |Δ| over the knobs of blocks the **label** enables (NaN if none), and
    the fraction of blocks whose thresholded gate differs."""
    lg = label[:, _GATE_IDX] >= 0.5
    pg = pred[:, _GATE_IDX] >= 0.5
    m = lg[:, _KNOB_BLOCK]
    d = np.abs(pred[:, _KNOB_IDX] - label[:, _KNOB_IDX])
    with np.errstate(invalid="ignore", divide="ignore"):
        l1 = np.where(m, d, 0.0).sum(axis=1) / m.sum(axis=1)
    l1[m.sum(axis=1) == 0] = np.nan
    return l1, (lg != pg).mean(axis=1)


class KnnMethod(Method):
    """Shared kNN caption retrieval; subclasses set ``name`` and ``model_key``."""

    model_key: str = ""  # key under cfg.eval.baselines.knn

    def __init__(self, cfg: Any = None, seed: int = 0) -> None:
        super().__init__(cfg, seed)
        if cfg is None:
            raise ValueError(f"{type(self).__name__} needs the eval config")
        kcfg = cfg.eval.baselines.knn
        self.model = str(kcfg[self.model_key])
        self.k_grid = [int(k) for k in kcfg.k_grid]
        if not self.k_grid or min(self.k_grid) < 1:
            raise ValueError("eval.baselines.knn.k_grid must be non-empty positive ints")
        self.k = sorted(self.k_grid)[len(self.k_grid) // 2]  # until selected on val
        self.cache_dir = str(cfg.eval.embeddings.dir)
        self._vectors: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._n_items = 0
        self._selection: dict[str, Any] | None = None

    # ------------------------------------------------------------------ cache access

    def _table(self, split: str) -> embed_cache.EmbeddingTable:
        try:
            return embed_cache.load_table(self.cache_dir, self.model, split, "text")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{self.name}: no {self.model} {split} text embeddings; {EMBED_HINT}"
            ) from e

    def _query_vectors(self, queries: Sequence[Query]) -> np.ndarray:
        try:
            return embed_cache.lookup_queries(self.cache_dir, self.model, "text", queries)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{self.name}: {e}") from e
        except KeyError as e:
            raise KeyError(f"{self.name}: {e.args[0]} ({EMBED_HINT})") from e

    # ------------------------------------------------------------------ Method API

    def fit(self, train: Sequence[SynthItem]) -> None:
        table = self._table("train")
        ids, texts, boards = [], [], []
        for it in train:
            for c in it.captions:
                ids.append(c.caption_id)
                texts.append(c.text)
                boards.append(it.label)
        if not ids:
            raise ValueError(f"{self.name}.fit: train split has no captions")
        stale = table.stale(
            {i: embed_cache.text_sha256(t) for i, t in zip(ids, texts, strict=True)}
        )
        if stale:
            raise ValueError(
                f"{self.name}: {len(stale)} train captions missing or stale in the "
                f"{self.model} cache (first: {stale[0]!r}); {EMBED_HINT}"
            )
        self._vectors = embed_cache.l2_normalize(table.lookup(ids))
        self._labels = _label_matrix(boards)
        self._n_items = len(train)

    def hparam_grid(self) -> list[dict[str, Any]]:
        return [{"k": k} for k in self.k_grid]

    def set_hparams(self, hparams: dict[str, Any]) -> None:
        k = int(hparams["k"])
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._selection = {"criterion": "runner", "chosen": {"k": k}}

    def _predict_vectors(self, qv: np.ndarray, k: int) -> np.ndarray:
        if self._vectors is None or self._labels is None:
            raise RuntimeError(f"{self.name}.predict called before fit")
        idx = top_k(qv, self._vectors, k)
        return aggregate(self._labels[idx])

    def predict(self, queries: Sequence[Query]) -> list[FloatBoard]:
        if not queries:
            return []
        out = self._predict_vectors(self._query_vectors(queries), self.k)
        return [complete_board(vector_board(row)) for row in out]

    def select(self, val: Sequence[SynthItem]) -> dict[str, Any]:
        """Pick k on val captions by mean masked param L1 + mean gate error (per item, then
        over items; NaN L1 terms, i.e. all-off labels, are skipped). Ties: smaller k."""
        if self._vectors is None or self._labels is None:
            raise RuntimeError(f"{self.name}.select called before fit")
        queries = [q for it in val for q in it.queries()]
        if not queries:
            raise ValueError(f"{self.name}.select: val has no captions")
        units = np.array([q.unit_id for q in queries])
        by_item = {it.item_id: it.label for it in val}
        labels = _label_matrix([by_item[u] for u in units])
        qv = self._query_vectors(queries)
        kmax = max(self.k_grid)
        idx = top_k(qv, self._vectors, kmax)
        _, inv = np.unique(units, return_inverse=True)
        scores = []
        for k in sorted(self.k_grid):
            pred = aggregate(self._labels[idx[:, :k]])
            l1, gerr = masked_errors(pred, labels)
            l1_item = _group_nanmean(l1, inv)
            g_item = _group_nanmean(gerr, inv)
            row = {
                "k": k,
                "param_l1": float(np.nanmean(l1_item)) if np.isfinite(l1_item).any() else 0.0,
                "gate_error": float(np.mean(g_item)),
            }
            row["score"] = row["param_l1"] + row["gate_error"]
            scores.append(row)
        best = min(scores, key=lambda r: (r["score"], r["k"]))
        self.k = int(best["k"])
        self._selection = {
            "criterion": "val masked param L1 + gate error (KnnMethod.select, no rendering)",
            "chosen": {"k": self.k},
            "scores": scores,
            "n_val_items": len(val),
            "n_val_captions": len(queries),
        }
        return dict(self._selection)

    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "hf_id": embed_cache.HF_IDS.get(self.model, self.model),
            "k": self.k,
            "train_captions": 0 if self._vectors is None else int(self._vectors.shape[0]),
            "train_items": self._n_items,
        }
        if self._selection is not None:
            out["selection"] = self._selection
        return out


def _group_nanmean(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    n = int(groups.max()) + 1 if groups.size else 0
    ok = np.isfinite(x)
    s = np.bincount(groups[ok], weights=x[ok], minlength=n)
    c = np.bincount(groups[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


class KnnClap(KnnMethod):
    name = "knn_clap"
    model_key = "clap_model"


class KnnBge(KnnMethod):
    name = "knn_bge"
    model_key = "bge_model"


__all__ = ["KnnBge", "KnnClap", "KnnMethod", "aggregate", "masked_errors", "top_k"]
