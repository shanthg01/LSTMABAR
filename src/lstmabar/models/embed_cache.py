"""Embedding cache (P5 task 5.5, D5; wave 0 contract, used by kNN, Text2FX, the CLAP score and
the P6 model).

Layout: ``<cache_dir>/<family>/<model>/<split>_<kind>.npz`` with ``family`` ``clap`` or
``bge``, ``kind`` one of :data:`KINDS`, and arrays

- ``ids``: ``(N,)`` str (:mod:`lstmabar.eval.data` key conventions),
- ``vectors``: ``(N, D)`` float32, L2-normalized,
- ``sha256``: ``(N,)`` str, the sha256 of the **source content** each vector was computed from
  (text: :func:`text_sha256`; audio: :func:`audio_sha256` of the exact input array), so a
  changed caption or level is detected as stale instead of silently reused,
- ``model``: the Hugging Face id, ``kind``: the kind.

Splits: ``train``, ``val``, ``test``, ``test_unseen_pedal``, ``gold`` (and ``demo`` if ever
needed).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KINDS: tuple[str, ...] = ("text", "audio", "wet")
"""``text`` = captions / gold instructions; ``audio`` = the model's dry input (per item at
its level, gold clips at −15 dBFS); ``wet`` = target renders (CLAP-score metric)."""

FAMILIES: dict[str, str] = {
    "larger_clap_general": "clap",
    "clap-htsat-unfused": "clap",
    "bge-base-en-v1.5": "bge",
}
HF_IDS: dict[str, str] = {
    "larger_clap_general": "laion/larger_clap_general",
    "clap-htsat-unfused": "laion/clap-htsat-unfused",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audio_sha256(x: np.ndarray) -> str:
    """Of the float32 samples actually embedded (before the encoder's own resampling)."""
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def cache_path(cache_dir: str | Path, model: str, split: str, kind: str) -> Path:
    if model not in FAMILIES:
        raise KeyError(f"unknown embedding model {model!r}; known: {sorted(FAMILIES)}")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    return Path(cache_dir) / FAMILIES[model] / model / f"{split}_{kind}.npz"


@dataclass
class EmbeddingTable:
    ids: list[str]
    vectors: np.ndarray  # (N, D) float32
    sha256: list[str]
    model: str
    kind: str

    def __post_init__(self) -> None:
        # An empty table must still be 2-D: pass np.zeros((0, dim)).
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.ids):
            raise ValueError(f"vectors {self.vectors.shape} do not match {len(self.ids)} ids")
        if len(self.sha256) != len(self.ids):
            raise ValueError("sha256 and ids differ in length")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate ids in embedding table")
        self._index = {k: i for i, k in enumerate(self.ids)}

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def lookup(self, keys: Sequence[str]) -> np.ndarray:
        """``(len(keys), D)`` vectors; ``KeyError`` naming the first missing id."""
        try:
            idx = [self._index[k] for k in keys]
        except KeyError as e:
            raise KeyError(f"{self.model}/{self.kind}: no embedding for id {e.args[0]!r}") from e
        return self.vectors[idx]

    def stale(self, expected: Mapping[str, str]) -> list[str]:
        """Ids in ``expected`` (id -> source sha256) that are missing or whose sha differs."""
        return [
            k
            for k, h in expected.items()
            if k not in self._index or self.sha256[self._index[k]] != h
        ]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            ids=np.array(self.ids, dtype=str),
            vectors=self.vectors,
            sha256=np.array(self.sha256, dtype=str),
            model=np.array(self.model),
            kind=np.array(self.kind),
        )
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingTable:
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                ids=[str(s) for s in z["ids"]],
                vectors=z["vectors"],
                sha256=[str(s) for s in z["sha256"]],
                model=str(z["model"]),
                kind=str(z["kind"]),
            )


def load_table(cache_dir: str | Path, model: str, split: str, kind: str) -> EmbeddingTable:
    path = cache_path(cache_dir, model, split, kind)
    if not path.exists():
        raise FileNotFoundError(
            f"embedding cache {model}/{split}_{kind} not built; run `lstmabar eval embed`"
        )
    return EmbeddingTable.load(path)


def lookup_queries(cache_dir, model: str, kind: str, queries) -> np.ndarray:
    """``(len(queries), D)`` vectors for ``eval.data.Query`` s (``kind`` ``text`` uses
    ``q.text_key``, ``audio`` uses ``q.audio_key``), loading each split's table once."""
    attr = {"text": "text_key", "audio": "audio_key"}[kind]
    tables: dict[str, EmbeddingTable] = {}
    rows = []
    for q in queries:
        if q.split not in tables:
            tables[q.split] = load_table(cache_dir, model, q.split, kind)
        rows.append(tables[q.split].lookup([getattr(q, attr)])[0])
    dim = next(iter(tables.values())).dim if tables else 0
    return np.stack(rows) if rows else np.zeros((0, dim), dtype=np.float32)


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


__all__ = [
    "FAMILIES",
    "HF_IDS",
    "KINDS",
    "EmbeddingTable",
    "audio_sha256",
    "cache_path",
    "l2_normalize",
    "load_table",
    "lookup_queries",
    "text_sha256",
]
