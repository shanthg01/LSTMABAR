"""Scoring a method on a split (wave 0 contract; wave 1 (a) implements it).

Per query: predict a board -> render it with ``default_pedalboard`` on the query's dry input
(at the query's level, D1) -> metrics against the target. Per-query scores are averaged into
one :class:`UnitScores` per statistical unit (synthetic item / gold item, D3) before any
statistics. Renders of predictions are never written to the dataset tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from lstmabar.data.schema import FloatBoard
from lstmabar.eval.baselines.base import Method
from lstmabar.eval.data import GoldItem, Query, SynthItem

SYNTH_SCORES: tuple[str, ...] = (
    "param_l1",
    "gate_acc",
    "mrstft",
    "centroid_oct",
    "odd_even_db",
    "hnr_db",
    "archetype_l1",
)
"""Per-unit keys of :func:`score_synthetic` (``gate_acc`` = :func:`metrics.gate_correct`;
gate F1 is computed over the whole split from ``preds``/``labels``)."""

GOLD_SCORES: tuple[str, ...] = (
    "afc_mrstft",
    "afc_descriptor",
    "region_l1",
    "mrstft_target",
)
"""Per-unit keys of :func:`score_gold` (2AFC with the primary and secondary distance, region
distance, MR-STFT to the target render). Per-axis 2AFC keys are added as
``afc_mrstft/<axis>`` when the item has a distractor on that axis."""


@dataclass
class UnitScores:
    unit_id: str
    split: str  # synthetic split name | "gold"
    method: str
    seed: int
    scores: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)  # gold_source, subsets, internal_only, ...

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SplitResult:
    split: str
    method: str
    seed: int
    units: list[UnitScores]
    preds: dict[str, FloatBoard]  # query_id -> board (gate F1, region checks, the report)
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. gate F1 per block


def predict_all(method: Method, queries: Sequence[Query], batch_size: int = 64):
    """``{query_id: board}`` for every query, in batches."""
    raise NotImplementedError


def score_synthetic(
    method: Method, items: Sequence[SynthItem], cfg, split: str, seed: int = 0
) -> SplitResult:
    raise NotImplementedError


def score_gold(method: Method, items: Sequence[GoldItem], cfg, seed: int = 0) -> SplitResult:
    """Gold 2AFC etc. Also scores val gold-proxy items (``proxy.load_proxy_items``: GoldItems
    with ``split == "val"``); the result's ``split`` is the items' split."""
    raise NotImplementedError


def score_val(method: Method, val: Sequence[SynthItem], cfg, seed: int = 0) -> SplitResult:
    """The D8 selection run: :func:`score_synthetic` on val plus :func:`score_gold` on the val
    proxy items (merged per unit: ``afc_*`` keys join the synthetic keys)."""
    raise NotImplementedError


def select_hparams(method: Method, val: Sequence[SynthItem], cfg, seed: int = 0) -> dict:
    """Try every ``method.hparam_grid()`` entry on val, keep the one minimizing
    ``proxy.selection_score`` (ties: first in grid order), apply it, and return
    ``{"chosen": {...}, "scores": [...]}``; falls back to ``method.select(val)`` when the grid
    is empty."""
    raise NotImplementedError


__all__ = [
    "GOLD_SCORES",
    "SYNTH_SCORES",
    "SplitResult",
    "UnitScores",
    "predict_all",
    "score_gold",
    "score_synthetic",
    "score_val",
    "select_hparams",
]
