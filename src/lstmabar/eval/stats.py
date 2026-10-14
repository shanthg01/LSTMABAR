"""P5 statistics (task 5.2, D3). Wave 0: signatures only; wave 1 (a) implements them.

The unit is the item (a synthetic ``item_id`` or a gold id): per-item scores are averaged over
the item's captions / clips / distractors before any of these run. Everything is NaN-aware
(NaN items are dropped and ``n`` counts the rest) and seeded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CI:
    mean: float
    lo: float
    hi: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Comparison:
    """Method A vs B on paired per-item scores (``diff = A - B``)."""

    diff: CI  # paired bootstrap CI of the mean difference
    p_wilcoxon: float  # two-sided Wilcoxon signed-rank p-value
    cliffs_delta: float  # in [-1, 1]
    d_paired: float  # mean paired difference / SD of the differences
    n: int

    def to_dict(self) -> dict:
        return {**asdict(self), "diff": self.diff.to_dict()}


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CI:
    """Percentile bootstrap CI of the mean over items. ``n < 2`` gives ``lo = hi = mean``
    (NaN mean for ``n = 0``)."""
    raise NotImplementedError


def paired_bootstrap(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CI:
    """Bootstrap CI of ``mean(a - b)`` resampling items (pairs with a NaN on either side are
    dropped). ``a`` and ``b`` must be aligned by item."""
    raise NotImplementedError


def wilcoxon_p(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on paired scores (NaN if all differences are
    zero or fewer than 2 pairs)."""
    raise NotImplementedError


def cliffs_delta(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Cliff's delta ``P(a > b) - P(a < b)`` over all (a_i, b_j) pairs."""
    raise NotImplementedError


def paired_d(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Mean paired difference / SD of the differences (NaN when the SD is 0)."""
    raise NotImplementedError


def compare(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Comparison:
    """Everything D3 reports for a paired comparison."""
    raise NotImplementedError


def seed_summary(per_seed: Sequence[float]) -> dict[str, float]:
    """``{"mean", "sd", "n_seeds"}`` of per-seed aggregate scores (reported next to the item
    CIs for trained methods, D3)."""
    raise NotImplementedError


def ci_overlap(a: CI, b: CI) -> bool:
    """True when two CIs overlap (the P6 exit-gate test is *non*-overlapping CIs)."""
    return not (a.hi < b.lo or b.hi < a.lo)


__all__ = [
    "CI",
    "Comparison",
    "bootstrap_ci",
    "ci_overlap",
    "cliffs_delta",
    "compare",
    "paired_bootstrap",
    "paired_d",
    "seed_summary",
    "wilcoxon_p",
]
