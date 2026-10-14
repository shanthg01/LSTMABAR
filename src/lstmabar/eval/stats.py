"""P5 statistics (task 5.2, D3).

The unit is the item (a synthetic ``item_id`` or a gold id): per-item scores are averaged over
the item's captions / clips / distractors before any of these run. Everything is NaN-aware
(NaN items are dropped and ``n`` counts the rest) and seeded.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

NAN = float("nan")
_CHUNK_ELEMS = 4_000_000  # resample-index elements per chunk (bounded memory)


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


def _clean(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return v[np.isfinite(v)]


def _pairs(a, b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"paired scores must align: {a.shape} vs {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def _boot_means(v: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Means of ``n_resamples`` with-replacement resamples of ``v`` (vectorized, chunked)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n = v.size
    rows = max(1, _CHUNK_ELEMS // n)
    out = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, rows):
        m = min(rows, n_resamples - start)
        idx = rng.integers(0, n, size=(m, n))
        out[start : start + m] = v[idx].mean(axis=1)
    return out


def _ci(v: np.ndarray, n_resamples: int, seed: int, alpha: float) -> CI:
    n = int(v.size)
    if n == 0:
        return CI(NAN, NAN, NAN, 0)
    mean = float(v.mean())
    if n < 2 or n_resamples < 1:
        return CI(mean, mean, mean, n)
    means = _boot_means(v, int(n_resamples), seed)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(mean, float(lo), float(hi), n)


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CI:
    """Percentile bootstrap CI of the mean over items. ``n < 2`` gives ``lo = hi = mean``
    (NaN mean for ``n = 0``)."""
    return _ci(_clean(values), n_resamples, seed, alpha)


def paired_bootstrap(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CI:
    """Bootstrap CI of ``mean(a - b)`` resampling items (pairs with a NaN on either side are
    dropped). ``a`` and ``b`` must be aligned by item."""
    x, y = _pairs(a, b)
    return _ci(x - y, n_resamples, seed, alpha)


def wilcoxon_p(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on paired scores (NaN if all differences are
    zero or fewer than 2 pairs)."""
    from scipy.stats import wilcoxon

    x, y = _pairs(a, b)
    d = x - y
    if d.size < 2 or not np.any(d != 0):
        return NAN
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        return NAN
    return p if math.isfinite(p) else NAN


def cliffs_delta(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Cliff's delta ``P(a > b) - P(a < b)`` over all (a_i, b_j) pairs."""
    x, y = _clean(a), np.sort(_clean(b))
    if x.size == 0 or y.size == 0:
        return NAN
    less = np.searchsorted(y, x, side="left")  # b_j < a_i
    greater = y.size - np.searchsorted(y, x, side="right")  # b_j > a_i
    return float((less.sum() - greater.sum()) / (x.size * y.size))


def paired_d(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Mean paired difference / SD of the differences (NaN when the SD is 0)."""
    x, y = _pairs(a, b)
    d = x - y
    if d.size < 2:
        return NAN
    sd = float(d.std(ddof=1))
    return float(d.mean() / sd) if sd > 0 else NAN


def compare(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Comparison:
    """Everything D3 reports for a paired comparison (on the pairs where both are finite)."""
    x, y = _pairs(a, b)
    return Comparison(
        diff=_ci(x - y, n_resamples, seed, alpha),
        p_wilcoxon=wilcoxon_p(x, y),
        cliffs_delta=cliffs_delta(x, y),
        d_paired=paired_d(x, y),
        n=int(x.size),
    )


def seed_summary(per_seed: Sequence[float]) -> dict[str, float]:
    """``{"mean", "sd", "n_seeds"}`` of per-seed aggregate scores (reported next to the item
    CIs for trained methods, D3). ``sd`` is the sample SD (NaN for fewer than 2 seeds)."""
    v = _clean(per_seed)
    return {
        "mean": float(v.mean()) if v.size else NAN,
        "sd": float(v.std(ddof=1)) if v.size >= 2 else NAN,
        "n_seeds": int(v.size),
    }


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
