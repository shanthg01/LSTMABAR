"""P5 statistics (task 5.2): known-answer tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lstmabar.eval import stats as S


def test_bootstrap_ci_covers_mean_and_is_seeded():
    v = np.random.default_rng(0).normal(1.0, 1.0, size=200)
    ci = S.bootstrap_ci(v, n_resamples=2000, seed=3)
    assert ci.n == 200 and ci.lo < ci.mean < ci.hi
    assert ci.mean == pytest.approx(v.mean())
    assert 0.1 < ci.hi - ci.lo < 0.5  # ~ 2 * 1.96 / sqrt(200) = 0.28
    assert S.bootstrap_ci(v, n_resamples=2000, seed=3) == ci
    assert S.bootstrap_ci(v, n_resamples=2000, seed=4) != ci
    assert set(ci.to_dict()) == {"mean", "lo", "hi", "n"}


def test_bootstrap_ci_nan_and_small():
    ci = S.bootstrap_ci([1.0, float("nan"), 3.0], n_resamples=500)
    assert ci.n == 2 and ci.mean == 2.0 and 1.0 <= ci.lo <= ci.hi <= 3.0
    one = S.bootstrap_ci([5.0])
    assert (one.mean, one.lo, one.hi, one.n) == (5.0, 5.0, 5.0, 1)
    empty = S.bootstrap_ci([float("nan")])
    assert empty.n == 0 and math.isnan(empty.mean)


def test_bootstrap_chunking_large_n():
    v = np.random.default_rng(1).uniform(size=3000)
    ci = S.bootstrap_ci(v, n_resamples=10_000, seed=0)
    assert ci.lo < v.mean() < ci.hi


def test_paired_bootstrap_and_tests():
    rng = np.random.default_rng(2)
    b = rng.normal(size=100)
    a = b + 0.5 + rng.normal(scale=0.1, size=100)
    a[3] = np.nan
    d = S.paired_bootstrap(a, b, n_resamples=2000)
    assert d.n == 99 and d.lo > 0.4 and d.hi < 0.6
    assert S.wilcoxon_p(a, b) < 1e-10
    assert S.paired_d(a, b) > 3
    c = S.compare(a, b, n_resamples=2000)
    assert c.n == 99 and c.diff == d and c.p_wilcoxon < 1e-10
    assert set(c.to_dict()) == {"diff", "p_wilcoxon", "cliffs_delta", "d_paired", "n"}
    with pytest.raises(ValueError):
        S.paired_bootstrap([1.0, 2.0], [1.0])


def test_degenerate_paired_cases_are_nan():
    assert math.isnan(S.wilcoxon_p([1.0, 2.0], [1.0, 2.0]))
    assert math.isnan(S.wilcoxon_p([1.0], [0.0]))
    assert math.isnan(S.paired_d([1.0, 2.0], [0.0, 1.0]))  # SD of the differences is 0
    c = S.compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], n_resamples=100)
    assert c.diff.mean == 0.0 and math.isnan(c.p_wilcoxon) and c.cliffs_delta == 0.0


def test_cliffs_delta_extremes_and_ties():
    assert S.cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert S.cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert S.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    # brute force on random data with ties
    rng = np.random.default_rng(5)
    a, b = rng.integers(0, 5, 40).astype(float), rng.integers(0, 5, 30).astype(float)
    brute = np.mean(np.sign(a[:, None] - b[None, :]))
    assert S.cliffs_delta(a, b) == pytest.approx(brute)
    assert math.isnan(S.cliffs_delta([], [1.0]))


def test_seed_summary_and_overlap():
    s = S.seed_summary([1.0, 2.0, 3.0, float("nan")])
    assert s == {"mean": 2.0, "sd": 1.0, "n_seeds": 3}
    assert math.isnan(S.seed_summary([1.0])["sd"])
    assert S.ci_overlap(S.CI(0, 0, 1, 2), S.CI(0, 0.5, 2, 2))
    assert not S.ci_overlap(S.CI(0, 0, 1, 2), S.CI(0, 1.5, 2, 2))
