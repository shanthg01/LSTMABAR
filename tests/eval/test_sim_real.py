"""P5 wave 1 (e): test_real sim-to-real gap (D11) on the eval fixture's two test_real items."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from lstmabar.eval import sim_real as S
from lstmabar.eval.data import load_test_real
from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("simreal")
    cfg = eval_fixture_config(
        root,
        [
            "eval.sim_real.input_peak_sweep_dbfs=[-18.0,-6.0]",
            "eval.stats.n_resamples=200",
            f"eval.report_dir={(root / 'reports').as_posix()}",
        ],
    )
    make_eval_fixture(cfg, seconds=0.5)
    return cfg, root


def test_bootstrap_ci_nan_aware():
    ci = S.bootstrap_ci([1.0, float("nan"), 3.0, None], n_resamples=500, seed=0)
    assert ci.n == 2 and ci.mean == 2.0 and ci.lo <= 2.0 <= ci.hi
    one = S.bootstrap_ci([5.0])
    assert (one.mean, one.lo, one.hi, one.n) == (5.0, 5.0, 5.0, 1)
    assert S.bootstrap_ci([]).n == 0
    a = S.bootstrap_ci(np.arange(10.0), n_resamples=300, seed=1)
    assert a == S.bootstrap_ci(np.arange(10.0), n_resamples=300, seed=1)  # seeded


def test_descriptor_terms():
    e = S.descriptor_errors(
        {"centroid_hz": 2000.0, "odd_even_db": 3.0, "hnr_db": None},
        {"centroid_hz": 1000.0, "odd_even_db": -1.0, "hnr_db": 10.0},
    )
    assert e["centroid_oct"] == pytest.approx(1.0)
    assert e["odd_even_db"] == pytest.approx(4.0)
    assert math.isnan(e["hnr_db"])
    sh = S.descriptor_shift({"centroid_hz": 500.0, "crest_factor_db": 6.0}, {"centroid_hz": 1000.0})
    assert sh["centroid_oct"] == pytest.approx(-1.0) and math.isnan(sh["crest_db"])


def test_select_items_deterministic_per_group(fx):
    cfg, _ = fx
    items = load_test_real(cfg)
    assert len(S.select_items(items, None, None)) == 2
    one = S.select_items(items, 1, None)
    assert len(one) == 2  # one per group, two groups
    assert [r.item.item_id for r in one] == [r.item.item_id for r in S.select_items(items, 5, 1)]


def _run(fx, monkeypatch, engines):
    cfg, root = fx
    import lstmabar.data.render as R

    monkeypatch.setattr(R, "librosa_available", lambda: False)  # no pYIN: fast, as in CI
    monkeypatch.setattr(S, "KB_ENGINES", {"ts808": engines, "rat": ("greybox",)})
    assert S.run(cfg, SimpleNamespace(limit=None, workers=1)) == 0
    res = json.loads((root / "reports" / "sim_real.json").read_text(encoding="utf-8"))
    md = (root / "reports" / "sim_real.md").read_text(encoding="utf-8")

    kb = res["kb"]["fixture:Dev0"]
    assert kb["kb_pedal"] == "ts808" and kb["n_items"] == 1 and not kb["unseen_pedal"]
    keys = {(c["engine"], c["level"]) for c in kb["cells"]}
    for eng in (*engines, "greybox_default"):
        assert {(eng, "-18"), (eng, "-6"), (eng, "best")} <= keys
    assert ("dry", "-") in keys
    by = {(c["engine"], c["level"]): c for c in kb["cells"]}
    for c in by.values():
        assert c["mrstft"]["n"] == 1 and c["mrstft"]["mean"] > 0
    for eng in engines:
        best = by[(eng, "best")]["mrstft"]["mean"]
        lv = min(by[(eng, lv)]["mrstft"]["mean"] for lv in ("-18", "-6"))
        assert best == pytest.approx(lv, rel=1e-5)

    # the internal-only non-KB item: aggregate family shifts only, no per-item rows
    assert "fixture:Dev1" in res["family_shifts"]
    assert res["groups"]["fixture:Dev1"]["internal_only"] is True
    ids = [r["item_id"] for r in res["per_item"]]
    assert ids == ["fx-real-0"]
    assert "fx-real-1" not in json.dumps(res) and "fx-real-1" not in md
    assert str(root) not in md and root.as_posix() not in json.dumps(res)
    assert "optimistic" in md


def test_pod_kind_and_crop_rule():
    it = SimpleNamespace(dataset="podset", dry_path="podset/input/s1_0-input.flac")
    assert S.pod_kind(it) == "s1"
    assert S.pod_kind(SimpleNamespace(dataset="egfxset", dry_path="x/a_1.flac")) is None


def test_run_greybox_aggregate_only_for_internal(fx, monkeypatch):
    _run(fx, monkeypatch, ("greybox",))


@pytest.mark.slow
def test_run_with_whitebox(fx, monkeypatch):
    _run(fx, monkeypatch, ("whitebox", "greybox"))
