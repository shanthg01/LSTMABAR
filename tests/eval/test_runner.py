"""P5 scoring runner on the eval fixture."""

from __future__ import annotations

import math

import pytest

from lstmabar.eval import data as D
from lstmabar.eval import runner as R
from lstmabar.eval.baselines.base import Method, board_vector, default_board


class Const(Method):
    name = "const"

    def predict(self, queries):
        return [default_board(enabled=0.0) for _ in queries]


class Oracle(Method):
    """Predicts the true board of each query (synthetic label / gold target render)."""

    name = "oracle"

    def __init__(self, boards):
        super().__init__()
        self.boards = boards

    def predict(self, queries):
        return [self.boards[q.query_id] for q in queries]


class Switch(Method):
    """hparam ``use_oracle`` selects between the oracle and the no-op board."""

    name = "switch"

    def __init__(self, boards):
        super().__init__()
        self.oracle = Oracle(boards)
        self.use_oracle = False

    def hparam_grid(self):
        return [{"use_oracle": False}, {"use_oracle": True}]

    def set_hparams(self, hparams):
        self.use_oracle = bool(hparams["use_oracle"])

    def predict(self, queries):
        if self.use_oracle:
            return self.oracle.predict(queries)
        return Const().predict(queries)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture

    root = tmp_path_factory.mktemp("evalrunner")
    cfg = eval_fixture_config(root, ["eval.workers=1"])
    make_eval_fixture(cfg, seconds=0.5)
    return cfg


def _synth_oracle(items):
    return {q.query_id: it.label for it in items for q in it.queries()}


def test_predict_all_batches_and_completes(fx):
    items = D.load_split(fx, "test")
    qs = [q for it in items for q in it.queries()]
    seen = []

    class Rec(Const):
        def predict(self, queries):
            seen.append(len(queries))
            return [{"drive": {"enabled": 1.0}} for _ in queries]

    out = R.predict_all(Rec(), qs, batch_size=3)
    assert list(out) == [q.query_id for q in qs]
    assert max(seen) == 3 and sum(seen) == len(qs)
    assert out[qs[0].query_id]["drive"]["enabled"] == 1.0
    assert set(out[qs[0].query_id]) == {"compressor", "drive", "eq"}

    class Bad(Const):
        def predict(self, queries):
            return []

    with pytest.raises(ValueError):
        R.predict_all(Bad(), qs)


def test_score_synthetic_oracle_is_perfect(fx):
    items = D.load_split(fx, "test")
    res = R.score_synthetic(Oracle(_synth_oracle(items)), items, fx, "test")
    assert res.split == "test" and len(res.units) == len(items)
    assert [u.unit_id for u in res.units] == [it.item_id for it in items]
    for u in res.units:
        assert set(R.SYNTH_SCORES) <= set(u.scores)
        assert u.scores["gate_acc"] == 1.0
        assert u.scores["mrstft"] == pytest.approx(0.0, abs=1e-3)
        assert u.scores["centroid_oct"] == pytest.approx(0.0, abs=1e-3)
        assert u.meta["n_queries"] == 2
        p = u.scores["param_l1"]
        assert math.isnan(p) or p == 0.0
    f1 = res.extra["gate_f1"]
    assert all(math.isnan(v) or v == 1.0 for v in f1.values())
    assert res.extra["n_queries"] == 2 * len(items)


def _fresh_ctx(cfg):
    ctx = R._context(cfg)
    ctx._renders.clear()
    ctx.n_renders = 0
    return ctx


def test_render_cache_one_render_per_unique_input_board(fx):
    items = D.load_split(fx, "test")
    ctx = _fresh_ctx(fx)
    res = R.score_synthetic(Const(), items, fx, "test")
    # one board for everything: one render per distinct input (item), not per caption
    assert ctx.n_renders == len({q.audio_key for it in items for q in it.queries()})
    assert res.extra["n_queries"] == 2 * len(items)
    assert all(u.scores["mrstft"] >= 0 for u in res.units)
    ctx = _fresh_ctx(fx)
    boards = _synth_oracle(items)
    R.score_synthetic(Oracle(boards), items, fx, "test")
    unique = {
        (q.audio_key, tuple(round(v, 6) for v in board_vector(boards[q.query_id])))
        for it in items
        for q in it.queries()
    }
    assert ctx.n_renders == len(unique)


def test_failed_renders_are_flagged_and_lose_2afc(fx, monkeypatch):
    monkeypatch.setattr(R._Ctx, "render", lambda self, q, b, h, t: (None, {}, {}))
    items = D.load_split(fx, "test")
    res = R.score_synthetic(Const(), items, fx, "test")
    assert res.extra["n_render_failed"] == 2 * len(items)
    for u in res.units:
        assert u.meta["render_failed"] == 2 and math.isnan(u.scores["mrstft"])
        assert u.scores["gate_acc"] >= 0  # parameter scores are still defined
    gold = D.load_gold_items(fx)
    g = R.score_gold(Const(), gold, fx)
    assert g.extra["n_render_failed"] == sum(len(x.clips) for x in gold)
    for u in g.units:
        assert u.scores["afc_mrstft"] == 0.0 and u.scores["afc_descriptor"] == 0.0
        assert all(v == 0.0 for k, v in u.scores.items() if k.startswith("afc_"))
        assert u.meta["render_failed"] == u.meta["n_clips"]


def test_score_gold_oracle_wins_2afc(fx):
    gold = D.load_gold_items(fx)
    boards = {
        q.query_id: dict(c.target.render["board_params"])
        for g in gold
        for q, c in zip(g.queries(), g.clips, strict=True)
    }
    res = R.score_gold(Oracle(boards), gold, fx)
    assert res.split == "gold" and [u.unit_id for u in res.units] == [g.gold_id for g in gold]
    for u, g in zip(res.units, gold, strict=True):
        s = u.scores
        assert set(R.GOLD_SCORES) <= set(s)
        assert s["afc_mrstft"] == 1.0
        assert s["mrstft_target"] == pytest.approx(0.0, abs=1e-3)
        assert s["region_l1"] == pytest.approx(0.0, abs=1e-9)
        axes = {d.axis for c in g.clips for d in c.distractors}
        assert {f"afc_mrstft/{a}" for a in axes} <= set(s)
        assert {f"afc_descriptor/{a}" for a in axes} <= set(s)
        assert all(s[f"afc_mrstft/{a}"] == 1.0 for a in axes)
        assert u.meta["gold_source"] == g.source and u.meta["subsets"] == g.subsets()
        assert u.meta["n_clips"] == len(g.clips) and u.meta["render_failed"] == 0
    assert res.extra["n_render_failed"] == 0


def test_score_gold_const(fx):
    gold = D.load_gold_items(fx)
    res = R.score_gold(Const(), gold, fx)
    for u in res.units:
        assert 0.0 <= u.scores["afc_mrstft"] <= 1.0
        assert u.scores["region_l1"] >= 0.0


def test_score_val_without_proxy_and_select(fx):
    val = D.load_split(fx, "val")
    with pytest.warns(UserWarning, match="proxy"):
        res = R.score_val(Const(), val, fx)
    assert res.split == "val" and res.extra["proxy"] is False
    m = Switch(_synth_oracle(val))
    with pytest.warns(UserWarning):
        out = R.select_hparams(m, val, fx)
    assert out["chosen"] == {"use_oracle": True} and m.use_oracle
    assert [s["hparams"] for s in out["scores"]] == m.hparam_grid()
    assert out["scores"][1]["score"] < out["scores"][0]["score"]


def test_score_val_merges_proxy_afc(fx, monkeypatch):
    import dataclasses
    from pathlib import Path

    from lstmabar.eval import proxy

    val = D.load_split(fx, "val")
    g0 = D.load_gold_items(fx)[0]
    items = [
        dataclasses.replace(
            g0,
            gold_id=v.item_id,
            split="val",
            source=proxy.PROXY_SOURCE,
            text_key=v.captions[0].caption_id,
            audio_key=v.item_id,
            clips=g0.clips[:1],
        )
        for v in val[:1]
    ]
    manifest = Path(str(fx.data.manifests_dir)) / f"{fx.eval.proxy.version}.jsonl"
    manifest.write_text("", encoding="utf-8")
    monkeypatch.setattr(proxy, "load_proxy_items", lambda cfg, limit=None: items)
    try:
        res = R.score_val(Const(), val, fx)
    finally:
        manifest.unlink()
    assert res.extra["proxy"] is True and res.extra["n_proxy"] == 1
    by_id = {u.unit_id: u for u in res.units}
    merged = by_id[val[0].item_id].scores
    assert {"afc_mrstft", "afc_descriptor", "mrstft", "param_l1"} <= set(merged)
    assert "region_l1" not in merged  # only afc_* keys join
    assert all(not k.startswith("afc_") for u in res.units[1:] for k in u.scores)


def test_select_hparams_all_nan_falls_back(fx, monkeypatch):
    monkeypatch.setattr(R, "_selection_score", lambda result, cfg: float("nan"))
    val = D.load_split(fx, "val")
    m = Switch(_synth_oracle(val))
    with pytest.warns(UserWarning, match="NaN"):
        out = R.select_hparams(m, val, fx)
    assert out["fallback"] is True and out["chosen"] == m.hparam_grid()[0]


def test_select_hparams_empty_grid_uses_select(fx):
    class Sel(Const):
        def select(self, val):
            return {"k": 3}

    assert R.select_hparams(Sel(), [], fx) == {"chosen": {"k": 3}, "scores": []}


@pytest.mark.slow
def test_worker_pool_matches_in_process(fx):
    from omegaconf import OmegaConf

    items = D.load_split(fx, "test")
    a = R.score_synthetic(Const(), items, fx, "test")
    cfg2 = OmegaConf.merge(fx, OmegaConf.from_dotlist(["eval.workers=2"]))
    b = R.score_synthetic(Const(), items, cfg2, "test")
    for ua, ub in zip(a.units, b.units, strict=True):
        assert ua.unit_id == ub.unit_id
        for k, v in ua.scores.items():
            assert (math.isnan(v) and math.isnan(ub.scores[k])) or v == pytest.approx(
                ub.scores[k], rel=1e-6
            )
