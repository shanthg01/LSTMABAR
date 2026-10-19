"""P5 report + ``lstmabar eval run`` / ``eval report`` on the eval fixture (task 5.4)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest

from lstmabar.eval import report as REP
from lstmabar.eval import run as RUN
from lstmabar.eval.fixtures import EVAL_CONFIG

ABS = re.compile(r"(?<![\w])[A-Za-z]:[\\/]|\\Users\\|/Users/|/tmp/|AppData")
FAST = ["eval.workers=1", "eval.stats.n_resamples=200"]


def _args(**kw) -> argparse.Namespace:
    base = dict(
        limit=None,
        methods=None,
        splits=None,
        workers=None,
        run_dir=None,
        fixtures=False,
        estimate=False,
        overrides=[],
    )
    return argparse.Namespace(**(base | kw))


def _copy(cfg):
    from omegaconf import OmegaConf

    return OmegaConf.create(OmegaConf.to_container(cfg))


def _assert_no_abs(text: str, *roots: Path) -> None:
    assert not ABS.search(text), ABS.search(text)
    for r in roots:
        assert str(r) not in text and r.as_posix() not in text


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture

    root = tmp_path_factory.mktemp("evalreport")
    cfg = eval_fixture_config(root, FAST)
    make_eval_fixture(cfg, seconds=0.5)
    return cfg, root


def test_scrub_and_cells():
    s = RUN.scrub("missing C:\\Users\\x\\data\\cache\\a.npz and /home/u/data/b.npz; ok a/b")
    assert "Users" not in s and "/home" not in s and "a.npz" in s and "ok a/b" in s
    assert RUN.scrub("${paths.data}/manifests/x.jsonl") == "${paths.data}/manifests/x.jsonl"
    assert REP.fmt_cell({"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 12}) == "0.500 [0.400, 0.600] (12)"
    assert REP.fmt_cell({"mean": None, "lo": None, "hi": None, "n": 0}) == "n/a, 0 items"
    multi = {"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 3, "seed_sd": 0.01, "n_seeds": 5}
    assert "seed SD 0.010 (5 seeds)" in REP.fmt_cell(multi)


def test_fixtures_cli_end_to_end(tmp_path):
    from lstmabar.cli import main

    out = tmp_path / "rep"
    rc = main(
        [
            "eval",
            "run",
            *FAST,  # dotlist overrides go right after the step
            f"eval.report_dir={out.as_posix()}",
            f"paths.runs={(tmp_path / 'runs').as_posix()}",
            "--config",
            str(EVAL_CONFIG),
            "--fixtures",
            "--methods",
            "noop",
            "mean",
        ]
    )
    assert rc == 0
    md = (out / "p5_baselines_smoke.md").read_text(encoding="utf-8")
    rep = json.loads((out / "p5_baselines_smoke.json").read_text(encoding="utf-8"))
    assert rep["schema"] == REP.REPORT_SCHEMA
    for key in ("created", "git", "config", "hashes", "methods", "rows", "comparisons"):
        assert key in rep
    assert set(rep["hashes"]) >= {"dataset_sha256", "gold_index_sha256", "embeddings"}
    assert [m["name"] for m in rep["methods"]] == ["noop", "mean"]
    assert all(m["status"] == "ok" for m in rep["methods"])
    splits = {r["split"] for r in rep["rows"]}
    assert splits == {"val", "test", "test_unseen_pedal", "gold"}
    # human column carried as n/a, 0 items
    human = [r for r in rep["rows"] if r["split"] == "gold" and r["gold_source"] == "human"]
    assert human and all(r["n"] == 0 and r["mean"] is None for r in human)
    head = [
        r
        for r in rep["rows"]
        if r["split"] == "gold"
        and r["metric"] == "afc_mrstft"
        and r["subset"] == "headline"
        and r["gold_source"] == "llm_draft"
    ]
    assert {r["method"] for r in head} == {"noop", "mean"} and all(r["n"] == 2 for r in head)
    assert any(r["metric"].startswith("afc_mrstft/") for r in rep["rows"])  # per axis
    assert any(r["metric"] == "gate_f1/macro" for r in rep["rows"])
    # mean vs noop on headline gold 2AFC and test MR-STFT
    comps = {(c["a"], c["b"], c["split"], c["metric"]) for c in rep["comparisons"]}
    assert ("mean", "noop", "gold", "afc_mrstft") in comps
    assert ("mean", "noop", "test", "mrstft") in comps
    # harmonic off for val selection (eval.select_harmonic: false) is recorded
    assert rep["harmonic"]["val"] is False
    assert any("OFF" in n and "val" in n for n in rep["notes"])
    assert any("v1 model" in n for n in rep["notes"])
    assert "## Gold 2AFC, headline subset" in md and "n/a, 0 items" in md
    assert md.index("Gold 2AFC") < md.index("Synthetic `test`") < md.index("Val selection")
    assert re.search(r"\d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\] \(\d+\)", md)
    _assert_no_abs(md, tmp_path)
    _assert_no_abs(json.dumps(rep), tmp_path)


def test_resume_not_run_and_report(fx, tmp_path, monkeypatch):
    cfg, root = fx
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _copy(cfg)
    cfg.eval.report_dir = str(tmp_path / "rep")
    cfg.paths.runs = str(tmp_path / "runs")
    assert RUN.run(cfg, _args(methods=["noop"], splits=["val", "test", "gold"])) == 0
    run_dir = next((tmp_path / "runs").glob("*/*"))
    done = sorted((run_dir / "scores" / "noop" / "seed0").glob("*.jsonl"))
    assert [p.stem for p in done] == ["gold", "test", "val"]
    mtimes = {p: p.stat().st_mtime_ns for p in done}

    methods = ["noop", "mean", "knn_clap", "text2fx", "llm_zero_shot"]
    args = _args(methods=methods, splits=["val", "test", "gold"], run_dir=str(run_dir))
    assert RUN.run(cfg, args) == 0
    assert {p: p.stat().st_mtime_ns for p in done} == mtimes  # finished parts skipped
    assert (run_dir / "scores" / "mean" / "seed0" / "gold.jsonl").exists()
    rep = json.loads((tmp_path / "rep" / "p5_baselines.json").read_text(encoding="utf-8"))
    by = {m["name"]: m for m in rep["methods"]}
    assert [m["name"] for m in rep["methods"]] == methods
    assert by["noop"]["status"] == by["mean"]["status"] == "ok"
    for name, frag in (
        ("knn_clap", "embedding cache missing"),
        ("text2fx", "limit (text2fx:"),  # fixture splits exceed max_items
        ("llm_zero_shot", "LlmCacheMiss"),
    ):
        assert by[name]["status"] == "not run"
        assert all(v.startswith("not run: ") and frag in v for v in by[name]["splits"].values())
    md = (tmp_path / "rep" / "p5_baselines.md").read_text(encoding="utf-8")
    assert "not run: limit" in md and "LlmCacheMiss" in md
    _assert_no_abs(md, root, tmp_path)
    _assert_no_abs(json.dumps(rep), root, tmp_path)

    # eval report: re-render from the run dir (no scoring)
    out2 = tmp_path / "rep2"
    rargs = _args(run_dir=str(run_dir), overrides=[f"eval.report_dir={out2.as_posix()}"])
    assert RUN.report(cfg, rargs) == 0
    rep2 = json.loads((out2 / "p5_baselines.json").read_text(encoding="utf-8"))
    assert rep2["rows"] == rep["rows"]
    assert [m["status"] for m in rep2["methods"]] == [m["status"] for m in rep["methods"]]


def test_multi_seed_method_reports_seed_sd(fx, tmp_path, monkeypatch):
    """A non-deterministic method runs eval.stats.min_seeds seeds; rows carry the seed SD."""
    from lstmabar.eval.baselines import base
    from lstmabar.eval.baselines.trivial import MeanParams

    class Jitter(MeanParams):
        name = "jitter"
        deterministic = False

        def predict(self, queries):
            boards = super().predict(queries)
            for b in boards:
                b["eq"]["high_db"] = min(1.0, 0.2 + 0.1 * self.seed)
                b["eq"]["enabled"] = 1.0
            return boards

    import sys
    import types

    mod = types.ModuleType("lstmabar_fake.jitter")
    mod.Jitter = Jitter
    monkeypatch.setitem(sys.modules, "lstmabar_fake.jitter", mod)
    monkeypatch.setitem(base.REGISTRY, "jitter", "lstmabar_fake.jitter:Jitter")
    cfg, _ = fx
    cfg = _copy(cfg)
    cfg.eval.report_dir = str(tmp_path / "rep")
    cfg.paths.runs = str(tmp_path / "runs")
    cfg.eval.stats.min_seeds = 2
    assert RUN.run(cfg, _args(methods=["jitter"], splits=["test"])) == 0
    rep = json.loads((tmp_path / "rep" / "p5_baselines.json").read_text(encoding="utf-8"))
    assert rep["methods"][0]["seeds"] == [0, 1]
    row = next(r for r in rep["rows"] if r["split"] == "test" and r["metric"] == "mrstft")
    assert row["n_seeds"] == 2 and row["seed_sd"] is not None


def test_estimate_mocked_sdk(fx, tmp_path, monkeypatch, capsys):
    from unittest import mock

    from lstmabar.eval.baselines import llm

    cfg, _ = fx
    client = mock.MagicMock()
    client.messages.count_tokens.return_value = mock.Mock(input_tokens=321)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm, "default_client_factory", lambda: client)
    cfg = _copy(cfg)
    cfg.paths.runs = str(tmp_path / "runs")
    assert RUN.run(cfg, _args(estimate=True, limit=2)) == 0
    out = capsys.readouterr().out
    assert "LLM zero-shot" in out and "count_tokens" in out and "within cap" in out
    assert client.messages.count_tokens.called
    assert not client.messages.batches.create.called
    assert not (tmp_path / "runs").exists()  # no run dir, no scoring


def _base(fx, tmp_path):
    cfg, root = fx
    cfg = _copy(cfg)
    cfg.eval.report_dir = str(tmp_path / "rep")
    cfg.paths.runs = str(tmp_path / "runs")
    return cfg, root


def test_resume_refuses_config_change_and_rescores_leftovers(fx, tmp_path):
    cfg, _ = _base(fx, tmp_path)
    assert RUN.run(cfg, _args(methods=["noop"], splits=["val", "test"], limit=2)) == 0
    run_dir = next((tmp_path / "runs").glob("*/*"))
    # --limit run: written as a smoke report, never over the real report name
    assert (tmp_path / "rep" / "p5_baselines_smoke.json").exists()
    assert not (tmp_path / "rep" / "p5_baselines.json").exists()
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["limit"] == 2

    # resume without --limit (or with any other scoring setting changed) is refused
    with pytest.raises(SystemExit, match="max_items"):
        RUN.run(cfg, _args(methods=["noop"], splits=["val", "test"], run_dir=str(run_dir)))
    other = _copy(cfg)
    other.eval.metrics.gate_threshold = 0.4
    with pytest.raises(SystemExit, match="gate_threshold"):
        RUN.run(other, _args(methods=["noop"], limit=2, run_dir=str(run_dir)))

    # a crash left only a .jsonl.tmp for test: resume re-scores it; val is kept
    seed_dir = run_dir / "scores" / "noop" / "seed0"
    test_file = seed_dir / "test.jsonl"
    test_file.replace(seed_dir / "test.jsonl.tmp")
    val_mtime = (seed_dir / "val.jsonl").stat().st_mtime_ns
    # a split file from another config (fingerprint mismatch) is not trusted either
    meta = seed_dir / "val.meta.json"
    m = json.loads(meta.read_text(encoding="utf-8"))
    args = _args(methods=["noop"], splits=["val", "test"], limit=2, run_dir=str(run_dir))
    assert RUN.run(cfg, args) == 0
    assert test_file.exists() and (seed_dir / "val.jsonl").stat().st_mtime_ns == val_mtime
    m["fingerprint"] = "stale"
    meta.write_text(json.dumps(m), encoding="utf-8")
    assert RUN.run(cfg, args) == 0
    assert (seed_dir / "val.jsonl").stat().st_mtime_ns != val_mtime


def test_no_scores_never_writes_a_report(fx, tmp_path, monkeypatch):
    cfg, _ = _base(fx, tmp_path)
    rep = tmp_path / "rep"
    rep.mkdir()
    (rep / "p5_baselines.md").write_text("real report", encoding="utf-8")
    assert RUN.run(cfg, _args(methods=["text2fx"], splits=["test"])) == 1
    assert (rep / "p5_baselines.md").read_text(encoding="utf-8") == "real report"


def test_other_errors_propagate(fx, tmp_path, monkeypatch):
    """Only not-implemented / embedding-cache / LLM-cache-miss failures become "not run"."""
    from lstmabar.eval.baselines import trivial

    def boom(self, train):
        raise FileNotFoundError("some data file")

    monkeypatch.setattr(trivial.MeanParams, "fit", boom)
    cfg, _ = _base(fx, tmp_path)
    with pytest.raises(FileNotFoundError):
        RUN.run(cfg, _args(methods=["mean"], splits=["test"]))


def test_test_real_section_and_notes(fx, tmp_path):
    cfg, _ = _base(fx, tmp_path)
    rep_dir = tmp_path / "rep"
    rep_dir.mkdir()
    cell = {"mean": 1.5, "lo": 1.0, "hi": 2.0, "n": 10}
    sim = {
        "protocol": {"sample_rate": 44100},
        "groups": {
            "egfx:RAT": {"n_total": 10, "n_evaluated": 10, "internal_only": False},
            "pod:ocd": {"n_total": 5, "n_evaluated": 5, "internal_only": True},
        },
        "kb": {
            "egfx:RAT": {
                "kb_pedal": "rat",
                "internal_only": False,
                "cells": [{"engine": "greybox", "level": "-18", "mrstft": cell}],
            }
        },
        "family_shifts": {
            "pod:ocd": {
                "family": "overdrive",
                "internal_only": True,
                "terms": {"centroid_oct": {"real": cell, "synthetic_family": cell}},
                "rows": [
                    {"item_id": "pod-secret-1", "x": 1.0},
                    {"item_id": "e1", "x": 2.0, "internal_only": False},
                ],
            }
        },
        "per_item": [{"item_id": "pod-secret-2", "internal_only": True}],
        "kb_knob_sweeps": [{"group": "pod:ocd"}],
    }
    (rep_dir / "sim_real.json").write_text(json.dumps(sim), encoding="utf-8")
    assert RUN.run(cfg, _args(methods=["noop"], splits=["test", "gold"])) == 0
    rep = json.loads((rep_dir / "p5_baselines.json").read_text(encoding="utf-8"))
    text = json.dumps(rep["test_real"])
    assert rep["test_real"]["status"] == "embedded"
    assert "pod-secret" not in text and "per_item" not in rep["test_real"]
    assert set(rep["test_real"]["omitted_sections"]) == {"per_item", "kb_knob_sweeps"}
    assert "e1" in text and "1.5" in text
    md = (rep_dir / "p5_baselines.md").read_text(encoding="utf-8")
    assert "pod-secret" not in md and "1.500 [1.000, 2.000] (10)" in md
    notes = " ".join(rep["notes"])
    assert "No real human gold items" in notes and "LLM-written test instructions" in notes
    assert "uncorrected for multiple comparisons" in notes
    gate = next(r for r in rep["rows"] if r["metric"] == "gate_f1/macro")
    assert gate["lo"] is None and gate["hi"] is None
    assert "embeddings" in rep["hashes"]["missing"]


def test_gold_note_with_human_items():
    from lstmabar.eval.runner import SplitResult, UnitScores

    units = [
        UnitScores(
            f"h{i}", "gold", "noop", 0, {}, {"gold_source": "human", "subsets": ["all", "headline"]}
        )
        for i in range(3)
    ]
    note = REP._gold_note([SplitResult("gold", "noop", 0, units, {})])
    assert "real human set (n = 3" in note and "supplementary" in note and "< ~30" in note


def test_estimate_without_key_never_builds_a_client(fx, tmp_path, monkeypatch, capsys):
    from lstmabar.eval.baselines import llm

    def factory():
        raise AssertionError("client factory must not be called without a key")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "default_client_factory", factory)
    cfg, _ = _base(fx, tmp_path)
    assert RUN.run(cfg, _args(estimate=True, limit=2)) == 0
    out = capsys.readouterr().out
    assert "approx" in out and "within cap" in out


def test_write_includes_methods_missing_from_run_json(tmp_path, monkeypatch):
    """Scores merged from another machine (or a crashed resume) appear in the report even if
    run.json's requested_methods does not list them; `eval report --methods` restricts."""
    from lstmabar.eval import run as R

    seen = {}

    def fake_build(results, cfg, extras):
        seen["methods"] = sorted({r.method for r in results})
        return {"rows": []}

    monkeypatch.setattr(
        R, "load_results", lambda d: [type("S", (), {"method": m})() for m in ("noop", "text2fx")]
    )
    monkeypatch.setattr(R, "load_method_records", lambda d: [{"name": "noop"}, {"name": "text2fx"}])
    import lstmabar.eval.report as rep

    monkeypatch.setattr(rep, "build_report", fake_build)
    monkeypatch.setattr(rep, "render_markdown", lambda r: "")
    monkeypatch.setattr(
        rep, "write_report", lambda r, d, n: (tmp_path / "x.md", tmp_path / "x.json")
    )
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"eval": {"report_dir": str(tmp_path), "report_name": "x"}})
    R._write(cfg, tmp_path, {"requested_methods": ["noop"]})
    assert seen["methods"] == ["noop", "text2fx"]
    R._write(cfg, tmp_path, {"requested_methods": ["noop"], "restrict_methods": True})
    assert seen["methods"] == ["noop"]
