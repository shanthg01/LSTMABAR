"""`lstmabar eval results`: per-unit seed averaging and cell formatting."""

import json

import numpy as np

from lstmabar.eval.results import _fmt, unit_scores


def _write(root, method, seed, split, rows):
    d = root / "scores" / method / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{split}.jsonl").write_text(
        "\n".join(json.dumps({"unit_id": u, "scores": {"x": v}, "meta": {}}) for u, v in rows)
        + "\n",
        encoding="utf-8",
    )


def test_unit_scores_average_over_seeds(tmp_path):
    _write(tmp_path, "m", 0, "test", [("a", 1.0), ("b", 3.0)])
    _write(tmp_path, "m", 1, "test", [("a", 3.0), ("b", 5.0)])
    ids, vals, seed_means = unit_scores(tmp_path, "m", "test", "x")
    assert ids == ["a", "b"]
    np.testing.assert_allclose(vals, [2.0, 4.0])
    assert seed_means == [2.0, 4.0]


def test_fmt_cells():
    assert _fmt(None) == "n/a"
    c = {"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 10, "seed_sd": 0.01, "n_seeds": 5}
    assert _fmt(c) == "0.500 [0.400, 0.600] (10); seed SD 0.010"
