"""`eval run` resume fingerprint: portable across data locations, sensitive to content."""

from __future__ import annotations

import json

from lstmabar.config import load_config
from lstmabar.eval.fixtures import EVAL_CONFIG
from lstmabar.eval.run import fingerprint, fingerprint_hash


def _cfg(data_root, *extra):
    return load_config(EVAL_CONFIG, [f"paths.data={data_root.as_posix()}", *extra])


def _index(root, sha):
    p = root / "manifests" / "v1" / "index.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"dataset_sha256": sha}), encoding="utf-8")


def test_same_content_at_two_locations_matches(tmp_path):
    a, b = tmp_path / "local" / "data", tmp_path / "drive" / "data"
    _index(a, "abc")
    _index(b, "abc")
    fa, fb = fingerprint(_cfg(a)), fingerprint(_cfg(b))
    assert fingerprint_hash(fa) == fingerprint_hash(fb)
    assert fa["content"]["dataset_sha256"] == "abc"
    assert str(a) not in json.dumps(fa) and a.as_posix() not in json.dumps(fa)


def test_different_dataset_or_settings_differ(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _index(a, "abc")
    _index(b, "xyz")
    assert fingerprint_hash(fingerprint(_cfg(a))) != fingerprint_hash(fingerprint(_cfg(b)))
    base = fingerprint_hash(fingerprint(_cfg(a)))
    assert base != fingerprint_hash(fingerprint(_cfg(a, "eval.max_items=3")))
    assert base == fingerprint_hash(fingerprint(_cfg(a, "eval.workers=7")))


def test_method_blocks_only_affect_their_method(tmp_path):
    a = tmp_path / "a"
    _index(a, "abc")
    base, colab = _cfg(a), _cfg(a, "eval.baselines.text2fx.max_items=null")
    assert fingerprint_hash(fingerprint(base)) == fingerprint_hash(fingerprint(colab))
    for m in ("noop", "mean", "knn_clap"):
        assert fingerprint_hash(fingerprint(base, m)) == fingerprint_hash(fingerprint(colab, m))
    assert fingerprint_hash(fingerprint(base, "text2fx")) != fingerprint_hash(
        fingerprint(colab, "text2fx")
    )
    knn = _cfg(a, "eval.baselines.knn.k_grid=[1,2]")
    assert fingerprint_hash(fingerprint(base, "knn_bge")) != fingerprint_hash(
        fingerprint(knn, "knn_bge")
    )


def test_resume_check_uses_the_recorded_portable_fingerprint(tmp_path):
    import pytest

    from lstmabar.eval.run import FINGERPRINT_FILE, _check_resume, _json_dump

    a, b = tmp_path / "local" / "data", tmp_path / "drive" / "data"
    _index(a, "abc")
    _index(b, "abc")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("name: x\n", encoding="utf-8")  # resolved, local
    _json_dump(fingerprint(_cfg(a)), run_dir / FINGERPRINT_FILE)
    _check_resume(run_dir, _cfg(b))  # same content elsewhere: allowed
    _check_resume(run_dir, _cfg(b, "eval.baselines.text2fx.max_items=null"))  # method block
    with pytest.raises(SystemExit, match="eval.max_items"):
        _check_resume(run_dir, _cfg(b, "eval.max_items=3"))
