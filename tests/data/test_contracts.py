"""P4 wave 0: data contracts (schema, levels, layout, splits, vocab, fixtures, CLI)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from lstmabar.cli import build_parser
from lstmabar.config import load_config
from lstmabar.data import levels
from lstmabar.data.cli import STEPS
from lstmabar.data.fixtures import fixture_config, make_fixture_dataset
from lstmabar.data.layout import DataLayout
from lstmabar.data.schema import (
    Caption,
    DryClip,
    RenderItem,
    TestRealItem,
    board_to_floats,
    floats_to_board,
    from_dict,
    read_jsonl,
    to_dict,
    write_jsonl,
)
from lstmabar.data.splits import assign_clip_split, unit_hash
from lstmabar.data.vocab import holdout_names, mentions, names_for, strip_allowed_names
from lstmabar.dsp.pedalboard import default_pedalboard

REPO_ROOT = Path(__file__).resolve().parents[2]


def _item(**kw) -> RenderItem:
    base = dict(
        item_id="c-r0",
        clip_id="c",
        split="train",
        seed=1,
        scenario="pedal",
        engine="greybox",
        pedal_id="big_muff",
        pedal_knobs={"sustain": 0.5},
        board_params={"drive": {"gain_db": 0.4, "enabled": 1.0}},
        board_params_derived=None,
        clamped=False,
        clamp_notes=[],
        input_peak_dbfs=-12.0,
        wet_gain_db=3.0,
        wet_path="v1/wet/shard-00000/c-r0.flac",
        descriptors_wet={"hnr_db": float("nan")},
    )
    base.update(kw)
    return RenderItem(**base)


def test_config_loads_and_mix_sums_to_one():
    cfg = load_config(REPO_ROOT / "configs" / "data" / "v1.yaml")
    assert cfg.audio.sample_rate == 44100
    assert math.isclose(sum(cfg.render.free_board_mix.values()), 1.0)
    assert cfg.render.holdout_pedal not in cfg.render.whitebox_pedals
    assert cfg.captions.max_usd <= 4.5


def test_jsonl_round_trip_and_nan_becomes_null(tmp_path):
    item = _item()
    cap = Caption("c-r0-c0", "c-r0", "template", "a warm, smooth overdrive", template_id="t1")
    write_jsonl(tmp_path / "items.jsonl", [item])
    write_jsonl(tmp_path / "caps.jsonl", [cap])
    line = (tmp_path / "items.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["descriptors_wet"]["hnr_db"] is None  # strict JSON, no NaN
    (back,) = read_jsonl(tmp_path / "items.jsonl", RenderItem)
    assert back.board_params == item.board_params and back.descriptors_wet == {"hnr_db": None}
    assert list(read_jsonl(tmp_path / "caps.jsonl", Caption)) == [cap]


def test_from_dict_rejects_unknown_fields():
    d = to_dict(_item())
    d["extra"] = 1
    with pytest.raises(KeyError):
        from_dict(RenderItem, d)


def test_all_records_constructible():
    for cls in (DryClip, TestRealItem):
        assert cls.__dataclass_fields__  # frozen dataclasses with explicit schemas


def test_board_float_round_trip():
    board = default_pedalboard()
    params = board.random_params(2, generator=torch.Generator().manual_seed(0))
    f = board_to_floats(params, index=1)
    back = floats_to_board(f)
    v1 = board.to_vector(back)
    v0 = board.to_vector(params)[1:2]
    assert torch.allclose(v1, v0)


def test_levels():
    x = np.sin(np.linspace(0, 20, 1000)) * 0.3
    y, g = levels.normalize_peak(x, -1.0)
    assert levels.peak_dbfs(y) == pytest.approx(-1.0)
    assert np.allclose(y, x * levels.db_to_gain(g))
    z = levels.apply_input_level(y, -18.0)
    assert levels.peak_dbfs(z) == pytest.approx(-18.0)
    assert levels.peak_dbfs(levels.inference_normalize(x)) == pytest.approx(-15.0)
    zero, g0 = levels.normalize_peak(np.zeros(10), -1.0)
    assert g0 == 0.0 and not zero.any()
    rng = np.random.default_rng(0)
    s = [levels.sample_input_peak(rng, -24, -6) for _ in range(100)]
    assert min(s) >= -24 and max(s) <= -6


def test_flac_round_trip_and_overload_guard(tmp_path):
    x, _ = levels.normalize_peak(np.random.default_rng(0).standard_normal(4410), -1.0)
    levels.write_flac(tmp_path / "a.flac", x, 44100, "PCM_24")
    y, sr = levels.read_audio(tmp_path / "a.flac")
    assert sr == 44100 and np.max(np.abs(y - x)) < 1e-6
    assert len(levels.sha256_file(tmp_path / "a.flac")) == 64
    with pytest.raises(ValueError):
        levels.write_flac(tmp_path / "b.flac", x * 2, 44100, "PCM_24")


def test_layout_paths_are_relative_posix(tmp_path):
    lay = DataLayout.from_config(fixture_config(tmp_path))
    assert lay.dry_relpath("x") == "v1/dry/x.flac"
    assert lay.wet_relpath(3, "x-r0") == "v1/wet/shard-00003/x-r0.flac"
    assert lay.resolve_rendered(lay.dry_relpath("x")).is_relative_to(tmp_path)
    assert not lay.shard_done(0)


def test_splits():
    tp = ("a", "b")
    assert assign_clip_split("guitar_techs", "a", "r1", tp, "c") == "train"
    assert assign_clip_split("own", "me", "r1", tp, "c") == "test"
    with pytest.raises(ValueError):
        assign_clip_split("egfxset", "a", "r1", tp, "c")
    with pytest.raises(ValueError):
        assign_clip_split("guitar_techs", "zz", "r1", tp, "c")
    heldout = [assign_clip_split("guitar_techs", "c", f"r{i}", tp, "c") for i in range(200)]
    assert set(heldout) == {"val", "test"}
    # Same recording -> same split, whatever else changes.
    assert assign_clip_split("guitar_techs", "c", "r7", tp, "c") == heldout[7]
    assert 0.0 <= unit_hash("x") < 1.0


def test_vocab():
    allowed = names_for(["ts808", "ds1"])
    text = "a TS808-style mid hump, not a DS-1 buzz"
    assert set(mentions(text, allowed)) == {"TS808", "DS-1"}
    assert not any(ch.isdigit() for ch in strip_allowed_names(text, allowed))
    assert "RAT" in holdout_names("rat") and "Fuzz Face" not in holdout_names("fuzz_face_si")
    assert mentions("grated cheese", ["RAT"]) == []


def test_fixture_dataset(tmp_path):
    cfg = fixture_config(tmp_path)
    clips = make_fixture_dataset(cfg, clips_per_performer=2, seconds=0.5)
    lay = DataLayout.from_config(cfg)
    assert len(clips) == 6 and lay.dry_manifest.exists()
    assert {c.split for c in clips if c.performer != "fx3"} == {"train"}
    assert {c.split for c in clips if c.performer == "fx3"} <= {"val", "test"}
    for c in clips:
        x, sr = levels.read_audio(lay.resolve_rendered(c.path))
        assert sr == 44100 and levels.peak_dbfs(x) == pytest.approx(-1.0, abs=0.01)
    assert list(read_jsonl(lay.dry_manifest, DryClip)) == clips


def test_cli_registers_data_steps():
    parser = build_parser()
    args = parser.parse_args(["data", "render", "render.workers=2"])
    assert args.step == "render" and args.overrides == ["render.workers=2"]
    assert set(STEPS) >= {"ingest", "calibrate", "render", "bench", "captions", "freeze"}
