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
from lstmabar.data.splits import assign_clip_split, unit_hash, val_recordings
from lstmabar.data.vocab import (
    caption_violations,
    holdout_names,
    mentions,
    names_for,
    strip_allowed_names,
)
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
        label_source="derived",
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


def _clip(**kw) -> DryClip:
    base = dict(
        clip_id="gt-p1-r1-0",
        source="guitar_techs",
        performer="p1",
        recording_id="r1",
        technique="chords",
        source_file="guitar_techs/p1/di/r1.wav",
        offset_s=1.5,
        duration_s=3.0,
        sample_rate=44100,
        source_gain_db=6.0,
        path="v1/dry/gt-p1-r1-0.flac",
    )
    base.update(kw)
    return DryClip(**base)


def test_dryclip_and_test_real_round_trip(tmp_path):
    real = TestRealItem(
        item_id="egf-rat-1",
        dataset="egfxset",
        device="RAT",
        family="distortion",
        kb_pedal_id="rat",
        settings={"distortion": 0.5},
        dry_path=DataLayout.test_real_relpath("egfxset", "clean", "1.wav"),
        wet_path=DataLayout.test_real_relpath("egfxset", "rat", "1.wav"),
        sample_rate=48000,
        internal_only=False,
        license="CC BY 4.0",
    )
    write_jsonl(tmp_path / "a.jsonl", [_clip()])
    write_jsonl(tmp_path / "b.jsonl", [real])
    assert list(read_jsonl(tmp_path / "a.jsonl", DryClip)) == [_clip()]
    assert list(read_jsonl(tmp_path / "b.jsonl", TestRealItem)) == [real]


@pytest.mark.parametrize("bad", [r"C:\x.flac", "C:/x.flac", "/abs/x.flac", r"a\b.flac", "../x"])
def test_record_paths_must_be_relative_posix(bad):
    with pytest.raises(ValueError):
        _clip(path=bad)
    with pytest.raises(ValueError):
        _item(wet_path=bad)


def test_numpy_and_tensor_values_serialize():
    d = to_dict(_item(input_peak_dbfs=np.float32(-12.0), descriptors_wet={"x": torch.tensor(1.5)}))
    assert json.dumps(d) and d["descriptors_wet"]["x"] == 1.5


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
    recs = [("a", "chords", f"a{i}") for i in range(20)]
    recs += [("b", "scales", f"b{i}") for i in range(10)] + [("b", "solo", "only")]
    vset = val_recordings(recs, 0.1)
    assert len(vset) == 3 and "only" not in vset  # 2 + 1; single-recording groups stay train
    assert val_recordings(reversed(recs), 0.1) == vset  # order-independent
    assert val_recordings(recs, 0.0) == set()
    # Shared content: the same take goes to val for every performer.
    shared = [(p, "chords", f"{p}_chords_{t}") for p in ("P1", "P2") for t in range(10)]
    vs = val_recordings(shared, 0.2)
    assert {r.split("_", 1)[1] for r in vs if r.startswith("P1")} == {
        r.split("_", 1)[1] for r in vs if r.startswith("P2")
    }
    assert assign_clip_split("guitar_techs", "a", "a0", tp, "c", vset) in {"train", "val"}
    assert {assign_clip_split("guitar_techs", "a", r, tp, "c", vset) for r in vset} == {"val"}
    assert assign_clip_split("guitar_techs", "c", "anything", tp, "c", vset) == "test"
    assert assign_clip_split("own", "me", "r1", tp, "c", vset) == "test"
    with pytest.raises(ValueError):
        assign_clip_split("egfxset", "a", "r1", tp, "c", vset)
    with pytest.raises(ValueError):
        assign_clip_split("guitar_techs", "zz", "r1", tp, "c", vset)
    assert 0.0 <= unit_hash("x") < 1.0


def test_vocab():
    allowed = names_for(["ts808", "ds1"])
    text = "a TS808-style mid hump, not a DS-1 buzz"
    assert set(mentions(text, allowed)) == {"TS808", "DS-1"}
    assert not any(ch.isdigit() for ch in strip_allowed_names(text, allowed))
    assert "RAT" in holdout_names("rat") and "Fuzz Face" not in holdout_names("fuzz_face_si")
    assert mentions("grated cheese", ["RAT"]) == []
    assert mentions("a DS‑1 buzz", allowed) == ["DS-1"]  # unicode hyphen
    assert "ProCo" in holdout_names("rat")
    assert caption_violations("a TS808 with a scooped DS-1 edge", allowed) == []
    for bad in (
        "boost it 6 dB",
        "cut the lows at a hundred hz",
        "around twelve o'clock",
        "dimed gain",
        "a 50% blend",
    ):
        assert caption_violations(bad, allowed), bad
    assert caption_violations("a ProCo-style grind", allowed, holdout_names("rat"))
    assert caption_violations("like one of those vintage fuzzes", allowed) == []


def test_fixture_dataset(tmp_path):
    cfg = fixture_config(tmp_path)
    clips = make_fixture_dataset(cfg, clips_per_performer=2, seconds=0.5)
    lay = DataLayout.from_config(cfg)
    assert len(clips) == 6 and lay.dry_manifest.exists()
    assert {c.split for c in clips if c.performer == "fx3"} == {"test"}
    assert {c.split for c in clips if c.performer != "fx3"} == {"train", "val"}
    for c in clips:
        x, sr = levels.read_audio(lay.resolve_rendered(c.path))
        assert sr == 44100 and levels.peak_dbfs(x) == pytest.approx(-1.0, abs=0.01)
    assert list(read_jsonl(lay.dry_manifest, DryClip)) == clips


def test_cli_registers_data_steps():
    parser = build_parser()
    args = parser.parse_args(["data", "render", "render.workers=2"])
    assert args.step == "render" and args.overrides == ["render.workers=2"]
    assert set(STEPS) >= {"ingest", "calibrate", "render", "bench", "captions", "freeze"}
