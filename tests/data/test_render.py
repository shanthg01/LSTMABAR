"""P4 task 4.2: renderer (spec sampling, labels, levels, analysis, shards)."""

import json
from argparse import Namespace
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from lstmabar.data import levels
from lstmabar.data import render as R
from lstmabar.data.calibrate import CalibrationTable, Label
from lstmabar.data.fixtures import fixture_config, make_fixture_dataset
from lstmabar.data.layout import DONE_MARKER, DataLayout
from lstmabar.data.schema import (
    FREE_SCENARIOS,
    DryClip,
    RenderItem,
    board_to_floats,
    read_jsonl,
)
from lstmabar.dsp.pedalboard import default_pedalboard
from lstmabar.physics.derive import derive
from lstmabar.physics.kb import load_kb
from tests.analysis.signals import constant_track, oscillator

KB = load_kb()
POTS = {pid: tuple(p.pots) for pid, p in KB.items()}
SMALL = [
    "audio.clip_seconds=0.5",
    "render.renders_per_clip.train=1",
    "render.renders_per_clip.val=1",
    "render.renders_per_clip.test=1",
    "render.unseen_renders_per_clip=1",
    "render.shard_size=3",
]


@pytest.fixture(scope="module")
def sampling(tmp_path_factory):
    return R.Sampling.from_config(fixture_config(tmp_path_factory.mktemp("cfg")))


def _clip(clip_id: str, split: str) -> DryClip:
    return DryClip(
        clip_id, "fixture", "p", clip_id, "", "f.wav", 0.0, 0.5, 44100, 0.0, "", "", split
    )


# --------------------------------------------------------------------------- spec sampling


def test_spec_mix_follows_config(sampling):
    specs = [R.sample_spec(f"c{i}", "train", 0, sampling, POTS) for i in range(3000)]
    pedal = [s for s in specs if s.scenario == "pedal"]
    assert len(pedal) / len(specs) == pytest.approx(sampling.pedal_fraction, abs=0.03)
    counts = Counter(s.pedal_id for s in pedal)
    assert set(counts) == set(KB) - {sampling.holdout_pedal}  # never the holdout
    assert max(counts.values()) - min(counts.values()) < 0.25 * len(pedal) / len(counts)
    free = Counter(s.scenario for s in specs if s.scenario != "pedal")
    for name, p in sampling.free_board_mix:
        assert free[name] / sum(free.values()) == pytest.approx(p, abs=0.04), name
    for s in specs:
        assert sampling.input_peak_dbfs[0] <= s.input_peak_dbfs <= sampling.input_peak_dbfs[1]
        if s.scenario == "pedal":
            want = "whitebox" if s.pedal_id in sampling.whitebox_pedals else "greybox"
            assert s.engine == want and s.free_board is None
            assert set(s.pedal_knobs) == set(POTS[s.pedal_id])
            assert all(0 <= v <= 1 for v in s.pedal_knobs.values())
        else:
            assert s.engine == "greybox" and s.pedal_id is None and s.pedal_knobs is None


def test_free_board_gates_per_scenario(sampling):
    want = {
        "bypass": (0, 0, 0),
        "eq": (0, 0, 1),
        "comp": (1, 0, 0),
        "comp_eq": (1, 0, 1),
    }
    drive_gates = []
    for k in range(400):
        for sc in FREE_SCENARIOS:
            s = R.sample_spec("c", "train", k, sampling, POTS, force=(sc, None))
            gates = tuple(s.free_board[b]["enabled"] for b in ("compressor", "drive", "eq"))
            knobs = [v for b in s.free_board.values() for n, v in b.items() if n != "enabled"]
            assert all(0 <= v <= 1 for v in knobs)
            if sc == "drive":
                drive_gates.append(gates)
            else:
                assert gates == want[sc], sc
    comp, drv, eq = np.array(drive_gates).T
    assert (drv == 1).all()
    assert comp.mean() == pytest.approx(sampling.p_comp, abs=0.08)
    assert eq.mean() == pytest.approx(sampling.p_eq, abs=0.08)
    with pytest.raises(ValueError):
        R.scenario_gates("fuzz", np.random.default_rng(0), 0.5, 0.5)


def test_plan_is_deterministic_and_holds_out_the_pedal(sampling):
    clips = [_clip(f"a{i}", "train") for i in range(30)]
    clips += [_clip(f"v{i}", "val") for i in range(10)] + [
        _clip(f"t{i}", "test") for i in range(10)
    ]
    plan = R.plan_specs(clips, sampling, POTS)
    assert plan == R.plan_specs(clips, sampling, POTS)
    assert R.plan_specs(clips[:5], sampling, POTS) == plan[: 5 * 6]  # per-item seeds
    rpc, unseen = sampling.renders_per_clip, sampling.unseen_renders_per_clip
    n = Counter(s.split for s in plan)
    assert n == {
        "train": 30 * rpc["train"],
        "val": 10 * rpc["val"],
        "test": 10 * rpc["test"],
        "test_unseen_pedal": 10 * unseen,
    }
    hold = sampling.holdout_pedal
    assert all(s.pedal_id != hold for s in plan if s.split != "test_unseen_pedal")
    assert all(s.pedal_id == hold for s in plan if s.split == "test_unseen_pedal")
    assert len({s.item_id for s in plan}) == len({s.seed for s in plan}) == len(plan)
    other = replace(sampling, dataset_seed=1)
    assert R.plan_specs(clips, other, POTS) != plan
    with pytest.raises(ValueError):
        R.plan_specs([_clip("x", "")], sampling, POTS)
    assert R.plan_hash(plan[:3]) != R.plan_hash(plan[1:4])
    sizes = [len(s) for s in R.shard_specs(plan, 100)]
    assert sum(sizes) == len(plan) and max(sizes) == 100 and len(sizes) == -(-len(plan) // 100)


def test_planned_counts(tmp_path, sampling):
    c = R.planned_counts(fixture_config(tmp_path), sampling)
    assert c["train"] == 3700 * 6 and c["test_unseen_pedal"] == 130 * 2 and c["clips"] == 4280


# --------------------------------------------------------------------------- labels / levels


def test_clamped_detection():
    notes = [
        "diode branches ignored; mostly hidden by the 60 dB clamp",
        "drive.gain_db 64.47 clamped to 60 dB: circuit gain into the clipper exceeds ...",
    ]
    assert R.clamp_notes(notes) == notes[1:]
    low = derive(KB["big_muff"], {"sustain": 0.0})
    assert any("clamp" in n for n in low.notes)  # Big Muff's static note mentions the clamp
    assert R.clamp_notes(low.notes) == []
    assert R.clamp_notes(derive(KB["big_muff"], {"sustain": 1.0}).notes)


@pytest.fixture
def ctx(tmp_path):
    cfg = fixture_config(tmp_path)
    clips = make_fixture_dataset(cfg, clips_per_performer=1, seconds=0.5)
    return R.RenderContext(cfg, R.Options(allow_derived_labels=False, pitch=False)), clips


def test_levels_and_wet_gain_restore(ctx, sampling):
    c, clips = ctx
    clip = clips[0]
    spec = R.sample_spec(clip.clip_id, "train", 0, sampling, POTS, force=("bypass", None))
    item = R.render_item(spec, clip, 0, c)
    stored, sr = levels.read_audio(c.layout.resolve_rendered(item.wet_path))
    assert sr == 44100 and levels.peak_dbfs(stored) == pytest.approx(c.wet_peak, abs=0.01)
    assert item.sha256 == levels.sha256_file(c.layout.resolve_rendered(item.wet_path))
    true_wet = stored * levels.db_to_gain(item.wet_gain_db)
    dry, _ = levels.read_audio(c.layout.resolve_rendered(clip.path))
    x = levels.apply_input_level(dry, spec.input_peak_dbfs)  # bypass: wet == render input
    assert levels.peak_dbfs(x) == pytest.approx(spec.input_peak_dbfs)
    assert np.max(np.abs(true_wet - x)) < 1e-5 * np.max(np.abs(x))
    assert item.wet_gain_db == pytest.approx(spec.input_peak_dbfs - c.wet_peak, abs=1e-3)
    assert item.descriptors_wet["rms_db"] == pytest.approx(item.descriptors_dry["rms_db"], abs=0.01)
    assert item.label_source == "free" and item.board_params == spec.free_board
    assert item.voiced_fraction is None and np.isnan(item.archetype_wet["saw"])


def test_bad_wet_raises_with_item_id(ctx, sampling, monkeypatch):
    c, clips = ctx
    spec = R.sample_spec(clips[0].clip_id, "train", 0, sampling, POTS, force=("eq", None))
    monkeypatch.setattr(c, "render_board", lambda b, x: np.full_like(x, np.nan))
    with pytest.raises(FloatingPointError, match=spec.item_id):
        R.render_item(spec, clips[0], 0, c)
    monkeypatch.setattr(c, "render_board", lambda b, x: np.zeros_like(x))
    with pytest.raises(ValueError, match=spec.item_id):
        R.render_item(spec, clips[0], 0, c)


def test_greybox_pedal_label_is_derived(ctx, sampling):
    c, clips = ctx
    spec = R.sample_spec(clips[0].clip_id, "train", 3, sampling, POTS, force=("pedal", "big_muff"))
    spec = replace(spec, pedal_knobs={"sustain": 1.0, "tone": 0.5, "volume": 0.5})
    item = R.render_item(spec, clips[0], 0, c)
    assert item.label_source == "derived" and item.engine == "greybox"
    assert item.board_params == item.board_params_derived
    assert item.clamped and all(R.CLAMP_MARKER in n for n in item.clamp_notes)
    assert item.board_params["compressor"]["enabled"] == 0.0
    assert item.board_params["eq"]["enabled"] == 1.0


def test_missing_calibration_table_fails_loudly(ctx):
    c, _ = ctx
    with pytest.raises(FileNotFoundError, match="allow-derived-labels"):
        c.table("ts808")
    with pytest.raises(FileNotFoundError):
        R._preflight(c.cfg, c.opts, c.kb)
    allowed = R.RenderContext(c.cfg, R.Options(allow_derived_labels=True, pitch=False))
    assert allowed.table("ts808") is None


class FakeTable:
    def __init__(self, pedal_id):
        self.pedal_id = pedal_id

    def lookup(self, knobs):
        board = board_to_floats(default_pedalboard().default_params(1))  # full layout
        board["compressor"]["enabled"] = 0.0
        board["drive"].update(gain_db=0.25, level_db=1.0)
        return Label(board=board, source="calibrated", clamp_notes=["level offset clamped"])


@pytest.mark.skipif(
    not __import__("lstmabar.physics.whitebox.diode_clipper", fromlist=["x"]).numba_available(),
    reason="white-box render needs numba (whitebox extra) to be fast",
)
def test_whitebox_item_uses_calibration_table(ctx, sampling, monkeypatch):
    c, clips = ctx
    for pid in sampling.whitebox_pedals:
        c.layout.calibration_table(pid).parent.mkdir(parents=True, exist_ok=True)
        c.layout.calibration_table(pid).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(CalibrationTable, "load", classmethod(lambda cls, p: FakeTable(p)))
    R._preflight(c.cfg, c.opts, c.kb)  # tables present now
    spec = R.sample_spec(clips[0].clip_id, "train", 5, sampling, POTS, force=("pedal", "ts808"))
    item = R.render_item(spec, clips[0], 0, c)
    assert item.engine == "whitebox" and item.label_source == "calibrated"
    assert item.board_params["drive"]["gain_db"] == 0.25
    assert item.label_clamp_notes == ["level offset clamped"]
    assert item.board_params_derived["drive"]["enabled"] == 1.0
    assert item.board_params_derived != item.board_params
    wet, _ = levels.read_audio(c.layout.resolve_rendered(item.wet_path))
    assert np.isfinite(wet).all() and levels.peak_dbfs(wet) == pytest.approx(-1.0, abs=0.01)


# --------------------------------------------------------------------------- analysis


def test_analyse_gating_and_track_cache(tmp_path):
    x = oscillator("square", 196.0, 1.0, 0.4)
    tr = constant_track(x, 196.0)
    desc, arch = R.analyse(x, 44100, tr, True, 10, 8)
    assert np.isfinite(desc["odd_even_db"]) and arch["square"] > 0.8
    assert sum(arch.values()) == pytest.approx(1.0)
    desc, arch = R.analyse(x, 44100, tr, False, 10, 8)  # low voiced fraction: NaN
    assert np.isnan(desc["hnr_db"]) and all(np.isnan(v) for v in arch.values())
    assert np.isfinite(desc["centroid_hz"])
    path = tmp_path / "f0" / "c.npz"
    R.save_track(path, tr, "abc")
    back = R.load_track(path, "abc")
    assert np.array_equal(back.f0_hz, tr.f0_hz) and back.hop_length == tr.hop_length
    assert R.load_track(path, "other") is None and R.load_track(tmp_path / "no.npz") is None
    assert R.voiced_fraction(tr) == 1.0 and R.voiced_fraction(None) is None


# --------------------------------------------------------------------------- end to end


def _items(layout: DataLayout) -> list[RenderItem]:
    return [
        i
        for p in sorted(layout.wet_dir.glob("shard-*"))
        for i in read_jsonl(p / R.ITEMS_FILE, RenderItem)
    ]


def test_run_on_fixtures_end_to_end_and_resume(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path)
    args = Namespace(fixtures=True, overrides=SMALL, workers=1, limit=None)
    assert R.run(cfg, args) == 0
    fcfg = R.fixture_setup(cfg, args)
    layout = DataLayout.from_config(fcfg)
    items = _items(layout)
    # 6 clips (R11): 2 train + 2 val + 2 test (+2 unseen) at one render each.
    assert len(items) == 8 and len(list(layout.wet_dir.glob("shard-*"))) == 3
    assert Counter(i.split for i in items)["test_unseen_pedal"] == 2
    for i in items:
        assert layout.resolve_rendered(i.wet_path).exists()
        assert (i.split == "test_unseen_pedal") == (i.pedal_id == "rat")
        if i.scenario == "pedal":
            assert i.label_source == "derived" and i.board_params_derived is not None
        else:
            assert i.label_source == "free" and i.board_params_derived is None
        if R.librosa_available():
            assert i.voiced_fraction is not None and layout.f0_cache(i.clip_id).exists()
            if i.voiced_fraction >= fcfg.analysis.min_voiced_fraction:
                assert sum(i.archetype_wet.values()) == pytest.approx(1.0)
        else:
            assert i.voiced_fraction is None and i.archetype_wet["saw"] is None
    # Resume: finished shards are skipped, a partial one is rewritten.
    plan = json.loads((layout.wet_dir / R.PLAN_FILE).read_text(encoding="utf-8"))
    assert plan["shards"] == 3 and plan["items"] == 8 and len(plan["shard_sha256"]) == 3
    first = (layout.shard_dir(0) / R.ITEMS_FILE).stat().st_mtime_ns
    (layout.shard_dir(2) / DONE_MARKER).unlink()
    (layout.shard_dir(2) / "junk.flac").write_bytes(b"x")
    (layout.shard_dir(1) / DONE_MARKER).write_text("{trunc", encoding="utf-8")  # unreadable
    calls = []
    orig = R.render_shard
    monkeypatch.setattr(R, "render_shard", lambda job: calls.append(job["shard"]) or orig(job))
    assert R.run(cfg, args) == 0
    assert calls == [1, 2] and not (layout.shard_dir(2) / "junk.flac").exists()
    assert (layout.shard_dir(0) / R.ITEMS_FILE).stat().st_mtime_ns == first
    assert _items(layout) == items  # deterministic re-render
    # A finished shard from a different plan (specs or render settings) is never mixed in.
    for extra in ("seed=7", "analysis.max_frames=9", "render.wet_peak_dbfs=-3.0"):
        with pytest.raises(RuntimeError, match="different plan"):
            R.run(cfg, Namespace(**{**vars(args), "overrides": [*SMALL, extra]}))
    # Shard dirs beyond the plan are flagged.
    (layout.shard_dir(9)).mkdir()
    with pytest.warns(UserWarning, match="beyond the plan"):
        R.run(cfg, args)
