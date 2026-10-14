"""Tiny synthetic evaluation fixture (tests and ``--fixtures`` smoke runs; no downloads, no
librosa): dry clips for three fake performers, free-board wet renders with captions for every
synthetic split, a three-item gold manifest (target + 2 distractors on 2 test clips at the
inference level), and two ``test_real`` items at 48 kHz.

Captions and gold instructions are fixed placeholder strings (never caption templates or real
gold text).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from lstmabar.audio import resample, synth_riff
from lstmabar.config import load_config
from lstmabar.data import levels
from lstmabar.data.fixtures import FIXTURE_PERFORMERS, make_fixture_dataset
from lstmabar.data.gold import GoldRender, expand, load_targets, region_from_dict, sample
from lstmabar.data.layout import DONE_MARKER, DataLayout
from lstmabar.data.render import ARCHETYPE_KEYS, analyse
from lstmabar.data.schema import (
    Caption,
    DryClip,
    FloatBoard,
    RenderItem,
    TestRealItem,
    board_to_floats,
    floats_to_board,
    to_dict,
    write_jsonl,
)
from lstmabar.dsp.pedalboard import default_pedalboard

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_CONFIG = REPO_ROOT / "configs" / "eval" / "p5.yaml"
TARGETS_FILE = REPO_ROOT / "data" / "gold" / "targets" / "targets_v1.yaml"
FIXTURE_GOLD_VERSION = "gold_fx"

RENDERS_PER_SPLIT = {"train": 4, "val": 2, "test": 2}
UNSEEN_PER_TEST_CLIP = 1

# Placeholder gold items (target, distractors) in the targets-file region schema.
FIXTURE_GOLD = (
    (
        "fx001",
        "make it brighter (fixture)",
        {"kind": "free", "gain": "none", "brightness": "bright"},
        [
            ("brightness", {"kind": "free", "gain": "none", "brightness": "dark"}),
            (
                "gain",
                {
                    "kind": "free",
                    "gain": "high",
                    "family": "distortion",
                    "clipping": "hard",
                    "brightness": "bright",
                },
            ),
        ],
    ),
    (
        "fx002",
        "dirty overdrive (fixture)",
        {
            "kind": "free",
            "gain": "medium",
            "family": "overdrive",
            "clipping": "soft",
            "brightness": "neutral",
        },
        [("gain", {"kind": "free", "gain": "none", "brightness": "neutral"})],
    ),
    (
        "fx003",
        "squash it (fixture)",
        {"kind": "free", "gain": "none", "brightness": "neutral", "compression": "heavy"},
        [("compression", {"kind": "free", "gain": "none", "brightness": "neutral"})],
    ),
)


def eval_fixture_config(root: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """``configs/eval/p5.yaml`` with every data/report dir under ``root``."""
    cfg = load_config(EVAL_CONFIG)
    root = Path(root)
    data = {
        k: str(root / k.removesuffix("_dir"))
        for k in (
            "sources_dir",
            "rendered_dir",
            "cache_dir",
            "test_real_dir",
            "manifests_dir",
            "calibration_dir",
        )
    }
    patch = {
        "data": data,
        "ingest": {
            "train_performers": list(FIXTURE_PERFORMERS[:2]),
            "heldout_performer": FIXTURE_PERFORMERS[2],
        },
        "paths": {"runs": str(root / "runs")},
        "gold": {
            "gold_dir": str(root / "gold"),
            "targets_file": str(TARGETS_FILE),
            "version": FIXTURE_GOLD_VERSION,
        },
        "eval": {
            "report_dir": str(root / "reports"),
            "gold_manifest": str(root / "manifests" / f"{FIXTURE_GOLD_VERSION}.jsonl"),
            "gold_index": str(root / "manifests" / f"{FIXTURE_GOLD_VERSION}.index.json"),
            "embeddings": {"dir": str(root / "cache")},
        },
    }
    cfg = OmegaConf.merge(cfg, patch)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _analysis(y: np.ndarray, cfg) -> tuple[dict, dict]:
    desc, arch = analyse(
        y, int(cfg.audio.sample_rate), None, False, int(cfg.analysis.n_harmonics), 8
    )
    return desc, arch


def _wet(board, params: FloatBoard, x: np.ndarray, wet_peak: float) -> tuple[np.ndarray, float]:
    """Render ``params`` on ``x``; returns the peak-normalized wet and its ``wet_gain_db``."""
    with torch.no_grad():
        y = board(torch.from_numpy(x.astype(np.float32))[None], floats_to_board(params))
    y = y[0].double().numpy()
    if not np.all(np.isfinite(y)) or np.max(np.abs(y)) == 0:
        y = x.copy()
    y_norm, gain_db = levels.normalize_peak(y, wet_peak)
    return y_norm, -gain_db


def _item(
    cfg, layout, board, clip: DryClip, x_stored, split, k, rng, params, shard_rel
) -> RenderItem:
    sr = int(cfg.audio.sample_rate)
    peak = float(rng.uniform(-24.0, -6.0))
    x = levels.apply_input_level(x_stored, peak)
    y, wet_gain_db = _wet(board, params, x, float(cfg.render.wet_peak_dbfs))
    item_id = f"{clip.clip_id}-{'u' if split == 'test_unseen_pedal' else 'r'}{k}"
    wet_rel = f"{shard_rel}/{item_id}.flac"
    levels.write_flac(layout.resolve_rendered(wet_rel), y, sr, str(cfg.render.wet_subtype))
    d_dry, a_dry = _analysis(x, cfg)
    d_wet, a_wet = _analysis(y, cfg)
    on = {b for b, v in params.items() if v["enabled"] >= 0.5}
    scenario = "drive" if "drive" in on else "comp_eq" if on >= {"compressor", "eq"} else "eq"
    return RenderItem(
        item_id=item_id,
        clip_id=clip.clip_id,
        split=split,
        seed=int(rng.integers(1 << 31)),
        scenario=scenario,
        engine="greybox",
        pedal_id=None,
        pedal_knobs=None,
        board_params=params,
        board_params_derived=None,
        label_source="free",
        clamped=False,
        clamp_notes=[],
        input_peak_dbfs=peak,
        wet_gain_db=wet_gain_db,
        wet_path=wet_rel,
        sha256=levels.sha256_file(layout.resolve_rendered(wet_rel)),
        voiced_fraction=None,
        descriptors_dry=d_dry,
        descriptors_wet=d_wet,
        archetype_dry=a_dry,
        archetype_wet=a_wet,
    )


def _caption_texts(params: FloatBoard) -> list[str]:
    drive = params["drive"]["enabled"] >= 0.5
    bright = params["eq"]["high_db"] >= 0.5 and params["eq"]["enabled"] >= 0.5
    a = f"fixture caption {'driven' if drive else 'clean'} {'bright' if bright else 'dark'}"
    return [a, a + " paraphrase"]


def make_eval_fixture(cfg: DictConfig, seconds: float = 1.0, seed: int = 0) -> dict[str, int]:
    """Write the whole fixture under the dirs in ``cfg`` (from :func:`eval_fixture_config`).
    Returns counts per split (+ ``gold_renders``, ``test_real``)."""
    layout = DataLayout.from_config(cfg)
    sr = int(cfg.audio.sample_rate)
    clips = make_fixture_dataset(cfg, clips_per_performer=2, seconds=seconds)
    board = default_pedalboard(sr)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    shard_rel = f"{layout.version}/wet/{layout.shard_name(0)}"
    shard_dir = layout.resolve_rendered(shard_rel)
    shard_dir.mkdir(parents=True, exist_ok=True)
    items: list[RenderItem] = []
    caps: list[Caption] = []
    counts: dict[str, int] = {}
    for clip in clips:
        x_stored, _ = levels.read_audio(layout.resolve_rendered(clip.path))
        jobs = [(clip.split, k) for k in range(RENDERS_PER_SPLIT[clip.split])]
        if clip.split == "test":
            jobs += [("test_unseen_pedal", k) for k in range(UNSEEN_PER_TEST_CLIP)]
        for split, k in jobs:
            params = board_to_floats(board.random_params(1, generator=gen, p_enabled=0.6))
            it = _item(cfg, layout, board, clip, x_stored, split, k, rng, params, shard_rel)
            items.append(it)
            counts[split] = counts.get(split, 0) + 1
            for j, text in enumerate(_caption_texts(params)):
                caps.append(
                    Caption(
                        caption_id=f"{it.item_id}-c{j}",
                        item_id=it.item_id,
                        kind="template" if j == 0 else "paraphrase",
                        text=text,
                        template_id="fx-00",
                    )
                )
    write_jsonl(shard_dir / "items.jsonl", items)
    write_jsonl(shard_dir / "captions.jsonl", caps)
    (shard_dir / DONE_MARKER).write_text("fixture\n", encoding="utf-8")
    counts["gold_renders"] = _make_gold(cfg, layout, board, clips, rng)
    counts["test_real"] = _make_test_real(cfg, layout, rng)
    return counts


def _make_gold(cfg, layout: DataLayout, board, clips: list[DryClip], rng) -> int:
    sr = int(cfg.audio.sample_rate)
    peak = float(cfg.audio.inference_peak_dbfs)
    targets = load_targets(TARGETS_FILE)
    test_clips = [c for c in clips if c.split == "test"]
    gold_dir = Path(str(cfg.gold.gold_dir))
    gold_dir.mkdir(parents=True, exist_ok=True)
    instr = [
        {"id": gid, "source": "llm_draft", "instruction": text, "target": None}
        for gid, text, _, _ in FIXTURE_GOLD
    ]
    (gold_dir / "instructions_llm_draft.yaml").write_text(
        "# eval fixture (placeholder text)\n" + yaml.safe_dump(instr, sort_keys=False),
        encoding="utf-8",
    )
    version = str(cfg.gold.version)
    records: list[GoldRender] = []
    for gid, _, tdict, dlist in FIXTURE_GOLD:
        regions = [("target", None, None, tdict)] + [
            ("distractor", i, axis, d) for i, (axis, d) in enumerate(dlist)
        ]
        for ci, clip in enumerate(test_clips):
            x_stored, _ = levels.read_audio(layout.resolve_rendered(clip.path))
            x = levels.apply_input_level(x_stored, peak)
            for role, di, axis, rdict in regions:
                region = region_from_dict(rdict, gid)
                _, params = sample(expand(region, targets, gid), rng)
                y, wet_gain_db = _wet(board, params, x, float(cfg.render.wet_peak_dbfs))
                suffix = "t" if role == "target" else f"d{di}"
                item_id = f"{gid}-c{ci}-{suffix}"
                wet_rel = f"{version}/wet/{layout.shard_name(0)}/{item_id}.flac"
                path = layout.resolve_rendered(wet_rel)
                levels.write_flac(path, y, sr, str(cfg.render.wet_subtype))
                d_wet, a_wet = _analysis(y, cfg)
                d_dry, a_dry = _analysis(x, cfg)
                ritem = RenderItem(
                    item_id=item_id,
                    clip_id=clip.clip_id,
                    split="gold",
                    seed=0,
                    scenario=expand(region, targets, gid).scenario,
                    engine="greybox",
                    pedal_id=None,
                    pedal_knobs=None,
                    board_params=params,
                    board_params_derived=None,
                    label_source="free",
                    clamped=False,
                    clamp_notes=[],
                    input_peak_dbfs=peak,
                    wet_gain_db=wet_gain_db,
                    wet_path=wet_rel,
                    sha256=levels.sha256_file(path),
                    descriptors_dry=d_dry,
                    descriptors_wet=d_wet,
                    archetype_dry=a_dry,
                    archetype_wet=a_wet,
                )
                records.append(
                    GoldRender(
                        item_id=item_id,
                        gold_id=gid,
                        gold_source="llm_draft",
                        role=role,
                        distractor_index=di,
                        axis=axis,
                        clip_index=ci,
                        clip_id=clip.clip_id,
                        in_scope=True,
                        ambiguous=gid == "fx003",
                        confidence="high",
                        unseen_pedal=False,
                        region=region.to_yaml(),
                        render=to_dict(ritem),
                    )
                )
    path = Path(str(cfg.eval.gold_manifest))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
    return len(records)


def _make_test_real(cfg, layout: DataLayout, rng) -> int:
    sr_real = 48000
    items = []
    for i, (kb, family, settings) in enumerate(
        (
            ("ts808", "overdrive", {"drive": 0.5, "tone": 0.5, "level": 0.5}),
            (None, "overdrive", {"gain": 0.7}),
        )
    ):
        x = resample(synth_riff("single_notes", 1.0, 44100, seed=50 + i), 44100, sr_real)
        x, _ = levels.normalize_peak(x, -6.0)
        y, _ = levels.normalize_peak(np.tanh(4.0 * x), -3.0)
        rel_dry = f"fixture/dev{i}/clean.flac"
        rel_wet = f"fixture/dev{i}/wet.flac"
        levels.write_flac(layout.test_real / rel_dry, x, sr_real, "PCM_24")
        levels.write_flac(layout.test_real / rel_wet, y, sr_real, "PCM_24")
        items.append(
            TestRealItem(
                item_id=f"fx-real-{i}",
                dataset="fixture",
                device=f"Dev{i}",
                family=family,
                kb_pedal_id=kb,
                settings=settings,
                dry_path=rel_dry,
                wet_path=rel_wet,
                sample_rate=sr_real,
                internal_only=i == 1,
                license="synthetic",
            )
        )
    write_jsonl(layout.test_real_manifest, items)
    return len(items)


__all__ = [
    "ARCHETYPE_KEYS",
    "EVAL_CONFIG",
    "FIXTURE_GOLD",
    "FIXTURE_GOLD_VERSION",
    "eval_fixture_config",
    "make_eval_fixture",
]
