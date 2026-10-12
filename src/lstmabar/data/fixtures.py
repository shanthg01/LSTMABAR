"""Tiny synthetic DI fixture set for tests and smoke runs (no downloads).

:func:`make_fixture_dataset` writes a handful of short synthetic riffs as dry clips in the
v1 layout, for three fake performers (two train, one held out), plus the dry-clip manifest.
Renderer, caption and split code are built and tested against it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from lstmabar.audio import RIFF_KINDS, synth_riff
from lstmabar.config import load_config
from lstmabar.data.layout import DataLayout
from lstmabar.data.levels import normalize_peak, sha256_file, write_flac
from lstmabar.data.schema import DryClip, write_jsonl
from lstmabar.data.splits import assign_splits

FIXTURE_PERFORMERS = ("fx1", "fx2", "fx3")
REPO_ROOT = Path(__file__).resolve().parents[3]  # src layout, editable install (uv)


def fixture_config(root: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """``configs/data/v1.yaml`` with every data dir under ``root`` and fixture performers."""
    cfg = load_config(REPO_ROOT / "configs" / "data" / "v1.yaml")
    root = Path(root)
    patch = {
        "data": {
            k: str(root / k.removesuffix("_dir"))
            for k in (
                "sources_dir",
                "rendered_dir",
                "cache_dir",
                "test_real_dir",
                "manifests_dir",
                "calibration_dir",
            )
        },
        "ingest": {
            "train_performers": list(FIXTURE_PERFORMERS[:2]),
            "heldout_performer": FIXTURE_PERFORMERS[2],
        },
        "paths": {"runs": str(root / "runs")},
    }
    cfg = OmegaConf.merge(cfg, patch)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def make_fixture_dataset(
    cfg: DictConfig, clips_per_performer: int = 2, seconds: float | None = None
) -> list[DryClip]:
    """Write synthetic dry clips + ``dry_clips.jsonl``; returns the records."""
    layout = DataLayout.from_config(cfg)
    sr = int(cfg.audio.sample_rate)
    seconds = float(seconds if seconds is not None else cfg.audio.clip_seconds)
    peak = float(cfg.ingest.dry_peak_dbfs)
    clips: list[DryClip] = []
    for p_idx, performer in enumerate(FIXTURE_PERFORMERS):
        for k in range(clips_per_performer):
            kind = RIFF_KINDS[(p_idx + k) % len(RIFF_KINDS)]
            seed = 100 * p_idx + k
            recording = f"{performer}-rec{k}"
            clip_id = f"fx-{performer}-rec{k}-0"
            x, gain_db = normalize_peak(synth_riff(kind, seconds, sr, seed=seed), peak)
            rel = layout.dry_relpath(clip_id)
            path = layout.resolve_rendered(rel)
            write_flac(path, x, sr, str(cfg.ingest.dry_subtype))
            clips.append(
                DryClip(
                    clip_id=clip_id,
                    source="fixture",
                    performer=performer,
                    recording_id=recording,
                    technique=kind,
                    source_file=f"fixture/{recording}.wav",
                    offset_s=0.0,
                    duration_s=seconds,
                    sample_rate=sr,
                    source_gain_db=gain_db,
                    path=rel,
                    sha256=sha256_file(path),
                    split="",
                    license="synthetic",
                )
            )
    splits = assign_splits(((c.source, c.performer, c.recording_id) for c in clips), cfg)
    clips = [replace(c, split=s) for c, s in zip(clips, splits, strict=True)]
    write_jsonl(layout.dry_manifest, clips)
    return clips


__all__ = ["FIXTURE_PERFORMERS", "fixture_config", "make_fixture_dataset"]
