"""Task 4.1 (wave 1a): Guitar-TECHS DI ingest -> onset-aligned dry clips + manifest.

Contract: ``run(cfg, args)`` lists the Zenodo record's files (R9), downloads one archive at a
time into ``data.sources_dir``, extracts the DI tracks only, deletes the archive, resamples to
``audio.sample_rate``, cuts ``audio.clip_seconds`` clips starting ``ingest.pre_onset_ms`` before
detected onsets (skipping clips quieter than ``ingest.min_rms_dbfs``), stores each at
``ingest.dry_peak_dbfs`` (:mod:`lstmabar.data.levels`) under :class:`DataLayout` and writes
``DryClip`` records (split assigned by :func:`lstmabar.data.splits.assign_from_config`) to
``DataLayout.dry_manifest``.
"""

from __future__ import annotations


def run(cfg, args) -> int:
    raise NotImplementedError("P4 task 4.1 (wave 1a)")
