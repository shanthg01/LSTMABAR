"""Task 4.2 (wave 1b): dry clip x (pedal preset | free board) x sampled knobs -> wet shards.

Contract: ``run(cfg, args)`` reads ``DataLayout.dry_manifest``, samples render specs per clip
(resolutions R3, R5, R6, R7; deterministic per-item seeds), renders TS808/DS-1 through the
white-box and everything else through the grey-box ``default_pedalboard``, labels white-box
items from the calibration tables (:mod:`lstmabar.data.calibrate`), measures descriptors and
archetypes (pYIN track computed once per dry clip, cached at ``DataLayout.f0_cache``), and
writes shards of ``render.shard_size`` items (FLAC + ``items.jsonl`` + ``DONE`` last) with
multiprocessing; finished shards are skipped on re-run.
"""

from __future__ import annotations


def run(cfg, args) -> int:
    raise NotImplementedError("P4 task 4.2 (wave 1b)")


def bench(cfg, args) -> int:
    """R8 throughput gate: time render + analysis per item on fixtures, extrapolate to 20k."""
    raise NotImplementedError("P4 task 4.2 (wave 1b)")
