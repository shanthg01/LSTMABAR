"""Split assignment (decision 11, R2, R6).

- Guitar-TECHS: the ``ingest.train_performers`` go to ``train``; the ``ingest.heldout_performer``
  is split by **recording** into ``val`` / ``test`` (a hash of the recording id, so no source
  recording is shared between val and test and the assignment does not depend on file order).
- Own DI (``source == "own"``): test only.
- ``test_unseen_pedal`` is not a clip split: it is the set of holdout-pedal renders made on
  ``test`` clips (``render.holdout_pedal``).

Task 4.6 (wave 2) adds the checks, hashes and data card on top of this.
"""

from __future__ import annotations

import hashlib


def unit_hash(key: str, seed: int = 0) -> float:
    """Deterministic value in [0, 1) from ``key`` (platform-independent)."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64


def assign_clip_split(
    source: str,
    performer: str,
    recording_id: str,
    train_performers: list[str] | tuple[str, ...],
    heldout_performer: str | None,
    val_fraction: float = 0.5,
    seed: int = 0,
) -> str:
    if source == "own":
        return "test"
    if source == "egfxset":
        raise ValueError("EGFxSet is test_real only (R2); it has no dry-clip split")
    if performer in train_performers:
        return "train"
    if heldout_performer is not None and performer == heldout_performer:
        return "val" if unit_hash(recording_id, seed) < val_fraction else "test"
    raise ValueError(f"performer {performer!r} is in neither train_performers nor heldout")


def assign_from_config(source: str, performer: str, recording_id: str, cfg) -> str:
    return assign_clip_split(
        source,
        performer,
        recording_id,
        list(cfg.ingest.train_performers),
        cfg.ingest.heldout_performer,
        float(cfg.splits.val_fraction),
        int(cfg.splits.seed),
    )


__all__ = ["assign_clip_split", "assign_from_config", "unit_hash"]
