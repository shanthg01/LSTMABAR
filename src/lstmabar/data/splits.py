"""Split assignment (decision 11, R2, R6).

- Guitar-TECHS: the ``ingest.train_performers`` go to ``train``; the ``ingest.heldout_performer``
  is split by **recording** into ``val`` / ``test``: recordings are sorted by a hash of their id
  and the first ``round(n * val_fraction)`` go to val (at least one each when n >= 2), so no
  source recording is shared between val and test, the result does not depend on file order,
  and neither split is empty by chance.
- Own DI (``source == "own"``): test only.
- ``test_unseen_pedal`` is not a clip split: it is the set of holdout-pedal renders made on
  ``test`` clips (``render.holdout_pedal``).

Task 4.6 (wave 2) adds the checks, hashes and data card on top of this.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping


def unit_hash(key: str, seed: int = 0) -> float:
    """Deterministic value in [0, 1) from ``key`` (platform-independent)."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64


def heldout_split_map(
    recording_ids: Iterable[str], val_fraction: float = 0.5, seed: int = 0
) -> dict[str, str]:
    """``{recording_id: "val" | "test"}`` for the held-out performer's recordings."""
    ids = sorted(set(recording_ids), key=lambda r: (unit_hash(r, seed), r))
    n = len(ids)
    n_val = round(n * val_fraction)
    if n >= 2:
        n_val = min(max(n_val, 1), n - 1)
    return {r: ("val" if i < n_val else "test") for i, r in enumerate(ids)}


def assign_clip_split(
    source: str,
    performer: str,
    recording_id: str,
    train_performers: list[str] | tuple[str, ...],
    heldout_performer: str | None,
    heldout_map: Mapping[str, str],
) -> str:
    if source == "own":
        return "test"
    if source == "egfxset":
        raise ValueError("EGFxSet is test_real only (R2); it has no dry-clip split")
    if performer in train_performers:
        return "train"
    if heldout_performer is not None and performer == heldout_performer:
        if recording_id not in heldout_map:
            raise KeyError(f"recording {recording_id!r} missing from the held-out split map")
        return heldout_map[recording_id]
    raise ValueError(f"performer {performer!r} is in neither train_performers nor heldout")


def assign_splits(clips: Iterable[tuple[str, str, str]], cfg) -> list[str]:
    """Splits for ``(source, performer, recording_id)`` triples using ``cfg.ingest`` /
    ``cfg.splits``; the held-out map is built from the held-out performer's recordings."""
    clips = list(clips)
    heldout = cfg.ingest.heldout_performer
    hmap = heldout_split_map(
        (r for _, p, r in clips if p == heldout),
        float(cfg.splits.val_fraction),
        int(cfg.splits.seed),
    )
    train = list(cfg.ingest.train_performers)
    return [assign_clip_split(s, p, r, train, heldout, hmap) for s, p, r in clips]


__all__ = ["assign_clip_split", "assign_splits", "heldout_split_map", "unit_hash"]
