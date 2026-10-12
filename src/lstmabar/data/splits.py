"""Split assignment (decision 11 as revised at P4 wave 2, R2, R6).

Guitar-TECHS facts drove the revision: the held-out player P3 recorded only 12 musical
excerpts (~8 min of DI), too little to split again into val and test, and out of domain for
model selection (different player, guitar and content). So:

- ``test``: every clip of the ``ingest.heldout_performer`` (performer-disjoint from train).
- ``val``: an in-domain, **by-recording** slice of the train performers: within each
  (performer, technique) group, recordings are sorted by a hash of their id and the first
  ``max(1, round(n * splits.val_recording_fraction))`` go to val (groups with a single
  recording stay in train). No recording is shared between splits, and the result does not
  depend on file order.
- ``train``: the remaining train-performer recordings.
- Own DI (``source == "own"``): test only.
- ``test_unseen_pedal`` is not a clip split: it is the set of holdout-pedal renders made on
  ``test`` clips (``render.holdout_pedal``).

Val is therefore performer-*overlapping* (it measures generalisation to new recordings, not
new players); test is the performer-disjoint set. The data card says so.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping


def unit_hash(key: str, seed: int = 0) -> float:
    """Deterministic value in [0, 1) from ``key`` (platform-independent)."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64


def val_recordings(
    recordings: Iterable[tuple[str, str, str]], fraction: float = 0.1, seed: int = 0
) -> set[str]:
    """Recording ids sent to val from ``(performer, technique, recording_id)`` triples."""
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for performer, technique, rec in recordings:
        groups[(performer, technique)].add(rec)
    out: set[str] = set()
    for recs in groups.values():
        if len(recs) < 2 or fraction <= 0:
            continue
        ids = sorted(recs, key=lambda r: (unit_hash(r, seed), r))
        n_val = min(max(round(len(ids) * fraction), 1), len(ids) - 1)
        out.update(ids[:n_val])
    return out


def assign_clip_split(
    source: str,
    performer: str,
    recording_id: str,
    train_performers: list[str] | tuple[str, ...],
    heldout_performer: str | None,
    val_set: set[str] | frozenset[str] | Mapping[str, object] = frozenset(),
) -> str:
    if source == "own":
        return "test"
    if source == "egfxset":
        raise ValueError("EGFxSet is test_real only (R2); it has no dry-clip split")
    if heldout_performer is not None and performer == heldout_performer:
        return "test"
    if performer in train_performers:
        return "val" if recording_id in val_set else "train"
    raise ValueError(f"performer {performer!r} is in neither train_performers nor heldout")


def assign_splits(clips: Iterable[tuple[str, ...]], cfg) -> list[str]:
    """Splits for ``(source, performer, recording_id[, technique])`` tuples using
    ``cfg.ingest`` / ``cfg.splits``. Call once over the full clip list (the val slice is chosen
    per (performer, technique) group)."""
    clips = [tuple(c) for c in clips]
    train = list(cfg.ingest.train_performers)
    vset = val_recordings(
        ((c[1], c[3] if len(c) > 3 else "", c[2]) for c in clips if c[1] in train),
        float(cfg.splits.val_recording_fraction),
        int(cfg.splits.seed),
    )
    heldout = cfg.ingest.heldout_performer
    return [assign_clip_split(c[0], c[1], c[2], train, heldout, vset) for c in clips]


__all__ = ["assign_clip_split", "assign_splits", "unit_hash", "val_recordings"]
