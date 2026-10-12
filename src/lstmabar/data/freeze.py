"""Tasks 4.6-4.7 (wave 2): split checks, content hashes, index, data card, vocab audit.

Contract: ``run(cfg, args)`` verifies split disjointness, hashes every dry/wet file into
``DataLayout.file_hashes``, writes ``DataLayout.index`` (counts per split/scenario/engine/pedal,
clamped counts, per-shard hashes, dataset hash) and regenerates ``DATA_CARD.md`` numbers.
"""

from __future__ import annotations


def run(cfg, args) -> int:
    raise NotImplementedError("P4 tasks 4.6-4.7 (wave 2)")
