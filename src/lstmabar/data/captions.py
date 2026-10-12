"""Task 4.3 (wave 1c): template captions + LLM paraphrase (resolution R1, decisions 5-6).

GOLD-BLIND: this module and anyone editing it must never read ``data/gold/``.

Contract: ``run(cfg, args)`` reads shard ``items.jsonl`` files and writes ``captions.jsonl``
next to them (``Caption`` records): templates from params/descriptors/KB vocab (no numbers;
pedal names only from :mod:`lstmabar.data.vocab`), then paraphrases via the Message Batches
API under ``captions.max_usd`` with the spend ledger at ``DataLayout.ledger``.
``--estimate`` prices a run without spending.
"""

from __future__ import annotations


def run(cfg, args) -> int:
    raise NotImplementedError("P4 task 4.3 (wave 1c)")
