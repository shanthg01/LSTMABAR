"""``reports/p5_baselines.{md,json}`` (task 5.4, D6). Wave 0: the JSON schema; wave 2 fills
in :func:`build_report` / :func:`render_markdown`.

JSON layout (``schema: 1``)::

    {
      "schema": 1,
      "created": "...", "git": {...}, "config": {...resolved eval config...},
      "hashes": {"dataset_sha256": ..., "gold_index_sha256": ..., "embeddings": {relpath: sha}},
      "methods": [{"name": ..., "info": {...}, "seeds": [0, ...]}],
      "rows": [Row, ...],               # long format, one cell per row
      "comparisons": [ComparisonRow, ...],
      "test_real": {...},               # D11 section (sim-to-real gap), aggregate only for
                                        # internal_only items
      "notes": ["...", ...]             # caveats the markdown prints verbatim
    }

A cell is ``method x split x metric x subset x gold_source``; ``gold_source`` is ``"-"`` for
synthetic splits and one of ``data.gold.SOURCES`` (``human`` carried from the start, "n/a,
0 items" until friend items are paired) for gold. No path in the report is absolute.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lstmabar.eval.runner import SplitResult

REPORT_SCHEMA = 1


@dataclass(frozen=True)
class Row:
    method: str
    split: str  # test | test_unseen_pedal | val | gold
    metric: str  # runner.SYNTH_SCORES / GOLD_SCORES key, or "gate_f1/<block>"
    subset: str  # "all" | data.GOLD_SUBSETS
    gold_source: str  # "-" | data.gold.SOURCES
    mean: float
    lo: float
    hi: float
    n: int
    seed_sd: float | None = None  # trained methods: SD of per-seed means (D3)
    n_seeds: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonRow:
    a: str
    b: str
    split: str
    metric: str
    subset: str
    gold_source: str
    comparison: dict[str, Any]  # stats.Comparison.to_dict()


def build_report(
    results: Sequence[SplitResult], cfg, extras: dict[str, Any] | None = None
) -> dict[str, Any]:
    raise NotImplementedError


def render_markdown(report: dict[str, Any]) -> str:
    raise NotImplementedError


def write_report(report: dict[str, Any], out_dir: str | Path, name: str) -> tuple[Path, Path]:
    raise NotImplementedError


__all__ = ["REPORT_SCHEMA", "ComparisonRow", "Row", "build_report", "render_markdown"]
