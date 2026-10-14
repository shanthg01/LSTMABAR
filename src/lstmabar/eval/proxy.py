"""Val "gold-proxy" for selection (D8). Wave 0: signatures; wave 1 (f) implements them.

Gold is test-only, so baselines, ``k`` and P6 early stopping select on val: MR-STFT to the
target wet + masked param L1, plus a **synthetic val 2AFC** whose distractors follow the gold
axis rules. :func:`label_to_region` maps a point label to a categorical ``gold.Region`` (levels
from the targets-file tables, nearest level when a value falls between table ranges) so
``gold.draft_distractors`` can be reused unchanged; distractors are sampled with ``gold.sample``
and rendered (grey-box) on the item's own dry clip at its own input level into
``data/rendered/<eval.proxy.version>/`` with a committed manifest
``data/manifests/<eval.proxy.version>.jsonl`` (``gold.GoldRender`` records, ``gold_source``
``"val_proxy"``).

Never select on any gold split.
"""

from __future__ import annotations

from collections.abc import Mapping

from lstmabar.data.gold import Region, Targets
from lstmabar.data.schema import RenderItem

PROXY_SOURCE = "val_proxy"


def label_to_region(item: RenderItem, targets: Targets) -> Region:
    """Categorical region of a val item's label.

    Pedal items (``scenario == "pedal"``) with a targets-file pedal table -> ``kind: pedal``
    with gain/brightness levels from the pedal's knob rotations (``item.pedal_knobs``); other
    pedal items and free-board items -> ``kind: free`` from ``board_params`` (gates decide
    ``gain: none`` / ``compression: none``; drive family and clipping from the nearest table
    entries).
    """
    raise NotImplementedError


def load_proxy_items(cfg, limit: int | None = None):
    """Val proxy units as ``eval.data.GoldItem`` s (``split="val"``, ``source=PROXY_SOURCE``,
    instruction = the item's first caption by ``caption_id``, ``input_peak_dbfs`` = the item's
    level, one clip whose target is the item's own wet as a ``GoldRender`` and whose
    distractors are the proxy renders), so ``runner.score_gold`` scores them unchanged.

    Keys: ``gold_id`` = the val ``item_id`` (so ``score_val`` merges per unit with the
    synthetic scores), ``text_key`` = that first ``caption_id`` and ``audio_key`` = the
    ``item_id`` (the val embedding-cache keys of ``SynthItem.queries``)."""
    raise NotImplementedError


def selection_score(result, cfg) -> float:
    """Scalar D8 selection objective of a ``runner.score_val`` result (lower is better):
    ``w_mrstft * mean MR-STFT to the target wet + w_param * mean masked param L1
    + w_afc * (1 - mean proxy 2AFC)`` with weights from ``eval.proxy.selection_weights``."""
    raise NotImplementedError


def region_levels(region: Region) -> Mapping[str, str]:
    """``{axis: level}`` of a region (for the report's per-axis breakdown)."""
    raise NotImplementedError


def run(cfg, args) -> int:
    """``lstmabar eval proxy``: draft + render the val proxy distractors (resumable)."""
    raise NotImplementedError


__all__ = [
    "PROXY_SOURCE",
    "label_to_region",
    "load_proxy_items",
    "region_levels",
    "run",
    "selection_score",
]
