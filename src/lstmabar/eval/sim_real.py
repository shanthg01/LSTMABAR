"""``test_real`` sim-to-real gap (D11). Wave 0: signatures; wave 1 (e) implements them.

For items with a ``kb_pedal_id``: render the KB pedal at the item's ``settings`` on its clean
input (white-box ts808; grey-box rat, and the derived grey-box board for ts808 too), resample
48 -> 44.1 kHz, loudness-match to the real wet (true input level unknown, so also sweep
``eval.sim_real.input_peak_sweep_dbfs``) and report MR-STFT and descriptor errors. EGFxSet
``settings`` keys are mapped onto KB knobs with their direction checked (TS: drive/tone/level;
RAT: distortion/filter/volume). BD-2 and pOD-set pedals (no KB entry): family-level descriptor
comparisons only. ``internal_only`` items are reported in aggregate only. Text-driven methods
are not evaluated on ``test_real`` in P5 (no instruction text).
"""

from __future__ import annotations

from typing import Any


def evaluate(cfg, limit: int | None = None) -> dict[str, Any]:
    """The report's ``test_real`` section (aggregate-only where any item is internal)."""
    raise NotImplementedError


def run(cfg, args) -> int:
    """``lstmabar eval sim-real``: writes ``reports/sim_real.{md,json}``."""
    raise NotImplementedError


__all__ = ["evaluate", "run"]
