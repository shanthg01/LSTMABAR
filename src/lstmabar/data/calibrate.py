"""Resolution R4 (wave 1b-cal): per-pedal white-box calibration tables for labels.

Contract: ``run(cfg, args)`` fits best-fit board params (``physics.fidelity`` machinery, at
``audio.sample_rate``) on a ``calibration.gain_points`` x ``calibration.tone_points`` knob grid
per pedal in ``render.whitebox_pedals``, jointly over ``calibration.sine_levels_dbfs``.
Each gain chain starts from the derived params at gain = 0 and is warm-started along the
gain axis. Tables are written to ``DataLayout.calibration_table(pedal)`` (JSON, committed).

The renderer only uses :meth:`CalibrationTable.load` and :meth:`CalibrationTable.lookup`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from lstmabar.data.schema import FloatBoard


@dataclass(frozen=True)
class Label:
    """Result of :meth:`CalibrationTable.lookup`."""

    board: FloatBoard  # normalized params, full BoardParams layout incl. "enabled"
    source: str  # "calibrated" (interpolated) | "calibrated_nearest" (fallback)
    clamp_notes: list[str] = field(default_factory=list)  # e.g. level offset hit the range


@dataclass(frozen=True)
class CalibrationTable:
    """Fitted normalized board params on a knob grid for one white-box pedal.

    ``gain_knob`` / ``tone_knob`` name the pedal's pots spanning the grid (e.g. ``drive`` /
    ``tone`` for ts808, ``dist`` / ``tone`` for ds1); ``level_knob`` is applied analytically.
    ``boards[i][j]`` is the fit at ``(gain_values[i], tone_values[j])``.
    """

    pedal_id: str
    gain_knob: str
    tone_knob: str
    level_knob: str | None
    gain_values: list[float]
    tone_values: list[float]
    boards: list[list[FloatBoard]]
    fit_error_db: list[list[float]] = field(default_factory=list)
    max_jump: float = 0.25
    meta: dict = field(default_factory=dict)  # config, git sha, sample rate, levels, f0s

    @classmethod
    def load(cls, path: str | Path) -> CalibrationTable:
        raise NotImplementedError("P4 resolution R4 (wave 1b-cal)")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("P4 resolution R4 (wave 1b-cal)")

    def lookup(self, knobs: Mapping[str, float]) -> Label:
        """Normalized label for full pedal ``knobs`` (rotations in [0, 1]).

        Bilinear interpolation of the grid fits (``source="calibrated"``), or the nearest grid
        point (``"calibrated_nearest"``) where neighbouring fits differ by more than
        ``max_jump``. The level knob's analytic offset is added to ``drive.level_db`` and
        clamped to Drive's range; a clamp is reported in ``Label.clamp_notes``.
        """
        raise NotImplementedError("P4 resolution R4 (wave 1b-cal)")


def run(cfg, args) -> int:
    raise NotImplementedError("P4 resolution R4 (wave 1b-cal)")


__all__ = ["CalibrationTable", "Label", "run"]
