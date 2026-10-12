"""Resolution R4 (wave 1b): per-pedal white-box calibration tables for labels.

Contract: ``run(cfg, args)`` fits best-fit board params (``physics.fidelity`` machinery) on a
``calibration.gain_points`` x ``calibration.tone_points`` knob grid per white-box pedal, jointly
over ``calibration.sine_levels_dbfs``, warm-started along the grid, and writes
``DataLayout.calibration_table(pedal)``. ``CalibrationTable.lookup(knobs)`` returns normalized
``FloatBoard`` params by bilinear interpolation, falling back to the nearest grid point where
neighbours jump by more than ``calibration.max_jump``.
"""

from __future__ import annotations


def run(cfg, args) -> int:
    raise NotImplementedError("P4 resolution R4 (wave 1b)")
