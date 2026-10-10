"""Offline, non-differentiable circuit simulations of real pedals (P2 decision 5).

Interface: a builder registered per pedal id with :func:`register_whitebox` returns a
:class:`WhiteBoxModel` for a knob setting; :func:`simulate` converts full-scale audio to
volts (:mod:`lstmabar.physics.calibration`), runs the model and converts back.

Circuit modules imported below register their builders on import.
"""

from lstmabar.physics.whitebox import ds1, ts808  # noqa: F401  (register "ds1", "ts808")
from lstmabar.physics.whitebox.base import (
    WHITEBOX,
    WhiteBoxModel,
    has_whitebox,
    register_whitebox,
    simulate,
)
from lstmabar.physics.whitebox.diode_clipper import (
    ClipperCircuit,
    DiodeClipper,
    DiodePair,
    feedback_clipper,
    shunt_clipper,
)

__all__ = [
    "WHITEBOX",
    "ClipperCircuit",
    "DiodeClipper",
    "DiodePair",
    "WhiteBoxModel",
    "feedback_clipper",
    "has_whitebox",
    "register_whitebox",
    "shunt_clipper",
    "simulate",
]
