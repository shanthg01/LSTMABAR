"""Offline, non-differentiable circuit simulations of real pedals (P2 decision 5).

Interface: a builder registered per pedal id with :func:`register_whitebox` returns a
:class:`WhiteBoxModel` for a knob setting; :func:`simulate` converts full-scale audio to
volts (:mod:`lstmabar.physics.calibration`), runs the model and converts back.
"""

from lstmabar.physics.whitebox.base import (
    WHITEBOX,
    WhiteBoxModel,
    has_whitebox,
    register_whitebox,
    simulate,
)

__all__ = ["WHITEBOX", "WhiteBoxModel", "has_whitebox", "register_whitebox", "simulate"]
