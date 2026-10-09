"""Differentiable pedalboard: filters, waveshapers, oversampling, effect blocks."""

from lstmabar.dsp import blocks
from lstmabar.dsp.pedalboard import BoardParams, Pedalboard, default_pedalboard

__all__ = ["BoardParams", "Pedalboard", "blocks", "default_pedalboard"]
