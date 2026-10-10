"""White-box model contract.

- Models are NumPy, float64, batched: input volts ``(B, T)`` -> output volts ``(B, T)``,
  time-aligned with the input (no added latency beyond what the circuit itself causes).
- Knob settings are fixed per model instance (built by a registered builder).
- Models handle their own oversampling internally and return audio at ``sample_rate``.
- Op-amps are ideal (no rails, no slew/bandwidth limit) unless a model documents otherwise.
- Registration: each circuit lives in its own module under ``physics/whitebox/`` and registers
  builders with ``@register_whitebox("<pedal id>")``; ``physics/whitebox/__init__.py`` imports
  those modules so the registry is filled on import.
- Device model parameters (diode Is/n, transistor Is/beta) live in a table keyed by part
  number in ``physics/whitebox/devices.py``, not in the pedal YAML.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import numpy as np

from lstmabar.physics.calibration import VOLTS_PER_FULL_SCALE
from lstmabar.physics.kb import Pedal


class WhiteBoxModel(ABC):
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    @abstractmethod
    def process_volts(self, v: np.ndarray) -> np.ndarray:
        """Simulate the circuit: ``(B, T)`` float64 volts in, ``(B, T)`` volts out."""


Builder = Callable[[Pedal, Mapping[str, float], int], WhiteBoxModel]
WHITEBOX: dict[str, Builder] = {}


def register_whitebox(pedal_id: str) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        if pedal_id in WHITEBOX:
            raise ValueError(f"white-box builder for {pedal_id!r} already registered")
        WHITEBOX[pedal_id] = fn
        return fn

    return deco


def has_whitebox(pedal_id: str) -> bool:
    return pedal_id in WHITEBOX


def simulate(
    pedal: Pedal,
    audio: np.ndarray,
    sample_rate: int = 44100,
    knobs: Mapping[str, float] | None = None,
    volts_per_fs: float = VOLTS_PER_FULL_SCALE,
    **build_kwargs,
) -> np.ndarray:
    """Run full-scale ``audio`` (``(T,)`` or ``(B, T)``) through ``pedal``'s white-box model.

    ``build_kwargs`` (e.g. ``oversample``, ``backend``) are passed to the pedal's builder.
    """
    if pedal.id not in WHITEBOX:
        raise KeyError(f"no white-box model for pedal {pedal.id!r}")
    knobs = pedal.knobs(knobs)
    x = np.asarray(audio, dtype=np.float64)
    squeeze = x.ndim == 1
    x = np.atleast_2d(x)
    if x.ndim != 2:
        raise ValueError(f"expected audio of shape (T,) or (B, T), got {x.shape}")
    model = WHITEBOX[pedal.id](pedal, knobs, sample_rate, **build_kwargs)
    y = model.process_volts(x * volts_per_fs) / volts_per_fs
    return y[0] if squeeze else y


__all__ = ["WHITEBOX", "WhiteBoxModel", "has_whitebox", "register_whitebox", "simulate"]
