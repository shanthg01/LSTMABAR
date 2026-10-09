"""Shared contracts for differentiable effect blocks.

Conventions used across ``lstmabar.dsp``:

- Audio tensors are mono, shape ``(B, T)``, float32, nominal range [-1, 1].
- Per-example parameters are tensors of shape ``(B,)``.
- Blocks receive **normalized** parameters in [0, 1] (what a sigmoid head emits) and map them
  to physical units through their :class:`ParamSpec`. Low-level functions in
  ``filters``/``waveshaper``/``oversample``/``compressor`` take **physical** units.
- Everything is differentiable with respect to audio and parameters, and runs on any device.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

Params = dict[str, Tensor]


@dataclass(frozen=True)
class ParamSpec:
    """A knob: physical range, unit and taper. ``default`` is in physical units."""

    name: str
    min: float
    max: float
    default: float
    unit: str = ""
    taper: Literal["linear", "log"] = "linear"

    def __post_init__(self) -> None:
        if not self.min < self.max:
            raise ValueError(f"{self.name}: min must be < max")
        if not self.min <= self.default <= self.max:
            raise ValueError(f"{self.name}: default outside [min, max]")
        if self.taper == "log" and self.min <= 0:
            raise ValueError(f"{self.name}: log taper needs min > 0")

    def denormalize(self, u: Tensor) -> Tensor:
        """Map normalized [0, 1] to physical units (input is clamped to [0, 1])."""
        u = u.clamp(0.0, 1.0)
        if self.taper == "log":
            return self.min * (self.max / self.min) ** u
        return self.min + (self.max - self.min) * u

    def normalize(self, v: Tensor) -> Tensor:
        """Map physical units to normalized [0, 1] (output is clamped to [0, 1])."""
        if self.taper == "log":
            u = torch.log(v / self.min) / math.log(self.max / self.min)
        else:
            u = (v - self.min) / (self.max - self.min)
        return u.clamp(0.0, 1.0)

    @property
    def default_normalized(self) -> float:
        return float(self.normalize(torch.tensor(self.default, dtype=torch.float64)))


class EffectBlock(nn.Module, ABC):
    """A differentiable effect with named knobs.

    Subclasses set ``param_specs`` and implement :meth:`process`, which receives physical
    parameter values. Callers use ``block(x, params)`` with normalized params; missing params
    fall back to their defaults.
    """

    param_specs: tuple[ParamSpec, ...] = ()

    def __init__(self, sample_rate: int = 44100) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    @property
    def param_names(self) -> list[str]:
        return [s.name for s in self.param_specs]

    def default_params(self, batch_size: int, device: torch.device | str = "cpu") -> Params:
        """Normalized default parameters, each of shape ``(batch_size,)``."""
        return {
            s.name: torch.full((batch_size,), s.default_normalized, device=device)
            for s in self.param_specs
        }

    def physical(
        self, params: Params | None, batch_size: int, device: torch.device | str = "cpu"
    ) -> Params:
        """Denormalize a (possibly partial) normalized parameter dict, filling defaults."""
        params = params or {}
        unknown = set(params) - set(self.param_names)
        if unknown:
            raise KeyError(f"{type(self).__name__}: unknown params {sorted(unknown)}")
        defaults = self.default_params(batch_size, device)
        return {
            s.name: s.denormalize(params.get(s.name, defaults[s.name])) for s in self.param_specs
        }

    def forward(self, x: Tensor, params: Params | None = None) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"expected audio of shape (B, T), got {tuple(x.shape)}")
        return self.process(x, self.physical(params, x.shape[0], x.device))

    @abstractmethod
    def process(self, x: Tensor, p: Params) -> Tensor:
        """Apply the effect. ``p`` holds physical values, each of shape ``(B,)``."""
