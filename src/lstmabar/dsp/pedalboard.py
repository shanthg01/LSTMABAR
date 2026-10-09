"""An ordered chain of effect blocks with per-block wet/dry gates.

Parameters are nested dicts ``{block_name: {knob_name: (B,) tensor}}`` of **normalized**
values in [0, 1], plus an optional per-block ``"enabled"`` gate in [0, 1]. Each block's output
is ``g * block(x) + (1 - g) * x``, so ``g = 0`` bypasses exactly, ``g = 1`` is fully wet and
intermediate values blend (which keeps the gate differentiable for a predictor head).
"""

import torch
from torch import Tensor, nn

from lstmabar.dsp.base import EffectBlock
from lstmabar.dsp.blocks import EQ3, Compressor, Drive

BoardParams = dict[str, dict[str, Tensor]]
ENABLED = "enabled"


class Pedalboard(nn.Module):
    """Batched, differentiable chain of :class:`EffectBlock` s with wet/dry gates."""

    def __init__(self, blocks: list[tuple[str, EffectBlock]], sample_rate: int = 44100) -> None:
        super().__init__()
        names = [name for name, _ in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate block names: {names}")
        for name, block in blocks:
            if block.sample_rate != sample_rate:
                raise ValueError(
                    f"block {name!r} has sample_rate {block.sample_rate}, board has {sample_rate}"
                )
            if ENABLED in block.param_names:
                raise ValueError(f"block {name!r} uses the reserved param name {ENABLED!r}")
        self.blocks = nn.ModuleDict(blocks)
        self.sample_rate = sample_rate

    @property
    def block_names(self) -> list[str]:
        return list(self.blocks.keys())

    # -- parameters -----------------------------------------------------------------------

    def _check_names(self, params: BoardParams) -> None:
        unknown = set(params) - set(self.blocks)
        if unknown:
            raise KeyError(f"Pedalboard: unknown blocks {sorted(unknown)}")
        for name, p in params.items():
            bad = set(p) - set(self.blocks[name].param_names) - {ENABLED}
            if bad:
                raise KeyError(f"Pedalboard: unknown params for {name!r}: {sorted(bad)}")

    def default_params(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> BoardParams:
        """Every knob at its default (normalized) and every block enabled."""
        out: BoardParams = {}
        for name, block in self.blocks.items():
            p = block.default_params(batch_size, device, dtype)
            p[ENABLED] = torch.ones(batch_size, device=device, dtype=dtype)
            out[name] = p
        return out

    def random_params(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        p_enabled: float = 0.7,
    ) -> BoardParams:
        """Knobs ~ U(0, 1); gates ~ Bernoulli(``p_enabled``) as 0./1.

        Draws happen on the CPU (so a CPU generator works for any target device) in one
        ``(batch_size, P)`` tensor in :meth:`layout` order, then move to ``device``.
        """
        layout = self.layout()
        u = torch.rand(batch_size, len(layout), generator=generator, dtype=torch.float64)
        for j, (_, knob) in enumerate(layout):
            if knob == ENABLED:
                u[:, j] = (u[:, j] < p_enabled).to(u.dtype)
        return self.from_vector(u.to(device=device, dtype=dtype))

    def layout(self) -> list[tuple[str, str]]:
        """``(block, knob)`` pairs: per block in order, its knobs then ``"enabled"``."""
        return [
            (name, knob)
            for name, block in self.blocks.items()
            for knob in [*block.param_names, ENABLED]
        ]

    def _filled(self, params: BoardParams, batch: int, like: Tensor) -> BoardParams:
        """``params`` with every missing knob/gate filled by its default, as ``(B,)``."""
        self._check_names(params)
        full = self.default_params(batch, like.device, like.dtype)
        for name, p in params.items():
            for knob, v in p.items():
                v = torch.as_tensor(v, device=like.device, dtype=like.dtype)
                if v.numel() == 1:
                    v = v.reshape(1).expand(batch)
                if v.shape != (batch,):
                    raise ValueError(f"{name}.{knob}: expected shape ({batch},), got {v.shape}")
                full[name][knob] = v
        return full

    @staticmethod
    def _batch_of(params: BoardParams) -> tuple[int, Tensor]:
        for p in params.values():
            for v in p.values():
                if isinstance(v, Tensor) and v.dim() == 1:
                    return v.shape[0], v
        raise ValueError("cannot infer batch size from params (no (B,) tensors)")

    def to_vector(self, params: BoardParams) -> Tensor:
        """``(B, P)`` tensor in :meth:`layout` order, with defaults for missing entries."""
        batch, like = self._batch_of(params)
        full = self._filled(params, batch, like)
        return torch.stack([full[name][knob] for name, knob in self.layout()], dim=-1)

    def from_vector(self, v: Tensor) -> BoardParams:
        """Inverse of :meth:`to_vector`: ``(B, P)`` → nested dict of ``(B,)`` columns."""
        layout = self.layout()
        if v.dim() != 2 or v.shape[-1] != len(layout):
            raise ValueError(f"expected (B, {len(layout)}), got {tuple(v.shape)}")
        out: BoardParams = {name: {} for name in self.blocks}
        for j, (name, knob) in enumerate(layout):
            out[name][knob] = v[:, j]
        return out

    def describe(self, params: BoardParams) -> dict[str, dict[str, float | list[float]]]:
        """Physical knob values (spec units) and gates, for display.

        Floats when the batch size is 1, otherwise lists of floats. Units are available from
        each block's ``param_specs``.
        """
        batch, like = self._batch_of(params)
        full = self._filled(params, batch, like.detach())
        out: dict[str, dict[str, float | list[float]]] = {}
        for name, block in self.blocks.items():
            values = {}
            for spec in block.param_specs:
                values[spec.name] = spec.denormalize(full[name][spec.name].detach())
            values[ENABLED] = full[name][ENABLED].detach().clamp(0.0, 1.0)
            out[name] = {
                k: float(t[0]) if batch == 1 else [float(e) for e in t] for k, t in values.items()
            }
        return out

    # -- processing -----------------------------------------------------------------------

    def forward(self, x: Tensor, params: BoardParams | None = None) -> Tensor:
        """``(B, T)`` → ``(B, T)``; missing blocks/knobs use defaults (gates default to 1)."""
        if x.dim() != 2:
            raise ValueError(f"expected audio of shape (B, T), got {tuple(x.shape)}")
        params = params or {}
        self._check_names(params)
        for name, block in self.blocks.items():
            p = dict(params.get(name, {}))
            gate = p.pop(ENABLED, None)
            if gate is None:
                x = block(x, p)
                continue
            g = torch.as_tensor(gate, device=x.device, dtype=x.dtype)
            if g.numel() == 1:
                g = g.reshape(1).expand(x.shape[0])
            if g.shape != (x.shape[0],):
                raise ValueError(f"{name}.enabled: expected shape ({x.shape[0]},), got {g.shape}")
            if not g.requires_grad and bool((g == 0).all()):
                continue  # whole batch bypassed: skip the work (output would equal x exactly)
            g = g.unsqueeze(-1)
            x = g * block(x, p) + (1.0 - g) * x
        return x


def default_pedalboard(sample_rate: int = 44100) -> Pedalboard:
    """Compressor → Drive → EQ (block names ``"compressor"``, ``"drive"``, ``"eq"``)."""
    return Pedalboard(
        [
            ("compressor", Compressor(sample_rate)),
            ("drive", Drive(sample_rate)),
            ("eq", EQ3(sample_rate)),
        ],
        sample_rate=sample_rate,
    )


__all__ = ["ENABLED", "BoardParams", "Pedalboard", "default_pedalboard"]
