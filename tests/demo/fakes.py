"""A toy board implementing the ``Pedalboard`` API, so demo tests don't need the real one."""

import torch
from torch import Tensor, nn

from lstmabar.dsp.base import EffectBlock, ParamSpec


class Gain(EffectBlock):
    param_specs = (ParamSpec("gain_db", -24.0, 24.0, 0.0, "dB"),)

    def process(self, x: Tensor, p: dict[str, Tensor]) -> Tensor:
        return x * 10.0 ** (p["gain_db"][:, None] / 20.0)


class Smooth(EffectBlock):
    """Two-tap moving average blended by ``mix``; has a log-taper knob for UI coverage."""

    param_specs = (
        ParamSpec("cutoff_hz", 100.0, 10000.0, 1000.0, "Hz", taper="log"),
        ParamSpec("mix", 0.0, 1.0, 0.5),
    )

    def process(self, x: Tensor, p: dict[str, Tensor]) -> Tensor:
        smoothed = (x + torch.roll(x, 1, dims=-1)) / 2
        m = p["mix"][:, None]
        return (1 - m) * x + m * smoothed


class FakeBoard(nn.Module):
    def __init__(self, sample_rate: int = 44100) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.blocks = nn.ModuleDict(
            {"drive": Gain(sample_rate), "eq": Smooth(sample_rate), "level": Gain(sample_rate)}
        )

    @property
    def block_names(self) -> list[str]:
        return list(self.blocks.keys())

    def default_params(self, batch_size, device="cpu", dtype=torch.float32):
        out = {}
        for name, blk in self.blocks.items():
            p = blk.default_params(batch_size, device, dtype)
            p["enabled"] = torch.ones(batch_size, device=device, dtype=dtype)
            out[name] = p
        return out

    def forward(self, x: Tensor, params=None) -> Tensor:
        params = params or {}
        for name, blk in self.blocks.items():
            p = dict(params.get(name, {}))
            gate = p.pop("enabled", torch.ones(1)).to(x).reshape(-1, 1)
            x = gate * blk(x, p) + (1 - gate) * x
        return x

    def describe(self, params):
        out = {}
        for name, blk in self.blocks.items():
            p = dict(params.get(name, {}))
            gate = p.pop("enabled", torch.ones(1))
            phys = blk.physical(p, torch.zeros(gate.shape[0], 1))
            d = {k: v.tolist() for k, v in phys.items()}
            d["enabled"] = gate.tolist()
            out[name] = d
        return out
