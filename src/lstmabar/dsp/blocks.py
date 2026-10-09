"""Effect blocks: normalized knobs in, physical units inside :meth:`EffectBlock.process`.

Each block declares its knobs as :class:`~lstmabar.dsp.base.ParamSpec` (physical range, unit,
taper, default) and is built only from the low-level differentiable primitives in
``filters``, ``waveshaper``, ``oversample`` and ``compressor``. Every spec range lies inside
the clamps of those primitives, so no knob position has a clamped (zero) gradient.
"""

import math

import torch
from torch import Tensor

from lstmabar.dsp.base import EffectBlock, Params, ParamSpec
from lstmabar.dsp.compressor import compress
from lstmabar.dsp.filters import apply_filters, biquad, biquad_coeffs, tilt
from lstmabar.dsp.oversample import FACTORS, oversampled
from lstmabar.dsp.waveshaper import waveshape

_BUTTERWORTH_Q = 1.0 / math.sqrt(2.0)


def _db_to_lin(db: Tensor) -> Tensor:
    return torch.pow(10.0, db / 20.0).unsqueeze(-1)


def _full_like(v: Tensor, value: float) -> Tensor:
    return torch.full_like(v, value, dtype=torch.promote_types(v.dtype, torch.float64))


class Gain(EffectBlock):
    """Clean gain stage."""

    param_specs = (ParamSpec("gain_db", -24.0, 24.0, 0.0, "dB"),)

    def process(self, x: Tensor, p: Params) -> Tensor:
        return x * _db_to_lin(p["gain_db"])


class Drive(EffectBlock):
    """Overdrive/distortion: pre-HPF → input gain → oversampled waveshaper → tilt → level.

    The pre-clip high-pass (Tube-Screamer-style ~720 Hz by default) sets how much low end
    reaches the clipper; ``tone_db`` is a post-clip tilt EQ pivoting at 1 kHz.
    """

    param_specs = (
        ParamSpec("pre_hpf_hz", 20.0, 1500.0, 720.0, "Hz", taper="log"),
        ParamSpec("gain_db", 0.0, 60.0, 20.0, "dB"),
        ParamSpec("softness", 0.0, 1.0, 0.3),
        ParamSpec("asymmetry", 0.0, 1.0, 0.0),
        ParamSpec("bias", -0.25, 0.25, 0.0),
        ParamSpec("tone_db", -12.0, 12.0, 0.0, "dB"),
        ParamSpec("level_db", -36.0, 6.0, -12.0, "dB"),
    )

    def __init__(self, sample_rate: int = 44100, oversample: int = 4) -> None:
        super().__init__(sample_rate)
        if oversample not in FACTORS:
            raise ValueError(f"oversample must be one of {FACTORS}, got {oversample}")
        self.oversample = oversample

    def process(self, x: Tensor, p: Params) -> Tensor:
        x = biquad(x, "highpass", p["pre_hpf_hz"], self.sample_rate)
        x = x * _db_to_lin(p["gain_db"])
        x = oversampled(
            lambda u: waveshape(u, p["softness"], p["asymmetry"], p["bias"]),
            x,
            self.oversample,
        )
        x = tilt(x, p["tone_db"], self.sample_rate)
        return x * _db_to_lin(p["level_db"])


class EQ3(EffectBlock):
    """Three-band EQ: low shelf at 120 Hz, sweepable peak (Q 0.9), high shelf at 3 kHz."""

    LOW_HZ = 120.0
    HIGH_HZ = 3000.0
    MID_Q = 0.9

    param_specs = (
        ParamSpec("low_db", -12.0, 12.0, 0.0, "dB"),
        ParamSpec("mid_db", -12.0, 12.0, 0.0, "dB"),
        ParamSpec("mid_hz", 250.0, 4000.0, 800.0, "Hz", taper="log"),
        ParamSpec("high_db", -12.0, 12.0, 0.0, "dB"),
    )

    def process(self, x: Tensor, p: Params) -> Tensor:
        sr = self.sample_rate
        f_mid = p["mid_hz"]
        q_shelf = _full_like(f_mid, _BUTTERWORTH_Q)
        low = biquad_coeffs("lowshelf", _full_like(f_mid, self.LOW_HZ), q_shelf, p["low_db"], sr)
        mid = biquad_coeffs("peak", f_mid, _full_like(f_mid, self.MID_Q), p["mid_db"], sr)
        high = biquad_coeffs(
            "highshelf", _full_like(f_mid, self.HIGH_HZ), q_shelf, p["high_db"], sr
        )
        return apply_filters(x, [low, mid, high])


class Compressor(EffectBlock):
    """Feed-forward soft-knee compressor (see :func:`lstmabar.dsp.compressor.compress`)."""

    param_specs = (
        ParamSpec("threshold_db", -60.0, 0.0, -20.0, "dB"),
        ParamSpec("ratio", 1.0, 20.0, 4.0, "", taper="log"),
        ParamSpec("attack_ms", 0.5, 100.0, 10.0, "ms", taper="log"),
        ParamSpec("release_ms", 10.0, 1000.0, 150.0, "ms", taper="log"),
        ParamSpec("makeup_db", 0.0, 24.0, 0.0, "dB"),
    )

    def process(self, x: Tensor, p: Params) -> Tensor:
        return compress(
            x,
            p["threshold_db"],
            p["ratio"],
            p["attack_ms"],
            p["release_ms"],
            self.sample_rate,
            makeup_db=p["makeup_db"],
        )


__all__ = ["EQ3", "Compressor", "Drive", "Gain"]
