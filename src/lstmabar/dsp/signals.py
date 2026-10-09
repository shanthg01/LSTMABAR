"""Deterministic synthetic test signals (no data files needed).

``pluck_riff`` is a cheap stand-in for a DI guitar recording: Karplus-Strong plucked notes
(each one a single ``scipy.signal.lfilter`` call) placed on a short riff. Used by the tests
and by the parameter-recovery procedure (:mod:`lstmabar.dsp.recovery`).
"""

import math

import numpy as np
import torch
from scipy.signal import lfilter
from torch import Tensor

# E2-A2-D3 power-chord-ish riff: (onset s, MIDI note, velocity)
DEFAULT_RIFF: tuple[tuple[float, int, float], ...] = (
    (0.00, 40, 1.0),
    (0.18, 47, 0.7),
    (0.36, 52, 0.8),
    (0.54, 45, 0.9),
    (0.72, 52, 0.6),
    (0.90, 50, 1.0),
    (1.08, 57, 0.7),
)


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12)


def karplus_strong(
    freq_hz: float,
    seconds: float,
    sample_rate: int = 44100,
    decay: float = 0.996,
    brightness: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """One plucked string: ``y[n] = e[n] + decay * (y[n-N] + y[n-N-1]) / 2``.

    The excitation ``e`` is one period of noise, one-pole low-passed (``brightness`` in (0, 1],
    1 = raw noise) and made zero-mean. Returns float64, peak-normalized to 1.
    """
    n = max(2, round(sample_rate / freq_hz - 0.5))  # two-tap average adds half a sample delay
    length = int(seconds * sample_rate)
    rng = np.random.default_rng(seed)
    burst = lfilter([brightness], [1.0, brightness - 1.0], rng.uniform(-1, 1, n))
    excitation = np.zeros(length)
    excitation[: min(n, length)] = (burst - burst.mean())[:length]
    a = np.zeros(n + 2)
    a[0] = 1.0
    a[n] = a[n + 1] = -0.5 * decay
    y = lfilter([1.0], a, excitation)
    return y / (np.abs(y).max() + 1e-12)


def pluck_riff(
    seconds: float = 1.25,
    sample_rate: int = 44100,
    peak: float = 0.5,
    seed: int = 0,
    riff: tuple[tuple[float, int, float], ...] = DEFAULT_RIFF,
) -> Tensor:
    """Monophonic Karplus-Strong riff, shape ``(T,)`` float32, peak ``peak``.

    Each note rings until the next onset (with a 5 ms release fade, like a fretting hand) or
    the end of the clip; the riff loops if ``seconds`` exceeds its length.
    """
    length = int(seconds * sample_rate)
    out = np.zeros(length)
    period = riff[-1][0] + (riff[-1][0] - riff[-2][0]) if len(riff) > 1 else seconds
    events = []
    k = 0
    while True:
        base = k * period
        if base >= seconds:
            break
        events += [(base + t, note, vel) for t, note, vel in riff if base + t < seconds]
        k += 1
    fade = int(0.005 * sample_rate)
    for i, (onset, note, vel) in enumerate(events):
        start = int(onset * sample_rate)
        stop = int(events[i + 1][0] * sample_rate) if i + 1 < len(events) else length
        tone = karplus_strong(
            midi_to_hz(note), (stop - start) / sample_rate, sample_rate, seed=seed * 997 + i
        )
        env = np.ones(stop - start)
        if i + 1 < len(events):
            env[-fade:] = np.linspace(1.0, 0.0, fade)
        out[start:stop] += vel * tone[: stop - start] * env
    out *= peak / (np.abs(out).max() + 1e-12)
    return torch.from_numpy(out.astype(np.float32))


def sine(
    freq_hz: float, seconds: float, sample_rate: int = 44100, amplitude: float = 1.0
) -> Tensor:
    """``amplitude * sin(2 pi f t)``, shape ``(T,)`` float32."""
    t = torch.arange(int(seconds * sample_rate), dtype=torch.float64) / sample_rate
    return (amplitude * torch.sin(2 * math.pi * freq_hz * t)).float()


__all__ = ["DEFAULT_RIFF", "karplus_strong", "midi_to_hz", "pluck_riff", "sine"]
