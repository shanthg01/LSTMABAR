"""Band-limited additive oscillators for the analysis tests."""

import numpy as np

SR = 44100


def oscillator(
    kind: str,
    f0: float,
    seconds: float = 0.5,
    amplitude: float = 0.5,
    sr: int = SR,
    max_harmonic: int | None = None,
) -> np.ndarray:
    """Additive ``sine``/``triangle``/``square``/``saw`` with every harmonic below Nyquist.

    ``amplitude`` is the H1 peak amplitude. Triangle uses alternating signs (the real
    waveform); the analysis ignores phase anyway.
    """
    t = np.arange(int(seconds * sr)) / sr
    top = int((0.5 * sr - 1) // f0) if max_harmonic is None else max_harmonic
    x = np.zeros_like(t)
    for k in range(1, top + 1):
        if kind == "sine":
            a = 1.0 if k == 1 else 0.0
        elif kind == "triangle":
            a = (-1) ** ((k - 1) // 2) / k**2 if k % 2 else 0.0
        elif kind == "square":
            a = 1.0 / k if k % 2 else 0.0
        elif kind == "saw":
            a = (-1) ** (k + 1) / k
        else:
            raise ValueError(kind)
        if a:
            x += a * np.sin(2 * np.pi * k * f0 * t)
    return amplitude * x


def harmonic_series(
    amps, f0: float, seconds: float = 0.5, sr: int = SR, seed: int = 0, dc: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of cosines with the given peak amplitudes and random phases → ``(x, phases)``."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    phases = rng.uniform(0, 2 * np.pi, len(amps))
    x = np.full_like(t, dc)
    for k, (a, ph) in enumerate(zip(amps, phases, strict=True), start=1):
        x += a * np.cos(2 * np.pi * k * f0 * t + ph)
    return x, phases
