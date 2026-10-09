"""Audio I/O and small numpy helpers shared by the demo and data code.

All functions work on mono float32 numpy arrays in [-1, 1]. No torch, no gradio.
"""

from math import gcd
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

RiffKind = Literal["single_notes", "power_chords", "clean_arpeggio"]
RIFF_KINDS: tuple[str, ...] = ("single_notes", "power_chords", "clean_arpeggio")

_EPS = 1e-12


def resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling of a 1-D signal (no-op when the rates match)."""
    x = np.asarray(x, dtype=np.float32)
    if orig_sr == target_sr or x.size == 0:
        return x
    g = gcd(int(orig_sr), int(target_sr))
    y = resample_poly(x, int(target_sr) // g, int(orig_sr) // g)
    return y.astype(np.float32)


def _int_to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype.kind == "f":
        return data.astype(np.float32)
    if data.dtype.kind == "u":
        info = np.iinfo(data.dtype)
        mid = (int(info.max) + 1) / 2
        return ((data.astype(np.float64) - mid) / mid).astype(np.float32)
    if data.dtype.kind == "i":
        scale = float(-np.iinfo(data.dtype).min)
        return (data.astype(np.float64) / scale).astype(np.float32)
    if data.dtype.kind == "b":
        return data.astype(np.float32)
    raise TypeError(f"unsupported audio dtype {data.dtype}")


def _downmix(data: np.ndarray) -> np.ndarray:
    """Mono from (T,), (T, C) or (C, T) with a small channel count."""
    if data.ndim == 1:
        return data
    if data.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {data.shape}")
    # Gradio and soundfile give (T, C); tolerate (C, T) when the first axis is clearly channels.
    if data.shape[0] <= 8 and data.shape[1] > data.shape[0]:
        data = data.T
    return data.mean(axis=1)


def to_mono_float(data: np.ndarray, sr: int, sample_rate: int = 44100) -> np.ndarray:
    """Convert raw audio (int or float, mono or multichannel) to mono float32 at ``sample_rate``.

    Matches what ``gradio.Audio(type="numpy")`` hands back: ``(sr, ndarray)`` with int16/int32
    or float samples, shape ``(T,)`` or ``(T, C)``.
    """
    x = _downmix(_int_to_float(np.asarray(data)))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return resample(x, int(sr), int(sample_rate))


def load_audio(
    path: str | Path, sample_rate: int = 44100, max_seconds: float | None = None
) -> np.ndarray:
    """Read an audio file as mono float32 at ``sample_rate``, optionally truncated."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if max_seconds is not None:
        data = data[: int(round(max_seconds * sr))]
    x = to_mono_float(data, sr, sample_rate)
    if max_seconds is not None:
        x = x[: int(round(max_seconds * sample_rate))]
    return x


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def peak_normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Scale so that max |x| == ``peak`` (silence is returned unchanged)."""
    x = np.asarray(x, dtype=np.float32)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < _EPS:
        return x.copy()
    return (x * (peak / m)).astype(np.float32)


def loudness_match(y: np.ndarray, ref: np.ndarray, max_peak: float = 0.99) -> np.ndarray:
    """Scale ``y`` to the RMS of ``ref``; pull it down if its peak would exceed ``max_peak``."""
    y = np.asarray(y, dtype=np.float32)
    ry, rr = rms(y), rms(ref)
    if ry < _EPS or rr < _EPS:
        return y.copy()
    out = y * (rr / ry)
    m = float(np.max(np.abs(out)))
    if m > max_peak:
        out = out * (max_peak / m)
    return out.astype(np.float32)


# --- Synthetic guitar riffs (Karplus-Strong) ---------------------------------------------------

_E2 = 82.4069  # low E string


def _midi_hz(semitones_above_e2: float) -> float:
    return _E2 * 2.0 ** (semitones_above_e2 / 12.0)


def _pluck(
    freq: float,
    n_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
    t60: float = 2.5,
    brightness: float = 0.6,
) -> np.ndarray:
    """One Karplus-Strong string pluck, vectorized with ``lfilter``.

    ``y[n] = x[n] + g/2 * (y[n-N] + y[n-N-1])`` with a lowpassed noise burst of one period as
    excitation; ``g`` is set so the note decays by 60 dB in roughly ``t60`` seconds.
    """
    period = max(2, int(round(sample_rate / freq - 0.5)))
    g = min(0.9999, 10.0 ** (-3.0 / (t60 * freq)))
    burst = rng.uniform(-1.0, 1.0, period)
    # One-pole lowpass on the burst: lower brightness -> softer, rounder attack.
    a = 1.0 - brightness
    burst = lfilter([1.0 - a], [1.0, -a], burst)
    excitation = np.zeros(n_samples)
    excitation[: min(period, n_samples)] = burst[:n_samples]
    den = np.zeros(period + 2)
    den[0] = 1.0
    den[period] = -0.5 * g
    den[period + 1] = -0.5 * g
    y = lfilter([1.0], den, excitation)
    return y


def _place(out: np.ndarray, note: np.ndarray, start: int, stop: int | None, fade: int) -> None:
    """Add ``note`` into ``out`` at ``start``; damp it with a short fade at ``stop`` if given."""
    end = len(out) if stop is None else min(len(out), stop + fade)
    seg = note[: end - start].copy()
    if stop is not None and stop < len(out):
        k = stop - start
        ramp_len = len(seg) - k
        if ramp_len > 0:
            seg[k:] *= np.linspace(1.0, 0.0, ramp_len)
    out[start : start + len(seg)] += seg


def synth_riff(
    kind: RiffKind, seconds: float = 4.0, sample_rate: int = 44100, seed: int = 0
) -> np.ndarray:
    """A short plucked-string guitar riff for demos and tests (mono float32, peak 0.8)."""
    if kind not in RIFF_KINDS:
        raise ValueError(f"unknown riff kind {kind!r}; choose from {RIFF_KINDS}")
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sample_rate))
    out = np.zeros(n)
    fade = int(0.01 * sample_rate)

    def _span(start: int, stop: int | None) -> int:
        # Only synthesize the part of the note that is audible (filter cost ~ length x period).
        end = n if stop is None else min(n, stop + fade)
        return max(end - start, 1)

    def jitter() -> int:
        return int(rng.integers(0, int(0.008 * sample_rate) + 1))

    if kind == "single_notes":
        # E minor pentatonic lick, eighth notes, each note muted by the next.
        lick = [12, 15, 17, 19, 22, 19, 17, 15, 12, 10, 12, 15, 17, 15, 12, 7]
        step = n // len(lick)
        for i, semi in enumerate(lick):
            start = i * step + jitter()
            if start >= n:
                break
            stop = (i + 1) * step if i + 1 < len(lick) else None
            note = _pluck(
                _midi_hz(semi), _span(start, stop), sample_rate, rng, t60=1.5, brightness=0.7
            )
            _place(out, note, start, stop, fade)
    elif kind == "power_chords":
        # E5 - G5 - A5 - C5 shapes (root, fifth, octave), downstrummed, choked at chord change.
        roots = [0, 3, 5, 8]
        hits_per_chord = 2
        step = n // (len(roots) * hits_per_chord)
        for i in range(len(roots) * hits_per_chord):
            root = roots[i // hits_per_chord]
            start = i * step
            stop = (i + 1) * step if i + 1 < len(roots) * hits_per_chord else None
            for j, interval in enumerate((0, 7, 12)):
                s = start + j * int(0.006 * sample_rate) + jitter()
                if s >= n:
                    continue
                note = _pluck(
                    _midi_hz(root + interval),
                    _span(s, stop),
                    sample_rate,
                    rng,
                    t60=2.0,
                    brightness=0.8,
                )
                _place(out, 0.6 * note, s, stop, fade)
    else:  # clean_arpeggio
        # Em - C - G - D arpeggios, notes left ringing.
        chords = [
            [0, 7, 12, 15, 19, 24],
            [8, 15, 20, 24, 27, 32],
            [3, 10, 15, 19, 22, 27],
            [10, 17, 22, 26, 29, 34],
        ]
        pattern = [0, 2, 3, 4, 5, 4, 3, 2]
        total = len(chords) * len(pattern)
        step = n // total
        for i in range(total):
            chord = chords[i // len(pattern)]
            semi = chord[pattern[i % len(pattern)]]
            start = i * step + jitter()
            if start >= n:
                break
            # Let notes ring until the chord changes.
            chord_end = (i // len(pattern) + 1) * len(pattern) * step
            stop = chord_end if chord_end < n else None
            note = _pluck(
                _midi_hz(semi), _span(start, stop), sample_rate, rng, t60=3.0, brightness=0.45
            )
            _place(out, 0.5 * note, start, stop, fade)

    return peak_normalize(out.astype(np.float32), 0.8)
