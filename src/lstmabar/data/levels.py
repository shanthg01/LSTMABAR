"""Level policy (P4 resolution R3) and FLAC storage helpers.

- Dry clips are stored peak-normalized to ``ingest.dry_peak_dbfs`` (−1 dBFS).
- Each render samples ``input_peak_dbfs ~ U(lo, hi)`` and scales the stored dry clip to that
  peak before the pedal (0 dBFS = 1 V peak, P2 calibration).
- Wet renders are stored peak-normalized to ``render.wet_peak_dbfs`` as 24-bit FLAC with the
  gain needed to restore them (``wet_gain_db``): FLAC is integer PCM and cannot hold values
  above full scale.
- At inference (demo/eval) inputs are peak-normalized to ``audio.inference_peak_dbfs``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

SILENCE_DBFS = -200.0


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def peak_dbfs(x: np.ndarray) -> float:
    """Sample peak in dBFS (``SILENCE_DBFS`` for an all-zero signal)."""
    p = float(np.max(np.abs(x))) if np.size(x) else 0.0
    return 20.0 * float(np.log10(p)) if p > 0 else SILENCE_DBFS


def normalize_peak(x: np.ndarray, target_dbfs: float) -> tuple[np.ndarray, float]:
    """Scale ``x`` to the target peak. Returns ``(y, gain_db)`` with ``y = x * 10**(gain_db/20)``.

    An all-zero signal is returned unchanged with ``gain_db = 0``.
    """
    p = peak_dbfs(x)
    if p <= SILENCE_DBFS:
        return np.asarray(x, dtype=np.float64).copy(), 0.0
    gain_db = target_dbfs - p
    return np.asarray(x, dtype=np.float64) * db_to_gain(gain_db), gain_db


def sample_input_peak(rng: np.random.Generator, lo: float, hi: float) -> float:
    if hi < lo:
        raise ValueError(f"input peak range [{lo}, {hi}] is empty")
    return float(rng.uniform(lo, hi))


def apply_input_level(dry: np.ndarray, input_peak_dbfs: float) -> np.ndarray:
    """Scale a stored dry clip (any peak) so its peak is exactly ``input_peak_dbfs``."""
    y, _ = normalize_peak(dry, input_peak_dbfs)
    return y


def inference_normalize(x: np.ndarray, target_dbfs: float = -15.0) -> np.ndarray:
    """The inference-time level rule (R3): same peak range as training, centred."""
    y, _ = normalize_peak(x, target_dbfs)
    return y


def write_flac(path: str | Path, x: np.ndarray, sample_rate: int, subtype: str) -> None:
    """Write mono audio; raises if any sample is outside [-1, 1] (would clip silently)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected mono (T,) audio, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("audio contains NaN/inf")
    if np.max(np.abs(x), initial=0.0) > 1.0:
        raise ValueError("audio exceeds full scale; normalize before writing")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sample_rate, format="FLAC", subtype=subtype)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float64 audio and its sample rate."""
    x, sr = sf.read(str(path), dtype="float64", always_2d=False)
    if x.ndim != 1:
        x = x.mean(axis=1)
    return x, int(sr)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


__all__ = [
    "apply_input_level",
    "db_to_gain",
    "inference_normalize",
    "normalize_peak",
    "peak_dbfs",
    "read_audio",
    "sample_input_peak",
    "sha256_file",
    "write_flac",
]
