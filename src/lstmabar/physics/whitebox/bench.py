"""White-box throughput benchmark (sizes P4 rendering).

Real-time factor = seconds of 44.1 kHz audio processed per CPU-second, aggregated over the
batch, for the TS808 model (feedback clipper) at drive max and a 1 V-peak guitar-like
input. Single-threaded: torch is pinned to one thread for the resampler.

    uv run python -m lstmabar.physics.whitebox.bench [--backend numpy] [--seconds 2]
"""

import argparse
import time

import numpy as np
import torch

from lstmabar.physics.kb import load_kb
from lstmabar.physics.whitebox import WHITEBOX
from lstmabar.physics.whitebox.diode_clipper import numba_available


def _test_signal(batch: int, seconds: float, sr: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = rng.uniform(82, 660, size=(batch, 1))
    env = np.exp(-3.0 * (t % 0.5))  # re-plucked every 0.5 s
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in (1, 2, 3)) * env
    return x / np.abs(x).max(axis=-1, keepdims=True)


def benchmark(
    batch_sizes=(1, 16, 64),
    seconds: float = 1.0,
    oversample: int = 4,
    backend: str = "auto",
    sample_rate: int = 44100,
) -> list[dict]:
    """Return one row per batch size: wall seconds and aggregate real-time factor."""
    torch.set_num_threads(1)
    pedal = load_kb()["ts808"]
    model = WHITEBOX["ts808"](pedal, pedal.knobs({"drive": 1.0}), sample_rate)
    model.oversample, model.backend = oversample, backend
    model.process_volts(_test_signal(1, 0.01, sample_rate))  # warm-up (numba compile)
    rows = []
    for b in batch_sizes:
        x = _test_signal(b, seconds, sample_rate)
        t0 = time.process_time()
        y = model.process_volts(x)
        dt = time.process_time() - t0
        assert np.isfinite(y).all()
        rows.append({"batch": b, "seconds": dt, "rtf": b * seconds / max(dt, 1e-9)})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="auto", choices=("auto", "numba", "numpy"))
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--oversample", type=int, default=4)
    args = ap.parse_args(argv)
    backend = args.backend
    if backend == "auto":
        backend = "numba" if numba_available() else "numpy"
    print(f"backend={backend} oversample={args.oversample} seconds={args.seconds}")
    for r in benchmark(seconds=args.seconds, oversample=args.oversample, backend=backend):
        print(f"batch {r['batch']:3d}: {r['seconds']:.3f} CPU-s, real-time factor {r['rtf']:.3g}x")


if __name__ == "__main__":
    main()
