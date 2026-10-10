import numpy as np
import pytest

from lstmabar.physics.kb import load_kb
from lstmabar.physics.whitebox import (
    WHITEBOX,
    WhiteBoxModel,
    has_whitebox,
    register_whitebox,
    simulate,
)


class _Halve(WhiteBoxModel):
    def __init__(self, sample_rate, seen):
        super().__init__(sample_rate)
        self.seen = seen

    def process_volts(self, v):
        self.seen.append(v.copy())
        return 0.5 * v


def test_simulate_calibrates_and_shapes():
    pedal = load_kb()["ts808"]
    seen = []
    pid = pedal.id
    saved = WHITEBOX.pop(pid, None)
    try:
        register_whitebox(pid)(lambda p, k, sr: _Halve(sr, seen))
        assert has_whitebox(pid)
        x = np.linspace(-1, 1, 8)
        y = simulate(pedal, x, volts_per_fs=2.0)
        assert y.shape == x.shape
        np.testing.assert_allclose(seen[0], 2.0 * x[None])
        np.testing.assert_allclose(y, 0.5 * x)
        assert simulate(pedal, np.zeros((3, 5))).shape == (3, 5)
        with pytest.raises(ValueError):
            register_whitebox(pid)(lambda p, k, sr: _Halve(sr, seen))
    finally:
        WHITEBOX.pop(pid, None)
        if saved is not None:
            WHITEBOX[pid] = saved


def test_simulate_unknown_pedal():
    pedal = load_kb()["ts808"]
    saved = WHITEBOX.pop(pedal.id, None)
    try:
        with pytest.raises(KeyError):
            simulate(pedal, np.zeros(4))
    finally:
        if saved is not None:
            WHITEBOX[pedal.id] = saved
