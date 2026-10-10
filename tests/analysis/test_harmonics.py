import numpy as np
import pytest
import torch

from lstmabar.analysis.harmonics import (
    harmonic_amplitudes,
    harmonic_fit,
    harmonic_profile,
    hnr,
    refine_f0,
    to_db,
    to_dbc,
)
from tests.analysis.signals import SR, harmonic_series, oscillator

# The P2 fidelity sweep pitches (decision 6) plus the edges of the 80-700 Hz range.
PITCHES = [80.0, 82.41, 164.81, 329.63, 659.26, 700.0]
AMPS = np.array([0.7, 0.08, 0.2, 0.01, 0.05, 0.003, 0.02, 0.0, 0.01, 0.002])


@pytest.mark.parametrize("seconds", [0.25, 0.5])
@pytest.mark.parametrize("f0", PITCHES)
def test_known_f0_accuracy(f0, seconds):
    """Every harmonic down to -50 dBc within 0.1 dB, with DC, noise and unmodelled harmonics."""
    x, phases = harmonic_series(AMPS, f0, seconds, dc=0.05, seed=int(f0))
    t = np.arange(len(x)) / SR
    for k in range(11, 60):  # content above n_harmonics must not leak into the fit
        if k * f0 < SR / 2:
            x += 0.01 / k * np.cos(2 * np.pi * k * f0 * t + k)
    x += 1e-4 * np.random.default_rng(1).standard_normal(len(x))  # -80 dB noise floor
    amps, ph = harmonic_fit(x, SR, f0, n_harmonics=10)
    present = AMPS > 0
    err_db = np.abs(to_db(amps[present]) - to_db(AMPS[present]))
    assert err_db.max() < 0.1, err_db
    assert amps[~present][0] < 1e-3  # the absent H8 reads as ~noise floor
    dphi = np.angle(np.exp(1j * (ph[:3] - phases[:3])))
    assert np.abs(dphi).max() < 1e-3


def test_full_scale_sine_is_0db_and_batched_torch():
    x = torch.stack([torch.sin(2 * torch.pi * f * torch.arange(SR // 2) / SR) for f in (110, 440)])
    amps = harmonic_amplitudes(x.float(), SR, 110.0, n_harmonics=4)  # row 2 is H4 of 110
    assert amps.shape == (2, 4)
    assert abs(to_db(amps[0, 0])) < 0.01
    assert to_dbc(amps[0])[1:].max() <= -60
    assert abs(to_db(amps[1, 3])) < 0.01


def test_above_nyquist_is_nan_and_floored():
    x = oscillator("square", 3000.0, 0.1, 1.0)
    amps = harmonic_amplitudes(x, SR, 3000.0, n_harmonics=10)
    assert np.isnan(amps[7:]).all() and np.isfinite(amps[:7]).all()  # 8 * 3 kHz > 22.05 kHz
    dbc = to_dbc(amps, floor_db=-60)
    assert (dbc[7:] == -60).all()
    assert abs(dbc[2] - 20 * np.log10(1 / 3)) < 0.05


def test_to_dbc_floor_and_bad_h1():
    dbc = to_dbc(np.array([1.0, 0.1, 0.0, 1e-9]), floor_db=-60)
    np.testing.assert_allclose(dbc, [0.0, -20.0, -60.0, -60.0])
    assert np.isnan(to_dbc(np.array([0.0, 0.1]))).all()


def test_too_short_and_bad_f0():
    with pytest.raises(ValueError):
        harmonic_amplitudes(np.zeros(100), SR, 82.0)
    with pytest.raises(ValueError):
        harmonic_amplitudes(np.zeros(SR), SR, 0.0)


def test_hnr_tracks_added_noise():
    x = oscillator("saw", 196.0, 0.5, 0.5)
    sig_power = np.mean(x**2)
    rng = np.random.default_rng(0)
    for target in (40.0, 20.0, 0.0):
        noise = rng.standard_normal(len(x)) * np.sqrt(sig_power / 10 ** (target / 10))
        assert abs(float(hnr(x + noise, SR, 196.0)) - target) < 1.0
    assert hnr(x, SR, 196.0) > 80  # clean, band-limited: near the window's dynamic range


def test_profile_known_f0_without_librosa():
    x = oscillator("square", 110.0, 1.0, 0.3)
    prof = harmonic_profile(x, SR, n_harmonics=10, f0=110.0)
    expected = np.where(np.arange(1, 11) % 2 == 1, 1 / np.arange(1, 11), 0.0)
    np.testing.assert_allclose(prof.amplitudes, expected, atol=1e-3)
    assert abs(prof.h1 - 0.3) < 1e-3 and prof.f0_hz == 110.0


def test_profile_pyin():
    pytest.importorskip("librosa")
    # Two notes, plus silence the tracker must skip.
    a = oscillator("saw", 110.0, 0.6, 0.3)
    b = oscillator("saw", 164.81, 0.6, 0.3)
    x = np.concatenate([np.zeros(SR // 5), a, b, np.zeros(SR // 5)])
    prof = harmonic_profile(x, SR, n_harmonics=8)
    np.testing.assert_allclose(prof.amplitudes, 1 / np.arange(1, 9), rtol=0.01)
    assert 0.5 < prof.voiced_fraction < 1.0
    # Refined f0 lands on the true pitches (pYIN alone is on a 10-cent grid).
    cents = 1200 * np.log2(prof.frame_f0[:, None] / np.array([110.0, 164.81]))
    assert (np.abs(cents).min(axis=1) < 1.0).mean() > 0.9
    assert prof.hnr_db > 60


def test_refine_f0():
    x = oscillator("square", 146.83, 0.1, 0.5)
    coarse = 146.83 * 2 ** (8 / 1200)
    assert abs(refine_f0(x, SR, coarse) - 146.83) < 0.01
