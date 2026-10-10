import sys

import numpy as np
import pytest

from lstmabar.analysis.archetypes import reference_spectra
from lstmabar.analysis.descriptors import (
    crest_factor,
    crest_factor_db,
    describe,
    harmonic_energy_fraction,
    harmonic_slope,
    odd_even_ratio,
    rms,
    rms_db,
    spectral_centroid,
    spectral_rolloff,
)
from lstmabar.analysis.harmonics import harmonic_amplitudes
from tests.analysis.signals import SR, oscillator

F0 = 220.0


def measured(kind: str) -> np.ndarray:
    return harmonic_amplitudes(oscillator(kind, F0), SR, F0, n_harmonics=10)


def test_odd_even_ratio():
    square, tri, saw = measured("square"), measured("triangle"), measured("saw")
    assert odd_even_ratio(square) > 60 and odd_even_ratio(tri) > 60  # no even content
    assert odd_even_ratio(saw) == pytest.approx(-3.0, abs=0.05)  # ideal saw over H2..H10
    assert odd_even_ratio(square) - odd_even_ratio(saw) > 60


def test_harmonic_slope():
    assert harmonic_slope(measured("square")) == pytest.approx(-6.02, abs=0.05)
    assert harmonic_slope(measured("saw")) == pytest.approx(-6.02, abs=0.05)
    assert harmonic_slope(measured("triangle")) == pytest.approx(-12.04, abs=0.05)
    assert np.isnan(harmonic_slope(measured("sine")))
    batch = harmonic_slope(reference_spectra(10)[1:])
    np.testing.assert_allclose(batch, [-12.04, -6.02, -6.02], atol=0.01)


def test_harmonic_energy_fraction():
    assert harmonic_energy_fraction(measured("sine")) < 1e-8
    assert harmonic_energy_fraction(measured("square")) == pytest.approx(0.155, abs=0.002)
    assert harmonic_energy_fraction(measured("saw")) == pytest.approx(0.355, abs=0.002)


def test_level_descriptors():
    sine = oscillator("sine", F0, 0.5, 1.0)
    square = oscillator("square", F0, 0.5, 1.0)
    assert rms(sine) == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert rms_db(sine) == pytest.approx(-3.01, abs=0.01)
    assert crest_factor(sine) == pytest.approx(np.sqrt(2), rel=1e-3)
    assert crest_factor_db(sine) == pytest.approx(3.01, abs=0.02)
    assert crest_factor(square) < crest_factor(sine)  # band-limited square: Gibbs ripple only
    assert rms(np.zeros((3, 100))).shape == (3,)


def test_spectral_descriptors():
    sine = oscillator("sine", 1000.0, 0.5, 0.5)
    assert spectral_centroid(sine, SR) == pytest.approx(1000.0, rel=0.05)
    assert spectral_rolloff(sine, SR) == pytest.approx(1000.0, rel=0.02)
    tri, square, saw = (oscillator(k, F0) for k in ("triangle", "square", "saw"))
    assert spectral_centroid(tri, SR) < spectral_centroid(square, SR) < spectral_centroid(saw, SR)
    assert spectral_rolloff(tri, SR) < spectral_rolloff(saw, SR)


def test_describe_known_f0():
    d = describe(oscillator("square", F0, 1.0, 0.5), SR, f0=F0)
    assert d["odd_even_db"] > 60
    assert d["slope_db_per_oct"] == pytest.approx(-6.02, abs=0.1)
    assert d["hnr_db"] > 60
    assert {"rms_db", "crest_factor_db", "centroid_hz", "rolloff_hz", "f0_hz"} <= d.keys()


HARMONIC_KEYS = {"f0_hz", "hnr_db", "odd_even_db", "slope_db_per_oct", "harmonic_energy_fraction"}


def test_describe_without_librosa_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "librosa", None)  # import librosa -> ImportError
    d = describe(oscillator("saw", F0, 0.5, 0.5), SR)
    assert HARMONIC_KEYS <= d.keys()
    assert all(np.isnan(d[k]) for k in HARMONIC_KEYS)
    assert np.isfinite(d["centroid_hz"]) and np.isfinite(d["rms_db"])


def test_silence():
    z = np.zeros(SR // 4)
    assert crest_factor(z) == 0.0 and crest_factor_db(z) == -np.inf
    assert rms_db(z) < -150
    d = describe(z, SR, f0=F0)
    assert all(np.isnan(d[k]) for k in HARMONIC_KEYS)  # all frames below min_rms_db
