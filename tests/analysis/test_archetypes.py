import numpy as np
import pytest
import torch

from lstmabar.analysis.archetypes import (
    ARCHETYPES,
    archetype_readout,
    noise_fraction,
    project_archetypes,
    reference_spectra,
)
from lstmabar.analysis.descriptors import harmonic_energy_fraction, odd_even_ratio
from lstmabar.analysis.harmonics import harmonic_amplitudes, hnr
from lstmabar.dsp.blocks import Drive
from tests.analysis.signals import SR, oscillator


@pytest.mark.parametrize("f0", [82.41, 220.0, 659.26])
@pytest.mark.parametrize("kind", ARCHETYPES)
def test_pure_oscillators_read_as_themselves(kind, f0):
    r = archetype_readout(oscillator(kind, f0, 0.5, 0.4), SR, f0=f0, whole_clip=True)
    assert r.weights[kind] > 0.9, r.weights
    assert r.dominant == kind
    assert r.noise < 1e-6 and r.residual < 1e-3
    assert sum(r.weights.values()) == pytest.approx(1.0)


def test_whole_clip_refines_nominal_f0():
    x = oscillator("square", 196.0, 0.5, 0.4)
    nominal = 196.0 * 2 ** (-6 / 1200)
    assert archetype_readout(x, SR, f0=nominal, whole_clip=True).weights["square"] > 0.9
    off = archetype_readout(x, SR, f0=nominal, whole_clip=True, refine=False)
    assert off.weights["square"] < 0.9 or off.noise > 0.1


def test_frame_profile_path_matches():
    r = archetype_readout(oscillator("triangle", 110.0, 1.0, 0.4), SR, f0=110.0)
    assert r.weights["triangle"] > 0.9


def test_level_invariance_and_nan_handling():
    refs = reference_spectra(10)
    a = project_archetypes(refs[2] * 0.01)
    b = project_archetypes(refs[2] * 3.0)
    assert a.weights == pytest.approx(b.weights)
    # Harmonics above Nyquist (NaN) are dropped from profile and references alike.
    sq = refs[2].copy()
    sq[6:] = np.nan
    assert project_archetypes(sq).weights["square"] > 0.99
    with pytest.raises(ValueError):
        project_archetypes(np.array([0.0, 0.5]))


def test_noise_fraction_and_vector():
    assert noise_fraction(0.0) == pytest.approx(0.5)
    assert noise_fraction(None) == 0.0 and noise_fraction(float("inf")) == 0.0
    assert noise_fraction(20.0) == pytest.approx(1 / 101)
    x = oscillator("saw", 196.0, 0.5, 0.3)
    noisy = x + np.random.default_rng(0).standard_normal(len(x)) * np.sqrt(np.mean(x**2))
    r = archetype_readout(noisy, SR, f0=196.0, whole_clip=True)
    assert r.noise == pytest.approx(0.5, abs=0.03)
    v = r.vector()
    assert v.shape == (5,) and v.sum() == pytest.approx(1.0) and (v >= 0).all()


# ----------------------------------------------------------------- through the Drive block

DRIVE_F0 = 220.0


def _knobs(n: int, **physical) -> dict[str, torch.Tensor]:
    out = {}
    for spec in Drive.param_specs:
        v = physical.get(spec.name, spec.default)
        v = torch.as_tensor(v, dtype=torch.float64).expand(n)
        out[spec.name] = spec.normalize(v).float()
    return out


def _drive(**sweep) -> np.ndarray:
    """Render a 220 Hz sine through Drive, one row per value of the single swept knob."""
    ((name, values),) = sweep.items()
    n = len(values)
    x = torch.from_numpy(oscillator("sine", DRIVE_F0, 0.5, 0.5)).float().expand(n, -1)
    params = _knobs(n, pre_hpf_hz=20.0, **{name: torch.tensor(values, dtype=torch.float64)})
    with torch.no_grad():
        y = Drive(SR)(x, params).numpy()
    edge = int(0.05 * SR)  # crop the HPF settling and oversampler edges
    return y[:, edge:-edge]


def test_drive_gain_moves_sine_toward_square():
    """Symmetric clipping walks sine -> triangle-like -> square as gain rises.

    The sine weight falls strictly until it reaches 0 (around +10 dB at this level, where the
    profile reads as mostly triangle), then stays 0; the square weight never falls.
    """
    gains = [0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0, 48.0]
    y = _drive(gain_db=gains)
    amps = harmonic_amplitudes(y, SR, DRIVE_F0, n_harmonics=10)
    reads = [
        project_archetypes(a, float(hnr(row, SR, DRIVE_F0))) for a, row in zip(amps, y, strict=True)
    ]
    sine_w = np.array([r.weights["sine"] for r in reads])
    square_w = np.array([r.weights["square"] for r in reads])
    content = harmonic_energy_fraction(amps)
    d_sine = np.diff(sine_w)
    assert (d_sine <= 1e-9).all(), sine_w
    assert (d_sine[sine_w[:-1] > 0] < 0).all(), sine_w  # strictly, while still positive
    assert (np.diff(square_w) >= -1e-9).all(), square_w
    assert (np.diff(content) > 0).all(), content
    assert sine_w[0] > 0.9 and square_w[-1] > 0.95
    assert all(r.noise < 0.01 for r in reads)  # clipping adds harmonics, not noise


def test_drive_asymmetry_adds_even_harmonics():
    y = _drive(asymmetry=[0.0, 0.3, 0.6, 1.0])
    amps = harmonic_amplitudes(y, SR, DRIVE_F0, n_harmonics=10)
    even = (amps[:, 1::2] ** 2).sum(-1) / amps[:, 0] ** 2
    assert even[0] < 1e-6  # symmetric clipper: odd harmonics only
    assert (np.diff(even) > 0).all(), even
    assert (np.diff(odd_even_ratio(amps)) < 0).all()
    saw_w = [project_archetypes(a).weights["saw"] for a in amps]
    assert saw_w[-1] > saw_w[0]
