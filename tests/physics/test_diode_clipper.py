"""Validation of the diode-clipper white-box solver (P2 decision 5)."""

import math

import numpy as np
import pytest

from lstmabar.physics.kb import load_kb
from lstmabar.physics.whitebox import (
    WHITEBOX,
    feedback_clipper,
    has_whitebox,
    shunt_clipper,
    simulate,
)
from lstmabar.physics.whitebox.devices import DIODES, VT, diode
from lstmabar.physics.whitebox.diode_clipper import numba_available, solve
from lstmabar.physics.whitebox.linear import rc_highpass, rc_lowpass

SR = 44100
F0 = 441.0  # period = 100 samples (400 at 4x): exact FFT bins and exact half-wave symmetry
N_FFT = 4400  # 44 periods
# TS808 clipping-stage values (pedals/ts808.yaml), drive at mid (A taper: 50k of 500k).
TS = dict(R_gain=4.7e3, C_gain=47e-9, R_fb=51e3 + 50e3, C_fb=51e-12, part="1S2473")


def sine(amp, f=F0, seconds=0.15, batch=None):
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * f * t)
    amp = np.atleast_1d(np.asarray(amp, float))
    return amp[:, None] * x[None] if batch is None else np.repeat(amp[:, None] * x, batch, 0)


def tone_amplitude(y, f, start=0.05, stop_margin=0.01):
    """Least-squares amplitude of the ``f`` component over the steady-state part."""
    i0, i1 = int(start * SR), y.shape[-1] - int(stop_margin * SR)
    t = np.arange(i0, i1) / SR
    basis = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)], 1)
    coef, *_ = np.linalg.lstsq(basis, y[..., i0:i1].T, rcond=None)
    return np.hypot(coef[0], coef[1])


def harmonics_dbc(y, n=10):
    """H1..Hn levels: H1 in dB re 1 V, H2.. in dBc, from an exact-bin window before the end."""
    seg = y[..., -N_FFT - 400 : -400]
    spec = np.abs(np.fft.rfft(seg, axis=-1)) * 2 / N_FFT
    k0 = int(round(F0 * N_FFT / SR))
    h = spec[..., [k0 * k for k in range(1, n + 1)]]
    out = 20 * np.log10(np.maximum(h, 1e-300))
    out[..., 1:] -= out[..., :1]
    return out


# --- devices and linear stages -----------------------------------------------------------------


def test_device_table():
    assert VT == pytest.approx(0.025693, abs=1e-5)
    assert diode("1N4148").forward_voltage(1e-3) == pytest.approx(0.58, abs=0.02)
    assert diode("1n914") is DIODES["1N914"]
    assert diode("1N34A").forward_voltage(1e-3) == pytest.approx(0.31, abs=0.02)
    assert diode("LED_RED").forward_voltage(1e-3) == pytest.approx(1.68, abs=0.03)
    assert diode("LED_RED").forward_voltage(1e-2) == pytest.approx(1.80, abs=0.03)
    assert diode("", device="ge") is DIODES["1N34A"]
    with pytest.raises(KeyError):
        diode("XYZ", device="mosfet")


def test_rc_prewarp_exact_at_corner():
    from scipy.signal import freqz

    for flt in (rc_lowpass(1e3, 0.22e-6), rc_highpass(4.7e3, 47e-9)):
        bz, az = flt.digital(SR)
        _, h = freqz(bz, az, worN=[flt.prewarp_hz], fs=SR)
        assert 20 * np.log10(abs(h[0])) == pytest.approx(-10 * np.log10(2), abs=1e-9)


# --- solver validation ------------------------------------------------------------------------

SMALL_FREQS = (100.0, 300.0, 1000.0, 3000.0, 6000.0)


@pytest.mark.parametrize(
    "model, amp",
    [
        (shunt_clipper(R=2.2e3, C=10e-9), 1e-3),
        (feedback_clipper(**TS), 1e-4),
        (feedback_clipper(**{**TS, "R_fb": 551e3}), 1e-5),
    ],
    ids=["shunt", "feedback", "feedback_max_drive"],
)
def test_small_signal_matches_analytic(model, amp):
    """Tiny input: diodes stay in their linear region; the response must equal the analytic
    s-domain response (diodes linearized at 0 V, i.e. their ~tens-of-MΩ conductance)."""
    x = np.concatenate([sine(amp, f, seconds=0.1) for f in SMALL_FREQS])
    y = model.process_volts(x)
    for i, f in enumerate(SMALL_FREQS):
        got = 20 * np.log10(tone_amplitude(y[i], f) / amp)
        want = 20 * np.log10(abs(model.response(f)))
        assert got == pytest.approx(want, abs=0.1), f"{f} Hz: {got:.3f} vs {want:.3f} dB"


def test_shunt_bounded_by_forward_voltage():
    for part, n, amp in (("1N914", 1, 20.0), ("1N914", 2, 20.0), ("LED_RED", 1, 20.0)):
        r = 2.2e3
        y = shunt_clipper(R=r, C=10e-9, part=part, n_pos=n, n_neg=n).process_volts(sine(amp))
        peak = np.abs(y[0, 2000:-2000]).max()
        v_f = diode(part).forward_voltage(amp / r, n)  # drop at the peak source current
        assert 0.9 * v_f < peak < 1.05 * v_f, (part, n, peak, v_f)


def test_feedback_clean_path_plus_bounded_term():
    x = sine([0.1, 1.0])
    y = feedback_clipper(**{**TS, "R_fb": 551e3}).process_volts(x)
    d = np.abs(y - x)[:, 2000:-2000].max(axis=1)
    i_max = 1.0 / TS["R_gain"]  # largest gain-leg current at 1 V in
    v_f = diode("1S2473").forward_voltage(i_max)
    assert np.all(d < 1.05 * v_f)
    assert d[1] > 0.8 * v_f  # far into clipping: linear gain would be ~100x
    assert d[0] > 0.6 * v_f


@pytest.mark.parametrize("config", ["shunt", "feedback"])
def test_symmetry_even_harmonics(config):
    if config == "shunt":
        sym = shunt_clipper(R=2.2e3, C=10e-9)
        asym = shunt_clipper(R=2.2e3, C=10e-9, n_pos=1, n_neg=2)
        amp = 3.0
    else:
        sym = feedback_clipper(**TS)
        asym = feedback_clipper(**TS, n_pos=1, n_neg=2)
        amp = 0.3
    h_sym = harmonics_dbc(sym.process_volts(sine(amp)))[0]
    h_asym = harmonics_dbc(asym.process_volts(sine(amp)))[0]
    assert h_sym[2] > -40  # H3: clearly clipping
    assert np.all(h_sym[1::2] < -80), h_sym  # H2, H4, ...
    assert h_asym[1] > -40, h_asym


def test_oversampling_convergence():
    """4x vs 8x: harmonics above -60 dBc agree within 0.05 dB and the waveforms within
    -60 dB (relative RMS), at full drive and 0.5 V in."""
    kw = {**TS, "R_fb": 551e3}
    x = sine([0.05, 0.5])
    y4 = feedback_clipper(**kw, oversample=4).process_volts(x)
    y8 = feedback_clipper(**kw, oversample=8).process_volts(x)
    seg = slice(2000, -2000)
    err = np.sqrt(np.mean((y8 - y4)[:, seg] ** 2, -1) / np.mean(y4[:, seg] ** 2, -1))
    assert np.all(20 * np.log10(err) < -60), 20 * np.log10(err)
    h4, h8 = harmonics_dbc(y4), harmonics_dbc(y8)
    keep = h4 > -60
    keep[:, 0] = True
    assert np.all(np.abs(h4 - h8)[keep] < 0.05), np.abs(h4 - h8)


def test_newton_handles_steep_steps():
    """A ±50 V step into the shunt clipper: Newton with pnjlim must converge (no overflow)."""
    x = np.zeros((1, 2000))
    x[0, 500:1000], x[0, 1000:1500] = 50.0, -50.0
    y = shunt_clipper(R=1e3, C=10e-9).process_volts(x)
    assert np.isfinite(y).all() and np.abs(y).max() < 1.2


@pytest.mark.skipif(not numba_available(), reason="numba not installed")
def test_numba_matches_numpy():
    rng = np.random.default_rng(0)
    j = rng.standard_normal((3, 400)) * 1e-3
    pair = feedback_clipper(**TS).circuit.diodes
    a = solve(j, 51e-12, 1 / 101e3, pair, 4 * SR, backend="numba")
    b = solve(j, 51e-12, 1 / 101e3, pair, 4 * SR, backend="numpy")
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


# --- TS808 via the registry ---------------------------------------------------------------------


def _thd_db(y):
    h = harmonics_dbc(y)
    return 10 * np.log10(np.sum(10 ** (h[..., 1:] / 10), -1))


def test_ts808_registered_and_drive_increases_harmonics():
    pedal = load_kb()["ts808"]
    assert has_whitebox("ts808") and "ts808" in WHITEBOX
    x = sine(0.1)
    y = simulate(pedal, x[0])
    assert y.shape == x[0].shape and np.isfinite(y).all()
    thd = [_thd_db(simulate(pedal, x, knobs={"drive": d}))[0] for d in (0.0, 0.5, 1.0)]
    assert thd[0] < thd[1] < thd[2], thd
    # level pot is a divider: halving the fraction halves the output
    lvl = pedal.pots["level"]
    y_a = simulate(pedal, x, knobs={"level": 1.0})
    y_b = simulate(pedal, x, knobs={"level": 0.5})
    np.testing.assert_allclose(y_b, y_a * lvl.fraction(0.5), rtol=1e-12, atol=1e-15)
    assert math.isfinite(_thd_db(y_a)[0])


def test_bench_smoke():
    from lstmabar.physics.whitebox.bench import benchmark

    rows = benchmark(batch_sizes=(2,), seconds=0.05)
    assert rows[0]["batch"] == 2 and rows[0]["rtf"] > 0
