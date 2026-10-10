"""Validation of the diode-clipper white-box solver (P2 decision 5)."""

import dataclasses
import itertools
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
from lstmabar.physics.whitebox.devices import DIODES, SUBSTITUTES, VT, diode
from lstmabar.physics.whitebox.diode_clipper import (
    numba_available,
    resample_down,
    solve,
)
from lstmabar.physics.whitebox.linear import rc_highpass, rc_lowpass

SR = 44100
F0 = 441.0  # period = 100 samples (400 at 4x): exact FFT bins and exact half-wave symmetry
N_FFT = 4400  # 44 periods
# TS808 clipping-stage values (pedals/ts808.yaml), drive at mid (A taper: 50k of 500k).
TS = dict(R_gain=4.7e3, C_gain=47e-9, R_fb=51e3 + 50e3, C_fb=51e-12, part="1S2473")
TS_MAX = {**TS, "R_fb": 551e3}

# The NumPy fallback runs at ~0.1x real time, so the heavier validation tests need numba
# (`uv sync --extra whitebox`, installed in CI); without it they are skipped and the
# robustness sweep is shortened.
NUMBA = numba_available()
needs_numba = pytest.mark.skipif(not NUMBA, reason="numba (whitebox extra) not installed")


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
    sub = diode("1s2473")
    assert sub.part == "1S2473" and "1S2473" in SUBSTITUTES and "1S2473" not in DIODES
    assert (sub.Is, sub.N) == (DIODES["1N4148"].Is, DIODES["1N4148"].N)
    for part, device in (("1N4001", "si"), ("XYZ", "mosfet"), ("", "mosfet")):
        with pytest.raises(KeyError):
            diode(part, device)


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
        (feedback_clipper(**TS_MAX), 1e-5),
    ],
    ids=["shunt", "feedback", "feedback_max_drive"],
)
@needs_numba
def test_small_signal_matches_analytic(model, amp):
    """Tiny input: diodes stay in their linear region; the response must equal the analytic
    s-domain response (diodes linearized at 0 V, i.e. their ~tens-of-MΩ conductance)."""
    x = np.concatenate([sine(amp, f, seconds=0.1) for f in SMALL_FREQS])
    y = model.process_volts(x)
    for i, f in enumerate(SMALL_FREQS):
        got = 20 * np.log10(tone_amplitude(y[i], f) / amp)
        want = 20 * np.log10(abs(model.response(f)))
        assert got == pytest.approx(want, abs=0.1), f"{f} Hz: {got:.3f} vs {want:.3f} dB"


@needs_numba
def test_shunt_bounded_by_forward_voltage():
    for part, n, amp in (("1N914", 1, 20.0), ("1N914", 2, 20.0), ("LED_RED", 1, 20.0)):
        r = 2.2e3
        y = shunt_clipper(R=r, C=10e-9, part=part, n_pos=n, n_neg=n).process_volts(sine(amp))
        peak = np.abs(y[0, 2000:-2000]).max()
        v_f = diode(part).forward_voltage(amp / r, n)  # drop at the peak source current
        assert 0.9 * v_f < peak < 1.05 * v_f, (part, n, peak, v_f)


@needs_numba
def test_feedback_clean_path_plus_bounded_term():
    x = sine([0.1, 1.0])
    y = feedback_clipper(**TS_MAX).process_volts(x)
    d = np.abs(y - x)[:, 2000:-2000].max(axis=1)
    i_max = 1.0 / TS["R_gain"]  # largest gain-leg current at 1 V in
    v_f = diode("1S2473").forward_voltage(i_max)
    assert np.all(d < 1.05 * v_f)
    assert d[1] > 0.8 * v_f  # far into clipping: linear gain would be ~100x
    assert d[0] > 0.6 * v_f


@needs_numba
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


@needs_numba
def test_oversampling_convergence():
    """4x vs 8x: harmonics above -60 dBc agree within 0.05 dB and the waveforms within
    -60 dB (relative RMS), at full drive and 0.5 V in."""
    kw = TS_MAX
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


# Robustness sweep (PR #22 review B1): every modelled part x string layout x configuration,
# with harsh inputs (wideband noise, large and small squares). Newton must converge on
# every step (``solve`` raises otherwise) and the output must stay finite.
SWEEP_STRINGS = [(1, 1), (1, 2), (1, 3), (0, 1), (2, 0)]
SWEEP = list(itertools.product(sorted(DIODES), SWEEP_STRINGS, ("shunt", "feedback")))


def _harsh(scale, seconds, seed=0):
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    sq = np.sign(np.sin(2 * np.pi * 100 * np.arange(n) / SR + 0.1))
    return np.stack([scale * rng.standard_normal(n), 5 * scale * sq, 0.5 * scale * sq])


def test_newton_robustness_sweep():
    seconds = 0.05 if NUMBA else 0.003
    for part, (n_pos, n_neg), config in SWEEP:
        if config == "shunt":
            m = shunt_clipper(R=1e3, C=1e-9, part=part, n_pos=n_pos, n_neg=n_neg)
            x = _harsh(10.0, seconds)  # sigma 10 V noise, 50 V squares
        else:
            m = feedback_clipper(**{**TS_MAX, "part": part}, n_pos=n_pos, n_neg=n_neg)
            x = _harsh(1.0, seconds)  # sigma 1 V (0 dBFS) noise, 5 V squares
        y = m.process_volts(x)
        assert np.isfinite(y).all(), (part, n_pos, n_neg, config)


@needs_numba
@pytest.mark.parametrize(
    "model, x",
    [
        (shunt_clipper(R=1e3, C=3.3e-9, n_pos=1, n_neg=3), _harsh(10.0, 0.1, seed=1)[:1]),
        (feedback_clipper(**{**TS_MAX, "part": "LED_RED"}), _harsh(1.0, 0.1, seed=2)[:1]),
        (shunt_clipper(R=1e3, C=1e-9, part="1N34A", n_pos=0, n_neg=1), _harsh(1.0, 0.1)[1:2]),
        (shunt_clipper(R=1e3, C=1e-9, part="1N914", n_pos=0, n_neg=1), _harsh(10.0, 0.1)[1:2]),
    ],
    ids=["si_1_3_noise", "led_feedback_noise", "ge_one_sided_square", "si_one_sided_square"],
)
def test_review_b1_cases_converge(model, x):
    """The four non-converging cases reported in the PR #22 review."""
    assert np.isfinite(model.process_volts(x)).all()


def _reference_ode(config, amp, seconds, kw):
    """Independent continuous-time reference: the circuit ODE (feedback: two states, the
    C_gain and C_fb voltages) integrated with Radau, sampled at 32x, band-limited to 44.1 kHz.
    """
    from scipy.integrate import solve_ivp

    fs_ref = 32 * SR
    t = np.arange(int(seconds * fs_ref)) / fs_ref
    w = 2 * np.pi * F0
    d = diode(kw.get("part", "1N914"))
    s = d.thermal_slope()

    def vin(tt):
        return amp * np.sin(w * tt)

    def i_d(v):
        return d.Is * (np.exp(v / s) - np.exp(-v / s))

    def g_d(v):
        return d.Is / s * (np.exp(v / s) + np.exp(-v / s))

    if config == "feedback":
        rg, cg, rf, cf = kw["R_gain"], kw["C_gain"], kw["R_fb"], kw["C_fb"]

        def f(tt, y):
            ig = (vin(tt) - y[0]) / rg
            return [ig / cg, (ig - y[1] / rf - i_d(y[1])) / cf]

        def jac(tt, y):
            return [[-1 / (rg * cg), 0.0], [-1 / (rg * cf), -(1 / rf + g_d(y[1])) / cf]]

        y0 = [0.0, 0.0]
    else:
        r, c = kw["R"], kw["C"]

        def f(tt, y):
            return [((vin(tt) - y[0]) / r - i_d(y[0])) / c]

        def jac(tt, y):
            return [[-(1 / r + g_d(y[0])) / c]]

        y0 = [0.0]
    sol = solve_ivp(f, (0, t[-1]), y0, "Radau", t_eval=t, jac=jac, rtol=1e-8, atol=1e-11)
    assert sol.success
    out = vin(t) + sol.y[1] if config == "feedback" else sol.y[0]
    return resample_down(resample_down(out[None], 8), 4)[0]


@pytest.mark.parametrize(
    "config, amp, kw",
    [("feedback", 1.0, TS_MAX), ("shunt", 3.0, dict(R=2.2e3, C=10e-9))],
)
def test_matches_independent_ode_reference(config, amp, kw):
    """Catches sign/topology errors shared by the solver and ``ClipperCircuit.response``."""
    seconds = 0.02
    ref = _reference_ode(config, amp, seconds, kw)
    model = feedback_clipper(**kw) if config == "feedback" else shunt_clipper(**kw)
    y = model.process_volts(sine(amp, seconds=seconds))[0]
    seg = slice(int(0.005 * SR), -int(0.002 * SR))  # skip start transient and resampler edges
    err = np.sqrt(np.mean((y - ref)[seg] ** 2) / np.mean(ref[seg] ** 2))
    assert 20 * np.log10(err) < -65, 20 * np.log10(err)  # measured: -74 fb, -82 shunt


@needs_numba
def test_numba_matches_numpy():
    rng = np.random.default_rng(0)
    j = rng.standard_normal((3, 400)) * 1e-3
    pair = feedback_clipper(**TS).circuit.diodes
    a = solve(j, 51e-12, 1 / 101e3, pair, 4 * SR, backend="numba")
    b = solve(j, 51e-12, 1 / 101e3, pair, 4 * SR, backend="numpy")
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@needs_numba
def test_full_chain_numba_matches_numpy():
    """End-to-end parity through DiodeClipper (resampling, transposes, pre/post filters)."""
    x = _harsh(0.5, 0.01, seed=3)
    post = (rc_lowpass(1e3, 0.22e-6),)
    for make in (
        lambda be: feedback_clipper(**TS, n_pos=1, n_neg=2, post=post, backend=be),
        lambda be: shunt_clipper(R=2.2e3, C=10e-9, part="LED_RED", backend=be),
    ):
        np.testing.assert_allclose(
            make("numba").process_volts(x), make("numpy").process_volts(x), rtol=0, atol=1e-12
        )


# --- TS808 via the registry ---------------------------------------------------------------------


def _thd_db(y):
    h = harmonics_dbc(y)
    return 10 * np.log10(np.sum(10 ** (h[..., 1:] / 10), -1))


@needs_numba
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


def test_ts808_requires_feedback_stage():
    pedal = load_kb()["ts808"]
    bad = dataclasses.replace(
        pedal, clipping=(dataclasses.replace(pedal.clipping[0], location="shunt"),)
    )
    with pytest.raises(ValueError, match="feedback"):
        WHITEBOX["ts808"](bad, bad.default_knobs(), SR)
    model = WHITEBOX["ts808"](pedal, pedal.default_knobs(), SR, oversample=8, backend="numpy")
    assert model.oversample == 8 and model.backend == "numpy"
    assert any("1N4148" in n for n in model.notes)


def test_bench_smoke():
    from lstmabar.physics.whitebox.bench import benchmark

    rows = benchmark(batch_sizes=(2,), seconds=0.05 if NUMBA else 0.005)
    assert rows[0]["batch"] == 2 and rows[0]["rtf"] > 0
