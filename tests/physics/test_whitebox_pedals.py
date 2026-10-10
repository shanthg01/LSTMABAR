"""DS-1 white-box, the generic diode-port solver and the TS808 tone stage."""

import numpy as np
import pytest

from lstmabar.analysis.harmonics import harmonic_amplitudes, to_dbc
from lstmabar.physics.kb import load_kb
from lstmabar.physics.networks import pot_split
from lstmabar.physics.whitebox import WHITEBOX, has_whitebox, shunt_clipper, simulate
from lstmabar.physics.whitebox.devices import diode
from lstmabar.physics.whitebox.diode_clipper import DiodePair, numba_available
from lstmabar.physics.whitebox.ds1 import (
    booster,
    clip_port,
    level_divider,
    tone_network,
    tone_network_nodal,
)
from lstmabar.physics.whitebox.linear import AnalogFilter, nodal_transfer, poly_det
from lstmabar.physics.whitebox.port import PortClipper
from lstmabar.physics.whitebox.ts808 import tone_stage

SR = 44100
needs_numba = pytest.mark.skipif(not numba_available(), reason="numba (whitebox extra) missing")
KB = load_kb()


def _sines(freqs, amp, seconds=0.25):
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * np.asarray(freqs, float)[:, None] * t)


def _gain_db(y, f, amp, start=0.1):
    return 20 * np.log10(harmonic_amplitudes(y[int(start * SR) :], SR, f, 1)[0] / amp)


# --- linear helpers -----------------------------------------------------------------------------


def test_poly_det_and_nodal_transfer_match_numeric_solve():
    # voltage divider R1 - node - (R2 || C) : H = 1 / (1 + R1 (1/R2 + sC))
    r1, r2, c = 1e3, 2e3, 1e-7
    h = nodal_transfer([[[c, 1 / r1 + 1 / r2]]], [[1 / r1]], out=0)
    f = np.array([10.0, 1e3, 1e4])
    want = 1 / (1 + r1 * (1 / r2 + 2j * np.pi * f * c))
    np.testing.assert_allclose(h.response(f), want, rtol=1e-12)
    # (s + 2)(5s + 6) - 3·4 = 5s² + 16s
    np.testing.assert_allclose(poly_det([[[1, 2], [3]], [[4], [5, 6]]]), [5, 16, 0])


# --- TS808 tone stage ---------------------------------------------------------------------------


def _ts_tone_mna(p, f, tone, open_loop=1e9):
    """Nodal solve of the TS808 tone stage with a finite-gain op-amp (independent check)."""
    rb, ra = pot_split(p.pots["tone"], tone)  # (+) end..wiper, wiper..(-) end
    s = 2j * np.pi * f
    r7, c5 = p.c("R_tone_lp"), p.c("C_tone_lp")
    z = p.c("R_tone_shunt") + 1 / (s * p.c("C_tone_shunt"))
    rf, rbias = p.c("R_tone_fb"), p.c("R_tone_bias")
    m = np.zeros((4, 4), complex)  # nodes A (+), N (-), W (wiper), O (output)
    m[0, [0, 2]] = 1 / r7 + s * c5 + 1 / rbias + 1 / rb, -1 / rb
    m[1, [1, 2, 3]] = 1 / ra + 1 / rf, -1 / ra, -1 / rf
    m[2, [0, 1, 2]] = -1 / rb, -1 / ra, 1 / ra + 1 / rb + 1 / z
    m[3, [0, 1, 3]] = -open_loop, open_loop, 1
    return np.linalg.solve(m, np.array([1 / r7, 0, 0, 0], complex))[3]


@pytest.mark.parametrize("tone", [0.0, 0.3, 0.5, 1.0])
def test_ts808_tone_stage_matches_nodal_solve(tone):
    p = KB["ts808"]
    h = tone_stage(p, tone)
    for f in (50.0, 300.0, 723.0, 3000.0, 8000.0):
        assert h.response(f) == pytest.approx(_ts_tone_mna(p, f, tone), rel=1e-6)


def test_ts808_tone_stage_brightens_and_dc_gain():
    p = KB["ts808"]
    rbias, r7 = p.c("R_tone_bias"), p.c("R_tone_lp")
    for t in (0.0, 1.0):
        assert abs(tone_stage(p, t).response(1e-3)) == pytest.approx(rbias / (r7 + rbias), 1e-6)
    hi = [abs(tone_stage(p, t).response(4000.0)) for t in (0.0, 0.5, 1.0)]
    assert hi[0] < hi[1] < hi[2]


@needs_numba
def test_ts808_tone_knob_moves_brightness():
    p = KB["ts808"]
    x = _sines([330.0], 0.3, 0.4)
    h = [
        to_dbc(harmonic_amplitudes(simulate(p, x, SR, {"tone": t})[0, 4410:], SR, 330.0))
        for t in (0.0, 1.0)
    ]
    assert h[1][2] > h[0][2] + 6  # H3 (990 Hz) much stronger with tone up


# --- generic port solver ------------------------------------------------------------------------


@needs_numba
def test_port_clipper_reduces_to_shunt_clipper():
    """An RC node with no other load: the port model must equal the one-capacitor solver."""
    r, c = 2.2e3, 1e-8
    shunt = shunt_clipper(R=r, C=c, part="1N4148", oversample_linear=True)
    port = PortClipper(
        diodes=DiodePair(diode("1N4148")),
        z_port=AnalogFilter((r,), (r * c, 1.0)),
        t_open=AnalogFilter((1.0,), (r * c, 1.0)),
        t_port_out=AnalogFilter((1.0,), (1.0,)),
    )
    x = _sines([110.0, 1000.0], 3.0, 0.1)
    np.testing.assert_allclose(port.process_volts(x), shunt.process_volts(x), atol=1e-7)


def test_port_clipper_numpy_backend_matches_numba_short():
    if not numba_available():
        pytest.skip("numba missing")
    p = KB["ds1"]
    x = _sines([220.0], 0.1, 0.01)
    a = WHITEBOX["ds1"](p, p.knobs(), SR, backend="numba").process_volts(x)
    b = WHITEBOX["ds1"](p, p.knobs(), SR, backend="numpy").process_volts(x)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


# --- DS-1 ---------------------------------------------------------------------------------------


def test_ds1_registered_and_booster_gain():
    assert has_whitebox("ds1")
    # shared with the derivation; ElectroSmash: ~35 dB above a few hundred Hz
    assert 20 * np.log10(abs(booster(KB["ds1"]).response(3000.0))) == pytest.approx(35, abs=1.5)


def _ds1_exact(p, f, knobs):
    """Independent complex nodal solve of R14 → node X (C10) → tone stack → level load."""
    ra, rb = pot_split(p.pots["tone"], knobs["tone"])  # L..W, W..H
    _, r_level = level_divider(p, knobs["level"])
    s = 2j * np.pi * f
    g14, c10 = 1 / p.c("R_clip"), p.c("C_clip")
    g16, c12, c11, g17 = (
        1 / p.c("R_tone_lp"),
        p.c("C_tone_lp"),
        p.c("C_tone_hp"),
        1 / p.c("R_tone_hp"),
    )
    m = np.array(
        [
            [g14 + s * c10 + g16 + s * c11, -g16, -s * c11, 0],
            [-g16, g16 + s * c12 + 1 / ra, 0, -1 / ra],
            [-s * c11, 0, s * c11 + g17 + 1 / rb, -1 / rb],
            [0, -1 / ra, -1 / rb, 1 / ra + 1 / rb + 1 / r_level],
        ],
        complex,
    )
    return np.linalg.solve(m, np.array([g14, 0, 0, 0], complex))


@pytest.mark.parametrize("tone", [0.0, 0.5, 1.0])
def test_ds1_network_transfer_functions_match_nodal_solve(tone):
    p = KB["ds1"]
    knobs = p.knobs({"tone": tone})
    _, r_level = level_divider(p, knobs["level"])
    t_open, _ = clip_port(p, tone, r_level)
    tone_h = tone_network(p, tone, r_level)  # derivation's two-leg blend helper
    tone_n = tone_network_nodal(p, tone, r_level)
    for f in (82.0, 500.0, 2000.0, 7000.0):
        v = _ds1_exact(p, f, knobs)
        # diodes off: node X = T_open · v_s, output W = T_tone · X
        assert t_open.response(f) == pytest.approx(v[0], rel=1e-6)
        assert tone_h.response(f) * v[0] == pytest.approx(v[3], rel=1e-6)
        assert tone_n.response(f) == pytest.approx(tone_h.response(f), rel=1e-6)


@pytest.mark.parametrize("order_filter", ["ts808_tone", "ds1_booster"])
def test_high_order_filters_discretize_to_their_analog_response(order_filter):
    """Unreduced network TFs run as second-order sections; their digital response must
    match the analog one well below Nyquist."""
    from scipy.signal import sosfreqz

    flt = tone_stage(KB["ts808"], 0.7) if order_filter == "ts808_tone" else booster(KB["ds1"])
    assert flt.order > 2
    fs = 4 * SR
    f = np.array([50.0, 300.0, 1000.0, 5000.0])
    _, h = sosfreqz(flt.sos(fs), worN=f, fs=fs)
    # Only the bilinear frequency warping (~0.6% at 5 kHz at 176 kHz) separates them.
    want = 20 * np.log10(abs(flt.response(f)))
    np.testing.assert_allclose(20 * np.log10(abs(h)), want, atol=0.05)


@needs_numba
@pytest.mark.parametrize("knobs", [{"dist": 0.0}, {"dist": 1.0}, {"tone": 0.0}, {"tone": 1.0}])
def test_ds1_small_signal_matches_analytic_chain(knobs):
    p = KB["ds1"]
    model = WHITEBOX["ds1"](p, p.knobs(knobs), SR)
    freqs = [100.0, 300.0, 1000.0, 3000.0, 6000.0]
    amp = 1e-6  # ≪ the diodes' thermal voltage even after ~60 dB of gain
    y = model.process_volts(_sines(freqs, amp))
    for i, f in enumerate(freqs):
        want = 20 * np.log10(abs(model.response(f)))
        assert _gain_db(y[i], f, amp) == pytest.approx(want, abs=0.1), f


@needs_numba
def test_ds1_finite_bounded_and_knobs_act():
    p = KB["ds1"]
    x = _sines([220.0, 220.0], 1.0, 0.4) * np.array([[1.0], [0.01]])
    y = simulate(p, x, SR)
    assert np.isfinite(y).all() and np.abs(y).max() < 1.0  # diode node ~±0.7 V, then losses

    def thd_db(knobs, level):
        out = simulate(p, _sines([220.0], level, 0.4), SR, knobs)[0, 4410:]
        h = harmonic_amplitudes(out, SR, 220.0)
        return 10 * np.log10(np.sum((h[1:] / h[0]) ** 2))

    # Dist raises harmonic content (small input, where the booster alone does not saturate
    # the diodes hard).
    thd = [thd_db({"dist": d}, 1e-3) for d in (0.0, 0.5, 1.0)]
    assert thd[0] < thd[1] < thd[2], thd

    # Tone moves brightness: upper harmonics relative to the fundamental rise with tone.
    def bright(t):
        out = simulate(p, _sines([220.0], 0.1, 0.4), SR, {"tone": t})[0, 4410:]
        return to_dbc(harmonic_amplitudes(out, SR, 220.0))[4]  # H5, 1.1 kHz

    b = [bright(t) for t in (0.0, 0.5, 1.0)]
    assert b[0] < b[1] < b[2], b
