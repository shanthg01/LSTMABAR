import math

import numpy as np
import pytest

from lstmabar.physics.kb import Pot
from lstmabar.physics.networks import (
    TF,
    cap,
    ce_collector_feedback_bias,
    ce_open_loop_gain,
    divider,
    lower_corner_hz,
    noninverting_gain,
    par,
    pot_split,
    rc_highpass_tf,
    rc_hz,
    rc_lowpass_tf,
    res,
    shunt_feedback_gain,
    thevenin_cr_highpass,
    thevenin_rc_lowpass,
    two_leg_blend,
)

F = np.geomspace(20, 20000, 50)


def _z_c(c):
    return 1 / (2j * np.pi * F * c)


def test_tf_arithmetic_matches_complex_evaluation():
    r1, r2, c1, c2 = 1e3, 4.7e3, 0.1e-6, 22e-9
    z = par(res(r1) + cap(c1), res(r2)) / (res(r1) - cap(c2)) * 3 + 1
    zc1, zc2 = _z_c(c1), _z_c(c2)
    ref = ((r1 + zc1) * r2 / (r1 + zc1 + r2)) / (r1 - zc2) * 3 + 1
    np.testing.assert_allclose(z(F), ref, rtol=1e-9)
    with pytest.raises(ZeroDivisionError):
        TF([1.0], [0.0])


def test_first_order_sections():
    lp, hp = rc_lowpass_tf(1e3, 0.1e-6), rc_highpass_tf(1e3, 0.1e-6)
    fc = rc_hz(1e3, 0.1e-6)
    assert math.isclose(lp.db(fc), -3.01, abs_tol=0.01)
    assert math.isclose(hp.db(fc), -3.01, abs_tol=0.01)
    np.testing.assert_allclose(divider(res(1e3), cap(0.1e-6))(F), lp(F), rtol=1e-12)
    assert math.isclose(lower_corner_hz(hp, 20000.0), fc, rel_tol=0.01)


def test_noninverting_gain_shelf():
    g = noninverting_gain(res(51e3 + 500e3), res(4.7e3) + cap(0.047e-6))
    assert math.isclose(g.mag(20000), 1 + 551e3 / 4.7e3, rel_tol=0.01)
    assert math.isclose(g.mag(0.05), 1.0, abs_tol=0.01)  # zero at ~6 Hz


def test_shunt_feedback_gain_limits():
    ideal = shunt_feedback_gain(res(470e3), res(10e3), 1e12, 1e9)
    assert math.isclose(ideal.mag(1000), 47.0, rel_tol=1e-6)
    finite = shunt_feedback_gain(res(470e3), res(10e3), 100e3, 65.0)
    assert finite.mag(1000) < 47.0
    # 47 / (1 + (1 + 47 + 4.7) / 65)
    assert math.isclose(finite.mag(1000), 47 / (1 + 52.7 / 65), rel_tol=1e-6)


def test_collector_feedback_bias():
    # Big Muff booster: R13 15k, R22 100, R9 470k, R14 47k -> collector ~6.8 V.
    ic = ce_collector_feedback_bias(9.0, 15e3, 100.0, 470e3, 47e3)
    assert math.isclose(9.0 - ic * 15e3, 6.8, abs_tol=0.1)


def test_pot_split():
    pot = Pot(100e3, "B")
    assert pot_split(pot, 0.25) == (25e3, 75e3)
    lo, hi = pot_split(pot, 0.0)
    assert lo == 1.0 and hi == 100e3


def test_shunt_feedback_gain_matches_hybrid_pi_nodal():
    """Nodes B, C, E with a transconductance g_m (β -> ∞, as in the closed form). The closed
    form ignores the feedback network loading the collector (R_c/Z_f ~ 3% here)."""
    r_s, c_s, r_f, c_f, r_b, r_c, r_e = 10e3, 0.1e-6, 470e3, 470e-12, 100e3, 15e3, 150.0
    i_c = ce_collector_feedback_bias(9.0, r_c, r_e, r_f, r_b)
    gm = i_c / 0.02585
    h = shunt_feedback_gain(
        par(res(r_f), cap(c_f)), res(r_s) + cap(c_s), r_b, ce_open_loop_gain(r_c, r_e, i_c)
    )
    for f in (100.0, 1000.0, 5000.0):
        s = 2j * np.pi * f
        y_s, y_f = 1 / (r_s + 1 / (s * c_s)), 1 / r_f + s * c_f
        y = np.array(
            [
                [y_s + y_f + 1 / r_b, -y_f, 0],  # B
                [gm - y_f, y_f + 1 / r_c, -gm],  # C
                [-gm, 0, gm + 1 / r_e],  # E
            ],
            dtype=complex,
        )
        v = np.linalg.solve(y, np.array([y_s, 0, 0], dtype=complex))
        assert math.isclose(20 * np.log10(abs(v[1])), float(h.db(f)), abs_tol=0.5), f


def test_two_leg_blend_matches_nodal_mid_rotation():
    """Big-Muff-like stack at 30% rotation with a finite RC load, against a 3-node MNA solve."""
    r_lp, c_lp, c_hp, r_hp, pot = 39e3, 10e-9, 4e-9, 22e3, 100e3
    r_lw, r_wh = 0.3 * pot, 0.7 * pot
    c_load, r_load = 0.1e-6, 81e3
    v_l, z_l = thevenin_rc_lowpass(r_lp, c_lp)
    v_h, z_h = thevenin_cr_highpass(c_hp, r_hp)
    out = two_leg_blend(v_l, z_l + r_lw, v_h, z_h + r_wh, cap(c_load) + r_load)
    for f in F:
        s = 2j * np.pi * f
        y_load = 1 / (r_load + 1 / (s * c_load))
        y = np.array(
            [
                [1 / r_lp + s * c_lp + 1 / r_lw, 0, -1 / r_lw],  # L
                [0, s * c_hp + 1 / r_hp + 1 / r_wh, -1 / r_wh],  # H
                [-1 / r_lw, -1 / r_wh, 1 / r_lw + 1 / r_wh + y_load],  # W
            ],
            dtype=complex,
        )
        v = np.linalg.solve(y, np.array([1 / r_lp, s * c_hp, 0], dtype=complex))
        assert abs(out(f) - v[2]) < 1e-9 * max(1.0, abs(v[2]))


def test_two_leg_blend_wiper_at_a_leg():
    """Wiper at the LP end, HP leg disconnected (huge pot section) and an open load: the
    output is the unloaded LP leg."""
    v_l, z_l = rc_lowpass_tf(39e3, 10e-9), par(res(39e3), cap(10e-9))
    v_h, z_h = rc_highpass_tf(22e3, 4e-9), par(res(22e3), cap(4e-9))
    out = two_leg_blend(v_l, z_l + 1.0, v_h, z_h + 1e12, res(1e15))
    np.testing.assert_allclose(out(F), v_l(F), rtol=1e-6)
