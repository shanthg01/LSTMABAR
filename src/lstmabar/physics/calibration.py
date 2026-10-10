"""Level calibration shared by the grey-box derivations and the white-box sims (P2 decision 3).

Digital full scale maps to a voltage at the pedal input: ``0 dBFS <-> VOLTS_PER_FULL_SCALE``
volts peak. White-box sims take ``x * volts_per_fs`` volts in and return volts, divided by
``volts_per_fs`` on the way out. The grey-box ``Drive`` clips at ±1, which corresponds to the
clipping device's voltage ``V_clip``, so a circuit stage gain ``G`` becomes

    drive.gain_db = 20·log10(G) + 20·log10(volts_per_fs / V_clip)

and the clipper's ±1 output is ``V_clip`` volts, i.e. ``V_clip / volts_per_fs`` full scale.
The configured value lives under ``physics.volts_per_full_scale`` in ``configs/base.yaml``.
"""

import math

VOLTS_PER_FULL_SCALE = 1.0
"""Default: 0 dBFS = 1 V peak at the pedal input (hot humbucker territory)."""

CLIP_VOLTS: dict[str, float] = {
    "si": 0.6,
    "ge": 0.3,
    "led": 1.7,
    "si_transistor": 0.6,
    "ge_transistor": 0.3,
}
"""Approximate forward clipping voltage per device (one device conducting). MOSFET values
depend on the part, so derivations take them from the pedal's cited source."""


def clip_volts(device: str, n_series: int = 1) -> float:
    """Clipping voltage of ``n_series`` devices of type ``device`` in series."""
    if device not in CLIP_VOLTS:
        raise KeyError(f"no default clipping voltage for device {device!r}")
    return CLIP_VOLTS[device] * n_series


def db(x: float) -> float:
    """Amplitude ratio -> dB (``-inf`` for 0)."""
    return 20.0 * math.log10(x) if x > 0 else -math.inf


def drive_gain_db(
    stage_gain: float, v_clip: float, volts_per_fs: float = VOLTS_PER_FULL_SCALE
) -> float:
    """Grey-box ``drive.gain_db`` for a circuit stage of linear gain ``stage_gain``."""
    return db(stage_gain) + db(volts_per_fs / v_clip)


def output_level_db(
    v_clip: float, post_gain: float = 1.0, volts_per_fs: float = VOLTS_PER_FULL_SCALE
) -> float:
    """Grey-box ``drive.level_db``: clipper output ±1 is ``v_clip`` volts, then ``post_gain``
    (tone stage, volume pot, output buffer), expressed in full-scale units."""
    return db(v_clip * post_gain / volts_per_fs)


__all__ = [
    "CLIP_VOLTS",
    "VOLTS_PER_FULL_SCALE",
    "clip_volts",
    "db",
    "drive_gain_db",
    "output_level_db",
]
