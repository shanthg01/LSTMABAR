"""Device model parameters for the white-box sims, keyed by part number.

Diodes use the Shockley equation ``i = Is · (exp(v / (N · Vt)) - 1)``. ``n_series`` identical
diodes in series share the voltage equally, so a string behaves like one diode with
``N_string = n_series · N`` and the same ``Is``.

Sources (values are representative, not measured on a specific unit):

- ``1N4148`` / ``1N914``: LTspice ``standard.dio`` library line
  ``D(Is=2.52n Rs=.568 N=1.752 Cjo=4p M=.4 tt=20n)`` (the two parts share one die and one
  model there). Forward drop at 1 mA ≈ 0.58 V.
- ``1S2473`` (the TS808's clipping diodes): a Japanese general-purpose Si switching diode
  with no widely published SPICE model; it is commonly treated as equivalent to the
  1N4148/1N914 class (TS clones substitute 1N914s), so it reuses the 1N4148 parameters.
  Substitute, flagged in :data:`SUBSTITUTES`; verify against a measured curve if one turns up.
- ``1N34A`` (germanium point contact): no authoritative SPICE model exists (published hobby
  models disagree by orders of magnitude in Is). Chosen so the forward drop is ≈0.31 V at
  1 mA with a soft knee (N = 1.3), matching typical datasheet curves and the 0.3 V Ge value
  in :data:`lstmabar.physics.calibration.CLIP_VOLTS`.
- ``LED_RED`` (generic 3–5 mm red LED): fitted to typical datasheet points, Vf ≈ 1.68 V at
  1 mA and ≈ 1.80 V at 10 mA (N = 2.0), consistent with the 1.7 V LED calibration value.

Series resistance ``Rs`` is recorded for reference but **not used** by the one-capacitor
solver: in the pedal circuits it is small against the surrounding resistors (≥1 kΩ) and only
matters at currents the clippers never reach.
"""

import math
from dataclasses import dataclass, replace

BOLTZMANN = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ROOM_TEMP_K = 298.15  # 25 °C
VT = BOLTZMANN * ROOM_TEMP_K / ELEMENTARY_CHARGE
"""Thermal voltage kT/q at 25 °C (≈ 25.69 mV)."""


@dataclass(frozen=True)
class Diode:
    part: str
    Is: float  # saturation current (A)
    N: float  # emission coefficient (ideality)
    Rs: float = 0.0  # series resistance (ohms); informational, not simulated

    def thermal_slope(self, n_series: int = 1, vt: float = VT) -> float:
        """``n_series · N · Vt``: the exponential's voltage scale for a string (volts)."""
        return n_series * self.N * vt

    def forward_voltage(self, current: float, n_series: int = 1) -> float:
        """Voltage across ``n_series`` diodes in series at forward ``current`` (A)."""
        return self.thermal_slope(n_series) * math.log1p(current / self.Is)


DIODES: dict[str, Diode] = {
    "1N4148": Diode("1N4148", Is=2.52e-9, N=1.752, Rs=0.568),
    "1N914": Diode("1N914", Is=2.52e-9, N=1.752, Rs=0.568),
    "1N34A": Diode("1N34A", Is=1.0e-7, N=1.3, Rs=0.0),
    "LED_RED": Diode("LED_RED", Is=6.3e-18, N=2.0, Rs=0.0),
}

SUBSTITUTES: dict[str, str] = {"1S2473": "1N4148"}
"""Deliberate equivalents: parts with no published model that borrow another part's
parameters. :func:`diode` returns the borrowed model under the requested part name."""

DEFAULT_PART: dict[str, str] = {"si": "1N914", "ge": "1N34A", "led": "LED_RED"}
"""Fallback part per KB device type when a clipping stage names no (known) part."""


def diode(part: str = "", device: str = "si") -> Diode:
    """Diode parameters for ``part`` (case-insensitive), resolving :data:`SUBSTITUTES`.

    An empty ``part`` uses the ``device`` type's default part (:data:`DEFAULT_PART`). A
    non-empty part that is neither modelled nor a listed substitute raises ``KeyError``, so
    a typo or an unmodelled part in a pedal file never silently simulates another device.
    """
    key = part.strip().upper()
    if not key:
        if device not in DEFAULT_PART:
            raise KeyError(f"no default diode for device {device!r}; name a part")
        return DIODES[DEFAULT_PART[device]]
    if key in DIODES:
        return DIODES[key]
    if key in SUBSTITUTES:
        return replace(DIODES[SUBSTITUTES[key]], part=key)
    raise KeyError(
        f"unknown diode part {part!r}: add it to DIODES (with a cited source) or to SUBSTITUTES"
    )


__all__ = ["DEFAULT_PART", "DIODES", "SUBSTITUTES", "VT", "Diode", "diode"]
