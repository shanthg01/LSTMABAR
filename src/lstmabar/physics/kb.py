"""Pedal knowledge base: one YAML file per pedal under ``pedals/``, validated by dataclasses.

File format (all keys except ``descriptors`` and ``notes`` are required)::

    id: ts808                       # lowercase, digits, underscores; equals the file stem
    name: Ibanez TS808 Tube Screamer
    family: overdrive               # one of FAMILIES
    topology: opamp_feedback_clip   # one of TOPOLOGIES
    clipping:                       # one entry per clipping stage, in signal order
      - stage: clip                 # free name, unique within the pedal
        device: si                  # one of DEVICES
        part: 1S2473                # optional part number
        location: feedback          # one of LOCATIONS
        n_pos: 1                    # devices in series conducting on positive swings
        n_neg: 1                    # ... and on negative swings (n_pos != n_neg: asymmetric)
    components:                     # semantic names -> schematic values with SI suffixes
      R_gain: 4.7k
      C_gain: 0.047u
    pots:                           # pedal knobs, keyed by knob id
      drive: {value: 500k, taper: A, default: 0.5, label: Drive}
    sources:                        # URLs the values were taken from
      - https://www.electrosmash.com/tube-screamer-analysis
    descriptors: [warm, mid-forward]  # optional; feeds P4 captions (never read data/gold/)
    notes: free text                # optional

Component values are strings such as ``"4.7k"``, ``"0.047u"``, ``"51p"``, ``"2.2M"`` or
plain numbers, parsed to SI base units (ohms, farads, henries, volts). A component's unit is
inferred from its name prefix (``R``, ``C``, ``L``, ``V``) and range-checked.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FAMILIES = ("overdrive", "distortion", "fuzz")
TOPOLOGIES = (
    "opamp_feedback_clip",  # diodes in the op-amp feedback loop (TS808)
    "opamp_shunt_clip",  # op-amp gain stage, then diodes to ground (RAT, DS-1)
    "two_transistor_fuzz",  # Fuzz Face
    "cascaded_transistor_clip",  # transistor stages with diode feedback (Big Muff)
)
DEVICES = ("si", "ge", "led", "mosfet", "si_transistor", "ge_transistor")
LOCATIONS = ("feedback", "shunt", "transistor")
TAPERS = ("A", "B", "C")

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SI_PREFIX = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "K": 1e3,
    "M": 1e6,
}
_VALUE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([pnuµmkKM]?)\s*$")

# Plausible ranges per component kind (SI base units), to catch unit typos ("0.047" vs "0.047u").
_RANGES = {
    "R": (1.0, 50e6),
    "C": (1e-13, 1e-2),
    "L": (1e-6, 10.0),
    "V": (0.1, 30.0),
}

# Two-segment approximation of a log ("audio") pot: 10% of the travel resistance at mid-rotation.
_A_TAPER_MID = 0.10


class KBError(ValueError):
    """A pedal file is malformed. The message names the file and the offending key."""


def parse_value(v: str | float | int) -> float:
    """``"4.7k"`` -> 4700.0, ``"0.047u"`` -> 4.7e-08, ``51`` -> 51.0. Raises ``ValueError``."""
    if isinstance(v, bool):
        raise ValueError(f"not a component value: {v!r}")
    if isinstance(v, int | float):
        out = float(v)
    else:
        m = _VALUE_RE.match(str(v))
        if not m:
            raise ValueError(f"not a component value: {v!r}")
        out = float(m.group(1)) * _SI_PREFIX.get(m.group(2), 1.0)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"component value must be positive and finite: {v!r}")
    return out


def taper_fraction(taper: str, pos: float) -> float:
    """Fraction of the pot's resistance between the low end and the wiper at rotation ``pos``.

    ``B`` is linear. ``A`` (audio) is the usual two-segment approximation through
    (0.5, 0.1); ``C`` (reverse audio) mirrors it through (0.5, 0.9).
    """
    if taper not in TAPERS:
        raise ValueError(f"unknown taper {taper!r}")
    pos = min(max(float(pos), 0.0), 1.0)
    if taper == "B":
        return pos
    if taper == "A":
        if pos <= 0.5:
            return pos * (2 * _A_TAPER_MID)
        return _A_TAPER_MID + (pos - 0.5) * 2 * (1 - _A_TAPER_MID)
    return 1.0 - taper_fraction("A", 1.0 - pos)


@dataclass(frozen=True)
class Pot:
    value: float  # ohms, end to end
    taper: str
    default: float = 0.5  # knob rotation in [0, 1]
    label: str = ""

    def fraction(self, pos: float) -> float:
        """Divider fraction at rotation ``pos`` (a volume pot's gain)."""
        return taper_fraction(self.taper, pos)

    def resistance(self, pos: float) -> float:
        """Resistance between the low end and the wiper at rotation ``pos`` (ohms)."""
        return self.value * self.fraction(pos)


@dataclass(frozen=True)
class ClipStage:
    stage: str
    device: str
    location: str
    n_pos: int = 1
    n_neg: int = 1
    part: str = ""

    @property
    def symmetric(self) -> bool:
        return self.n_pos == self.n_neg


@dataclass(frozen=True)
class Pedal:
    id: str
    name: str
    family: str
    topology: str
    clipping: tuple[ClipStage, ...]
    components: Mapping[str, float]
    pots: Mapping[str, Pot]
    sources: tuple[str, ...]
    descriptors: tuple[str, ...] = ()
    notes: str = ""
    path: Path | None = field(default=None, compare=False)

    def c(self, name: str) -> float:
        """Component value in SI base units; ``KeyError`` names the pedal if missing."""
        try:
            return self.components[name]
        except KeyError:
            raise KeyError(f"{self.id}: missing component {name!r}") from None

    def default_knobs(self) -> dict[str, float]:
        return {k: p.default for k, p in self.pots.items()}


# --- Parsing ------------------------------------------------------------------------------------


def _req(d: Mapping, key: str, where: str) -> Any:
    if key not in d:
        raise KBError(f"{where}: missing required key {key!r}")
    return d[key]


def _choice(v: Any, options: tuple[str, ...], where: str) -> str:
    if v not in options:
        raise KBError(f"{where}: {v!r} is not one of {options}")
    return v


def _component(name: str, raw: Any, where: str) -> float:
    try:
        v = parse_value(raw)
    except ValueError as e:
        raise KBError(f"{where}.{name}: {e}") from None
    lo, hi = _RANGES.get(name[:1], (0.0, math.inf))
    if not lo <= v <= hi:
        raise KBError(f"{where}.{name}: {raw!r} -> {v:g} outside plausible range [{lo:g}, {hi:g}]")
    return v


def _pot(name: str, raw: Any, where: str) -> Pot:
    w = f"{where}.{name}"
    if not isinstance(raw, Mapping):
        raise KBError(f"{w}: expected a mapping")
    unknown = set(raw) - {"value", "taper", "default", "label"}
    if unknown:
        raise KBError(f"{w}: unknown keys {sorted(unknown)}")
    default = float(raw.get("default", 0.5))
    if not 0.0 <= default <= 1.0:
        raise KBError(f"{w}.default: {default} outside [0, 1]")
    return Pot(
        value=_component("R", _req(raw, "value", w), w),
        taper=_choice(_req(raw, "taper", w), TAPERS, f"{w}.taper"),
        default=default,
        label=str(raw.get("label", name.replace("_", " ").title())),
    )


def _clip(i: int, raw: Any, where: str) -> ClipStage:
    w = f"{where}.clipping[{i}]"
    if not isinstance(raw, Mapping):
        raise KBError(f"{w}: expected a mapping")
    unknown = set(raw) - {"stage", "device", "part", "location", "n_pos", "n_neg"}
    if unknown:
        raise KBError(f"{w}: unknown keys {sorted(unknown)}")
    n_pos, n_neg = int(raw.get("n_pos", 1)), int(raw.get("n_neg", 1))
    if n_pos < 0 or n_neg < 0 or n_pos + n_neg == 0:
        raise KBError(f"{w}: n_pos/n_neg must be >= 0 and not both 0")
    return ClipStage(
        stage=str(_req(raw, "stage", w)),
        device=_choice(_req(raw, "device", w), DEVICES, f"{w}.device"),
        location=_choice(_req(raw, "location", w), LOCATIONS, f"{w}.location"),
        n_pos=n_pos,
        n_neg=n_neg,
        part=str(raw.get("part", "")),
    )


_TOP_KEYS = {
    "id",
    "name",
    "family",
    "topology",
    "clipping",
    "components",
    "pots",
    "sources",
    "descriptors",
    "notes",
}


def pedal_from_dict(raw: Mapping, where: str = "<dict>", path: Path | None = None) -> Pedal:
    """Validate a parsed YAML mapping and build a :class:`Pedal`. Raises :class:`KBError`."""
    if not isinstance(raw, Mapping):
        raise KBError(f"{where}: top level must be a mapping")
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise KBError(f"{where}: unknown keys {sorted(unknown)}")
    pid = str(_req(raw, "id", where))
    if not _ID_RE.match(pid):
        raise KBError(f"{where}.id: {pid!r} must match {_ID_RE.pattern}")
    if path is not None and path.stem != pid:
        raise KBError(f"{where}.id: {pid!r} must equal the file stem {path.stem!r}")

    clipping_raw = _req(raw, "clipping", where)
    if not isinstance(clipping_raw, list) or not clipping_raw:
        raise KBError(f"{where}.clipping: expected a non-empty list")
    clipping = tuple(_clip(i, c, where) for i, c in enumerate(clipping_raw))
    stages = [c.stage for c in clipping]
    if len(set(stages)) != len(stages):
        raise KBError(f"{where}.clipping: duplicate stage names {stages}")

    comps_raw = _req(raw, "components", where)
    if not isinstance(comps_raw, Mapping):
        raise KBError(f"{where}.components: expected a mapping")
    components = {
        str(k): _component(str(k), v, f"{where}.components") for k, v in comps_raw.items()
    }

    pots_raw = _req(raw, "pots", where)
    if not isinstance(pots_raw, Mapping) or not pots_raw:
        raise KBError(f"{where}.pots: expected a non-empty mapping")
    pots = {str(k): _pot(str(k), v, f"{where}.pots") for k, v in pots_raw.items()}

    sources = _req(raw, "sources", where)
    if not isinstance(sources, list) or not sources:
        raise KBError(f"{where}.sources: expected a non-empty list of URLs")
    for s in sources:
        if not str(s).startswith(("http://", "https://")):
            raise KBError(f"{where}.sources: {s!r} is not a URL")

    descriptors = raw.get("descriptors", []) or []
    if not isinstance(descriptors, list):
        raise KBError(f"{where}.descriptors: expected a list")

    return Pedal(
        id=pid,
        name=str(_req(raw, "name", where)),
        family=_choice(_req(raw, "family", where), FAMILIES, f"{where}.family"),
        topology=_choice(_req(raw, "topology", where), TOPOLOGIES, f"{where}.topology"),
        clipping=clipping,
        components=components,
        pots=pots,
        sources=tuple(str(s) for s in sources),
        descriptors=tuple(str(d) for d in descriptors),
        notes=str(raw.get("notes", "") or ""),
        path=path,
    )


def load_pedal(path: str | Path) -> Pedal:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise KBError(f"{path.name}: invalid YAML: {e}") from None
    return pedal_from_dict(raw, where=path.name, path=path)


def default_pedals_dir() -> Path:
    """``pedals/`` at the repository root (for editable installs and tests)."""
    return Path(__file__).resolve().parents[3] / "pedals"


def load_kb(directory: str | Path | None = None) -> dict[str, Pedal]:
    """Load and validate every ``*.yaml`` in ``directory`` (default: repo ``pedals/``)."""
    directory = Path(directory) if directory is not None else default_pedals_dir()
    if not directory.is_dir():
        raise KBError(f"pedal directory not found: {directory.name}")
    kb: dict[str, Pedal] = {}
    for path in sorted(directory.glob("*.yaml")):
        pedal = load_pedal(path)
        kb[pedal.id] = pedal
    return kb


__all__ = [
    "DEVICES",
    "FAMILIES",
    "LOCATIONS",
    "TAPERS",
    "TOPOLOGIES",
    "ClipStage",
    "KBError",
    "Pedal",
    "Pot",
    "default_pedals_dir",
    "load_kb",
    "load_pedal",
    "parse_value",
    "pedal_from_dict",
    "taper_fraction",
]
