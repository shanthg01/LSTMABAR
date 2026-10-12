"""Pedal names allowed in captions, shared by the caption post-filter (task 4.3) and the
leakage tests (task 4.7).

Captions must not contain numbers (dB, Hz, knob values). Pedal model names that contain
digits ("TS808", "DS-1", "RAT2", "BD-2") are the only exception, and only as whole names
from this allowlist. Captions for splits other than ``test_unseen_pedal`` must not name the
holdout pedal at all.
"""

from __future__ import annotations

import re

PEDAL_ALIASES: dict[str, tuple[str, ...]] = {
    "ts808": ("TS808", "TS-808", "Tube Screamer", "Tubescreamer"),
    "ds1": ("DS-1", "DS1", "Boss DS-1"),
    "rat": ("RAT", "ProCo RAT", "Pro Co RAT", "RAT2", "RAT 2"),
    "big_muff": ("Big Muff", "Big Muff Pi", "Muff"),
    "fuzz_face_si": ("Fuzz Face", "silicon Fuzz Face"),
    "fuzz_face_ge": ("Fuzz Face", "germanium Fuzz Face"),
}
"""Names a caption may use for each KB pedal (case-insensitive whole-word match)."""

OTHER_DEVICE_NAMES: tuple[str, ...] = ("BD-2", "Blues Driver", "TS Mini")
"""Real-pedal test-set devices without a KB entry (never used in training captions)."""


def _pattern(name: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE)


def names_for(pedal_ids: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for pid in pedal_ids:
        out.extend(PEDAL_ALIASES.get(pid, ()))
    return out


def mentions(text: str, names: list[str] | tuple[str, ...]) -> list[str]:
    """Names from ``names`` that occur in ``text`` as whole words."""
    return [n for n in names if _pattern(n).search(text)]


def strip_allowed_names(text: str, allowed: list[str] | tuple[str, ...]) -> str:
    """``text`` with allowed pedal names removed (longest first), for digit checks."""
    for n in sorted(allowed, key=len, reverse=True):
        text = _pattern(n).sub(" ", text)
    return text


def holdout_names(holdout_pedal: str) -> list[str]:
    """Names that would leak the holdout pedal (its aliases minus names it shares with a
    training pedal, e.g. none for ``rat``)."""
    own = set(PEDAL_ALIASES.get(holdout_pedal, ()))
    shared = {n for pid, ns in PEDAL_ALIASES.items() if pid != holdout_pedal for n in ns}
    return sorted(own - shared)


__all__ = [
    "OTHER_DEVICE_NAMES",
    "PEDAL_ALIASES",
    "holdout_names",
    "mentions",
    "names_for",
    "strip_allowed_names",
]
