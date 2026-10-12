"""Caption vocabulary rules shared by the caption post-filter (task 4.3) and the leakage
tests (task 4.7), so the two cannot diverge (decision 5, R1).

Captions must not contain numbers (digits, spelled-out numbers except "one", which is too
common in ordinary English, units such as dB/Hz/ms/%).
Pedal model names with digits ("TS808", "DS-1", "RAT2") are the only exception, and only as
whole names from :data:`PEDAL_ALIASES`. Captions outside ``test_unseen_pedal`` must not name
the holdout pedal. Templates must not use the KB ``name`` field verbatim (it carries version
tags such as "Big Muff Pi (V3)"); they use these aliases.
"""

from __future__ import annotations

import re

PEDAL_ALIASES: dict[str, tuple[str, ...]] = {
    "ts808": ("TS808", "TS-808", "Tube Screamer", "Tubescreamer"),
    "ds1": ("DS-1", "DS1", "Boss DS-1"),
    "rat": ("RAT", "RAT2", "ProCo RAT", "Pro Co RAT", "ProCo", "Pro Co"),
    "big_muff": ("Big Muff", "Big Muff Pi", "Muff"),
    "fuzz_face_si": ("Fuzz Face", "silicon Fuzz Face"),
    "fuzz_face_ge": ("Fuzz Face", "germanium Fuzz Face"),
}
"""Names a caption may use for each KB pedal (case-insensitive whole-word match)."""

OTHER_DEVICE_NAMES: tuple[str, ...] = ("BD-2", "Blues Driver", "TS Mini")
"""Real-pedal test-set devices without a KB entry (never used in training captions)."""

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")

NUMBER_WORDS = (
    "zero two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty "
    "ninety hundred thousand percent dimed"
).split()

_UNIT = re.compile(r"(?<!\w)(db|dbfs|hz|khz|ms|kohm|ohms?|volts?|%)(?!\w)", re.IGNORECASE)
_NUMBER_WORD = re.compile(r"(?<!\w)(" + "|".join(NUMBER_WORDS) + r")(?!\w)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Unicode dashes -> ASCII hyphen (so "DS‑1" matches "DS-1")."""
    return text.translate(_DASHES)


def _pattern(name: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE)


def names_for(pedal_ids: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for pid in pedal_ids:
        out.extend(PEDAL_ALIASES.get(pid, ()))
    return out


def mentions(text: str, names: list[str] | tuple[str, ...]) -> list[str]:
    """Names from ``names`` that occur in ``text`` as whole words."""
    text = normalize_text(text)
    return [n for n in names if _pattern(n).search(text)]


def strip_allowed_names(text: str, allowed: list[str] | tuple[str, ...]) -> str:
    """``text`` with allowed pedal names removed (longest first), for digit checks."""
    text = normalize_text(text)
    for n in sorted(allowed, key=len, reverse=True):
        text = _pattern(n).sub(" ", text)
    return text


def holdout_names(holdout_pedal: str) -> list[str]:
    """Names that would leak the holdout pedal (its aliases minus names it shares with a
    training pedal)."""
    own = set(PEDAL_ALIASES.get(holdout_pedal, ()))
    shared = {n for pid, ns in PEDAL_ALIASES.items() if pid != holdout_pedal for n in ns}
    return sorted(own - shared)


def caption_violations(
    text: str, allowed_names: list[str] | tuple[str, ...], forbidden_names=()
) -> list[str]:
    """Reasons ``text`` breaks the caption rules (empty list = OK).

    ``allowed_names``: pedal names that may appear (digits inside them are fine).
    ``forbidden_names``: names that must not appear (e.g. :func:`holdout_names`).
    """
    out = [f"names forbidden pedal {n!r}" for n in mentions(text, list(forbidden_names))]
    rest = strip_allowed_names(text, allowed_names)
    if re.search(r"\d", rest):
        out.append("contains a digit")
    out += [f"contains unit {m.group(0)!r}" for m in _UNIT.finditer(rest)]
    out += [f"contains number word {m.group(0)!r}" for m in _NUMBER_WORD.finditer(rest)]
    return out


__all__ = [
    "NUMBER_WORDS",
    "OTHER_DEVICE_NAMES",
    "PEDAL_ALIASES",
    "caption_violations",
    "holdout_names",
    "mentions",
    "names_for",
    "normalize_text",
    "strip_allowed_names",
]
