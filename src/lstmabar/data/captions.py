"""Task 4.3 (wave 1c): template captions + LLM paraphrase (resolution R1, decisions 5-6).

BLIND TO THE HELD-OUT SET: this module and anyone editing it must never read the held-out
instruction set (human-written test instructions).
Templates and the paraphrase prompt were written from the KB, the board's knob semantics and
the descriptor definitions only (``tests/data/test_caption_hygiene.py`` enforces that this
file references no such path).

Contract: ``run(cfg, args)`` reads shard ``items.jsonl`` files and writes ``captions.jsonl``
next to them (``Caption`` records): templates from params/descriptors/KB vocab (no numbers;
pedal names only from :mod:`lstmabar.data.vocab`), then paraphrases via the Message Batches
API under ``captions.max_usd`` with the spend ledger at ``DataLayout.ledger``.
``--estimate`` prices a run without spending.

**Templates.** Each item gets ``captions.templates_per_item`` absolute instructions ("make
it ...", "give me ...", "I want ...", or a bare description), built deterministically from a
per-item RNG (``seed``, ``item_id``, template index). Word choices come from binned values;
the thresholds are chosen from the physical meaning of each quantity, not tuned on any test
data:

- *Drive amount* from ``drive.gain_db`` (0-60 dB): below 18 dB the waveshaper is only
  pushed into clipping by the loudest peaks of a -24...-6 dBFS input (light / edge of
  breakup); above 36 dB even the quietest inputs are driven well past the knee (heavy);
  in between, medium.
- *Treble* = ``drive.tone_db`` (post-clip tilt, pivot 1 kHz; drive on) + ``eq.high_db``
  (3 kHz shelf; EQ on). |sum| >= 4 dB is clearly audible on a full-range source, so it
  names the tone bright / dark; >= 9 dB adds an intensifier.
- *Mids* (``eq.mid_db``) and *bass* (``eq.low_db``): the same 4 dB / 9 dB rule for mids
  (scooped / mid-forward); 5 dB for the 120 Hz shelf (bassy / thin), which is less audible.
- *Compression*: estimated gain reduction at the render's input peak,
  ``max(0, input_peak_dbfs - threshold_db) * (1 - 1 / ratio)``: < 2 dB is inaudible (not
  mentioned), 2-6 dB light, 6-12 dB moderate, > 12 dB heavy (squashed).
- *Measured brightness* (wet ``centroid_hz``): below 1 kHz (the tilt pivot) dark, above
  3 kHz (the EQ high-shelf corner) bright. A knob-derived treble word wins; if the two
  disagree neither is used.
- *Harmonic character* (only when ``voiced_fraction >= analysis.min_voiced_fraction`` and
  the values are measured): added harmonic energy ``harmonic_energy_fraction`` wet minus
  dry (wet alone if the dry value is missing): < 0.03 adds no audible distortion character
  (smooth / clean), >= 0.15 (more than a square wave's ~0.16 relative to a sine) fuzzy /
  saturated; in between gritty if odd harmonics dominate (wet ``odd_even_db`` >= 6 dB,
  symmetric clipping) else warm / smooth (even-rich, asymmetric).
- *Noise* (same voicing rule): wet ``hnr_db`` < 6 dB (noise within 6 dB of the harmonics)
  with a drop of >= 6 dB from the dry clip: raspy / fizzy. Clean scenarios with HNR > 25 dB
  may be called pure.
- KB ``descriptors`` of the item's pedal (pedal items), dropping ones that contradict a
  chosen brightness / mids word.

Analysis values that are ``None``/NaN are never captioned. Pedal names appear in only
``captions.pedal_name_fraction`` of pedal items, only as :data:`vocab.PEDAL_ALIASES`, and the
holdout pedal (``render.holdout_pedal``) is never named outside ``test_unseen_pedal``. Every
caption passes :func:`vocab.item_caption_violations` (the rule shared with the leakage tests).

**Paraphrase.** Unique template texts not yet in the cache are sent through the Message
Batches API (``captions.model``, thinking disabled, effort low, JSON-schema output keyed by
a short local id, ``captions.per_request`` per request) with :data:`SYSTEM_PROMPT` (version
``captions.prompt_version``). Raw replies are cached by (template text, model, prompt
version) in ``DataLayout.llm_cache``; a reply that is malformed or fails the post-filter is
dropped and only the template remains (no paid retry).

**Spend guards (R1).** ``--estimate`` prices the run (input via the free ``count_tokens``
endpoint on the largest request x number of requests; output = captions x
``captions.output_tokens_per_caption`` or the pilot's measured value) and spends nothing.
Every submit is recorded in the ledger (``DataLayout.ledger``) *before* the batch is created;
a batch is in flight until its results are in, then it counts at its actual usage priced by
``captions.price_per_mtok``. The cap is hard: a submit is refused when spent + in flight +
new batch > ``captions.max_usd``, with in-flight and new batches counted at their **worst
case** (input estimate + every request at its ``max_tokens``); the estimate is shown and
logged alongside. The create call runs without SDK retries (a retry after a lost response
could create a second, unrecorded batch); an ambiguous failure leaves the entry in flight
for the owner to close (see :class:`SpendLedger`), a definite 4xx closes it at cost 0. One
run at a time (:class:`LedgerLock`). ``--pilot`` runs ``captions.pilot_items`` items and
records the measured output tokens per caption. Errored/expired requests are resubmitted at
most ``captions.max_resubmits`` times (counted across resumed runs); invalid requests are
cached as failed and auth/billing/permission errors are logged, neither is resubmitted. The
API key is read only from ``ANTHROPIC_API_KEY`` by the SDK client and never logged or
written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lstmabar.data.layout import DONE_MARKER, DataLayout
from lstmabar.data.schema import Caption, RenderItem, read_jsonl, write_jsonl
from lstmabar.data.vocab import (
    PEDAL_ALIASES,
    allowed_names,
    forbidden_names,
    item_caption_violations,
    mentions,
)

# --------------------------------------------------------------------------- KB vocabulary

FAMILY: dict[str, str] = {
    "ts808": "overdrive",
    "ds1": "distortion",
    "rat": "distortion",
    "big_muff": "fuzz",
    "fuzz_face_si": "fuzz",
    "fuzz_face_ge": "fuzz",
}
"""KB ``family`` per pedal (mirrors ``pedals/*.yaml``; checked against the KB in tests)."""

KB_DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "ts808": ("warm", "mid-forward", "smooth", "tight low end"),
    "ds1": ("hard-clipped", "scooped mids", "bright", "aggressive", "crunchy"),
    "rat": ("gritty", "hard-clipped", "mid-forward", "tight low end", "sweepable treble"),
    "big_muff": ("sustaining", "scooped mids", "thick", "saturated", "compressed"),
    "fuzz_face_si": ("bright", "harsh", "aggressive", "high-gain fuzz", "touch-sensitive"),
    "fuzz_face_ge": (
        "warm",
        "creamy",
        "touch-sensitive",
        "asymmetric",
        "cleans up with guitar volume",
    ),
}
"""KB ``descriptors`` per pedal (mirrors ``pedals/*.yaml``; checked against the KB in tests,
kept here so captioning needs no KB parsing)."""

# KB descriptors that read as a noun phrase ("with ...") rather than an adjective.
_KB_PHRASES = {
    "tight low end": "a tight low end",
    "scooped mids": "scooped mids",
    "sweepable treble": "a sweepable treble",
    "cleans up with guitar volume": "that cleans up with the guitar volume",
}
_KB_BRIGHT = {"bright", "harsh"}
_KB_DARK = {"warm", "creamy"}
_KB_SCOOP = {"scooped mids"}
_KB_MIDS = {"mid-forward"}
_KB_BASS = {"tight low end"}
_KB_SKIP = {"sweepable treble"}  # a property of the pedal, not of a sound
_KB_COMP = {"compressed", "sustaining"}
_KB_GAIN = {"saturated", "high-gain fuzz"}
_KB_HIGH_GAIN = {"aggressive", "harsh", "hard-clipped", "compressed", "sustaining", "thick"}
_KB_SMOOTH = {"smooth", "creamy"}
_KB_ROUGH = {"gritty", "harsh", "aggressive", "crunchy", "hard-clipped"}
_KB_ADJ = {"high-gain fuzz": "high-gain"}  # KB phrases that already contain a family noun
_NAME_SKIP = {"ProCo", "Pro Co"}  # brand-only aliases: valid names, but not used by templates


# --------------------------------------------------------------------------- thresholds

GAIN_LIGHT_DB = 18.0
GAIN_HEAVY_DB = 36.0
TREBLE_DB = 4.0
MIDS_DB = 4.0
BASS_DB = 5.0
STRONG_DB = 9.0
GR_MIN_DB, GR_LIGHT_DB, GR_HEAVY_DB = 2.0, 6.0, 12.0
CENTROID_DARK_HZ, CENTROID_BRIGHT_HZ = 1000.0, 3000.0
HEF_CLEAN, HEF_FUZZ = 0.03, 0.15
ODD_EVEN_GRITTY_DB = 6.0
HNR_NOISY_DB, HNR_DROP_DB, HNR_PURE_DB = 6.0, 6.0, 25.0

# Physical knob ranges of the default pedalboard (dsp/blocks.py ParamSpec; checked in tests).
KNOB_RANGES: dict[str, dict[str, tuple[float, float, str]]] = {
    "compressor": {
        "threshold_db": (-60.0, 0.0, "linear"),
        "ratio": (1.0, 20.0, "log"),
        "makeup_db": (0.0, 24.0, "linear"),
    },
    "drive": {
        "gain_db": (0.0, 60.0, "linear"),
        "tone_db": (-12.0, 12.0, "linear"),
        "softness": (0.0, 1.0, "linear"),
    },
    "eq": {
        "low_db": (-12.0, 12.0, "linear"),
        "mid_db": (-12.0, 12.0, "linear"),
        "high_db": (-12.0, 12.0, "linear"),
    },
}


def physical(board: Mapping[str, Mapping[str, float]], block: str, knob: str) -> float | None:
    """Physical value of a normalized knob (``None`` if missing or not finite)."""
    u = _num(board.get(block, {}).get(knob))
    if u is None:
        return None
    lo, hi, taper = KNOB_RANGES[block][knob]
    u = min(max(u, 0.0), 1.0)
    return lo * (hi / lo) ** u if taper == "log" else lo + (hi - lo) * u


def enabled(board: Mapping[str, Mapping[str, float]], block: str) -> bool:
    v = _num(board.get(block, {}).get("enabled"))
    return v is not None and v >= 0.5


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- attributes


@dataclass
class Attributes:
    """Binned, caption-ready facts about one item (words only, never numbers)."""

    scenario: str
    family: str | None = None  # overdrive | distortion | fuzz (drive scenarios)
    drive: str | None = None  # light | medium | heavy
    treble: str | None = None  # bright | dark (+ "very" if strong)
    treble_strong: bool = False
    mids: str | None = None  # scooped | forward
    mids_strong: bool = False
    bass: str | None = None  # bassy | thin
    comp: str | None = None  # light | moderate | heavy
    character: str | None = None  # clean | smooth | gritty | fuzzy
    noise: str | None = None  # noisy | pure
    kb: list[str] = field(default_factory=list)


def _treble_from_params(board, scenario: str) -> float | None:
    total, any_ = 0.0, False
    if enabled(board, "drive") and scenario in ("pedal", "drive"):
        t = physical(board, "drive", "tone_db")
        if t is not None:
            total, any_ = total + t, True
    if enabled(board, "eq"):
        h = physical(board, "eq", "high_db")
        if h is not None:
            total, any_ = total + h, True
    return total if any_ else None


def _voiced(item: RenderItem, min_voiced: float) -> bool:
    v = _num(item.voiced_fraction)
    return v is not None and v >= min_voiced


def attributes(item: RenderItem, min_voiced: float = 0.3) -> Attributes:
    """Bin an item's knobs and measured descriptors (see the module docstring)."""
    board = item.board_params
    sc = item.scenario
    a = Attributes(scenario=sc)
    driven = sc in ("pedal", "drive") and enabled(board, "drive")

    if sc == "pedal" and item.pedal_id in FAMILY:
        a.family = FAMILY[item.pedal_id]
    if driven:
        g = physical(board, "drive", "gain_db")
        if g is not None:
            a.drive = "light" if g < GAIN_LIGHT_DB else "heavy" if g > GAIN_HEAVY_DB else "medium"
        if a.family is None:  # free-board drive: name the family from gain and clip shape
            soft = physical(board, "drive", "softness")
            hard = soft is not None and soft < 0.35
            a.family = (
                "distortion" if a.drive == "heavy" or (hard and a.drive != "light") else "overdrive"
            )

    # treble: knobs first, measured centroid second, nothing if they disagree
    t = _treble_from_params(board, sc)
    p_word = None
    if t is not None and abs(t) >= TREBLE_DB:
        p_word = "bright" if t > 0 else "dark"
    c = _num(item.descriptors_wet.get("centroid_hz"))
    m_word = None
    if c is not None and c > 0:
        m_word = "dark" if c < CENTROID_DARK_HZ else "bright" if c > CENTROID_BRIGHT_HZ else None
    if p_word and m_word and p_word != m_word:
        a.treble = None
    else:
        a.treble = p_word or m_word
        a.treble_strong = bool(p_word) and t is not None and abs(t) >= STRONG_DB

    if enabled(board, "eq"):
        m = physical(board, "eq", "mid_db")
        if m is not None and abs(m) >= MIDS_DB:
            a.mids = "forward" if m > 0 else "scooped"
            a.mids_strong = abs(m) >= STRONG_DB
        lo = physical(board, "eq", "low_db")
        if lo is not None and abs(lo) >= BASS_DB:
            a.bass = "bassy" if lo > 0 else "thin"

    if enabled(board, "compressor"):
        thr = physical(board, "compressor", "threshold_db")
        ratio = physical(board, "compressor", "ratio")
        peak = _num(item.input_peak_dbfs)
        if thr is not None and ratio is not None and peak is not None:
            gr = max(0.0, peak - thr) * (1.0 - 1.0 / ratio)
            if gr >= GR_MIN_DB:
                a.comp = (
                    "light" if gr < GR_LIGHT_DB else "heavy" if gr > GR_HEAVY_DB else "moderate"
                )

    if _voiced(item, min_voiced):
        w, d = item.descriptors_wet, item.descriptors_dry
        hef_w = _num(w.get("harmonic_energy_fraction"))
        if driven and hef_w is not None:
            hef_d = _num(d.get("harmonic_energy_fraction"))
            added = hef_w - hef_d if hef_d is not None else hef_w
            if added < HEF_CLEAN:
                a.character = "smooth"
            elif added >= HEF_FUZZ:
                a.character = "fuzzy"
            else:
                oe = _num(w.get("odd_even_db"))
                if oe is not None:
                    a.character = "gritty" if oe >= ODD_EVEN_GRITTY_DB else "smooth"
        hnr_w = _num(w.get("hnr_db"))
        if hnr_w is not None:
            hnr_d = _num(d.get("hnr_db"))
            if driven and hnr_w < HNR_NOISY_DB and (hnr_d is None or hnr_d - hnr_w >= HNR_DROP_DB):
                a.noise = "noisy"
            elif not driven and hnr_w > HNR_PURE_DB:
                a.noise = "pure"

    if sc == "pedal" and item.pedal_id in KB_DESCRIPTORS:
        # KB words describe the pedal in general; drop the ones on an axis this item already
        # has a word for (it may contradict them) and the high-gain ones on light settings.
        drop = set(_KB_SKIP)
        if a.treble:
            drop |= _KB_BRIGHT | _KB_DARK
        if a.mids:
            drop |= _KB_SCOOP | _KB_MIDS
        if a.bass:
            drop |= _KB_BASS
        if a.comp:
            drop |= _KB_COMP
        if a.drive:
            drop |= _KB_GAIN  # the gain axis already has a word
        if a.drive == "light":
            drop |= _KB_HIGH_GAIN
        if a.character:  # the measured character word wins over the pedal's general one
            drop |= _KB_SMOOTH | _KB_ROUGH
        a.kb = [k for k in KB_DESCRIPTORS[item.pedal_id] if k not in drop]
    return a


# --------------------------------------------------------------------------- words

_DRIVE_ADJ = {
    "light": ["light", "mild", "gentle", "low-gain", "subtle", "slightly broken-up"],
    "medium": ["medium-gain", "crunchy", "solid", "punchy"],
    "heavy": ["heavy", "high-gain", "saturated", "thick", "full-on", "cranked"],
}
_DRIVE_MOD = {
    "light": ["with just a little grit", "right at the edge of breakup", "with a touch of dirt"],
    "medium": ["with a healthy amount of crunch", "with a good bit of bite"],
    "heavy": ["with tons of gain", "pushed really hard", "with lots of saturation"],
}
_FAMILY_NOUN = {
    "overdrive": ["overdrive", "overdriven tone", "drive tone", "overdrive sound"],
    "distortion": ["distortion", "distorted tone", "distortion sound", "dirty tone"],
    "fuzz": ["fuzz", "fuzz tone", "fuzzy sound", "fuzz sound"],
}
_FAMILY_ADJ = {"overdrive": "overdriven", "distortion": "distorted", "fuzz": "fuzzy"}
_CLEAN_NOUN = ["clean tone", "clean sound", "clean guitar tone", "clean signal"]
_TREBLE_ADJ = {
    "bright": ["bright", "trebly", "sparkly", "crisp", "cutting"],
    "dark": ["dark", "mellow", "warm", "muted", "rounded"],
}
_TREBLE_MOD = {
    "bright": ["with lots of top end", "with sparkling highs", "with a crisp top end"],
    "dark": ["with the highs rolled off", "with a soft top end", "with no harsh treble"],
}
_MIDS_ADJ = {"scooped": ["scooped", "mid-scooped"], "forward": ["mid-forward", "mid-heavy"]}
_MIDS_MOD = {
    "scooped": ["with scooped mids", "with the mids pulled back", "with hollowed-out mids"],
    "forward": ["with pushed mids", "with a strong midrange", "with a mid hump"],
}
_BASS_ADJ = {"bassy": ["bassy", "fat", "full-bodied"], "thin": ["thin", "lean", "tight"]}
_BASS_MOD = {
    "bassy": ["with a big low end", "with plenty of bass", "with a fat bottom end"],
    "thin": ["with a lean low end", "with the bass cut back", "without much low end"],
}
_COMP_ADJ = {
    "light": ["lightly compressed", "gently compressed"],
    "moderate": ["compressed", "evened-out"],
    "heavy": ["heavily compressed", "squashed", "squeezed"],
}
_COMP_MOD = {
    "light": ["with a touch of compression", "with gentle compression"],
    "moderate": ["with some compression", "with the dynamics evened out"],
    "heavy": ["with heavy compression", "with the dynamics squashed flat", "with lots of squash"],
}
_CHAR_ADJ = {
    "smooth": ["smooth", "creamy", "round"],
    "gritty": ["gritty", "raw", "biting", "edgy"],
    "fuzzy": ["fuzzy", "woolly", "buzzy", "furry"],
    "clean": ["clean", "clear"],
}
_NOISE_ADJ = {"noisy": ["raspy", "fizzy", "hairy"], "pure": ["pure", "pristine", "glassy"]}
_NOISE_MOD = {
    "noisy": ["with some fizz on top", "with a raspy edge", "with a bit of hiss and fizz"],
    "pure": ["with no noise", "nice and pristine"],
}
_INTENSIFIER = ["very", "really", "super"]

_BYPASS = [
    "keep it clean, no effects",
    "just the dry guitar, nothing on it",
    "leave the tone untouched",
    "bypass everything, I want the plain guitar",
    "no pedals, just the straight signal",
    "give me the plain unprocessed guitar",
    "I want the raw clean signal with no processing",
    "turn all the effects off",
    "the natural sound of the guitar, nothing added",
    "make it completely clean and dry",
]

# Frames. {np}: a noun phrase ("a warm overdrive with scooped mids"); {adjs}: an adjective
# list ("warm and overdriven"); {np_bare}: the noun phrase without the article.
_NP_FRAMES = [
    "make it {np}",
    "give me {np}",
    "I want {np}",
    "I'd like {np}",
    "{np_cap}",
    "{np_cap}, please",
    "go for {np}",
    "turn this into {np}",
    "dial in {np}",
    "can I get {np}?",
    "I'm after {np}",
    "something like {np}",
    "set me up with {np}",
    "let's hear {np}",
    "I'm looking for {np}",
    "how about {np}",
    "I need {np} here",
    "switch to {np}",
    "give this part {np}",
    "this riff needs {np}",
    "give my guitar {np}",
    "I'm going for {np}",
    "try {np}",
]
_ADJ_FRAMES = [
    "make it {adjs}",
    "make it sound {adjs}",
    "I want it {adjs}",
    "make my guitar sound {adjs}",
    "I want the tone {adjs}",
    "it should sound {adjs}",
    "make the guitar {adjs}",
    "{adjs_cap}, please",
    "I'd like it {adjs}",
    "get the tone {adjs}",
]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _join_adjs(adjs: Sequence[str]) -> str:
    if len(adjs) <= 1:
        return "".join(adjs)
    if len(adjs) == 2:
        return f"{adjs[0]} and {adjs[1]}"
    return ", ".join(adjs[:-1]) + f" and {adjs[-1]}"


@dataclass
class _Parts:
    adjs: list[str]  # pre-noun adjectives
    head: str  # noun ("overdrive", "Tube Screamer overdrive", "clean tone")
    mods: list[str]  # trailing "with ..." phrases
    pred_adjs: list[str]  # adjectives usable predicatively ("make it ...")


def _pick(rng: random.Random, xs: Sequence[str]) -> str:
    return xs[rng.randrange(len(xs))]


def _parts(a: Attributes, rng: random.Random, name: str | None) -> _Parts:
    """Assign each attribute to adjective or modifier form, randomly but deterministically."""
    adjs: list[str] = []
    mods: list[str] = []
    pred: list[str] = []

    def put(adj_choices, mod_choices, p_adj=0.6):
        if mod_choices and (not adj_choices or rng.random() > p_adj):
            mods.append(_pick(rng, mod_choices))
        else:
            w = _pick(rng, adj_choices)
            adjs.append(w)
            pred.append(w)

    driven = a.family is not None
    if a.drive:
        put(_DRIVE_ADJ[a.drive], _DRIVE_MOD[a.drive], 0.7)
    if a.character and driven and rng.random() < 0.8:
        put(_CHAR_ADJ[a.character], [], 1.0)
    if a.treble:
        choices = _TREBLE_ADJ[a.treble]
        if a.treble_strong and rng.random() < 0.5:
            choices = [f"{_pick(rng, _INTENSIFIER)} {w}" for w in choices]
        put(choices, _TREBLE_MOD[a.treble])
    if a.mids:
        put(_MIDS_ADJ[a.mids], _MIDS_MOD[a.mids], 0.4)
    if a.bass:
        put(_BASS_ADJ[a.bass], _BASS_MOD[a.bass], 0.5)
    if a.comp:
        put(_COMP_ADJ[a.comp], _COMP_MOD[a.comp], 0.5)
    if a.noise and rng.random() < 0.7:
        put(_NOISE_ADJ[a.noise], _NOISE_MOD[a.noise], 0.5)
    if a.kb:
        for d in rng.sample(a.kb, k=min(len(a.kb), rng.choice((0, 1, 1, 2)))):
            if d in _KB_PHRASES:
                phrase = _KB_PHRASES[d]
                mods.append(phrase if phrase.startswith("that") else f"with {phrase}")
            else:
                adjs.append(_KB_ADJ.get(d, d))
                pred.append(_KB_ADJ.get(d, d))

    # dedupe words already present (e.g. "warm" from treble and KB), keep order
    adjs = list(dict.fromkeys(adjs))
    pred = list(dict.fromkeys(pred))
    # keep phrases readable: at most three adjectives and two modifiers
    while len(adjs) > 3:
        w = adjs.pop(rng.randrange(len(adjs)))
        if w in pred and len(pred) > 3:
            pred.remove(w)
    mods = mods[:2] if len(mods) <= 2 else rng.sample(mods, 2)

    if driven:
        noun = _pick(rng, _FAMILY_NOUN[a.family])
        if name:
            style = rng.random()
            if a.family in name.lower():  # "Fuzz Face": no "Fuzz Face fuzz"
                head = name if style < 0.7 else f"{name}-style {noun}"
            elif style < 0.4:
                head = f"{name} {a.family}"
            elif style < 0.7:
                head = f"{name}-style {a.family}"
            else:
                head = noun
                mods.insert(0, f"like a {name}")
        else:
            head = noun
        pred.append(_FAMILY_ADJ[a.family])
    else:
        head = _pick(rng, _CLEAN_NOUN)
        if "clean" not in pred:
            pred.append("clean")
    pred = list(dict.fromkeys(pred))
    return _Parts(adjs=adjs, head=head, mods=mods, pred_adjs=pred)


def _np(parts: _Parts) -> str:
    adj = ", ".join(parts.adjs)
    core = f"{adj} {parts.head}" if adj else parts.head
    np_ = f"{_article(core)} {core}"
    mods = list(parts.mods)
    if mods and mods[0].startswith("like "):
        np_ += " " + mods.pop(0)
    if mods:
        # "with X and Y", not "with X and with Y"
        rest = [m.removeprefix("with ") if m.startswith("with ") else m for m in mods[1:]]
        np_ += " " + " and ".join([mods[0], *rest])
    return np_


def _render_template(a: Attributes, rng: random.Random, name: str | None) -> tuple[str, str]:
    """``(text, template_id)``. ``template_id`` names the frame family (for analysis)."""
    if a.scenario == "bypass":
        k = rng.randrange(len(_BYPASS))
        return _BYPASS[k], f"bypass-{k:02d}"
    parts = _parts(a, rng, name)
    # adjective frames carry no noun, so they are used only when no pedal is named
    use_adj = name is None and len(parts.pred_adjs) >= 2 and rng.random() < 0.3
    if use_adj:
        k = rng.randrange(len(_ADJ_FRAMES))
        preds = (
            parts.pred_adjs[:3]
            if len(parts.pred_adjs) <= 3
            else (rng.sample(parts.pred_adjs[:-1], 2) + parts.pred_adjs[-1:])
        )
        adjs = _join_adjs(preds)
        text = _ADJ_FRAMES[k].format(adjs=adjs, adjs_cap=_cap(adjs))
        return text, f"adj-{k:02d}"
    k = rng.randrange(len(_NP_FRAMES))
    np_ = _np(parts)
    text = _NP_FRAMES[k].format(np=np_, np_cap=_cap(np_))
    return text, f"np-{k:02d}"


# --------------------------------------------------------------------------- rules


def holdout_pedal(cfg) -> str | None:
    return cfg.render.get("holdout_pedal") if "render" in cfg else None


def name_rules(item: RenderItem, named: bool, holdout: str | None) -> tuple[list[str], list[str]]:
    """``(allowed, forbidden)`` pedal names for an item's captions (the shared rule in
    :func:`vocab.allowed_names` / :func:`vocab.forbidden_names`)."""
    args = (item.pedal_id, named, holdout, item.split)
    return allowed_names(*args), forbidden_names(*args)


def check_caption(text: str, item: RenderItem, named: bool, holdout: str | None) -> list[str]:
    """Post-filter: :func:`vocab.item_caption_violations` for this item."""
    return item_caption_violations(text, item.pedal_id, named, holdout, item.split)


def _item_rng(item: RenderItem, salt: str) -> random.Random:
    h = hashlib.sha256(f"{item.seed}:{item.item_id}:{salt}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def names_pedal(item: RenderItem, fraction: float, holdout: str | None) -> bool:
    """Whether an item's captions name its pedal (deterministic per item)."""
    if item.scenario != "pedal" or item.pedal_id not in PEDAL_ALIASES:
        return False
    if holdout and item.pedal_id == holdout and item.split != "test_unseen_pedal":
        return False
    return _item_rng(item, "name").random() < fraction


def template_captions(
    item: RenderItem,
    n: int = 1,
    pedal_name_fraction: float = 0.25,
    holdout: str | None = None,
    min_voiced: float = 0.3,
) -> list[Caption]:
    """``n`` deterministic template captions (``<item_id>-c0`` ...) for one item."""
    named = names_pedal(item, pedal_name_fraction, holdout)
    allowed, _ = name_rules(item, named, holdout)
    attrs = attributes(item, min_voiced)
    out: list[Caption] = []
    for k in range(n):
        rng = _item_rng(item, f"template-{k}")
        names = [n for n in allowed if n not in _NAME_SKIP]
        name = _pick(rng, names) if named and names else None
        text, tid = _render_template(attrs, rng, name)
        text = re.sub(r"\s+", " ", text).strip()
        bad = check_caption(text, item, named, holdout)
        if bad:  # a template bug, never data-dependent: fail loudly
            raise ValueError(f"template caption {text!r} for {item.item_id} breaks rules: {bad}")
        out.append(
            Caption(
                caption_id=f"{item.item_id}-c{k}",
                item_id=item.item_id,
                kind="template",
                text=text,
                template_id=tid,
                names_pedal=bool(name),
            )
        )
    return out


# --------------------------------------------------------------------------- synthetic items


def synthetic_items(n: int, holdout: str | None = "rat", seed: int = 0) -> list[RenderItem]:
    """Random but plausible ``RenderItem``s (no audio) for tests, examples and estimates.

    Scenario mix follows R5; descriptors are drawn from broad ranges and are sometimes
    missing (``None``), voiced fractions are sometimes low.
    """
    rng = random.Random(seed)
    pedals = [p for p in PEDAL_ALIASES if p != holdout]
    free = {"bypass": 0.05, "eq": 0.25, "comp": 0.15, "comp_eq": 0.15, "drive": 0.40}
    items: list[RenderItem] = []
    for i in range(n):
        split = rng.choice(["train"] * 6 + ["val", "test", "test_unseen_pedal"])
        if split == "test_unseen_pedal" and holdout:
            scenario, pedal = "pedal", holdout
        elif rng.random() < 0.7:
            scenario, pedal = "pedal", rng.choice(pedals)
        else:
            scenario = rng.choices(list(free), weights=list(free.values()))[0]
            pedal = None
        on = {
            "compressor": scenario in ("comp", "comp_eq")
            or (scenario == "drive" and rng.random() < 0.5),
            "drive": scenario in ("pedal", "drive"),
            "eq": scenario in ("pedal", "eq", "comp_eq")
            or (scenario == "drive" and rng.random() < 0.5),
        }
        board = {
            b: {k: rng.random() for k in knobs} | {"enabled": float(on[b])}
            for b, knobs in KNOB_RANGES.items()
        }

        def maybe(lo, hi, p_none=0.1):
            return None if rng.random() < p_none else rng.uniform(lo, hi)

        desc = {
            "centroid_hz": maybe(400.0, 5000.0),
            "rolloff_hz": maybe(1000.0, 9000.0),
            "crest_factor_db": maybe(3.0, 20.0),
            "rms_db": maybe(-40.0, -6.0),
            "f0_hz": maybe(80.0, 700.0, 0.3),
            "hnr_db": maybe(-5.0, 40.0, 0.3),
            "odd_even_db": maybe(-15.0, 25.0, 0.3),
            "slope_db_per_oct": maybe(-20.0, -2.0, 0.3),
            "harmonic_energy_fraction": maybe(0.0, 0.5, 0.3),
        }
        dry = {k: (None if v is None else v * rng.uniform(0.5, 1.0)) for k, v in desc.items()}
        items.append(
            RenderItem(
                item_id=f"syn-{i:06d}-r0",
                clip_id=f"syn-{i:06d}",
                split=split,
                seed=rng.randrange(2**31),
                scenario=scenario,
                engine="greybox",
                pedal_id=pedal,
                pedal_knobs={"gain": rng.random()} if pedal else None,
                board_params=board,
                board_params_derived=None,
                label_source="derived" if pedal else "free",
                clamped=False,
                clamp_notes=[],
                input_peak_dbfs=rng.uniform(-24.0, -6.0),
                wet_gain_db=0.0,
                wet_path=f"v1/wet/shard-00000/syn-{i:06d}-r0.flac",
                voiced_fraction=None if rng.random() < 0.1 else rng.random(),
                descriptors_dry=dry,
                descriptors_wet=desc,
            )
        )
    return items


# --------------------------------------------------------------------------- paraphrase

SYSTEM_PROMPT = """\
You reword short requests that guitarists type into a tone-shaping tool. Each input \
describes the guitar sound someone wants. For every input, write one rewording that sounds \
like a real player asking for that sound in their own words.

Rules:
- Keep the meaning: the same amount of gain, the same brightness, mids, low end, compression \
and character. Do not add qualities that are not in the input, and do not drop any.
- Describe the target sound (an absolute request), not a change relative to the current \
sound.
- Never use digits, number words, units or knob settings.
- If the input names a pedal, keep that exact name or leave it out. Never add or change a \
pedal, brand, amp or artist name.
- Keep it short: one phrase or sentence, plain text, no quotes, lists or emoji.
- Vary the wording and sentence structure; do not just shuffle the input's words.

Return one entry per input id."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "captions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["captions"],
    "additionalProperties": False,
}


def _local_id(i: int) -> str:
    """Short request-local key (letters only, so ids carry no digits into the prompt)."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(ord("a") + r) + s
    return s


def max_tokens_for(n_captions: int) -> int:
    """Per-request ``max_tokens``: ~2x the expected ~45 output tokens per caption plus JSON
    overhead. It bounds the worst-case cost that the cap check uses."""
    return 256 + 100 * n_captions


def request_params(texts: Sequence[str], cc) -> tuple[dict[str, Any], dict[str, str]]:
    """Message params for one request and its ``{local id: template text}`` map."""
    ids = {_local_id(i): t for i, t in enumerate(texts)}
    payload = json.dumps(
        {"captions": [{"id": k, "text": t} for k, t in ids.items()]}, ensure_ascii=False
    )
    params = {
        "model": str(cc.model),
        "max_tokens": max_tokens_for(len(texts)),
        "thinking": {"type": "disabled"},
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": payload}],
    }
    return params, ids


def chunk(texts: Sequence[str], size: int) -> list[list[str]]:
    return [list(texts[i : i + size]) for i in range(0, len(texts), size)]


PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + "\n" + json.dumps(OUTPUT_SCHEMA, sort_keys=True)).encode()
).hexdigest()[:16]
"""Fingerprint of the prompt + output schema. Part of every cache key, so editing either
never reuses stale replies; ``tests/data/test_captions.py`` pins it to ``prompt_version``
"p1" so an edit without a version bump fails the tests."""


def cache_key(text: str, model: str, prompt_version: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt_version}\n{PROMPT_HASH}\n{text}".encode()).hexdigest()


class ParaphraseCache:
    """Append-only JSONL of raw paraphrase replies keyed by (template, model, prompt version,
    prompt hash).

    ``text`` is ``None`` for a malformed/missing reply or a non-retryable request error
    (cached so it is never paid again).
    """

    def __init__(self, path: Path, model: str, prompt_version: str) -> None:
        self.path = Path(path)
        self.model = model
        self.prompt_version = prompt_version
        self._data: dict[str, str | None] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self._data[r["key"]] = r.get("text")

    def key(self, template: str) -> str:
        return cache_key(template, self.model, self.prompt_version)

    def __contains__(self, template: str) -> bool:
        return self.key(template) in self._data

    def get(self, template: str) -> str | None:
        return self._data.get(self.key(template))

    def put_many(self, entries: Mapping[str, str | None], status: str | None = None) -> None:
        if not entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            for template, text in entries.items():
                k = self.key(template)
                self._data[k] = text
                rec = {
                    "key": k,
                    "model": self.model,
                    "prompt_version": self.prompt_version,
                    "prompt_hash": PROMPT_HASH,
                    "template": template,
                    "text": text,
                    "status": status or ("ok" if text is not None else "malformed"),
                }
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def usd(input_tokens: float, output_tokens: float, price: Mapping[str, float]) -> float:
    return (input_tokens * float(price["input"]) + output_tokens * float(price["output"])) / 1e6


class SpendLedger:
    """JSONL spend ledger. Events per local ``ref``: ``submitted`` (estimate and worst case,
    written before the batch is created), ``created`` (batch id), ``ended`` (actual usage and
    cost; cost 0 for a create call the API definitely rejected).

    A ref without ``ended`` is in flight. It counts at its estimate for display and at its
    worst case (every request using its full ``max_tokens``) for the cap check, also when the
    process died before the batch id was recorded or the create call failed ambiguously.

    Clearing a stuck in-flight entry (owner action): look the batch up in the Console (or
    ``client.messages.batches.list()``); if it exists, append its real cost, e.g.
    ``{"event": "ended", "ref": "<ref>", "cost_usd": <real cost>, "note": "manual"}``; if no
    batch was created, append the same with ``"cost_usd": 0``. Never delete ledger lines.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def state(self) -> dict[str, dict[str, Any]]:
        refs: dict[str, dict[str, Any]] = {}
        for e in self.events():
            refs.setdefault(e["ref"], {}).update(e)
            refs[e["ref"]].setdefault("events", []).append(e["event"])
        return refs

    def spent(self) -> float:
        return sum(float(r.get("cost_usd", 0.0)) for r in self.state().values() if _ended(r))

    def in_flight(self) -> float:
        """In-flight batches at their estimate (for display)."""
        return sum(
            float(r.get("estimate_usd", 0.0)) for r in self.state().values() if not _ended(r)
        )

    def in_flight_worst(self) -> float:
        """In-flight batches at their worst case (for the cap check)."""
        return sum(
            float(r.get("worst_usd", r.get("estimate_usd", 0.0)))
            for r in self.state().values()
            if not _ended(r)
        )

    def pending(self) -> list[dict[str, Any]]:
        return [r for r in self.state().values() if not _ended(r)]

    def append(self, event: str, ref: str, **fields: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"event": event, "ref": ref, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")} | fields
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def _ended(r: Mapping[str, Any]) -> bool:
    return "ended" in r.get("events", ())


class BudgetExceeded(RuntimeError):
    pass


class SubmitUncertain(RuntimeError):
    """The batch create call failed in a way that may still have created a batch."""


class LedgerLocked(RuntimeError):
    pass


class LedgerLock:
    """One paraphrase run at a time: an exclusive lock file next to the ledger."""

    def __init__(self, ledger_path: Path, label: str = "captions") -> None:
        self.path = Path(ledger_path).with_suffix(".lock")
        self.label = label

    def __enter__(self) -> LedgerLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise LedgerLocked(
                f"{self.path.name} exists: another {self.label} run is active (or one crashed; "
                "delete the lock file once no run is active)"
            ) from e
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid {os.getpid()} since {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        return self

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)


@dataclass
class Estimate:
    n_captions: int
    n_requests: int
    input_tokens: int
    output_tokens: int
    usd: float
    input_source: str  # "count_tokens" | "approx"
    output_per_caption: float
    output_source: str  # "config" | "pilot"
    max_output_tokens: int = 0  # sum of the requests' max_tokens
    worst_usd: float = 0.0  # input estimate + every request at its max_tokens

    def lines(self, ledger: SpendLedger, max_usd: float) -> list[str]:
        spent, flight, worst = ledger.spent(), ledger.in_flight(), ledger.in_flight_worst()
        return [
            f"paraphrases to request: {self.n_captions} in {self.n_requests} requests",
            f"input tokens: {self.input_tokens} ({self.input_source})",
            f"output tokens: {self.output_tokens} "
            f"({self.output_per_caption:g} per caption, {self.output_source})",
            f"estimated cost: ${self.usd:.4f}",
            f"worst case (all requests at max_tokens = {self.max_output_tokens} output tokens): "
            f"${self.worst_usd:.4f}",
            f"ledger: spent ${spent:.4f}, in flight ${flight:.4f} (worst ${worst:.4f}); "
            f"cap ${max_usd:.2f}; after this run ${spent + flight + self.usd:.4f} "
            f"(cap check uses ${spent + worst + self.worst_usd:.4f})",
        ]


def _pilot_path(layout: DataLayout, cc) -> Path:
    return layout.llm_cache / f"pilot-{cc.model}-{cc.prompt_version}.json"


def output_per_caption(layout: DataLayout, cc) -> tuple[float, str]:
    p = _pilot_path(layout, cc)
    if p.exists():
        return float(
            json.loads(p.read_text(encoding="utf-8"))["output_tokens_per_caption"]
        ), "pilot"
    return float(cc.output_tokens_per_caption), "config"


def _approx_tokens(params: Mapping[str, Any]) -> int:
    """Rough input-token count without the API (~3.5 characters per token, plus the schema)."""
    chars = len(params["system"]) + len(json.dumps(params["messages"]))
    chars += len(json.dumps(params["output_config"]))
    return math.ceil(chars / 3.5) + 50


def count_input_tokens(client, params: Mapping[str, Any]) -> int:
    """Input tokens of one request via the free ``count_tokens`` endpoint."""
    r = client.messages.count_tokens(
        model=params["model"],
        system=params["system"],
        messages=params["messages"],
        thinking=params["thinking"],
        output_config=params["output_config"],
    )
    return int(r.input_tokens)


def estimate(texts: Sequence[str], cc, layout: DataLayout, client=None) -> Estimate:
    groups = chunk(texts, int(cc.per_request))
    n_cap = len(texts)
    per_cap, out_src = output_per_caption(layout, cc)
    if not groups:
        return Estimate(0, 0, 0, 0, 0.0, "none", per_cap, out_src)
    biggest = max(groups, key=lambda g: sum(len(t) for t in g))
    params, _ = request_params(biggest, cc)
    per_req, src = _approx_tokens(params), "approx"
    if client is not None:
        try:
            per_req, src = count_input_tokens(client, params), "count_tokens"
        except Exception as e:  # noqa: BLE001 - any API/network failure: fall back
            _log(f"count_tokens failed ({type(e).__name__}); input tokens approximated")
    tin = per_req * len(groups)
    tout = math.ceil(n_cap * per_cap)
    max_out = sum(max_tokens_for(len(g)) for g in groups)
    price = cc.price_per_mtok
    return Estimate(
        n_cap,
        len(groups),
        tin,
        tout,
        usd(tin, tout, price),
        src,
        per_cap,
        out_src,
        max_out,
        usd(tin, max_out, price),
    )


def _batch_file(layout: DataLayout, ref: str) -> Path:
    return layout.llm_cache / "batches" / f"{ref}.json"


def _parse_reply(message) -> dict[str, str] | None:
    """``{local id: text}`` from a succeeded message, ``None`` if malformed."""
    if getattr(message, "stop_reason", None) not in (None, "end_turn"):
        return None
    text = next((b.text for b in message.content if getattr(b, "type", None) == "text"), None)
    if text is None:
        return None
    try:
        data = json.loads(text)
        out = {}
        for e in data["captions"]:
            if isinstance(e.get("id"), str) and isinstance(e.get("text"), str):
                out[e["id"]] = e["text"].strip()
        return out
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def _usage_tokens(message) -> tuple[int, int]:
    u = getattr(message, "usage", None)
    if u is None:
        return 0, 0
    tin = sum(
        int(getattr(u, k, 0) or 0)
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return tin, int(getattr(u, "output_tokens", 0) or 0)


@dataclass
class BatchOutcome:
    replies: dict[str, str | None]  # template -> raw paraphrase (None = malformed)
    retry: list[str]  # templates whose requests errored/expired/canceled (retryable)
    invalid: list[str]  # invalid_request errors: cached as failed, never resubmitted
    failed: list[str]  # other non-retryable errors (auth, billing, ...): logged, not cached
    input_tokens: int
    output_tokens: int


_INVALID = {"invalid_request_error", "invalid_request"}
_NON_RETRYABLE = {
    "authentication_error",
    "billing_error",
    "permission_error",
    "not_found_error",
}


def _error_type(result) -> str:
    """Error type of an errored batch result (``result.error`` is an ``ErrorResponse`` whose
    ``.error.type`` is e.g. ``"invalid_request_error"``)."""
    err = getattr(result, "error", None)
    inner = getattr(err, "error", err)
    return str(getattr(inner, "type", "") or "")


def collect(client, batch_id: str, mapping: Mapping[str, Mapping[str, str]]) -> BatchOutcome:
    """Read a finished batch's results (any order) keyed by ``custom_id``."""
    out = BatchOutcome({}, [], [], [], 0, 0)
    seen: set[str] = set()
    for res in client.messages.batches.results(batch_id):
        cid = res.custom_id
        ids = mapping.get(cid)
        if ids is None:
            continue
        seen.add(cid)
        r = res.result
        etype = _error_type(r) if r.type == "errored" else ""
        if r.type == "succeeded":
            tin, tout = _usage_tokens(r.message)
            out.input_tokens += tin
            out.output_tokens += tout
            parsed = _parse_reply(r.message)
            for lid, template in ids.items():
                out.replies[template] = (parsed or {}).get(lid) or None
        elif etype in _INVALID:
            out.invalid.extend(ids.values())
        elif etype in _NON_RETRYABLE:
            out.failed.extend(ids.values())
        else:  # errored (server side, overloaded, rate limit), expired, canceled
            out.retry.extend(ids.values())
    for cid, ids in mapping.items():  # results never returned: treat as retryable
        if cid not in seen:
            out.retry.extend(ids.values())
    return out


def _log(msg: str) -> None:
    print(f"[captions] {msg}", file=sys.stderr, flush=True)


def _definitely_rejected(exc: Exception) -> bool:
    """True if a create call failed with a definite client error (4xx other than a timeout or
    conflict): the API did not create a batch. Connection errors, timeouts and 5xx are
    ambiguous (the batch may exist)."""
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 409)


class Paraphraser:
    """Submits uncached templates under the spend cap and fills the cache."""

    def __init__(
        self,
        cc,
        layout: DataLayout,
        client,
        poll_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cc = cc
        self.layout = layout
        self.client = client
        self.poll_seconds = poll_seconds
        self.sleep = sleep
        self.ledger = SpendLedger(layout.ledger)
        self.cache = ParaphraseCache(
            layout.llm_cache / "paraphrases.jsonl", str(cc.model), str(cc.prompt_version)
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.captions_returned = 0

    # -- one batch --------------------------------------------------------------------
    def submit(self, texts: Sequence[str], attempt: int = 0) -> str:
        """Price, check the cap (worst case), record, create. Returns the ledger ref."""
        est = estimate(texts, self.cc, self.layout, self.client)
        spent, flight = self.ledger.spent(), self.ledger.in_flight_worst()
        cap = float(self.cc.max_usd)
        if spent + flight + est.worst_usd > cap:
            raise BudgetExceeded(
                f"refusing to submit: spent ${spent:.4f} + in flight (worst case) "
                f"${flight:.4f} + this batch's worst case ${est.worst_usd:.4f} "
                f"(estimate ${est.usd:.4f}) > cap ${cap:.2f}"
            )
        ref = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        requests, mapping = [], {}
        for i, group in enumerate(chunk(texts, int(self.cc.per_request))):
            params, ids = request_params(group, self.cc)
            cid = f"a{attempt}-{i:05d}"
            requests.append({"custom_id": cid, "params": params})
            mapping[cid] = ids
        bf = _batch_file(self.layout, ref)
        bf.parent.mkdir(parents=True, exist_ok=True)
        bf.write_text(json.dumps({"ref": ref, "attempt": attempt, "mapping": mapping}), "utf-8")
        self.ledger.append(
            "submitted",
            ref,
            estimate_usd=est.usd,
            worst_usd=est.worst_usd,
            n_requests=est.n_requests,
            n_captions=est.n_captions,
            est_input_tokens=est.input_tokens,
            est_output_tokens=est.output_tokens,
            max_output_tokens=est.max_output_tokens,
            model=str(self.cc.model),
            prompt_version=str(self.cc.prompt_version),
            prompt_hash=PROMPT_HASH,
            attempt=attempt,
        )
        try:
            # no SDK-level retries: a retried create after a lost response would make a
            # second (paid) batch that the ledger does not know about
            batch = self.client.with_options(max_retries=0).messages.batches.create(
                requests=requests
            )
        except Exception as e:
            if _definitely_rejected(e):
                self.ledger.append(
                    "ended",
                    ref,
                    cost_usd=0.0,
                    status="rejected",
                    error=type(e).__name__,
                    status_code=getattr(e, "status_code", None),
                )
                raise
            _log(
                f"batch create for ledger ref {ref} failed ambiguously ({type(e).__name__}); "
                "it stays in flight at its worst case. Check the Console or "
                "client.messages.batches.list() and close the entry by hand (SpendLedger)."
            )
            raise SubmitUncertain(ref) from e
        self.ledger.append("created", ref, batch_id=batch.id)
        _log(f"submitted batch {batch.id}: {est.n_captions} captions, est ${est.usd:.4f}")
        return ref

    def wait_and_collect(self, ref: str) -> BatchOutcome:
        st = self.ledger.state()[ref]
        info = json.loads(_batch_file(self.layout, ref).read_text("utf-8"))
        batch_id = st["batch_id"]
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                break
            self.sleep(self.poll_seconds)
        out = collect(self.client, batch_id, info["mapping"])
        cost = usd(out.input_tokens, out.output_tokens, self.cc.price_per_mtok)
        # cache first: a crash after this re-collects the batch but never pays twice
        self.cache.put_many(out.replies)
        self.cache.put_many(dict.fromkeys(out.invalid), status="invalid_request")
        if out.failed:
            _log(f"{len(out.failed)} paraphrases failed with non-retryable errors (not cached)")
        self.ledger.append(
            "ended",
            ref,
            batch_id=batch_id,
            input_tokens=out.input_tokens,
            output_tokens=out.output_tokens,
            cost_usd=cost,
            n_replies=len(out.replies),
            n_retry=len(out.retry),
            n_invalid=len(out.invalid),
            n_failed=len(out.failed),
            status="ended",
        )
        self.input_tokens += out.input_tokens
        self.output_tokens += out.output_tokens
        self.captions_returned += len(out.replies)
        _log(f"batch {batch_id} ended: {len(out.replies)} replies, ${cost:.4f}")
        return out

    # -- full run ---------------------------------------------------------------------
    def resume_pending(self) -> dict[str, int]:
        """Collect batches left in flight by an earlier run.

        Returns ``{template: attempt}`` for retryable failures, so resubmission continues
        the count instead of restarting it.
        """
        retry: dict[str, int] = {}
        for r in self.ledger.pending():
            if "batch_id" not in r or not _batch_file(self.layout, r["ref"]).exists():
                _log(
                    f"ledger ref {r['ref']} has no batch id (process died before creation, or "
                    "the create call failed ambiguously); it stays counted at its worst case "
                    "until closed by hand (see SpendLedger)"
                )
                continue
            attempt = int(r.get("attempt", 0))
            for t in self.wait_and_collect(r["ref"]).retry:
                retry[t] = max(retry.get(t, 0), attempt)
        return retry

    def _loop(self, todo: list[str], attempt: int, skip: set[str]) -> None:
        while todo:
            ref = self.submit(todo, attempt)
            out = self.wait_and_collect(ref)
            skip.update(out.failed)
            todo = [t for t in dict.fromkeys(out.retry) if t not in self.cache]
            if todo and attempt >= int(self.cc.max_resubmits):
                _log(f"{len(todo)} paraphrases still failing after {attempt} resubmits; skipped")
                skip.update(todo)
                return
            attempt += 1

    def run(self, templates: Iterable[str]) -> None:
        with LedgerLock(self.ledger.path):
            wanted = list(dict.fromkeys(templates))
            resumed = self.resume_pending()
            skip: set[str] = set()
            max_re = int(self.cc.max_resubmits)
            # resumed retries continue their resubmit count
            by_attempt: dict[int, list[str]] = {}
            for t in wanted:
                if t in resumed and t not in self.cache:
                    if resumed[t] >= max_re:
                        skip.add(t)
                    else:
                        by_attempt.setdefault(resumed[t] + 1, []).append(t)
            for attempt in sorted(by_attempt):
                self._loop(by_attempt[attempt], attempt, skip)
            fresh = [t for t in wanted if t not in self.cache and t not in skip]
            fresh = [t for t in fresh if t not in resumed]
            self._loop(fresh, 0, skip)


def default_client_factory():
    """Anthropic SDK client (``llm`` extra). The key is read from ``ANTHROPIC_API_KEY`` only."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise RuntimeError("captions paraphrase needs the llm extra: uv sync --extra llm") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# --------------------------------------------------------------------------- shards / run


def shard_dirs(layout: DataLayout) -> list[Path]:
    """Finished shards (``DONE`` present) with an ``items.jsonl``, in order."""
    if not layout.wet_dir.exists():
        return []
    return [
        d
        for d in sorted(layout.wet_dir.glob("shard-*"))
        if (d / DONE_MARKER).exists() and (d / "items.jsonl").exists()
    ]


def build_captions(
    items: Sequence[RenderItem],
    cfg,
    cache: ParaphraseCache | None = None,
    stats: dict[str, int] | None = None,
) -> list[Caption]:
    """Template captions + cached paraphrases that pass the post-filter."""
    cc = cfg.captions
    holdout = holdout_pedal(cfg)
    n_t, n_p = int(cc.templates_per_item), int(cc.paraphrases_per_item)
    frac = float(cc.pedal_name_fraction)
    min_voiced = float(cfg.analysis.min_voiced_fraction) if "analysis" in cfg else 0.3
    stats = stats if stats is not None else {}
    for k in ("paraphrased", "rejected", "malformed", "missing", "same_as_template"):
        stats.setdefault(k, 0)
    out: list[Caption] = []
    for item in items:
        temps = template_captions(item, n_t, frac, holdout, min_voiced)
        out += temps
        if not n_p or cache is None:
            continue
        named = names_pedal(item, frac, holdout)
        allowed, _ = name_rules(item, named, holdout)
        for j in range(n_p):
            src = temps[j % len(temps)]
            if src.text not in cache:
                stats["missing"] += 1
                continue
            text = cache.get(src.text)
            if text is None:
                stats["malformed"] += 1
                continue
            text = re.sub(r"\s+", " ", text).strip()
            if text.casefold() == src.text.casefold():
                stats["same_as_template"] += 1
                continue
            if check_caption(text, item, named, holdout):
                stats["rejected"] += 1
                continue
            stats["paraphrased"] += 1
            out.append(
                Caption(
                    caption_id=f"{item.item_id}-c{n_t + j}",
                    item_id=item.item_id,
                    kind="paraphrase",
                    text=text,
                    template_id=src.template_id,
                    names_pedal=bool(mentions(text, allowed)),
                    model=cache.model,
                    prompt_version=cache.prompt_version,
                )
            )
    return out


def paraphrase_sources(items: Sequence[RenderItem], cfg) -> list[str]:
    """Unique template texts that get paraphrased, in first-seen order."""
    cc = cfg.captions
    n_t, n_p = int(cc.templates_per_item), int(cc.paraphrases_per_item)
    if n_p <= 0:
        return []
    holdout = holdout_pedal(cfg)
    min_voiced = float(cfg.analysis.min_voiced_fraction) if "analysis" in cfg else 0.3
    out: dict[str, None] = {}
    for item in items:
        temps = template_captions(item, n_t, float(cc.pedal_name_fraction), holdout, min_voiced)
        for j in range(n_p):
            out[temps[j % len(temps)].text] = None
    return list(out)


def _pilot_subset(items: Sequence[RenderItem], n: int) -> list[RenderItem]:
    key = lambda it: hashlib.sha256(it.item_id.encode()).hexdigest()  # noqa: E731
    return sorted(items, key=key)[:n]


def _write_atomic(path: Path, caps: Sequence[Caption]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    write_jsonl(tmp, caps)
    os.replace(tmp, path)


def run(cfg, args, client_factory: Callable[[], Any] | None = None) -> int:
    cc = cfg.captions
    layout = DataLayout.from_config(cfg)
    limit = getattr(args, "limit", None)
    est_only = bool(getattr(args, "estimate", False))
    pilot = bool(getattr(args, "pilot", False))
    factory = client_factory or default_client_factory

    # items: shards, or synthetic ones with --fixtures (estimates for hypothetical sizes)
    shard_items: list[tuple[Path, list[RenderItem]]] = []
    if getattr(args, "fixtures", False):
        items = synthetic_items(limit or 200, holdout_pedal(cfg), int(cfg.get("seed", 0)))
    else:
        total = 0
        for d in shard_dirs(layout):
            its = list(read_jsonl(d / "items.jsonl", RenderItem))
            if limit is not None and total + len(its) > limit:
                break  # only whole shards get captions files
            shard_items.append((d, its))
            total += len(its)
        items = [it for _, its in shard_items for it in its]
        if not items:
            _log("no finished shards found (render first, or use --fixtures)")
            return 1
    if pilot:
        items = _pilot_subset(items, int(cc.pilot_items))

    cache = ParaphraseCache(
        layout.llm_cache / "paraphrases.jsonl", str(cc.model), str(cc.prompt_version)
    )
    sources = paraphrase_sources(items, cfg)
    todo = [t for t in sources if t not in cache]
    _log(
        f"{len(items)} items, {len(sources)} unique templates to paraphrase, {len(todo)} not cached"
    )

    if getattr(args, "fixtures", False) and not est_only:
        for c in build_captions(items[:20], cfg):  # examples only; never calls the API
            print(c.text)
        return 0

    if est_only:
        client = None
        if todo:
            try:
                client = factory()
            except RuntimeError as e:
                _log(f"no API client ({e}); input tokens approximated from characters")
        est = estimate(todo, cc, layout, client)
        for line in est.lines(SpendLedger(layout.ledger), float(cc.max_usd)):
            print(line)
        return 0

    if todo:
        client = factory()
        para = Paraphraser(cc, layout, client, float(cc.get("poll_seconds", 60.0)))
        try:
            para.run(todo)
        except BudgetExceeded as e:
            _log(str(e))
            return 2
        except (SubmitUncertain, LedgerLocked) as e:
            _log(f"{type(e).__name__}: {e}")
            return 3
        cache = para.cache
        if pilot and para.captions_returned:
            per = para.output_tokens / para.captions_returned
            p = _pilot_path(layout, cc)
            p.write_text(
                json.dumps(
                    {
                        "output_tokens_per_caption": per,
                        "input_tokens": para.input_tokens,
                        "output_tokens": para.output_tokens,
                        "captions": para.captions_returned,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _log(f"pilot: {per:.1f} output tokens per caption (written to {p.name})")
    if pilot or getattr(args, "fixtures", False):
        return 0  # pilot / synthetic runs only fill the cache

    stats: dict[str, int] = {}
    n_written = 0
    for d, its in shard_items:
        caps = build_captions(its, cfg, cache, stats)
        _write_atomic(d / "captions.jsonl", caps)
        n_written += len(caps)
    tried = stats["paraphrased"] + stats["rejected"] + stats["malformed"]
    tried += stats["same_as_template"]
    rate = (tried - stats["paraphrased"]) / tried if tried else 0.0
    _log(
        f"wrote {n_written} captions in {len(shard_items)} shards; paraphrases: {stats}; "
        f"reject rate {rate:.1%}"
    )
    return 0


__all__ = [
    "FAMILY",
    "KB_DESCRIPTORS",
    "OUTPUT_SCHEMA",
    "SYSTEM_PROMPT",
    "Attributes",
    "BudgetExceeded",
    "LedgerLock",
    "LedgerLocked",
    "PROMPT_HASH",
    "SubmitUncertain",
    "Estimate",
    "ParaphraseCache",
    "Paraphraser",
    "SpendLedger",
    "attributes",
    "build_captions",
    "check_caption",
    "collect",
    "default_client_factory",
    "estimate",
    "name_rules",
    "names_pedal",
    "paraphrase_sources",
    "request_params",
    "run",
    "synthetic_items",
    "template_captions",
]
