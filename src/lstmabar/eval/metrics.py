"""P5 metrics (task 5.1, D2).

Conventions:
- Boards are complete normalized ``FloatBoard`` s (``eval.baselines.base.complete_board``).
- Audio is mono ``(T,)`` float at ``audio.sample_rate``; audio metrics peak-normalize both
  signals first (stored renders are peak-normalized, so absolute level is not compared).
- Every function returns plain floats (``NaN`` when a term is undefined, e.g. harmonic
  descriptors when the dry voiced fraction is below ``analysis.min_voiced_fraction``); the
  stats layer is NaN-aware.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from lstmabar.data.gold import Region, Targets, _norm, expand
from lstmabar.data.schema import FloatBoard
from lstmabar.dsp.losses import multi_resolution_stft_loss
from lstmabar.dsp.pedalboard import ENABLED

DESCRIPTOR_TERMS: tuple[str, ...] = ("centroid_oct", "odd_even_db", "hnr_db", "archetype_l1")
"""Keys of :func:`descriptor_errors` (|Δ centroid| in octaves, |Δ odd/even| dB, |Δ HNR| dB,
archetype-vector L1)."""

DESCRIPTOR_UNITS: dict[str, float] = {"centroid_oct": 1.0, "odd_even_db": 6.0, "hnr_db": 6.0}
"""Scale of each :func:`descriptor_distance` term (1 octave, 6 dB, 6 dB)."""

NAN = float("nan")


def _finite(v) -> bool:
    return v is not None and isinstance(v, int | float | np.floating) and math.isfinite(float(v))


def _on(v: float, threshold: float = 0.5) -> bool:
    return float(v) >= threshold


# ============================================================================ parameters


def param_l1(pred: FloatBoard, label: FloatBoard) -> dict[str, float]:
    """Normalized L1 per knob, masked by the **label's** gates (knobs of a block the label
    disables are skipped; gates themselves are scored by :func:`gate_f1`).

    Returns ``{"block.knob": |Δ|, ..., "mean": mean over the unmasked knobs}`` (``mean`` is
    NaN when every block is off).
    """
    out: dict[str, float] = {}
    for block, knobs in label.items():
        if not _on(knobs.get(ENABLED, 0.0)):
            continue
        for knob, v in knobs.items():
            if knob == ENABLED:
                continue
            out[f"{block}.{knob}"] = abs(float(pred[block][knob]) - float(v))
    out["mean"] = float(np.mean(list(out.values()))) if out else NAN
    return out


def gate_f1(
    preds: Sequence[FloatBoard], labels: Sequence[FloatBoard], threshold: float = 0.5
) -> dict[str, float]:
    """Per-block F1 of ``enabled >= threshold`` over a set of items, plus ``"macro"``.
    A block with no positive labels and no positive predictions scores NaN."""
    if len(preds) != len(labels):
        raise ValueError(f"{len(preds)} predictions for {len(labels)} labels")
    blocks: list[str] = []
    for lab in labels:
        blocks += [b for b in lab if b not in blocks]
    out: dict[str, float] = {}
    for b in blocks:
        tp = fp = fn = 0
        for p, lab in zip(preds, labels, strict=True):
            yp = _on(p[b][ENABLED], threshold)
            yt = _on(lab[b][ENABLED], threshold)
            tp += yp and yt
            fp += yp and not yt
            fn += yt and not yp
        den = 2 * tp + fp + fn
        out[b] = 2 * tp / den if den else NAN
    vals = [v for v in out.values() if math.isfinite(v)]
    out["macro"] = float(np.mean(vals)) if vals else NAN
    return out


def gate_correct(pred: FloatBoard, label: FloatBoard, threshold: float = 0.5) -> float:
    """Per-item fraction of blocks whose thresholded gate matches the label (bootstrappable)."""
    if not label:
        return NAN
    hits = [
        _on(pred[b][ENABLED], threshold) == _on(k[ENABLED], threshold) for b, k in label.items()
    ]
    return float(np.mean(hits))


def _interval_gap(u: float, lo: float, hi: float) -> float:
    lo, hi = min(lo, hi), max(lo, hi)
    return max(lo - u, 0.0, u - hi)


def _constrained(region: Region, targets: Targets) -> dict[str, set[str]]:
    """``{block: knobs}`` a free region constrains: its level axes, overrides and the voicing
    of its gated blocks (``tables.fixed`` and spec defaults are skipped)."""
    t = targets.tables
    out: dict[str, set[str]] = {"compressor": set(), "drive": set(), "eq": set()}
    out["eq"] |= set(t["brightness"][region.brightness])
    out["eq"] |= set(t["mids"][region.mids])
    if region.compression != "none":
        out["compressor"] |= set(t["compression"][region.compression])
    if region.gain != "none":
        out["drive"] |= set(t["family"][region.family]) | {"gain_db"}
        if region.clipping:
            out["drive"] |= set(t["clipping"][region.clipping])
    for key in region.overrides:
        block, _, knob = key.partition(".")
        out.setdefault(block, set()).add(knob)
    return out


def _pinned(targets: Targets) -> dict[str, set[str]]:
    """Knobs held constant by ``tables.fixed`` (output levels: renders are peak-normalized)."""
    return {b: set(k) for b, k in (targets.tables.get("fixed") or {}).items()}


def _board_distance(pred: FloatBoard, ref: FloatBoard, skip: Mapping[str, set[str]]) -> float:
    terms: list[float] = []
    for block, knobs in ref.items():
        on = _on(knobs.get(ENABLED, 0.0))
        terms.append(float(_on(pred[block][ENABLED]) != on))
        if not on:
            continue
        for knob, v in knobs.items():
            if knob == ENABLED or knob in skip.get(block, ()):
                continue
            terms.append(abs(float(pred[block][knob]) - float(v)))
    return float(np.mean(terms)) if terms else NAN


def region_distance(
    pred: FloatBoard,
    region: Region,
    targets: Targets,
    references: Sequence[FloatBoard] = (),
) -> float:
    """Normalized L1 from ``pred`` to a gold target region (D2).

    Free regions: mean over the knobs the region constrains (its level axes, overrides and
    gated blocks; knobs a free region pins at their default are skipped) of the distance from
    the predicted normalized value to the region's normalized interval (0 inside), plus the
    gate mismatches of the region's blocks. Pedal regions (boxes in pedal-knob space): the
    distance to the nearest of ``references`` (the rendered targets' board labels,
    ``GoldRender.render["board_params"]``), masked by the reference's gates.

    Both forms average one term per block gate (0/1 mismatch at threshold 0.5) together with
    the knob terms of the blocks the target enables. The knobs ``tables.fixed`` holds constant
    (output levels) are skipped for pedal references too. NaN for a pedal region without
    references.
    """

    if region.kind == "pedal":
        if not references:
            return NAN
        skip = _pinned(targets)
        return float(min(_board_distance(pred, r, skip) for r in references))
    c = expand(region, targets)
    want = _constrained(region, targets)
    terms: list[float] = []
    for block, gate in c.gates.items():
        on = bool(gate)
        terms.append(float(_on(pred[block][ENABLED]) != on))
        if not on:
            continue
        for knob in sorted(want.get(block, ())):
            lo, hi = c.ranges[block][knob]
            u = float(pred[block][knob])
            terms.append(_interval_gap(u, _norm(block, knob, lo), _norm(block, knob, hi)))
    return float(np.mean(terms)) if terms else NAN


# ============================================================================ audio


def _peak_normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return x / p if p > 0 and math.isfinite(p) else x


def mrstft(pred: np.ndarray, target: np.ndarray, fft_sizes: Sequence[int] = (512, 1024, 2048)):
    """MR-STFT distance (``dsp.losses.multi_resolution_stft_loss``: spectral convergence +
    log-magnitude L1) between peak-normalized ``pred`` and ``target`` (trimmed to the shorter
    length). Lower is closer. NaN if either signal is non-finite."""

    a, b = _peak_normalized(pred), _peak_normalized(target)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return NAN
    with torch.no_grad():
        ta = torch.from_numpy(a.astype(np.float32))[None]
        tb = torch.from_numpy(b.astype(np.float32))[None]
        return float(multi_resolution_stft_loss(ta, tb, tuple(int(s) for s in fft_sizes)))


def _abs_diff(a, b) -> float:
    return abs(float(a) - float(b)) if _finite(a) and _finite(b) else NAN


def descriptor_errors(
    pred_desc: Mapping[str, float | None],
    target_desc: Mapping[str, float | None],
    pred_arch: Mapping[str, float | None] | None = None,
    target_arch: Mapping[str, float | None] | None = None,
    harmonic: bool = True,
) -> dict[str, float]:
    """:data:`DESCRIPTOR_TERMS` errors. ``harmonic=False`` (dry voiced fraction below the
    threshold) or a missing/None value makes the harmonic terms and the archetype L1 NaN;
    ``centroid_oct`` only needs both centroids."""
    a, b = pred_desc.get("centroid_hz"), target_desc.get("centroid_hz")
    out = {
        "centroid_oct": (
            abs(math.log2(float(a) / float(b)))
            if _finite(a) and _finite(b) and float(a) > 0 and float(b) > 0
            else NAN
        ),
        "odd_even_db": NAN,
        "hnr_db": NAN,
        "archetype_l1": NAN,
    }
    if not harmonic:
        return out
    out["odd_even_db"] = _abs_diff(pred_desc.get("odd_even_db"), target_desc.get("odd_even_db"))
    out["hnr_db"] = _abs_diff(pred_desc.get("hnr_db"), target_desc.get("hnr_db"))
    if pred_arch is not None and target_arch is not None and target_arch:
        diffs = [_abs_diff(pred_arch.get(k), v) for k, v in target_arch.items()]
        if diffs and all(math.isfinite(d) for d in diffs):
            out["archetype_l1"] = float(sum(diffs))
    return out


def descriptor_distance(
    pred_desc: Mapping[str, float | None],
    ref_desc: Mapping[str, float | None],
    harmonic: bool = True,
) -> float:
    """Scalar descriptor distance for the secondary 2AFC: the mean of the available
    :func:`descriptor_errors` terms after scaling each by a fixed unit (1 octave, 6 dB, 6 dB;
    archetype L1 unscaled). NaN if no term is available."""
    e = descriptor_errors(pred_desc, ref_desc, harmonic=harmonic)
    terms = [e[k] / u for k, u in DESCRIPTOR_UNITS.items() if math.isfinite(e[k])]
    return float(np.mean(terms)) if terms else NAN


def analyse_render(
    y: np.ndarray, cfg, track=None, harmonic: bool = True
) -> tuple[dict[str, float], dict[str, float]]:
    """``(descriptors, archetype)`` of a predicted render with the same analysis settings as
    the dataset (``data.render.analyse`` with ``cfg.analysis``); ``track`` is the dry clip's
    cached pYIN track (renders are time-aligned with their dry input). Harmonic terms are
    computed only when ``harmonic`` and a track is given."""
    from lstmabar.data.render import analyse

    a = cfg.analysis
    return analyse(
        np.asarray(y, dtype=np.float64),
        int(cfg.audio.sample_rate),
        track,
        bool(harmonic and track is not None),
        int(a.n_harmonics),
        int(a.max_frames),
    )


# ============================================================================ text adherence


def two_afc(d_target: float, d_distractors: Sequence[float]) -> float:
    """Gold 2AFC for one (item, clip): the fraction of distractors with
    ``d(pred, target) < d(pred, distractor)``; ties count 0.5; NaN distances are skipped
    (NaN if none remain). Chance = 0.5."""
    if not _finite(d_target):
        return NAN
    t = float(d_target)
    wins = [1.0 if t < d else 0.5 if t == d else 0.0 for d in d_distractors if _finite(d)]
    return float(np.mean(wins)) if wins else NAN


def clap_score(text_emb: np.ndarray, audio_emb: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of ``(N, D)`` text and audio embeddings (secondary metric
    only, spike A)."""
    t = np.atleast_2d(np.asarray(text_emb, dtype=np.float64))
    a = np.atleast_2d(np.asarray(audio_emb, dtype=np.float64))
    if t.shape != a.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {a.shape}")
    nt = np.linalg.norm(t, axis=1)
    na = np.linalg.norm(a, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sum(t * a, axis=1) / (nt * na)


__all__ = [
    "DESCRIPTOR_TERMS",
    "DESCRIPTOR_UNITS",
    "analyse_render",
    "clap_score",
    "descriptor_distance",
    "descriptor_errors",
    "gate_correct",
    "gate_f1",
    "mrstft",
    "param_l1",
    "region_distance",
    "two_afc",
]
