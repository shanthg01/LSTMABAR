"""P5 metrics (task 5.1, D2). Wave 0: signatures only; wave 1 (a) implements them.

Conventions:
- Boards are complete normalized ``FloatBoard`` s (``eval.baselines.base.complete_board``).
- Audio is mono ``(T,)`` float at ``audio.sample_rate``; audio metrics peak-normalize both
  signals first (stored renders are peak-normalized, so absolute level is not compared).
- Every function returns plain floats (``NaN`` when a term is undefined, e.g. harmonic
  descriptors when the dry voiced fraction is below ``analysis.min_voiced_fraction``); the
  stats layer is NaN-aware.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from lstmabar.data.gold import Region, Targets
from lstmabar.data.schema import FloatBoard

DESCRIPTOR_TERMS: tuple[str, ...] = ("centroid_oct", "odd_even_db", "hnr_db", "archetype_l1")
"""Keys of :func:`descriptor_errors` (|Δ centroid| in octaves, |Δ odd/even| dB, |Δ HNR| dB,
archetype-vector L1)."""


# ============================================================================ parameters


def param_l1(pred: FloatBoard, label: FloatBoard) -> dict[str, float]:
    """Normalized L1 per knob, masked by the **label's** gates (knobs of a block the label
    disables are skipped; gates themselves are scored by :func:`gate_f1`).

    Returns ``{"block.knob": |Δ|, ..., "mean": mean over the unmasked knobs}`` (``mean`` is
    NaN when every block is off).
    """
    raise NotImplementedError


def gate_f1(
    preds: Sequence[FloatBoard], labels: Sequence[FloatBoard], threshold: float = 0.5
) -> dict[str, float]:
    """Per-block F1 of ``enabled >= threshold`` over a set of items, plus ``"macro"``.
    A block with no positive labels and no positive predictions scores NaN."""
    raise NotImplementedError


def gate_correct(pred: FloatBoard, label: FloatBoard, threshold: float = 0.5) -> float:
    """Per-item fraction of blocks whose thresholded gate matches the label (bootstrappable)."""
    raise NotImplementedError


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
    """
    raise NotImplementedError


# ============================================================================ audio


def mrstft(pred: np.ndarray, target: np.ndarray, fft_sizes: Sequence[int] = (512, 1024, 2048)):
    """MR-STFT distance (``dsp.losses.multi_resolution_stft_loss``: spectral convergence +
    log-magnitude L1) between peak-normalized ``pred`` and ``target`` (trimmed to the shorter
    length). Lower is closer."""
    raise NotImplementedError


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
    raise NotImplementedError


def descriptor_distance(
    pred_desc: Mapping[str, float | None],
    ref_desc: Mapping[str, float | None],
    harmonic: bool = True,
) -> float:
    """Scalar descriptor distance for the secondary 2AFC: the mean of the available
    :func:`descriptor_errors` terms after scaling each by a fixed unit (1 octave, 6 dB, 6 dB;
    archetype L1 unscaled). NaN if no term is available."""
    raise NotImplementedError


def analyse_render(
    y: np.ndarray, cfg, track=None, harmonic: bool = True
) -> tuple[dict[str, float], dict[str, float]]:
    """``(descriptors, archetype)`` of a predicted render with the same analysis settings as
    the dataset (``data.render.analyse`` with ``cfg.analysis``); ``track`` is the dry clip's
    cached pYIN track (renders are time-aligned with their dry input)."""
    raise NotImplementedError


# ============================================================================ text adherence


def two_afc(d_target: float, d_distractors: Sequence[float]) -> float:
    """Gold 2AFC for one (item, clip): the fraction of distractors with
    ``d(pred, target) < d(pred, distractor)``; ties count 0.5; NaN distances are skipped
    (NaN if none remain). Chance = 0.5."""
    raise NotImplementedError


def clap_score(text_emb: np.ndarray, audio_emb: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of ``(N, D)`` text and audio embeddings (secondary metric
    only, spike A)."""
    raise NotImplementedError


__all__ = [
    "DESCRIPTOR_TERMS",
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
