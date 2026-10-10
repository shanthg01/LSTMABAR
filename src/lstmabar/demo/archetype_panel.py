"""Pure helpers behind the demo's archetype panel (P3 task 3.5; no gradio import).

For each rendered signal (dry, grey-box, white-box) the panel shows the archetype readout
(sine/triangle/square/saw + noise, :func:`~lstmabar.analysis.archetypes.archetype_readout`'s
method) and the H1–H10 levels in dBc. To keep the cost bounded, only the loudest
:data:`ANALYSIS_SECONDS` excerpt is analysed (the same stretch of every signal), pYIN runs once
on the dry excerpt and its track is reused for the processed signals (same notes, and
distortion can confuse the pitch tracker), and at most :data:`MAX_FRAMES` frames are fitted.
The callback computes it once per render.

Chords, noise or silence have no stable pitch; the panel then says so instead of failing, and
the same happens when librosa (the ``analysis`` extra) is not installed.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from lstmabar.analysis.archetypes import ARCHETYPES, ArchetypeReadout, project_archetypes

log = logging.getLogger(__name__)

ANALYSIS_SECONDS = 3.0
MAX_FRAMES = 16
N_HARMONICS = 10
DBC_FLOOR = -60.0
COMPONENTS: tuple[str, ...] = (*ARCHETYPES, "noise")

NO_PITCH = (
    "No stable pitch found in the analysed excerpt (chords, noise or silence?), so there is "
    "no archetype readout. Try single notes, e.g. the `single_notes` example riff."
)
NO_LIBROSA = (
    "The archetype panel needs the pitch tracker from the analysis extra: "
    "`uv sync --extra analysis`."
)
ANALYSIS_ERROR = "The archetype analysis failed for this clip."


@dataclass
class SignalAnalysis:
    label: str
    readout: ArchetypeReadout | None  # None: no stable pitch in this signal
    dbc: np.ndarray | None = None  # (N_HARMONICS,) H1..H10 in dBc
    f0_hz: float = float("nan")


@dataclass
class ArchetypePanel:
    signals: list[SignalAnalysis] = field(default_factory=list)
    message: str = ""
    seconds: float = 0.0


def loudest_window(x: np.ndarray, sample_rate: int, seconds: float = ANALYSIS_SECONDS) -> slice:
    """The ``seconds``-long stretch of ``x`` with the highest RMS (whole clip if shorter)."""
    n = int(round(seconds * sample_rate))
    if len(x) <= n:
        return slice(0, len(x))
    hop = max(1, sample_rate // 4)
    energy = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64) ** 2)])
    starts = np.arange(0, len(x) - n + 1, hop)
    best = int(starts[np.argmax(energy[starts + n] - energy[starts])])
    return slice(best, best + n)


def analyse_signals(
    signals: Mapping[str, np.ndarray],
    sample_rate: int,
    seconds: float = ANALYSIS_SECONDS,
    track_fn=None,
    profile_fn=None,
) -> ArchetypePanel:
    """Archetype readout + H1–H10 dBc for each signal; the first one is the reference (dry).

    Signals may differ in length (the white-box render is shorter); the excerpt is taken from
    the stretch they all cover. ``track_fn``/``profile_fn`` replace
    :func:`~lstmabar.analysis.harmonics.track_f0` / ``harmonic_profile`` (tests).
    """
    if not signals:
        return ArchetypePanel(message=NO_PITCH)
    try:
        from lstmabar.analysis import harmonics
    except Exception:  # pragma: no cover - broken install
        log.exception("harmonics module unavailable")
        return ArchetypePanel(message=ANALYSIS_ERROR)
    track_fn = track_fn or harmonics.track_f0
    profile_fn = profile_fn or harmonics.harmonic_profile

    common = min(len(x) for x in signals.values())
    ref = np.asarray(next(iter(signals.values())))[:common]
    win = loudest_window(ref, sample_rate, seconds)
    excerpt_s = (win.stop - win.start) / sample_rate
    try:
        track = track_fn(ref[win], sample_rate)
    except ImportError:
        return ArchetypePanel(message=NO_LIBROSA)
    except Exception:
        log.exception("pitch tracking failed")
        return ArchetypePanel(message=ANALYSIS_ERROR)

    out = []
    for label, x in signals.items():
        seg = np.asarray(x, dtype=np.float64)[:common][win]
        try:
            prof = profile_fn(
                seg, sample_rate, n_harmonics=N_HARMONICS, track=track, max_frames=MAX_FRAMES
            )
            readout = project_archetypes(prof.amplitudes, prof.hnr_db)
        except ValueError:  # no voiced / loud frames, or no usable H1
            out.append(SignalAnalysis(label, None))
            continue
        except Exception:
            log.exception("harmonic profile failed for %s", label)
            out.append(SignalAnalysis(label, None))
            continue
        out.append(SignalAnalysis(label, readout, prof.dbc(DBC_FLOOR), float(prof.f0_hz)))
    if all(s.readout is None for s in out):
        return ArchetypePanel(message=NO_PITCH, seconds=excerpt_s)
    return ArchetypePanel(out, "", excerpt_s)


# --- Formatting -----------------------------------------------------------------------------


def _bar(share: float, width: int = 10) -> str:
    n = int(round(min(max(share, 0.0), 1.0) * width))
    return "█" * n + "·" * (width - n)


def archetype_markdown(panel: ArchetypePanel) -> str:
    """Markdown table: one column per signal, one row per archetype (+ noise), with bars."""
    if not panel.signals:
        return panel.message or NO_PITCH
    cols = [s.label for s in panel.signals]
    lines = [
        "| | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    vectors = [s.readout.vector() if s.readout is not None else None for s in panel.signals]
    for i, comp in enumerate(COMPONENTS):
        cells = []
        for v in vectors:
            cells.append("no stable pitch" if v is None else f"`{_bar(v[i])}` {v[i]:.2f}")
        lines.append(f"| {comp} | " + " | ".join(cells) + " |")
    dom = [s.readout.dominant if s.readout is not None else "–" for s in panel.signals]
    lines.append("| **dominant** | " + " | ".join(dom) + " |")
    f0 = next((s.f0_hz for s in panel.signals if np.isfinite(s.f0_hz)), float("nan"))
    head = f"Archetype readout of the loudest {panel.seconds:.1f} s excerpt"
    if np.isfinite(f0):
        head += f" (median f0 {f0:.0f} Hz)"
    return head + ":\n\n" + "\n".join(lines)


def archetype_rows(panel: ArchetypePanel) -> list[dict]:
    """Long-format rows ``{signal, component, share}`` for a stacked bar chart."""
    rows = []
    for s in panel.signals:
        if s.readout is None:
            continue
        for comp, share in zip(COMPONENTS, s.readout.vector(), strict=True):
            rows.append({"signal": s.label, "component": comp, "share": float(share)})
    return rows


def harmonic_rows(panel: ArchetypePanel) -> list[dict]:
    """Long-format rows ``{signal, harmonic, dbc}`` (H1..H10) for a line chart."""
    rows = []
    for s in panel.signals:
        if s.dbc is None:
            continue
        for k, v in enumerate(s.dbc, start=1):
            if np.isfinite(v):
                rows.append({"signal": s.label, "harmonic": k, "dbc": float(v)})
    return rows


__all__ = [
    "ANALYSIS_SECONDS",
    "COMPONENTS",
    "NO_LIBROSA",
    "NO_PITCH",
    "ArchetypePanel",
    "SignalAnalysis",
    "analyse_signals",
    "archetype_markdown",
    "archetype_rows",
    "harmonic_rows",
    "loudest_window",
]
