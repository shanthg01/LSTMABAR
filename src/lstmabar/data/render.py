"""Task 4.2 (wave 1b): dry clip x (pedal preset | free board) x sampled knobs -> wet shards.

Contract: ``run(cfg, args)`` reads ``DataLayout.dry_manifest``, samples render specs per clip
(resolutions R3, R5, R6, R7; deterministic per-item seeds), renders TS808/DS-1 through the
white-box and everything else through the grey-box ``default_pedalboard``, labels white-box
items from the calibration tables (:mod:`lstmabar.data.calibrate`), measures descriptors and
archetypes (pYIN track computed once per dry clip, cached at ``DataLayout.f0_cache``), and
writes shards of ``render.shard_size`` items (FLAC + ``items.jsonl`` + ``DONE`` last) with
multiprocessing; finished shards are skipped on re-run.

Details:

- **Specs** (:func:`plan_specs`, pure): per clip ``render.renders_per_clip[split]`` items
  ``<clip_id>-r<k>``; ``test`` clips get ``render.unseen_renders_per_clip`` more, of
  ``render.holdout_pedal``, in split ``test_unseen_pedal``. Each item's draws come from its own
  seed (:func:`item_seed` of ``(cfg.seed, clip_id, k)``), so a spec never depends on other
  clips or on the shard layout. The holdout pedal is only ever drawn for unseen items.
- **Labels.** Grey-box pedals: ``derive`` -> ``preset_params`` (``label_source="derived"``).
  White-box pedals: the calibration table's lookup (``calibrated`` / ``calibrated_nearest``),
  with the derived label kept in ``board_params_derived``; a missing table is an error unless
  derived labels are explicitly allowed (``--allow-derived-labels``; implied by
  ``--fixtures``). Free board: the sampled params (``"free"``). ``clamped`` is set only by
  derive notes containing ``" clamped to "`` (R6).
- **Levels** (R3): the stored dry clip is scaled to the item's ``input_peak_dbfs``; the wet is
  stored peak-normalized at ``render.wet_peak_dbfs`` (24-bit FLAC) with ``wet_gain_db``.
- **Analysis** (decision 5): one pYIN track per dry clip, reused for every wet render of it
  (the engines are time-aligned). Harmonic analysis runs on the peak-normalized (stored)
  signals, so its frame gating does not depend on the sampled level; ``rms_db`` is reported
  at the true level (dry: at the render input; wet: the true wet). When the dry clip's voiced
  fraction is below ``analysis.min_voiced_fraction`` the harmonic descriptors and the
  archetype readout are NaN (``null`` in JSON); spectral descriptors are always measured.
  The dry analysis is computed once per clip (per worker) and shifted to each item's level.
- **Shards**: ``DONE`` (written atomically, last) holds the hash of the shard's spec list plus
  the render fingerprint (:func:`render_fingerprint`: analysis/render/physics settings and
  the calibration tables' sha256); a finished shard from a different plan is an error (never
  silently mixed). Partial shards (no or unreadable ``DONE``) are rewritten. ``wet/plan.json``
  records the shard count and hashes; shard dirs beyond the plan trigger a warning.
- **Disabled blocks** keep knob values in ``board_params`` (random for free board, defaults
  or derived values for pedals); consumers must mask by ``"enabled"``.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from lstmabar.analysis.archetypes import ARCHETYPES, project_archetypes
from lstmabar.analysis.descriptors import harmonic_descriptors, signal_descriptors
from lstmabar.analysis.harmonics import F0Track, HarmonicProfile, harmonic_profile, track_f0
from lstmabar.data import levels
from lstmabar.data.calibrate import CalibrationTable
from lstmabar.data.layout import DONE_MARKER, DataLayout
from lstmabar.data.schema import (
    FREE_SCENARIOS,
    DryClip,
    FloatBoard,
    RenderItem,
    board_to_floats,
    floats_to_board,
    from_dict,
    read_jsonl,
    to_dict,
    write_jsonl,
)
from lstmabar.dsp.pedalboard import ENABLED, default_pedalboard
from lstmabar.physics.derive import DeriveContext, derive, preset_params
from lstmabar.physics.kb import Pedal, default_pedals_dir, load_kb

CLAMP_MARKER = " clamped to "
UNSEEN_SPLIT = "test_unseen_pedal"
ARCHETYPE_KEYS: tuple[str, ...] = (*ARCHETYPES, "noise")
ITEMS_FILE = "items.jsonl"

# ============================================================================ spec sampling


@dataclass(frozen=True)
class RenderSpec:
    """Everything random about one render, drawn from its seed (no audio, no labels)."""

    item_id: str
    clip_id: str
    split: str
    seed: int
    scenario: str  # "pedal" | FREE_SCENARIOS
    engine: str  # "whitebox" | "greybox"
    pedal_id: str | None
    pedal_knobs: dict[str, float] | None
    free_board: FloatBoard | None  # normalized params incl. gates (free scenarios only)
    input_peak_dbfs: float


@dataclass(frozen=True)
class Sampling:
    """The ``render.*`` settings spec sampling depends on."""

    renders_per_clip: dict[str, int]
    unseen_renders_per_clip: int
    pedal_fraction: float
    free_board_mix: tuple[tuple[str, float], ...]  # in FREE_SCENARIOS order
    p_eq: float
    p_comp: float
    holdout_pedal: str
    whitebox_pedals: tuple[str, ...]
    input_peak_dbfs: tuple[float, float]
    sample_rate: int
    dataset_seed: int

    @classmethod
    def from_config(cls, cfg) -> Sampling:
        r = cfg.render
        mix = {str(k): float(v) for k, v in r.free_board_mix.items()}
        unknown = set(mix) - set(FREE_SCENARIOS)
        if unknown:
            raise ValueError(f"render.free_board_mix: unknown scenarios {sorted(unknown)}")
        total = sum(mix.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"render.free_board_mix must sum to 1, got {total}")
        lo, hi = (float(v) for v in r.input_peak_dbfs)
        return cls(
            renders_per_clip={str(k): int(v) for k, v in r.renders_per_clip.items()},
            unseen_renders_per_clip=int(r.unseen_renders_per_clip),
            pedal_fraction=float(r.pedal_fraction),
            free_board_mix=tuple((s, mix.get(s, 0.0)) for s in FREE_SCENARIOS),
            p_eq=float(r.free_drive_p_eq),
            p_comp=float(r.free_drive_p_comp),
            holdout_pedal=str(r.holdout_pedal),
            whitebox_pedals=tuple(str(p) for p in r.whitebox_pedals),
            input_peak_dbfs=(lo, hi),
            sample_rate=int(cfg.audio.sample_rate),
            dataset_seed=int(cfg.seed),
        )


def item_seed(dataset_seed: int, clip_id: str, k: int) -> int:
    """Per-item seed in [0, 2**63) from ``(dataset seed, clip id, render index)``."""
    h = hashlib.sha256(f"{dataset_seed}:{clip_id}:{k}".encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def scenario_gates(scenario: str, rng: np.random.Generator, p_eq: float, p_comp: float):
    """``{block: 0./1.}`` gates of a free-board scenario (R5). Draws only for ``drive``."""
    on = {
        "bypass": (),
        "eq": ("eq",),
        "comp": ("compressor",),
        "comp_eq": ("compressor", "eq"),
    }
    if scenario == "drive":
        eq = bool(rng.random() < p_eq)
        comp = bool(rng.random() < p_comp)
        return {"compressor": float(comp), "drive": 1.0, "eq": float(eq)}
    if scenario not in on:
        raise ValueError(f"unknown free-board scenario {scenario!r}")
    return {b: float(b in on[scenario]) for b in ("compressor", "drive", "eq")}


@lru_cache(maxsize=4)
def _board(sample_rate: int):
    return default_pedalboard(sample_rate)


def _free_board(seed: int, scenario: str, rng: np.random.Generator, s: Sampling) -> FloatBoard:
    board = _board(s.sample_rate)
    gen = torch.Generator().manual_seed(seed)
    params = board.random_params(1, generator=gen)  # knobs U(0, 1); its gates are replaced
    floats = board_to_floats(params)
    for block, g in scenario_gates(scenario, rng, s.p_eq, s.p_comp).items():
        floats[block][ENABLED] = g
    return floats


def sample_spec(
    clip_id: str,
    split: str,
    k: int,
    s: Sampling,
    pedal_pots: Mapping[str, Sequence[str]],
    unseen: bool = False,
    force: tuple[str, str | None] | None = None,
) -> RenderSpec:
    """Spec of render ``k`` of a clip. ``pedal_pots`` maps KB pedal ids to their knob names.

    ``unseen`` draws the holdout pedal (split ``test_unseen_pedal``); ``force`` =
    ``(scenario, pedal_id)`` skips the scenario draw (benchmarks).
    """
    seed = item_seed(s.dataset_seed, clip_id, k)
    rng = np.random.default_rng(seed)
    pedals = sorted(p for p in pedal_pots if p != s.holdout_pedal)
    if force is not None:
        scenario, pedal_id = force
    elif unseen:
        scenario, pedal_id = "pedal", s.holdout_pedal
    elif rng.random() < s.pedal_fraction:
        scenario, pedal_id = "pedal", pedals[int(rng.integers(len(pedals)))]
    else:
        names = [n for n, _ in s.free_board_mix]
        probs = np.array([p for _, p in s.free_board_mix])
        scenario, pedal_id = names[int(rng.choice(len(names), p=probs / probs.sum()))], None
    knobs = board = None
    if scenario == "pedal":
        if pedal_id not in pedal_pots:
            raise KeyError(f"pedal {pedal_id!r} is not in the KB")
        knobs = {name: float(rng.random()) for name in sorted(pedal_pots[pedal_id])}
        engine = "whitebox" if pedal_id in s.whitebox_pedals else "greybox"
    else:
        board = _free_board(seed, scenario, rng, s)
        engine = "greybox"
    peak = levels.sample_input_peak(rng, *s.input_peak_dbfs)
    return RenderSpec(
        item_id=f"{clip_id}-r{k}",
        clip_id=clip_id,
        split=UNSEEN_SPLIT if unseen else split,
        seed=seed,
        scenario=scenario,
        engine=engine,
        pedal_id=pedal_id,
        pedal_knobs=knobs,
        free_board=board,
        input_peak_dbfs=peak,
    )


def plan_specs(
    clips: Sequence[DryClip], s: Sampling, pedal_pots: Mapping[str, Sequence[str]]
) -> list[RenderSpec]:
    """All render specs for ``clips``, in manifest order (a clip's items are consecutive)."""
    if s.holdout_pedal not in pedal_pots:
        raise KeyError(f"render.holdout_pedal {s.holdout_pedal!r} is not in the KB")
    specs: list[RenderSpec] = []
    for clip in clips:
        if clip.split not in s.renders_per_clip:
            raise ValueError(f"clip {clip.clip_id}: no renders_per_clip for split {clip.split!r}")
        n = s.renders_per_clip[clip.split]
        specs += [sample_spec(clip.clip_id, clip.split, k, s, pedal_pots) for k in range(n)]
        if clip.split == "test":
            specs += [
                sample_spec(clip.clip_id, clip.split, k, s, pedal_pots, unseen=True)
                for k in range(n, n + s.unseen_renders_per_clip)
            ]
    return specs


def shard_specs(specs: Sequence[RenderSpec], shard_size: int) -> list[list[RenderSpec]]:
    if shard_size < 1:
        raise ValueError("render.shard_size must be >= 1")
    return [list(specs[i : i + shard_size]) for i in range(0, len(specs), shard_size)]


def plan_hash(specs: Sequence[RenderSpec]) -> str:
    blob = json.dumps([asdict(s) for s in specs], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def clamp_notes(notes: Sequence[str]) -> list[str]:
    """The derive notes that report an actual clamp (R6); other notes may mention "clamp"."""
    return [n for n in notes if CLAMP_MARKER in n]


# ============================================================================ analysis


def save_track(path: Path, track: F0Track, sha256: str = "") -> None:
    """Atomic write (parallel workers may track the same clip at a shard boundary)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(
        tmp,
        times=track.times,
        f0_hz=track.f0_hz,
        voiced=np.asarray(track.voiced, dtype=bool),
        voiced_prob=track.voiced_prob,
        hop_length=track.hop_length,
        frame_length=track.frame_length,
        sha256=np.array(sha256),
    )
    try:
        os.replace(tmp, path)
    except PermissionError:  # Windows: another worker holds the file open; its copy is equal
        tmp.unlink(missing_ok=True)


def load_track(path: Path, sha256: str = "") -> F0Track | None:
    """Cached track, or ``None`` if missing or computed for different audio."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if sha256 and str(z["sha256"]) and str(z["sha256"]) != sha256:
            return None
        return F0Track(
            times=z["times"],
            f0_hz=z["f0_hz"],
            voiced=z["voiced"].astype(bool),
            voiced_prob=z["voiced_prob"],
            hop_length=int(z["hop_length"]),
            frame_length=int(z["frame_length"]),
        )


def voiced_fraction(track: F0Track | None) -> float | None:
    if track is None:
        return None
    v = np.asarray(track.voiced, dtype=bool)
    return float(v.mean()) if v.size else 0.0


def _nan_archetype() -> dict[str, float]:
    return dict.fromkeys(ARCHETYPE_KEYS, float("nan"))


def analyse(
    y: np.ndarray,
    sr: int,
    track: F0Track | None,
    harmonic: bool,
    n_harmonics: int,
    max_frames: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """``(descriptors, archetype)`` of ``y`` with one shared harmonic profile."""
    desc = signal_descriptors(y, sr)
    prof: HarmonicProfile | None = None
    if harmonic and track is not None:
        try:
            prof = harmonic_profile(
                y, sr, n_harmonics=n_harmonics, track=track, max_frames=max_frames
            )
        except ValueError:
            prof = None
    desc |= harmonic_descriptors(prof)
    arch = _nan_archetype()
    if prof is not None:
        try:
            v = project_archetypes(prof.amplitudes, prof.hnr_db).vector()
            arch = {k: float(x) for k, x in zip(ARCHETYPE_KEYS, v, strict=True)}
        except ValueError:
            pass
    return desc, arch


def librosa_available() -> bool:
    try:
        import librosa  # noqa: F401
    except ImportError:
        return False
    return True


# ============================================================================ rendering


@dataclass(frozen=True)
class Options:
    allow_derived_labels: bool = False
    pitch: bool = True  # False: no pYIN (librosa missing, fixtures only)


class RenderContext:
    """Per-process state: KB, board, calibration tables, dry clips and their analysis."""

    def __init__(self, cfg, opts: Options) -> None:
        self.cfg = cfg
        self.opts = opts
        self.layout = DataLayout.from_config(cfg)
        self.sr = int(cfg.audio.sample_rate)
        self.kb = load_pedals(cfg)
        self.board = default_pedalboard(self.sr)
        self.derive_ctx = DeriveContext.from_config(cfg)
        self.volts_per_fs = float(cfg.physics.volts_per_full_scale)
        self.oversample = int(cfg.render.whitebox_oversample)
        self.wet_peak = float(cfg.render.wet_peak_dbfs)
        self.wet_subtype = str(cfg.render.wet_subtype)
        a = cfg.analysis
        self.n_harmonics = int(a.n_harmonics)
        self.max_frames = int(a.max_frames)
        self.min_vf = float(a.min_voiced_fraction)
        self._tables: dict[str, CalibrationTable | None] = {}
        self._dry: dict[str, tuple[np.ndarray, float, F0Track | None, dict, dict]] = {}

    def table(self, pedal_id: str) -> CalibrationTable | None:
        if pedal_id not in self._tables:
            path = self.layout.calibration_table(pedal_id)
            if path.exists():
                self._tables[pedal_id] = CalibrationTable.load(path)
            elif self.opts.allow_derived_labels:
                self._tables[pedal_id] = None
            else:
                raise FileNotFoundError(
                    f"calibration table for {pedal_id!r} not found ({path.name}); run "
                    "`lstmabar data calibrate` or pass --allow-derived-labels"
                )
        return self._tables[pedal_id]

    def track(self, clip: DryClip, x: np.ndarray, timings: dict | None = None):
        if not self.opts.pitch:
            return None
        path = self.layout.f0_cache(clip.clip_id)
        tr = load_track(path, clip.sha256)
        if tr is None:
            t0 = time.perf_counter()
            tr = track_f0(x, self.sr)
            _tick(timings, "pyin", t0)
            save_track(path, tr, clip.sha256)
        return tr

    def dry(self, clip: DryClip, timings: dict | None = None):
        """``(stored dry, its peak dBFS, track, descriptors, archetype)``, cached per clip."""
        if clip.clip_id not in self._dry:
            x, sr = levels.read_audio(self.layout.resolve_rendered(clip.path))
            if sr != self.sr:
                raise ValueError(f"{clip.clip_id}: sample rate {sr}, expected {self.sr}")
            tr = self.track(clip, x, timings)
            vf = voiced_fraction(tr)
            t0 = time.perf_counter()
            harmonic = vf is not None and vf >= self.min_vf
            desc, arch = analyse(x, self.sr, tr, harmonic, self.n_harmonics, self.max_frames)
            _tick(timings, "dry_analysis", t0)
            self._dry = {clip.clip_id: (x, levels.peak_dbfs(x), tr, desc, arch)}  # keep one
        return self._dry[clip.clip_id]

    def render_board(self, board: FloatBoard, x: np.ndarray) -> np.ndarray:
        # One item at a time: batching items through the board gave no speed-up on one CPU
        # thread per worker (8 items batched took ~5x one item), and parallelism comes from
        # the worker processes.
        with torch.no_grad():
            xt = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
            y = self.board(xt, floats_to_board(board))
        return y[0].double().numpy()

    def render_whitebox(self, pedal: Pedal, knobs: Mapping[str, float], x: np.ndarray):
        from lstmabar.physics.whitebox import simulate

        return simulate(
            pedal, x, self.sr, knobs, volts_per_fs=self.volts_per_fs, oversample=self.oversample
        )


def _tick(timings: dict | None, key: str, t0: float) -> None:
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - t0


def load_pedals(cfg) -> dict[str, Pedal]:
    """KB from ``paths.pedals`` (relative to the CWD), else the repository's ``pedals/``."""
    path = Path(str(cfg.paths.pedals))
    return load_kb(path if path.is_dir() else default_pedals_dir())


def render_item(
    spec: RenderSpec,
    clip: DryClip,
    shard: int,
    ctx: RenderContext,
    timings: dict | None = None,
    out_root: Path | None = None,
) -> RenderItem:
    """Render, label, store and analyse one item. ``out_root`` overrides the rendered root."""
    dry, dry_peak, track, desc_dry, arch_dry = ctx.dry(clip, timings)
    x = levels.apply_input_level(dry, spec.input_peak_dbfs)
    derived = None
    notes: list[str] = []
    label_notes: list[str] = []
    if spec.scenario == "pedal":
        pedal = ctx.kb[spec.pedal_id]
        t0 = time.perf_counter()
        preset = derive(pedal, spec.pedal_knobs, ctx.derive_ctx)
        derived = board_to_floats(preset_params(preset))
        notes = clamp_notes(preset.notes)
        _tick(timings, "derive", t0)
        t0 = time.perf_counter()
        if spec.engine == "whitebox":
            wet = ctx.render_whitebox(pedal, spec.pedal_knobs, x)
            _tick(timings, "render_whitebox", t0)
            table = ctx.table(spec.pedal_id)
            if table is None:
                board, source = derived, "derived"
            else:
                label = table.lookup(spec.pedal_knobs)
                board, source = label.board, label.source
                label_notes = list(label.clamp_notes)
        else:
            wet = ctx.render_board(derived, x)
            _tick(timings, "render_greybox", t0)
            board, source = derived, "derived"
    else:
        t0 = time.perf_counter()
        board, source = spec.free_board, "free"
        wet = ctx.render_board(board, x)
        _tick(timings, "render_greybox", t0)

    t0 = time.perf_counter()
    wet = np.asarray(wet, dtype=np.float64)
    if not np.all(np.isfinite(wet)):
        raise FloatingPointError(f"{spec.item_id}: wet render contains NaN/inf")
    if levels.peak_dbfs(wet) <= levels.SILENCE_DBFS:
        raise ValueError(f"{spec.item_id}: wet render is silent")
    stored, norm_db = levels.normalize_peak(wet, ctx.wet_peak)
    gain_db = -norm_db  # restore gain: true wet = stored * 10**(gain_db / 20)
    rel = ctx.layout.wet_relpath(shard, spec.item_id)
    path = (out_root or ctx.layout.rendered) / rel
    levels.write_flac(path, stored, ctx.sr, ctx.wet_subtype)
    sha = levels.sha256_file(path)
    _tick(timings, "write", t0)

    t0 = time.perf_counter()
    vf = voiced_fraction(track)
    harmonic = vf is not None and vf >= ctx.min_vf
    desc_wet, arch_wet = analyse(stored, ctx.sr, track, harmonic, ctx.n_harmonics, ctx.max_frames)
    desc_wet["rms_db"] += gain_db  # true wet level (analysis ran on the stored file)
    desc_dry = dict(desc_dry)
    desc_dry["rms_db"] += spec.input_peak_dbfs - dry_peak  # level at the render input
    _tick(timings, "wet_analysis", t0)

    return RenderItem(
        item_id=spec.item_id,
        clip_id=spec.clip_id,
        split=spec.split,
        seed=spec.seed,
        scenario=spec.scenario,
        engine=spec.engine,
        pedal_id=spec.pedal_id,
        pedal_knobs=spec.pedal_knobs,
        board_params=board,
        board_params_derived=derived,
        label_source=source,
        clamped=bool(notes),
        clamp_notes=notes,
        input_peak_dbfs=spec.input_peak_dbfs,
        wet_gain_db=gain_db,
        wet_path=rel,
        sha256=sha,
        label_clamp_notes=label_notes,
        voiced_fraction=vf,
        descriptors_dry=desc_dry,
        descriptors_wet=desc_wet,
        archetype_dry=dict(arch_dry),
        archetype_wet=arch_wet,
    )


# ============================================================================ shards / workers

_CTX: RenderContext | None = None
_CTX_KEY: str | None = None


def _context(cfg_dict: dict, opts: dict) -> RenderContext:
    """One context per process (rebuilt only if the config changes)."""
    global _CTX, _CTX_KEY
    key = json.dumps([cfg_dict, opts], sort_keys=True, default=str)
    if _CTX is None or key != _CTX_KEY:
        _CTX = RenderContext(OmegaConf.create(cfg_dict), Options(**opts))
        _CTX_KEY = key
    return _CTX


def _init_worker() -> None:
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", message=".*numba.*")


def render_shard(job: dict) -> dict:
    """Render one shard (top-level and picklable: the multiprocessing worker function)."""
    ctx = _context(job["cfg"], job["opts"])
    shard = int(job["shard"])
    clips = {c["clip_id"]: from_dict(DryClip, c) for c in job["clips"]}
    specs = [RenderSpec(**s) for s in job["specs"]]
    sdir = ctx.layout.shard_dir(shard)
    if sdir.exists():
        shutil.rmtree(sdir)  # partial shard from an interrupted run
    sdir.mkdir(parents=True)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    items = [render_item(s, clips[s.clip_id], shard, ctx, timings) for s in specs]
    write_jsonl(sdir / ITEMS_FILE, items)
    done = {"plan_sha256": job["plan_sha256"], "items": len(items)}
    _write_atomic(sdir / DONE_MARKER, json.dumps(done, sort_keys=True) + "\n")
    return {"shard": shard, "items": len(items), "seconds": time.perf_counter() - t0}


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _done_hash(layout: DataLayout, shard: int) -> str | None:
    """The plan hash a finished shard was rendered from; ``None`` if missing or unreadable
    (treated as partial and re-rendered)."""
    path = layout.shard_dir(shard) / DONE_MARKER
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["plan_sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


PLAN_FILE = "plan.json"


def render_fingerprint(cfg, opts: Options) -> dict:
    """Everything besides the specs that changes a shard's content: render/analysis settings,
    physics calibration, label options and the calibration tables' content hashes."""
    layout = DataLayout.from_config(cfg)
    r = cfg.render
    tables = {}
    for pid in r.whitebox_pedals:
        path = layout.calibration_table(str(pid))
        tables[str(pid)] = levels.sha256_file(path) if path.exists() else None
    return {
        "sample_rate": int(cfg.audio.sample_rate),
        "physics": OmegaConf.to_container(cfg.physics, resolve=True),
        "analysis": OmegaConf.to_container(cfg.analysis, resolve=True),
        "render": {
            k: OmegaConf.to_container(r, resolve=True)[k]
            for k in (
                "wet_peak_dbfs",
                "wet_subtype",
                "whitebox_pedals",
                "whitebox_oversample",
                "input_peak_dbfs",
            )
        },
        "calibration_tables": tables,
        "options": asdict(opts),
    }


def shard_hash(specs: Sequence[RenderSpec], fingerprint: Mapping) -> str:
    blob = json.dumps({"specs": plan_hash(specs), "fp": fingerprint}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# ============================================================================ CLI steps


def fixture_setup(cfg, args, root: Path | None = None) -> DictConfig:
    """Fixture config (``data.cache_dir``/fixtures) with the CLI overrides; builds the
    synthetic dry clips once."""
    from lstmabar.data.fixtures import fixture_config, make_fixture_dataset

    root = root or Path(str(cfg.data.cache_dir)) / "fixtures"
    fcfg = fixture_config(root, list(getattr(args, "overrides", None) or []))
    if not DataLayout.from_config(fcfg).dry_manifest.exists():
        make_fixture_dataset(fcfg, clips_per_performer=2)
    return fcfg


def _options(cfg, args) -> Options:
    fixtures = bool(getattr(args, "fixtures", False))
    allow = bool(getattr(args, "allow_derived_labels", False)) or fixtures
    pitch = librosa_available()
    if not pitch:
        if not fixtures:
            raise ImportError(
                "rendering needs librosa for the pYIN tracks: `uv sync --extra analysis`"
            )
        warnings.warn("librosa missing: harmonic analysis skipped (fixtures only)", stacklevel=2)
    return Options(allow_derived_labels=allow, pitch=pitch)


def _preflight(cfg, opts: Options, kb: Mapping[str, Pedal]) -> None:
    from lstmabar.physics.whitebox import has_whitebox
    from lstmabar.physics.whitebox.diode_clipper import numba_available

    layout = DataLayout.from_config(cfg)
    for pid in cfg.render.whitebox_pedals:
        if pid not in kb or not has_whitebox(str(pid)):
            raise KeyError(f"render.whitebox_pedals: no white-box model for {pid!r}")
        if not layout.calibration_table(str(pid)).exists():
            msg = f"calibration table for {pid!r} missing"
            if not opts.allow_derived_labels:
                raise FileNotFoundError(
                    f"{msg}: run `lstmabar data calibrate` or pass --allow-derived-labels"
                )
            warnings.warn(f"{msg}: labelling {pid} items with derived params", stacklevel=2)
    if not numba_available():
        warnings.warn(
            "numba not installed: the white-box runs the slow NumPy loop "
            "(`uv sync --extra whitebox`)",
            stacklevel=2,
        )


def _clips(cfg, args) -> list[DryClip]:
    layout = DataLayout.from_config(cfg)
    if not layout.dry_manifest.exists():
        raise FileNotFoundError("dry-clip manifest not found: run `lstmabar data ingest` first")
    clips = list(read_jsonl(layout.dry_manifest, DryClip))
    limit = getattr(args, "limit", None)
    return clips[:limit] if limit else clips


def run(cfg, args) -> int:
    if getattr(args, "fixtures", False):
        cfg = fixture_setup(cfg, args)
    opts = _options(cfg, args)
    kb = load_pedals(cfg)
    _preflight(cfg, opts, kb)
    layout = DataLayout.from_config(cfg)
    clips = _clips(cfg, args)
    sampling = Sampling.from_config(cfg)
    specs = plan_specs(clips, sampling, {pid: tuple(p.pots) for pid, p in kb.items()})
    shards = shard_specs(specs, int(cfg.render.shard_size))
    by_id = {c.clip_id: c for c in clips}
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    fp = render_fingerprint(cfg, opts)
    hashes = [shard_hash(shard, fp) for shard in shards]
    layout.wet_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "shards": len(shards),
        "items": len(specs),
        "shard_size": int(cfg.render.shard_size),
        "plan_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
        "shard_sha256": hashes,
        "fingerprint": fp,
    }
    _write_atomic(layout.wet_dir / PLAN_FILE, json.dumps(plan, indent=1, sort_keys=True) + "\n")
    stale = sorted(
        p.name
        for p in layout.wet_dir.glob("shard-*")
        if p.is_dir() and int(p.name.split("-")[1]) >= len(shards)
    )
    if stale:
        warnings.warn(
            f"{len(stale)} shard dirs beyond the plan ({stale[0]}...): not part of this "
            "dataset; delete them",
            stacklevel=2,
        )
    jobs = []
    for i, shard in enumerate(shards):
        h = hashes[i]
        done = _done_hash(layout, i)
        if done == h:
            continue
        if done is not None:
            raise RuntimeError(
                f"{layout.shard_name(i)} is DONE but was rendered from a different plan "
                "(config or manifest changed); delete the wet dir or bump data.version"
            )
        clip_ids = dict.fromkeys(s.clip_id for s in shard)
        jobs.append(
            {
                "cfg": cfg_dict,
                "opts": asdict(opts),
                "shard": i,
                "plan_sha256": h,
                "specs": [asdict(s) for s in shard],
                "clips": [to_dict(by_id[c]) for c in clip_ids],
            }
        )
    _print_plan(specs, len(shards), len(jobs))
    workers = int(getattr(args, "workers", None) or cfg.render.workers)
    workers = max(1, min(workers, len(jobs)))
    t0 = time.perf_counter()
    if workers == 1:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)  # same numerics/threading as the pool workers
        try:
            for job in jobs:
                _report(render_shard(job), t0)
        finally:
            torch.set_num_threads(threads)
    else:
        ctx = multiprocessing.get_context("spawn")  # Windows-safe; no forked torch state
        with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as pool:
            futures = [pool.submit(render_shard, job) for job in jobs]
            for fut in as_completed(futures):
                _report(fut.result(), t0)
    print(f"done: {len(shards)} shards ({time.perf_counter() - t0:.0f} s)")
    return 0


def _print_plan(specs: Sequence[RenderSpec], n_shards: int, n_todo: int) -> None:
    counts: dict[str, int] = defaultdict(int)
    for s in specs:
        counts[f"{s.split}/{s.pedal_id or s.scenario}"] += 1
    print(f"plan: {len(specs)} items in {n_shards} shards ({n_todo} to render)")
    for k in sorted(counts):
        print(f"  {k:32s} {counts[k]:6d}")


def _report(res: dict, t0: float) -> None:
    print(
        f"shard {res['shard']:5d}: {res['items']} items in {res['seconds']:.1f} s "
        f"(elapsed {time.perf_counter() - t0:.0f} s)",
        flush=True,
    )


# ============================================================================ R8 bench

BENCH_REPORT = Path("reports") / "render_bench.md"


def planned_counts(cfg, s: Sampling) -> dict[str, float]:
    """Planned v1 clip and item counts (R7) from ``bench.planned_clips``."""
    pc = {str(k): int(v) for k, v in cfg.bench.planned_clips.items()}
    items = {sp: pc.get(sp, 0) * s.renders_per_clip.get(sp, 0) for sp in ("train", "val", "test")}
    items[UNSEEN_SPLIT] = pc.get("test", 0) * s.unseen_renders_per_clip
    return {"clips": float(sum(pc.values())), **{k: float(v) for k, v in items.items()}}


def bench(cfg, args) -> int:
    """R8 throughput gate: time render + analysis per item on fixtures, extrapolate to v1."""
    import tempfile

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="lstmabar-bench-") as tmp:
        tmp = Path(tmp)
        if (
            getattr(args, "fixtures", False)
            or not DataLayout.from_config(cfg).dry_manifest.exists()
        ):
            cfg = fixture_setup(cfg, args, root=tmp / "fixtures")
        opts = Options(allow_derived_labels=True, pitch=librosa_available())
        if not opts.pitch:
            raise ImportError("bench measures pYIN: `uv sync --extra analysis`")
        cfg = OmegaConf.merge(cfg, {"data": {"cache_dir": str(tmp / "cache")}})
        result = _bench(cfg, args, opts, tmp)
    print(result)
    BENCH_REPORT.parent.mkdir(parents=True, exist_ok=True)
    BENCH_REPORT.write_text(result + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {BENCH_REPORT.as_posix()}")
    return 0


def _bench(cfg, args, opts: Options, tmp: Path) -> str:
    from lstmabar.physics.whitebox.diode_clipper import numba_available

    s = Sampling.from_config(cfg)
    ctx = RenderContext(cfg, opts)
    pots = {pid: tuple(p.pots) for pid, p in ctx.kb.items()}
    clips = _clips(cfg, args)[: int(getattr(args, "limit", None) or cfg.bench.max_clips)]
    sr = ctx.sr
    per_pedal = int(cfg.bench.items_per_pedal)
    frames_opts = sorted({int(v) for v in cfg.bench.max_frames_options} | {ctx.max_frames})[::-1]

    # Warm-up (JIT: numba kernels, librosa's pYIN), untimed.
    first, _ = levels.read_audio(ctx.layout.resolve_rendered(clips[0].path))
    warm = levels.apply_input_level(first[: sr // 2], -12.0)
    track_f0(warm, sr)
    for pid in s.whitebox_pedals:
        ctx.render_whitebox(ctx.kb[pid], {}, warm)

    # Per-clip costs: pYIN + dry analysis at each max_frames option.
    clip_t: dict[str, list[float]] = defaultdict(list)
    tracks: dict[str, F0Track] = {}
    for clip in clips:
        x, _ = levels.read_audio(ctx.layout.resolve_rendered(clip.path))
        t0 = time.perf_counter()
        tracks[clip.clip_id] = track_f0(x, sr)
        clip_t["pyin"].append(time.perf_counter() - t0)
        save_track(ctx.layout.f0_cache(clip.clip_id), tracks[clip.clip_id], clip.sha256)
        for mf in frames_opts:
            t0 = time.perf_counter()
            analyse(x, sr, tracks[clip.clip_id], True, ctx.n_harmonics, mf)
            clip_t[f"dry_analysis@{mf}"].append(time.perf_counter() - t0)

    # Per-item costs: every KB pedal x items_per_pedal, and each free scenario once.
    cases = [("pedal", pid) for pid in sorted(pots) for _ in range(per_pedal)]
    cases += [(sc, None) for sc in FREE_SCENARIOS]
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    results: dict[int, list[tuple[dict, dict]]] = defaultdict(list)  # wet analysis per option
    for k, (scenario, pid) in enumerate(cases):
        clip = clips[k % len(clips)]
        spec = sample_spec(
            clip.clip_id, clip.split or "train", 1000 + k, s, pots, force=(scenario, pid)
        )
        t: dict[str, float] = {}
        ctx.dry(clip)  # dry cost measured above; keep it out of the item timing
        item = render_item(spec, clip, 0, ctx, t, out_root=tmp / "wet")
        stored, _ = levels.read_audio(tmp / "wet" / item.wet_path)
        for mf in frames_opts:
            t0 = time.perf_counter()
            out = analyse(stored, sr, tracks[clip.clip_id], True, ctx.n_harmonics, mf)
            t[f"wet_analysis@{mf}"] = time.perf_counter() - t0
            results[mf].append(out)
        t.pop("wet_analysis", None)
        cat = spec.engine if scenario == "pedal" else "free"
        if scenario == "pedal" and pid == s.holdout_pedal:
            cat = f"holdout_{spec.engine}"
        rows[cat].append(t)
        rows[f"pedal:{pid}" if pid else f"free:{scenario}"].append(t)

    noise = _analysis_noise(results, frames_opts)
    return _bench_report(cfg, s, ctx, clips, clip_t, rows, frames_opts, noise, numba_available())


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([r.get(key, 0.0) for r in rows])) if rows else 0.0


def _analysis_noise(results: Mapping[int, list], frames_opts: Sequence[int]) -> dict[int, str]:
    """Wet-analysis change vs the largest ``max_frames`` option, per option (bench report)."""
    ref = results[max(frames_opts)]
    out = {}
    for mf in frames_opts:
        d_oe, d_arch = [], []
        for (desc, arch), (rdesc, rarch) in zip(results[mf], ref, strict=True):
            d_oe.append(abs(desc["odd_even_db"] - rdesc["odd_even_db"]))
            d_arch.append(sum(abs(arch[k] - rarch[k]) for k in ARCHETYPE_KEYS))
        d_oe = np.array(d_oe)[np.isfinite(d_oe)]
        d_arch = np.array(d_arch)[np.isfinite(d_arch)]
        if not len(d_oe) or not len(d_arch):
            out[mf] = "n/a | n/a"
            continue
        out[mf] = (
            f"{np.median(d_oe):.2f} / {np.max(d_oe):.2f} | "
            f"{np.median(d_arch):.3f} / {np.max(d_arch):.3f}"
        )
    return out


def _bench_report(cfg, s, ctx, clips, clip_t, rows, frames_opts, noise, numba: bool) -> str:
    workers = int(cfg.render.workers)
    counts = planned_counts(cfg, s)
    n_items = sum(v for k, v in counts.items() if k != "clips")
    n_regular = counts["train"] + counts["val"] + counts["test"]
    n_pedals = len([p for p in ctx.kb if p != s.holdout_pedal])
    wb = [p for p in s.whitebox_pedals if p != s.holdout_pedal and p in ctx.kb]
    share_wb = s.pedal_fraction * len(wb) / max(n_pedals, 1)
    share = {
        "whitebox": share_wb,
        "greybox": s.pedal_fraction - share_wb,
        "free": 1 - s.pedal_fraction,
    }
    holdout_cat = "holdout_" + ("whitebox" if s.holdout_pedal in s.whitebox_pedals else "greybox")
    budget_h = float(cfg.bench.budget_hours)

    def item_cost(cat: str, mf: int) -> float:
        r = rows[cat]
        return sum(
            _mean(r, k)
            for k in ("derive", "render_whitebox", "render_greybox", "write", f"wet_analysis@{mf}")
        )

    def total_hours(mf: int) -> float:
        per_clip = float(np.mean(clip_t["pyin"])) + float(np.mean(clip_t[f"dry_analysis@{mf}"]))
        cpu = counts["clips"] * per_clip
        cpu += n_regular * sum(share[c] * item_cost(c, mf) for c in share)
        cpu += counts[UNSEEN_SPLIT] * item_cost(holdout_cat, mf)
        return cpu / workers / 3600

    lines = [
        "# Render throughput benchmark (R8)",
        "",
        "Generated by `lstmabar data bench"
        + (" --fixtures" if clips and clips[0].source == "fixture" else "")
        + "`. Single process, 1 torch thread, wall-clock `perf_counter` per stage on "
        f"{len(clips)} dry clips of {clips[0].duration_s:g} s ({clips[0].source}); "
        f"white-box {ctx.oversample}x oversampled, numba {'on' if numba else 'OFF'}; "
        f"`analysis.max_frames` = {ctx.max_frames}. JIT warm-up excluded. Timings are noisy "
        "if other jobs share the CPU.",
        "",
        "## Per-item cost (seconds, mean)",
        "",
        "| category | n | derive | render | write FLAC | wet analysis |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    cats = ["whitebox", "greybox", holdout_cat, "free"] + sorted(k for k in rows if ":" in k)
    for c in cats:
        r = rows.get(c, [])
        if not r:
            continue
        render = _mean(r, "render_whitebox") + _mean(r, "render_greybox")
        lines.append(
            f"| {c} | {len(r)} | {_mean(r, 'derive'):.3f} | {render:.3f} | "
            f"{_mean(r, 'write'):.3f} | {_mean(r, f'wet_analysis@{ctx.max_frames}'):.3f} |"
        )
    lines += [
        "",
        f"Per dry clip (once): pYIN {np.mean(clip_t['pyin']):.3f} s, dry analysis "
        f"{np.mean(clip_t[f'dry_analysis@{ctx.max_frames}']):.3f} s.",
        "",
        "## Extrapolation to the planned v1 (R7)",
        "",
        f"Planned clips (`bench.planned_clips`): {dict(cfg.bench.planned_clips)} -> "
        f"{int(counts['clips'])} clips, {int(n_items)} items "
        f"(train {int(counts['train'])}, val {int(counts['val'])}, test {int(counts['test'])}, "
        f"{UNSEEN_SPLIT} {int(counts[UNSEEN_SPLIT])}). Item mix from the config "
        f"(R5): white-box {share['whitebox']:.0%}, grey-box pedal {share['greybox']:.0%}, "
        f"free board {share['free']:.0%}; unseen items are {holdout_cat}. "
        f"Wall time = CPU time / `render.workers` ({workers}), assuming linear scaling.",
        "",
        f"Analysis noise: change of the wet analysis vs `max_frames={max(frames_opts)}` on the "
        f"same {sum(len(rows[c]) for c in (*share, holdout_cat))} bench renders: |Δ odd/even| "
        "in dB and the L1 distance of the 5-way archetype vector, as median / max.",
        "",
        "| analysis.max_frames | wet analysis s/item | full v1 (h) | per 20k items (h) | "
        f"within {budget_h:g} h | odd/even abs Δ dB (med / max) | archetype L1 (med / max) |",
        "|---:|---:|---:|---:|---|---|---|",
    ]
    for mf in frames_opts:
        h = total_hours(mf)
        wa = sum(share[c] * _mean(rows[c], f"wet_analysis@{mf}") for c in share)
        mark = " (current)" if mf == ctx.max_frames else ""
        lines.append(
            f"| {mf}{mark} | {wa:.3f} | {h:.1f} | {h * 20000 / n_items:.1f} | "
            f"{'yes' if h <= budget_h else 'no'} | {noise[mf]} |"
        )
    current = total_hours(ctx.max_frames)
    headroom = float(cfg.bench.headroom)
    target = budget_h * headroom
    ok = [mf for mf in frames_opts if total_hours(mf) <= target]
    lines += [
        "",
        f"Choice rule (R8): the largest `max_frames` whose estimate is within {headroom:g} x "
        f"the budget (= {target:g} h; `bench.headroom` covers timing noise and the unverified "
        "linear scaling over workers).",
        "",
    ]
    if ok and max(ok) != ctx.max_frames:
        lines.append(
            f"**Gate: {'pass' if current <= budget_h else 'fail'}** at the current "
            f"`max_frames={ctx.max_frames}` (~{current:.1f} h). Recommended: "
            f"`analysis.max_frames={max(ok)}` (~{total_hours(max(ok)):.1f} h)."
        )
    elif ok:
        lines.append(
            f"**Gate: pass.** The current `max_frames={ctx.max_frames}` renders v1 in "
            f"~{current:.1f} h and is the largest option within {target:g} h."
        )
    else:
        lines.append(
            f"**Gate: fail** (~{current:.1f} h even at the smallest max_frames): reduce the "
            "analysis further (subset) or render on more cores."
        )
    return "\n".join(lines)


__all__ = [
    "CLAMP_MARKER",
    "Options",
    "RenderContext",
    "RenderSpec",
    "Sampling",
    "analyse",
    "bench",
    "clamp_notes",
    "item_seed",
    "load_track",
    "plan_hash",
    "plan_specs",
    "planned_counts",
    "render_item",
    "render_shard",
    "run",
    "sample_spec",
    "save_track",
    "scenario_gates",
    "shard_specs",
]
