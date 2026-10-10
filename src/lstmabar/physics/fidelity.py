"""Grey-box fidelity check (execution plan P2.5, kickoff decision 6).

How close can the differentiable grey-box (``Drive`` + ``EQ3`` of
:func:`~lstmabar.dsp.pedalboard.default_pedalboard`, compressor off) get to a pedal's
white-box circuit simulation, and how close do the circuit derivations
(:func:`lstmabar.physics.derive.derive`) put it without fitting?

Procedure, per pedal with a white-box model and per knob setting (gain knob min/mid/max at
tone mid, plus tone min/max at gain mid):

1. **Inputs.** Sines at 82.41, 164.81, 329.63, 659.26 Hz (E2, E3, E4, E5) × -30, -20, -10,
   0 dBFS (0 dBFS = ``volts_per_fs`` volts peak, P2 decision 3), each ``settle`` + ``seconds``
   + ``tail`` long, starting at zero phase. Only the middle ``seconds`` are analysed: the
   ``settle`` pre-roll (0.15 s) covers the onset transient (≥ 15 time constants of the
   slowest modelled pole, the TS808's 16 Hz input coupling, τ = 10 ms; the grey-box
   pre-HPF bottoms out at 20 Hz), and the ``tail`` (20 ms ≥ the oversamplers' 32-sample
   half-width at either rate) drops the zero-phase resampler's end transient. Plus a
   Karplus-Strong riff (:func:`lstmabar.dsp.signals.pluck_riff`) for MR-STFT.
2. **Metric** (per clip, then averaged): ``|ΔH1|`` in dB (absolute level, no floor, so a
   silent render shows up) and ``|ΔHk|`` in dBc for k = 2..10 with both sides floored at
   ``floor_dbc`` (-60 dBc), from :func:`lstmabar.analysis.harmonics.harmonic_amplitudes` at
   the exact f0. The *harmonic error* of a setting is the mean of those 10 terms over the 16
   clips; its max is the largest single term. Reported per input level too, with the signed
   H1 error (grey minus white) that exposes level-dependent gaps such as the TS808 clean
   path. MR-STFT (:func:`lstmabar.dsp.losses.multi_resolution_stft_loss`) on the riff is
   reported alongside.
3. **Best fit.** One grey-box parameter set per (pedal, setting), fitted jointly to all
   pitches × levels (+ the riff). Loss = the metric itself, made differentiable by a torch
   harmonic projection (:class:`HarmonicProjector`, the same weighted least-squares fit as
   ``harmonic_amplitudes``) with an L1 on the 10 terms, + ``mrstft_weight`` × MR-STFT on the
   riff (edges cropped). Fitting uses the first ``fit_seconds`` of each analysed window with
   a shorter grey-box pre-roll ``fit_settle`` (for speed); the reported metric always uses
   the full window and pre-roll. Multi-start: the
   derived parameters (when the pedal has a deriver; else every knob at 0.5) plus
   ``restarts`` uniform-random starts, all optimized together (Adam on sigmoid logits, cosine
   decay, as in :mod:`lstmabar.dsp.recovery`). The start with the lowest final metric wins.
4. **Derived.** The same metric for :func:`~lstmabar.physics.derive.derive` +
   :func:`~lstmabar.physics.derive.preset_params`, unfitted; skipped (with a note) for
   pedals without a deriver.
5. **Gate** (P2 exit): best-fit harmonic error, averaged over the pedal's settings, ≤
   ``gate_db`` for TS808 and DS-1; pedals listed in :data:`DIAGNOSTIC` are reported without a
   gate.

Default rate 22.05 kHz (like :mod:`lstmabar.dsp.recovery`): H10 of the highest test pitch is
6.6 kHz, below the 11 kHz Nyquist. The white-box runs 8× oversampled there (176 kHz, the
same clipper rate as 4× at 44.1 kHz); the grey-box ``Drive`` keeps its 4×.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from lstmabar.analysis.harmonics import blackman_harris, harmonic_amplitudes, to_db, to_dbc
from lstmabar.dsp.losses import multi_resolution_stft_loss
from lstmabar.dsp.pedalboard import ENABLED, BoardParams, Pedalboard, default_pedalboard
from lstmabar.dsp.signals import pluck_riff
from lstmabar.physics.calibration import VOLTS_PER_FULL_SCALE
from lstmabar.physics.derive import DERIVERS, DeriveContext, derive, preset_params
from lstmabar.physics.kb import Pedal, load_kb
from lstmabar.physics.whitebox import WHITEBOX

FIT_BLOCKS = ("drive", "eq")
GAIN_KNOB = {"ts808": "drive", "ds1": "dist", "rat": "distortion"}
TONE_KNOB = {"ts808": "tone", "ds1": "tone", "rat": "filter"}
DIAGNOSTIC = ("rat",)
"""White-box pedals reported without a gate (P2: a RAT sim with ideal op-amps is diagnostic)."""
GATED = ("ts808", "ds1")


@dataclass
class FidelityConfig:
    seed: int = 0
    sample_rate: int = 22050
    pitches_hz: tuple[float, ...] = (82.41, 164.81, 329.63, 659.26)
    levels_dbfs: tuple[float, ...] = (-30.0, -20.0, -10.0, 0.0)
    seconds: float = 0.5
    settle: float = 0.15
    tail: float = 0.02
    fit_seconds: float = 0.2
    fit_settle: float = 0.05
    """Grey-box pre-roll during fitting (its slowest pole, the 20 Hz pre-HPF floor, has
    τ ≈ 8 ms); the reported metric always uses the full ``settle``."""
    n_harmonics: int = 10
    floor_dbc: float = -60.0
    riff_seconds: float = 1.0
    riff_peak: float = 0.5
    riff_crop: int = 512
    whitebox_oversample: int = 8
    drive_oversample: int = 4
    restarts: int = 2
    steps: int = 100
    lr: float = 0.05
    mrstft_weight: float = 1.0
    gate_db: float = 3.0
    volts_per_fs: float = VOLTS_PER_FULL_SCALE
    pedals: tuple[str, ...] | None = None
    """Pedal ids to check; ``None`` = every pedal with a white-box model."""
    settings: tuple[str, ...] | None = None
    """Knob-setting labels to run (see :func:`knob_settings`); ``None`` = all five."""
    quick: bool = False

    @classmethod
    def quick_config(cls, **kw) -> FidelityConfig:
        """A seconds-long smoke configuration (2 pitches × 2 levels, 1 setting, few steps)."""
        base = dict(
            pitches_hz=(164.81, 329.63),
            levels_dbfs=(-20.0, 0.0),
            seconds=0.2,
            settle=0.08,
            fit_seconds=0.1,
            fit_settle=0.04,
            riff_seconds=0.5,
            restarts=1,
            steps=8,
            settings=("gain mid",),
            quick=True,
        )
        base.update(kw)
        return cls(**base)


# --- Metric -------------------------------------------------------------------------------------


def harmonic_errors(
    amps_grey: np.ndarray, amps_white: np.ndarray, floor_dbc: float = -60.0
) -> np.ndarray:
    """Per-term absolute errors ``(..., K)``: ``|ΔH1|`` dB (no floor), then ``|ΔHk|`` dBc
    with both sides floored at ``floor_dbc``."""
    h1 = np.abs(to_db(amps_grey[..., :1]) - to_db(amps_white[..., :1]))
    dbc = np.abs(to_dbc(amps_grey, floor_dbc)[..., 1:] - to_dbc(amps_white, floor_dbc)[..., 1:])
    return np.concatenate([h1, dbc], axis=-1)


def summarize(
    amps_grey: np.ndarray,
    amps_white: np.ndarray,
    levels: np.ndarray,
    floor_dbc: float = -60.0,
) -> dict:
    """Summary of :func:`harmonic_errors` over clips ``(C, K)`` with input level per clip."""
    err = harmonic_errors(amps_grey, amps_white, floor_dbc)
    signed_h1 = (to_db(amps_grey[:, 0]) - to_db(amps_white[:, 0])).astype(float)
    out = {
        "mean_db": float(np.mean(err)),
        "max_db": float(np.max(err)),
        "h1_mean_db": float(np.mean(err[:, 0])),
        "h1_max_db": float(np.max(err[:, 0])),
        "dbc_mean_db": float(np.mean(err[:, 1:])),
        "dbc_max_db": float(np.max(err[:, 1:])),
        "per_level": {},
    }
    for lv in sorted(set(levels.tolist())):
        m = levels == lv
        out["per_level"][f"{lv:g}"] = {
            "mean_db": float(np.mean(err[m])),
            "h1_mean_db": float(np.mean(err[m, 0])),
            "h1_signed_db": float(np.mean(signed_h1[m])),
            "dbc_mean_db": float(np.mean(err[m, 1:])),
        }
    return out


class HarmonicProjector:
    """Differentiable version of :func:`~lstmabar.analysis.harmonics.harmonic_amplitudes`.

    Precomputes, per clip, the weighted least-squares projection (Blackman-Harris weights,
    design ``[1, cos(2π k f0 t), sin(2π k f0 t)]``, centred time axis) onto harmonics
    ``1..K`` of a known ``f0``; calling it is one batched matmul. ``f0s`` has one entry per
    clip (row of the input). Harmonics must lie below Nyquist.
    """

    def __init__(self, f0s, n_samples: int, sample_rate: float, n_harmonics: int = 10):
        sw = np.sqrt(blackman_harris(n_samples))[:, None]
        tc = (np.arange(n_samples) - (n_samples - 1) / 2) / sample_rate
        k = np.arange(1, n_harmonics + 1)
        mats = {}
        for f0 in sorted(set(float(f) for f in f0s)):
            if n_harmonics * f0 >= 0.499 * sample_rate:
                raise ValueError(f"H{n_harmonics} of {f0} Hz is above Nyquist at {sample_rate}")
            arg = 2 * np.pi * f0 * np.outer(tc, k)
            design = np.concatenate([np.ones((n_samples, 1)), np.cos(arg), np.sin(arg)], axis=1)
            mats[f0] = np.linalg.pinv(design * sw) * sw.T  # (1 + 2K, T)
        self.k = n_harmonics
        self.proj = torch.from_numpy(np.stack([mats[float(f)] for f in f0s]))  # (C, 1+2K, T)

    def __call__(self, x: Tensor, eps: float = 1e-20) -> Tensor:
        """``x`` ``(..., C, T)`` → peak amplitudes ``(..., C, K)``."""
        coef = torch.einsum("cjt,...ct->...cj", self.proj.to(x.dtype), x)
        a, b = coef[..., 1 : 1 + self.k], coef[..., 1 + self.k :]
        return torch.sqrt(a.square() + b.square() + eps)


def harmonic_loss(amps_grey: Tensor, amps_white: Tensor, floor_dbc: float = -60.0) -> Tensor:
    """Mean of :func:`harmonic_errors` in torch over the last two axes (clips, terms)."""
    db_g = 20 * torch.log10(amps_grey)
    db_w = 20 * torch.log10(amps_white)
    h1 = (db_g[..., :1] - db_w[..., :1]).abs()
    rel_g = (db_g[..., 1:] - db_g[..., :1]).clamp_min(floor_dbc)
    rel_w = (db_w[..., 1:] - db_w[..., :1]).clamp_min(floor_dbc)
    return torch.cat([h1, (rel_g - rel_w).abs()], dim=-1).mean(dim=(-2, -1))


# --- Signals ------------------------------------------------------------------------------------


@dataclass
class Signals:
    sines: np.ndarray  # (C, T) full scale, float64
    f0s: np.ndarray  # (C,)
    levels: np.ndarray  # (C,) dBFS
    riff: np.ndarray  # (T_riff,)
    start: int  # first analysed sample
    length: int  # analysed samples (metric)
    fit_length: int  # analysed samples (fit)
    tail: int


def make_signals(cfg: FidelityConfig) -> Signals:
    sr = cfg.sample_rate
    start, length = round(cfg.settle * sr), round(cfg.seconds * sr)
    tail = round(cfg.tail * sr)
    t = np.arange(start + length + tail) / sr
    f0s, levels, rows = [], [], []
    for lv in cfg.levels_dbfs:
        for f in cfg.pitches_hz:
            rows.append(10 ** (lv / 20) * np.sin(2 * np.pi * f * t))
            f0s.append(f)
            levels.append(lv)
    riff = pluck_riff(cfg.riff_seconds, sr, peak=cfg.riff_peak, seed=cfg.seed).double().numpy()
    return Signals(
        np.stack(rows),
        np.array(f0s),
        np.array(levels),
        riff,
        start,
        length,
        min(round(cfg.fit_seconds * sr), length),
        tail,
    )


# --- Knob settings ------------------------------------------------------------------------------


def knob_settings(pedal: Pedal) -> list[tuple[str, dict[str, float]]]:
    """Gain knob min/mid/max at tone mid, then tone min/max at gain mid (other knobs default)."""
    g, t = GAIN_KNOB[pedal.id], TONE_KNOB[pedal.id]
    raw = [
        ("gain min", {g: 0.0, t: 0.5}),
        ("gain mid", {g: 0.5, t: 0.5}),
        ("gain max", {g: 1.0, t: 0.5}),
        ("tone min", {g: 0.5, t: 0.0}),
        ("tone max", {g: 0.5, t: 1.0}),
    ]
    return [(label, pedal.knobs(k)) for label, k in raw]


# --- Grey-box -----------------------------------------------------------------------------------


def _board(cfg: FidelityConfig) -> Pedalboard:
    board = default_pedalboard(cfg.sample_rate)
    board.blocks["drive"].oversample = cfg.drive_oversample
    return board


def fit_knobs(board: Pedalboard) -> list[tuple[str, str]]:
    return [(b, k) for b, k in board.layout() if b in FIT_BLOCKS and k != ENABLED]


def board_params(knobs: list[tuple[str, str]], u: Tensor) -> BoardParams:
    """Params for normalized knob values ``u`` ``(B, K)``: compressor off, Drive + EQ on."""
    n = u.shape[0]
    p: BoardParams = {"compressor": {ENABLED: u.new_zeros(n)}}
    for name in FIT_BLOCKS:
        p[name] = {ENABLED: u.new_ones(n)}
    for j, (name, knob) in enumerate(knobs):
        p[name][knob] = u[:, j]
    return p


def physical(board: Pedalboard, knobs: list[tuple[str, str]], u: Tensor) -> dict[str, float]:
    """``{"block.knob": physical value}`` for one normalized vector ``u`` ``(K,)``."""
    out = {}
    for j, (b, k) in enumerate(knobs):
        spec = next(s for s in board.blocks[b].param_specs if s.name == k)
        out[f"{b}.{k}"] = float(spec.denormalize(u[j].double()))
    return out


def derived_vector(
    pedal: Pedal, knobs: dict[str, float], fit: list[tuple[str, str]], ctx: DeriveContext
) -> tuple[Tensor, tuple[str, ...]]:
    preset = derive(pedal, knobs, ctx)
    params = preset_params(preset)
    if params["compressor"][ENABLED].item() != 0.0:
        raise ValueError(f"{pedal.id}: derived preset enables the compressor")
    u = torch.stack([params[b][k][0] for b, k in fit]).float()
    return u, tuple(preset.notes)


@torch.no_grad()
def render(board: Pedalboard, knobs, u: Tensor, x: np.ndarray) -> np.ndarray:
    """Grey-box renders of clips ``x`` ``(C, T)`` for each row of ``u`` ``(S, K)``:
    ``(S, C, T)``."""
    s, c = u.shape[0], x.shape[0]
    xt = torch.from_numpy(x).float().repeat(s, 1)
    y = board(xt, board_params(knobs, u.repeat_interleave(c, dim=0)))
    return y.reshape(s, c, -1).double().numpy()


def _riff_mrstft(y: np.ndarray, target: np.ndarray, crop: int) -> float:
    a = torch.from_numpy(y[..., crop : y.shape[-1] - crop]).reshape(1, -1).float()
    b = torch.from_numpy(target[crop : target.shape[-1] - crop]).reshape(1, -1).float()
    return float(multi_resolution_stft_loss(a, b))


def _amps(y: np.ndarray, sig: Signals, cfg: FidelityConfig) -> np.ndarray:
    seg = y[..., sig.start : sig.start + sig.length]
    return np.stack(
        [
            harmonic_amplitudes(seg[..., i, :], cfg.sample_rate, float(f), cfg.n_harmonics)
            for i, f in enumerate(sig.f0s)
        ],
        axis=-2,
    )


def evaluate(
    board, knobs, u: Tensor, sig: Signals, white_amps, white_riff, cfg: FidelityConfig
) -> list[dict]:
    """Metric summary for each row of ``u`` ``(S, K)``."""
    y = render(board, knobs, u, sig.sines)
    yr = render(board, knobs, u, sig.riff[None])[:, 0]
    amps = _amps(y, sig, cfg)
    out = []
    for i in range(u.shape[0]):
        s = summarize(amps[i], white_amps, sig.levels, cfg.floor_dbc)
        s["mrstft_riff"] = _riff_mrstft(yr[i], white_riff, cfg.riff_crop)
        out.append(s)
    return out


def fit(
    board: Pedalboard,
    knobs,
    u0: Tensor,
    sig: Signals,
    white: np.ndarray,
    white_riff: np.ndarray,
    cfg: FidelityConfig,
) -> tuple[Tensor, list[float]]:
    """Adam on logits from starts ``u0`` ``(S, K)`` (all starts optimized in one batch; each
    start's loss depends only on its own logits). Returns ``(u (S, K), final losses)``."""
    s, c = u0.shape[0], sig.sines.shape[0]
    # The grey-box sees the same sines from ``fit_settle`` before the analysed window (a
    # shorter pre-roll than the white-box's, for speed; phase-continuous with it).
    lead = min(round(cfg.fit_settle * cfg.sample_rate), sig.start)
    first, stop = sig.start - lead, sig.start + sig.fit_length + sig.tail
    x = torch.from_numpy(sig.sines[:, first:stop]).float().repeat(s, 1)
    proj = HarmonicProjector(sig.f0s, sig.fit_length, cfg.sample_rate, cfg.n_harmonics)
    seg = slice(lead, lead + sig.fit_length)
    target = white[:, sig.start : sig.start + sig.fit_length]
    target_amps = proj(torch.from_numpy(target).float())
    xr = torch.from_numpy(sig.riff).float().expand(s, -1)
    cr = cfg.riff_crop
    tr = torch.from_numpy(white_riff[cr : len(white_riff) - cr]).float().expand(s, -1)

    logits = torch.logit(u0.clamp(0.02, 0.98)).clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr * 0.05)

    def losses() -> Tensor:
        u = logits.sigmoid()
        y = board(x, board_params(knobs, u.repeat_interleave(c, dim=0))).reshape(s, c, -1)
        harm = harmonic_loss(proj(y[..., seg]), target_amps, cfg.floor_dbc)  # (S,)
        if cfg.mrstft_weight == 0:
            return harm
        yr = board(xr, board_params(knobs, u))[:, cr : xr.shape[-1] - cr]
        mr = torch.stack(
            [multi_resolution_stft_loss(yr[i : i + 1], tr[i : i + 1]) for i in range(s)]
        )
        return harm + cfg.mrstft_weight * mr

    for _ in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        losses().sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        final = losses()
    return logits.detach().sigmoid(), final.tolist()


# --- Driver -------------------------------------------------------------------------------------


@dataclass
class FidelityResult:
    config: FidelityConfig
    pedals: dict
    gate: dict
    seconds_elapsed: float
    torch_threads: int = field(default_factory=torch.get_num_threads)


def _white(pedal, knobs, sig: Signals, cfg: FidelityConfig) -> tuple[np.ndarray, np.ndarray]:
    model = WHITEBOX[pedal.id](pedal, knobs, cfg.sample_rate, oversample=cfg.whitebox_oversample)
    v = cfg.volts_per_fs
    y = model.process_volts(sig.sines * v) / v
    yr = model.process_volts(sig.riff[None] * v)[0] / v
    if not (np.isfinite(y).all() and np.isfinite(yr).all()):
        raise RuntimeError(f"{pedal.id}: non-finite white-box output at {knobs}")
    return y, yr


def check_setting(
    pedal: Pedal, label: str, knobs: dict[str, float], sig: Signals, cfg: FidelityConfig, idx: int
) -> dict:
    board = _board(cfg)
    fk = fit_knobs(board)
    white, white_riff = _white(pedal, knobs, sig, cfg)
    white_amps = _amps(white, sig, cfg)
    ctx = DeriveContext(volts_per_fs=cfg.volts_per_fs, sample_rate=cfg.sample_rate)
    row: dict = {"label": label, "knobs": knobs, "notes": []}

    if pedal.id in DERIVERS:
        u_d, notes = derived_vector(pedal, knobs, fk, ctx)
        row["derived"] = evaluate(board, fk, u_d[None], sig, white_amps, white_riff, cfg)[0]
        row["derived_params"] = physical(board, fk, u_d)
        row["derive_notes"] = list(notes)
        first, first_name = u_d, "derived"
    else:
        row["derived"] = None
        row["derived_params"] = None
        row["notes"].append(f"no deriver for {pedal.id!r}: derived error skipped")
        first, first_name = torch.full((len(fk),), 0.5), "all knobs 0.5"

    gen = torch.Generator().manual_seed(cfg.seed * 7919 + idx)
    rand = 0.1 + 0.8 * torch.rand(cfg.restarts, len(fk), generator=gen)
    u0 = torch.cat([first[None], rand])
    u_fit, final_loss = fit(board, fk, u0, sig, white, white_riff, cfg)
    evals = evaluate(board, fk, u_fit, sig, white_amps, white_riff, cfg)
    best = min(range(len(evals)), key=lambda i: evals[i]["mean_db"])
    row["starts"] = [
        {
            "init": first_name if i == 0 else f"random {i}",
            "final_loss": final_loss[i],
            "mean_db": evals[i]["mean_db"],
        }
        for i in range(len(evals))
    ]
    row["best_start"] = row["starts"][best]["init"]
    row["best"] = evals[best]
    row["best_params"] = physical(board, fk, u_fit[best])
    row["white_h1_dbfs"] = {
        f"{lv:g}": float(np.mean(to_db(white_amps[sig.levels == lv, 0]))) for lv in cfg.levels_dbfs
    }
    return row


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_fidelity(cfg: FidelityConfig | None = None, log=_log) -> FidelityResult:
    cfg = cfg or FidelityConfig()
    start = time.perf_counter()
    torch.manual_seed(cfg.seed)
    kb = load_kb()
    ids = cfg.pedals or tuple(p for p in kb if p in WHITEBOX)
    sig = make_signals(cfg)
    pedals = {}
    for pid in ids:
        pedal = kb[pid]
        settings = [
            (lb, k) for lb, k in knob_settings(pedal) if cfg.settings is None or lb in cfg.settings
        ]
        rows = []
        for i, (label, knobs) in enumerate(settings):
            t0 = time.perf_counter()
            row = check_setting(pedal, label, knobs, sig, cfg, idx=len(pedals) * 100 + i)
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
            d = row["derived"]
            log(
                f"{pid} {label}: best {row['best']['mean_db']:.2f} dB"
                + (f", derived {d['mean_db']:.2f} dB" if d else ", derived n/a")
                + f" ({row['seconds']:.0f}s)"
            )
        pedals[pid] = {
            "name": pedal.name,
            "has_deriver": pid in DERIVERS,
            "diagnostic": pid in DIAGNOSTIC,
            "notes": list(
                getattr(WHITEBOX[pid](pedal, pedal.knobs(), cfg.sample_rate), "notes", ())
            ),
            "settings": rows,
        }
    gate = {}
    for pid, p in pedals.items():
        means = [r["best"]["mean_db"] for r in p["settings"]]
        avg = float(np.mean(means))
        gate[pid] = {
            "gated": pid in GATED,
            "best_mean_db": avg,
            "worst_setting_mean_db": float(np.max(means)),
            "settings_over": [
                r["label"] for r in p["settings"] if r["best"]["mean_db"] > cfg.gate_db
            ],
            "passed": bool(avg <= cfg.gate_db),
            "derived_mean_db": (
                float(np.mean([r["derived"]["mean_db"] for r in p["settings"]]))
                if p["has_deriver"]
                else None
            ),
        }
    return FidelityResult(cfg, pedals, gate, time.perf_counter() - start)


# --- Report -------------------------------------------------------------------------------------


def _f(v, nd=2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _metric_rows(pid: str, p: dict) -> list[str]:
    rows = []
    for r in p["settings"]:
        b, d = r["best"], r["derived"]
        knobs = ", ".join(f"{k} {v:g}" for k, v in r["knobs"].items())
        rows.append(
            f"| {r['label']} ({knobs}) | {b['mean_db']:.2f} | {b['max_db']:.1f} | "
            f"{b['h1_mean_db']:.2f} | {b['dbc_mean_db']:.2f} | {b['mrstft_riff']:.3f} | "
            f"{_f(d and d['mean_db'])} | {_f(d and d['max_db'], 1)} | "
            f"{_f(d and d['h1_mean_db'])} | {_f(d and d['dbc_mean_db'])} | "
            f"{_f(d and d['mrstft_riff'], 3)} |"
        )
    return rows


def _level_rows(p: dict, which: str) -> list[str]:
    rows = []
    for r in p["settings"]:
        m = r[which]
        if m is None:
            continue
        cells = " | ".join(
            f"{v['mean_db']:.2f} / {v['h1_signed_db']:+.2f} / {v['dbc_mean_db']:.2f}"
            for v in m["per_level"].values()
        )
        rows.append(f"| {r['label']} | {cells} |")
    return rows


def _param_rows(p: dict) -> list[str]:
    names = list(p["settings"][0]["best_params"])
    head = "| knob | " + " | ".join(r["label"] for r in p["settings"]) + " |"
    rows = [head, "|---|" + "---|" * len(p["settings"])]
    for n in names:
        cells = []
        for r in p["settings"]:
            fit_v = r["best_params"][n]
            der = r["derived_params"]
            cells.append(f"{fit_v:.3g}" + (f" ({der[n]:.3g})" if der else ""))
        rows.append(f"| `{n}` | " + " | ".join(cells) + " |")
    return rows


def format_report(r: FidelityResult, discussion: str = "") -> str:
    c = r.config
    levels = [f"{lv:g}" for lv in c.levels_dbfs]
    lines = [
        "# Grey-box fidelity (P2.5 exit gate)",
        "",
        "Best-fit and derived (unfitted) grey-box (`Drive` + `EQ3`) against the white-box "
        "circuit simulations, on the harmonic metric of P2 decision 6.",
        "",
        "## Gate",
        "",
        f"Gate: best-fit harmonic error, averaged over the pedal's knob settings, "
        f"≤ {c.gate_db:g} dB for {', '.join(GATED)}.",
        "",
        "| pedal | gated | best-fit mean (dB) | worst setting (dB) | settings > gate | "
        "derived mean (dB) | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for pid, g in r.gate.items():
        verdict = ("PASS" if g["passed"] else "FAIL") if g["gated"] else "diagnostic"
        lines.append(
            f"| {pid} | {'yes' if g['gated'] else 'no'} | {g['best_mean_db']:.2f} | "
            f"{g['worst_setting_mean_db']:.2f} | {', '.join(g['settings_over']) or 'none'} | "
            f"{_f(g['derived_mean_db'])} | **{verdict}** |"
        )
    cmd = "lstmabar fidelity" + (" --quick" if c.quick else "") + f" --seed {c.seed}"
    lines += [
        "",
        f"Generated by `{cmd}`; runtime {r.seconds_elapsed:.0f} s on CPU "
        f"({r.torch_threads} torch threads).",
        "",
        "## Procedure",
        "",
        f"- Inputs: sines at {', '.join(f'{f:g}' for f in c.pitches_hz)} Hz × "
        f"{', '.join(levels)} dBFS (0 dBFS = {c.volts_per_fs:g} V peak at the pedal input), "
        f"{c.settle:g} s settle + {c.seconds:g} s analysed + {c.tail:g} s tail; the settle "
        "pre-roll (≥ 15 time constants of the slowest modelled pole) and the tail (≥ the "
        "resamplers' 32-sample edge) are excluded from analysis. Riff: "
        f"{c.riff_seconds:g} s `pluck_riff`, peak {c.riff_peak:g}, MR-STFT with "
        f"{c.riff_crop} samples cropped at both edges.",
        f"- Rate {c.sample_rate} Hz (H{c.n_harmonics} of the top pitch below Nyquist); "
        f"white-box {c.whitebox_oversample}x oversampled, grey-box Drive "
        f"{c.drive_oversample}x.",
        f"- Metric per clip: |ΔH1| dB (absolute level, no floor) and |ΔHk| dBc, k = "
        f"2..{c.n_harmonics}, both sides floored at {c.floor_dbc:g} dBc "
        "(`analysis.harmonics.harmonic_amplitudes` at the exact f0). *mean* = mean of the "
        f"{c.n_harmonics} terms over all {len(c.pitches_hz) * len(c.levels_dbfs)} clips; "
        "*max* = largest single term.",
        f"- Best fit: one Drive + EQ3 parameter set (11 knobs, compressor off) per (pedal, "
        "setting), fitted jointly to every pitch and level. Loss = the metric (differentiable "
        f"harmonic projection on the first {c.fit_seconds:g} s of each analysed window, "
        f"grey-box pre-roll {c.fit_settle:g} s) + "
        f"{c.mrstft_weight:g} × MR-STFT on the riff. Multi-start: derived parameters (or all "
        f"knobs 0.5 without a deriver) + {c.restarts} random starts U(0.1, 0.9); Adam lr "
        f"{c.lr:g}, cosine decay, {c.steps} steps on sigmoid logits. Best start by final "
        "metric on the full window.",
        "- Derived: `physics.derive.derive` → `preset_params`, unfitted.",
        "",
    ]
    for pid, p in r.pedals.items():
        tag = " (diagnostic, not gated)" if p["diagnostic"] else ""
        lines += [
            f"## {p['name']} (`{pid}`){tag}",
            "",
            "White-box notes: " + ("; ".join(p["notes"]) or "none") + ".",
            "",
            "| setting | best mean | best max | best H1 | best dBc | best MR-STFT | "
            "derived mean | derived max | derived H1 | derived dBc | derived MR-STFT |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
            *_metric_rows(pid, p),
            "",
            "Per input level, best fit: harmonic error / signed H1 error (grey − white) / "
            "dBc error, in dB:",
            "",
            "| setting | " + " | ".join(f"{lv} dBFS" for lv in levels) + " |",
            "|---|" + "---|" * len(levels),
            *_level_rows(p, "best"),
            "",
        ]
        if p["has_deriver"]:
            lines += [
                "Per input level, derived:",
                "",
                "| setting | " + " | ".join(f"{lv} dBFS" for lv in levels) + " |",
                "|---|" + "---|" * len(levels),
                *_level_rows(p, "derived"),
                "",
            ]
        lines += [
            "White-box output level (mean H1, dBFS) per input level: "
            + "; ".join(
                f"{r['label']}: " + ", ".join(f"{v:.1f}" for v in r["white_h1_dbfs"].values())
                for r in p["settings"]
            )
            + ".",
            "",
            "Fitted parameters (derived in parentheses):"
            if p["has_deriver"]
            else "Fitted parameters:",
            "",
            *_param_rows(p),
            "",
            "Multi-start (final mean harmonic error, dB): "
            + "; ".join(
                f"{r['label']}: "
                + ", ".join(f"{s['init']} {s['mean_db']:.2f}" for s in r["starts"])
                + f" → best {r['best_start']}"
                for r in p["settings"]
            )
            + ".",
            "",
        ]
        notes = sorted({n for r in p["settings"] for n in r["notes"]})
        if notes:
            lines += ["Notes: " + "; ".join(notes) + ".", ""]
    if discussion:
        lines += ["## Discussion", "", discussion.strip(), ""]
    return "\n".join(lines)


def load_result(path: str | Path) -> FidelityResult:
    """Read a ``greybox_fidelity.json`` back (e.g. to re-render the Markdown)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    c = data.pop("config")
    for k in ("pitches_hz", "levels_dbfs", "pedals", "settings"):
        if c.get(k) is not None:
            c[k] = tuple(c[k])
    return FidelityResult(config=FidelityConfig(**c), **data)


def write_report(
    r: FidelityResult, out_dir: str | Path = "reports", discussion: str = ""
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md, js = out / "greybox_fidelity.md", out / "greybox_fidelity.json"
    md.write_text(format_report(r, discussion), encoding="utf-8")
    js.write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")
    return md, js


def gate_passed(r: FidelityResult) -> bool:
    return all(g["passed"] for g in r.gate.values() if g["gated"])


__all__ = [
    "DIAGNOSTIC",
    "GATED",
    "FidelityConfig",
    "FidelityResult",
    "HarmonicProjector",
    "check_setting",
    "format_report",
    "gate_passed",
    "harmonic_errors",
    "harmonic_loss",
    "knob_settings",
    "load_result",
    "make_signals",
    "run_fidelity",
    "summarize",
    "write_report",
]
