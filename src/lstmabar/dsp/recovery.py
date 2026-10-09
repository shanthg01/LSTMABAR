"""Parameter recovery (execution plan P1.7, the P1 exit gate).

Can gradient descent on a multi-resolution STFT loss find the knob settings that produced a
render? Procedure:

1. ``trials`` target settings for the Drive + EQ knobs are drawn ``U(lo, hi)`` (normalized),
   avoiding the range extremes; the compressor is gated off and every gate is fixed (not
   optimized).
2. Each target is rendered from a synthetic Karplus-Strong guitar riff
   (:func:`lstmabar.dsp.signals.pluck_riff`, a different pluck-noise seed per trial).
3. Fresh logits (init 0, i.e. every knob at 0.5 through a sigmoid) are optimized with Adam on
   the MR-STFT loss between estimate and target, all trials batched together. Each trial's
   loss depends only on its own logits, so the batch loss (the mean over trials) gives each
   trial exactly its own gradient up to a constant factor, which Adam is invariant to.
4. **Audio match** (the exit-gate criterion): final loss ``<= rel_threshold * initial``
   **and** ``<= abs_threshold``. Losses are computed per example on the audio with ``crop``
   samples removed at both edges (filter/oversampling edge transients).
5. **Parameter space** (reported, not gated): the fraction of trials with every knob within
   ``param_tol`` of its target, with and without the :data:`DEGENERATE_KNOBS`, per-knob
   rates, and a diagnosis of each trial that misses a non-exempt knob.

The loss is not monotone in parameter distance, so an audio match does not bound knob
errors. :func:`perturbation_floor` gives the absolute threshold an audio-side scale (the loss
of renders whose knobs are all slightly off). See ``reports/param_recovery.md``.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from lstmabar.dsp.losses import multi_resolution_stft_loss
from lstmabar.dsp.pedalboard import ENABLED, BoardParams, Pedalboard, default_pedalboard
from lstmabar.dsp.signals import pluck_riff

RECOVERED_BLOCKS = ("drive", "eq")
DEGENERATE_KNOBS = ("drive.asymmetry", "drive.bias")
"""Knobs exempted from the "all knobs recovered" count: on this material their errors move
together (rank correlation ~ +0.99) and trials that miss *only* these knobs still reach
near-floor loss, i.e. different (asymmetry, bias) pairs give the same magnitude spectrum.
Both shape the even harmonics; a magnitude-only loss cannot tell which one produced them."""


@dataclass
class RecoveryConfig:
    trials: int = 24
    steps: int = 300
    seed: int = 0
    lr: float = 0.08
    seconds: float = 1.0
    sample_rate: int = 22050
    """Half the project rate halves the cost; every knob frequency (<= 4 kHz) and the
    drive's harmonics of interest stay well below the 11 kHz Nyquist."""
    oversample: int = 4
    crop: int = 512
    target_lo: float = 0.15
    target_hi: float = 0.85
    rel_threshold: float = 0.1
    abs_threshold: float = 0.08
    gate: float = 0.9
    """Exit gate: required fraction of audio-matched trials."""
    param_tol: float = 0.05
    """Parameter-space criterion: a knob is recovered when ``|u_hat - u| <= param_tol``."""


@dataclass
class RecoveryResult:
    config: RecoveryConfig
    knobs: list[str]
    target: list[list[float]]
    estimate: list[list[float]]
    initial_loss: list[float]
    final_loss: list[float]
    success: list[bool]
    success_rate: float
    passed: bool
    per_param: dict[str, dict[str, float]]
    correlated_pairs: list[tuple[str, str, float]]
    perturbation_floor: dict[str, float]
    param_space: dict
    seconds_elapsed: float
    loss_curve: list[float] = field(default_factory=list)


def _crop(x: Tensor, crop: int) -> Tensor:
    """``x[:, crop:-crop]``; ``crop = 0`` keeps everything (``x[:, 0:-0]`` would be empty)."""
    if crop < 0:
        raise ValueError(f"crop must be >= 0, got {crop}")
    return x[:, crop : x.shape[-1] - crop]


def per_example_loss(pred: Tensor, target: Tensor, crop: int) -> Tensor:
    """MR-STFT loss of each example separately, ``(B,)``, on ``[crop:-crop]``."""
    pred, target = _crop(pred, crop), _crop(target, crop)
    return torch.stack(
        [multi_resolution_stft_loss(pred[i : i + 1], target[i : i + 1]) for i in range(len(pred))]
    )


def _board(cfg: RecoveryConfig) -> Pedalboard:
    board = default_pedalboard(cfg.sample_rate)
    board.blocks["drive"].oversample = cfg.oversample
    return board


def _knobs(board: Pedalboard) -> list[tuple[str, str]]:
    return [(b, k) for b, k in board.layout() if b in RECOVERED_BLOCKS and k != ENABLED]


def _params(board: Pedalboard, knobs: list[tuple[str, str]], u: Tensor) -> BoardParams:
    """Board params with the recovered knobs from ``u`` ``(B, K)``, compressor off."""
    batch = u.shape[0]
    p: BoardParams = {"compressor": {ENABLED: u.new_zeros(batch)}}
    for name in RECOVERED_BLOCKS:
        p[name] = {ENABLED: u.new_ones(batch)}
    for j, (name, knob) in enumerate(knobs):
        p[name][knob] = u[:, j]
    return p


def _inputs(cfg: RecoveryConfig) -> Tensor:
    return torch.stack(
        [
            pluck_riff(cfg.seconds, cfg.sample_rate, seed=cfg.seed * 1000 + i)
            for i in range(cfg.trials)
        ]
    )


@torch.no_grad()
def perturbation_floor(
    cfg: RecoveryConfig, deltas: tuple[float, ...] = (0.01, 0.03, 0.05)
) -> dict[str, float]:
    """Median loss when every knob is offset by ``±delta`` (random signs) from the target.

    Calibrates what an absolute MR-STFT value means: a render with every knob within
    ``delta`` of the target scores about this much.
    """
    board, gen = _board(cfg), torch.Generator().manual_seed(cfg.seed + 7)
    knobs = _knobs(board)
    x = _inputs(cfg)
    u = cfg.target_lo + (cfg.target_hi - cfg.target_lo) * torch.rand(
        cfg.trials, len(knobs), generator=gen
    )
    y = board(x, _params(board, knobs, u))
    out = {}
    for d in deltas:
        sign = torch.randint(0, 2, u.shape, generator=gen) * 2 - 1
        y_hat = board(x, _params(board, knobs, (u + d * sign).clamp(0, 1)))
        out[f"{d:g}"] = float(per_example_loss(y_hat, y, cfg.crop).median())
    return out


PATH_POINTS = (0.0, 0.25, 0.5, 0.75, 1.0)


@torch.no_grad()
def _param_space(cfg, board, knobs, x, target, u_true, u_hat, final, success) -> dict:
    """Parameter-space recovery rates plus a diagnosis of every trial that misses a
    non-exempt knob.

    Diagnosis: the missed non-exempt knobs are moved in a straight line from the estimate to
    their targets (every other knob stays at its estimate) and the loss is evaluated at
    ``PATH_POINTS``. A loss that rises along the path means the estimate sits in a basin
    separate from the target's: a local minimum in which the other knobs compensate.
    """
    names = [f"{b}.{k}" for b, k in knobs]
    specs = {
        f"{b}.{s.name}": s for b, k in knobs for s in board.blocks[b].param_specs if s.name == k
    }
    tol = cfg.param_tol
    ok = (u_hat - u_true).abs() <= tol  # (N, K)
    keep = torch.tensor([n not in DEGENERATE_KNOBS for n in names])
    all_ok = ok.all(1)
    all_ok_ex = ok[:, keep].all(1)
    only_degenerate = all_ok_ex & ~all_ok
    misses = []
    for i in torch.nonzero(~all_ok_ex).flatten().tolist():
        missed = [j for j in range(len(names)) if keep[j] and not ok[i, j]]
        mask = torch.zeros(len(names))
        mask[missed] = 1.0
        path = torch.stack([u_hat[i] + t * mask * (u_true[i] - u_hat[i]) for t in PATH_POINTS])
        xi = x[i : i + 1].expand(len(PATH_POINTS), -1)
        ti = target[i : i + 1].expand(len(PATH_POINTS), -1)
        losses = per_example_loss(board(xi, _params(board, knobs, path)), ti, cfg.crop)
        misses.append(
            {
                "trial": i,
                "audio_match": bool(success[i]),
                "final_loss": float(final[i]),
                "knobs": [
                    {
                        "knob": names[j],
                        "unit": specs[names[j]].unit,
                        "target": float(specs[names[j]].denormalize(u_true[i, j])),
                        "estimate": float(specs[names[j]].denormalize(u_hat[i, j])),
                        "error": float(u_hat[i, j] - u_true[i, j]),
                    }
                    for j in missed
                ],
                "path_loss": losses.tolist(),
            }
        )
    exact = final[all_ok]
    return {
        "tol": tol,
        "exempt": list(DEGENERATE_KNOBS),
        "all_within_tol": float(all_ok.float().mean()),
        "all_within_tol_exempt": float(all_ok_ex.float().mean()),
        "per_knob_within_tol": {n: float(ok[:, j].float().mean()) for j, n in enumerate(names)},
        "median_loss_all_within_tol": float(exact.median()) if len(exact) else float("nan"),
        "only_degenerate_missed_trials": torch.nonzero(only_degenerate).flatten().tolist(),
        "only_degenerate_missed_losses": final[only_degenerate].tolist(),
        "misses": misses,
    }


def run_recovery(cfg: RecoveryConfig | None = None, log=print) -> RecoveryResult:
    cfg = cfg or RecoveryConfig()
    start = time.perf_counter()
    torch.manual_seed(cfg.seed)
    board = _board(cfg)
    knobs = _knobs(board)
    gen = torch.Generator().manual_seed(cfg.seed)
    u_true = cfg.target_lo + (cfg.target_hi - cfg.target_lo) * torch.rand(
        cfg.trials, len(knobs), generator=gen
    )
    x = _inputs(cfg)
    with torch.no_grad():
        target = board(x, _params(board, knobs, u_true))

    logits = torch.zeros_like(u_true, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr * 0.05)
    curve = []
    log_every = max(1, cfg.steps // 10)
    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        y = board(x, _params(board, knobs, logits.sigmoid()))
        loss = multi_resolution_stft_loss(_crop(y, cfg.crop), _crop(target, cfg.crop))
        loss.backward()
        opt.step()
        sched.step()
        curve.append(loss.item())
        if step % log_every == 0 or step == cfg.steps - 1:
            log(f"step {step:4d}  mean loss {curve[-1]:.4f}  ({time.perf_counter() - start:.0f}s)")

    with torch.no_grad():
        u_hat = logits.sigmoid()
        final = per_example_loss(board(x, _params(board, knobs, u_hat)), target, cfg.crop)
        initial = per_example_loss(
            board(x, _params(board, knobs, torch.full_like(u_hat, 0.5))), target, cfg.crop
        )
    success = (final <= cfg.rel_threshold * initial) & (final <= cfg.abs_threshold)
    n_ok = int(success.sum())
    rate = n_ok / cfg.trials

    names = [f"{b}.{k}" for b, k in knobs]
    err = u_hat - u_true
    abs_err = err.abs()
    per_param = {
        n: {
            "median": float(abs_err[:, j].median()),
            "p90": float(torch.quantile(abs_err[:, j], 0.9)),
        }
        for j, n in enumerate(names)
    }
    # Spearman (rank) correlation of signed errors: robust to the few large-error trials.
    ranks = err.argsort(0).argsort(0).to(err.dtype)
    corr = torch.corrcoef(ranks.T)
    pairs = [
        (names[i], names[j], float(corr[i, j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if math.isfinite(float(corr[i, j])) and abs(float(corr[i, j])) >= 0.5
    ]
    pairs.sort(key=lambda t: -abs(t[2]))
    param_space = _param_space(cfg, board, knobs, x, target, u_true, u_hat, final, success)
    floor = perturbation_floor(cfg)
    return RecoveryResult(
        config=cfg,
        knobs=names,
        target=u_true.tolist(),
        estimate=u_hat.tolist(),
        initial_loss=initial.tolist(),
        final_loss=final.tolist(),
        success=success.tolist(),
        success_rate=rate,
        passed=n_ok >= math.ceil(cfg.gate * cfg.trials - 1e-9),
        per_param=per_param,
        correlated_pairs=pairs,
        perturbation_floor=floor,
        param_space=param_space,
        seconds_elapsed=time.perf_counter() - start,
        loss_curve=curve,
    )


def format_report(r: RecoveryResult) -> str:
    c, ps = r.config, r.param_space
    n = len(r.success)
    n_ok = sum(r.success)
    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    tol = ps["tol"]
    exempt = ", ".join(f"`{k}`" for k in ps["exempt"])
    lines = [
        "# Parameter recovery (P1.7 exit gate)",
        "",
        f"**Exit gate (audio match): {'PASS' if r.passed else 'FAIL'}** — {n_ok}/{n} trials "
        f"({pct(r.success_rate)}; gate {pct(c.gate)}) reach a render that matches the target "
        "audio.",
        "",
        f"**Parameter space:** {pct(ps['all_within_tol'])} of trials have every knob within "
        f"±{tol} (normalized) of its target; {pct(ps['all_within_tol_exempt'])} when the "
        f"degenerate knobs {exempt} are exempted (see below).",
        "",
        "The gate is an *audio-match* criterion. Matching audio does not imply matching knobs: "
        "several passing trials found a different knob setting that sounds the same to the "
        "loss (see *Trials that miss a knob*).",
        "",
        "Generated by `lstmabar recover "
        f"--trials {c.trials} --steps {c.steps} --seed {c.seed}`; "
        f"runtime {r.seconds_elapsed:.0f} s on CPU.",
        "",
        "## Procedure",
        "",
        f"- Board: `default_pedalboard({c.sample_rate})` with the compressor gated off; Drive "
        f"({c.oversample}x oversampled) and EQ gated on. Gates are fixed, not optimized.",
        f"- {len(r.knobs)} knobs recovered; targets ~ U({c.target_lo}, {c.target_hi}) "
        "normalized, one trial per batch row.",
        f"- Input: {c.seconds} s Karplus-Strong riff (`lstmabar.dsp.signals.pluck_riff`), "
        "different pluck noise per trial.",
        f"- Estimate: logits initialised at 0 (all knobs 0.5), Adam lr {c.lr} with cosine "
        f"decay, {c.steps} steps on the MR-STFT loss (FFT 512/1024/2048), "
        f"{c.crop} samples cropped at both edges.",
        f"- Audio match: final loss <= {c.rel_threshold} x initial **and** <= {c.abs_threshold}.",
        "",
        "## Audio-match threshold",
        "",
        "For scale, the median MR-STFT loss of a render whose every knob is offset by ±delta "
        "(random signs) from the target:",
        "",
        "| delta (normalized) | median loss |",
        "|---|---|",
        *[f"| {d} | {v:.4f} |" for d, v in r.perturbation_floor.items()],
        "",
        f"A loss of {c.abs_threshold} is a smaller spectral difference than nudging every knob "
        "by 1% of its travel, and ~20x below the median initial (all knobs at 0.5) loss of "
        f"{_median(r.initial_loss):.3f}: the two renders have nearly identical multi-resolution "
        "magnitude spectra. This is a statement about the audio only. The loss is not monotone "
        "in parameter distance, so a low loss does **not** bound the knob errors (passing "
        "trials below miss single knobs by up to "
        f"{_max_miss(ps, audio_match=True):.2f} normalized).",
        "",
        f"Losses: initial median {_median(r.initial_loss):.4f}; final median "
        f"{_median(r.final_loss):.4f}, max {max(r.final_loss):.4f}. Not audio-matched: "
        f"{[i for i, s in enumerate(r.success) if not s] or 'none'}.",
        "",
        f"## Parameter space (tolerance ±{tol} normalized)",
        "",
        f"- All knobs within tolerance: {pct(ps['all_within_tol'])} of trials "
        f"(median final loss of those trials {ps['median_loss_all_within_tol']:.4f}).",
        f"- All knobs except {exempt} within tolerance: {pct(ps['all_within_tol_exempt'])}.",
        "",
        "| knob | within ±" + f"{tol} | median abs err | p90 abs err |",
        "|---|---|---|---|",
        *[
            f"| `{k}` | {pct(ps['per_knob_within_tol'][k])} | {v['median']:.3f} | {v['p90']:.3f} |"
            for k, v in r.per_param.items()
        ],
        "",
        "### Exempted (degenerate) knobs",
        "",
        f"{exempt} both set the clipper's even-harmonic content (negative-half ceiling vs "
        "operating point), and a magnitude-only loss cannot tell which produced it. Evidence "
        "from this run: their signed errors are rank-correlated "
        f"({_pair_r(r, *ps['exempt'])}), and the trials that miss *only* these knobs "
        f"({ps['only_degenerate_missed_trials'] or 'none'}) still end at loss "
        f"{_fmt_list(ps['only_degenerate_missed_losses'])} — as low as trials that recover "
        "every knob — so the audio really is (near-)identical along that direction.",
        "",
        "### Trials that miss a non-exempt knob",
        "",
        "Physical target → estimate for each missed knob, the final loss, and the loss along a "
        "straight path that moves only the missed knobs from the estimate (t=0) to their "
        "targets (t=1), all other knobs left at their estimates:",
        "",
        "| trial | audio match | final loss | missed knobs (target → estimate) | path loss "
        "t=0 … 1 | path shape |",
        "|---|---|---|---|---|---|",
        *[_miss_row(m) for m in ps["misses"]],
        "",
        "Path shapes:",
        "",
        "- **barrier** — the loss rises away from the estimate, then falls below it near the "
        "target: the estimate is a *local minimum* separated from the target's lower basin. "
        "These are not unidentifiable settings; the optimizer got stuck.",
        "- **coupled** — the loss rises all the way: the missed knobs are tied to compensating "
        "ones (shelves, `drive.tone_db` tilt, `drive.level_db`), so correcting them alone is "
        "worse. The estimate's loss is still well above the target's (0), so this is either "
        "another local minimum or a slowly converging valley; this probe cannot tell which.",
        "- **flat** — the missed knobs barely change the loss at the estimate (a near-"
        "degenerate direction, e.g. a peak frequency when its gain is ~0 dB).",
        "- **descending** — the loss falls straight toward the target: not yet converged.",
        "",
        f"Of the {len(ps['misses'])} trials listed, {_count_with(ps, 'eq.mid_hz')} miss "
        "`eq.mid_hz`: the sweepable peak (Q 0.9) is the hardest knob, because from the "
        "all-0.5 start (~1 kHz) the peak can lock onto the wrong part of the spectrum (or its "
        "gain is driven toward 0 dB) while the fixed shelves, the drive's tilt and the level "
        "absorb the remaining difference.",
        "",
        "## Correlated errors (Spearman |r| >= 0.5 across trials)",
        "",
        "With most errors ~0, the ranks are driven by the trials that did not converge "
        "exactly, so read these as directions of the residual error:",
        "",
        *([f"- `{a}` vs `{b}`: r = {v:+.2f}" for a, b, v in r.correlated_pairs] or ["- none"]),
        "",
        "`drive.gain_db` vs `drive.level_db` is not among the strongly traded pairs: the "
        "clipper makes the spectrum depend on the input gain, not just the output level.",
        "",
    ]
    return "\n".join(lines)


def _fmt_list(v: list[float]) -> str:
    return "[" + ", ".join(f"{e:.4f}" for e in v) + "]" if v else "n/a"


def _pair_r(r: RecoveryResult, a: str, b: str) -> str:
    for x, y, v in r.correlated_pairs:
        if {x, y} == {a, b}:
            return f"r = {v:+.2f}"
    return "|r| < 0.5"


def _max_miss(ps: dict, audio_match: bool) -> float:
    errs = [
        abs(k["error"]) for m in ps["misses"] if m["audio_match"] == audio_match for k in m["knobs"]
    ]
    return max(errs, default=0.0)


def _miss_row(m: dict) -> str:
    knobs = "; ".join(
        f"`{k['knob']}` {k['target']:.1f} → {k['estimate']:.1f} {k['unit']}".rstrip()
        for k in m["knobs"]
    )
    path = " / ".join(f"{v:.3f}" for v in m["path_loss"])
    match = "yes" if m["audio_match"] else "no"
    shape = _path_shape(m["path_loss"])
    return f"| {m['trial']} | {match} | {m['final_loss']:.4f} | {knobs} | {path} | {shape} |"


def _path_shape(path: list[float]) -> str:
    start, peak, end = path[0], max(path), path[-1]
    if peak - min(path) < 0.25 * start:
        return "flat"
    if end < start and peak > start:
        return "barrier"
    if end > start:
        return "coupled"
    return "descending"


def _count_with(ps: dict, knob: str) -> int:
    return sum(any(k["knob"] == knob for k in m["knobs"]) for m in ps["misses"])


def _median(v: list[float]) -> float:
    return float(torch.tensor(v).median())


def load_result(path: str | Path) -> RecoveryResult:
    """Read a ``param_recovery.json`` back (e.g. to re-render the Markdown)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["config"] = RecoveryConfig(**data["config"])
    data["correlated_pairs"] = [tuple(p) for p in data["correlated_pairs"]]
    return RecoveryResult(**data)


def write_report(r: RecoveryResult, out_dir: str | Path = "reports") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md, js = out / "param_recovery.md", out / "param_recovery.json"
    md.write_text(format_report(r), encoding="utf-8")
    data = asdict(r)
    data.pop("loss_curve")
    js.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return md, js


__all__ = [
    "RecoveryConfig",
    "RecoveryResult",
    "format_report",
    "load_result",
    "per_example_loss",
    "perturbation_floor",
    "run_recovery",
    "write_report",
]
