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
4. A trial succeeds when its final loss is ``<= rel_threshold * initial`` **and**
   ``<= abs_threshold``. Losses are computed per example on the audio with ``crop`` samples
   removed at both edges (filter/oversampling edge transients).

The absolute threshold is calibrated in :func:`perturbation_floor`: the loss of a render
whose knobs all sit a small random offset away from the target. See ``reports/``.
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
    """Exit gate: required fraction of successful trials."""


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
    seconds_elapsed: float
    loss_curve: list[float] = field(default_factory=list)


def per_example_loss(pred: Tensor, target: Tensor, crop: int) -> Tensor:
    """MR-STFT loss of each example separately, ``(B,)``, on ``[crop:-crop]``."""
    if crop:
        pred, target = pred[:, crop:-crop], target[:, crop:-crop]
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
        c = cfg.crop
        loss = multi_resolution_stft_loss(y[:, c:-c], target[:, c:-c])
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
        seconds_elapsed=time.perf_counter() - start,
        loss_curve=curve,
    )


def format_report(r: RecoveryResult) -> str:
    c = r.config
    n_ok = sum(r.success)
    lines = [
        "# Parameter recovery (P1.7 exit gate)",
        "",
        f"**Result: {'PASS' if r.passed else 'FAIL'}** — {n_ok}/{len(r.success)} trials recovered "
        f"({100 * r.success_rate:.0f}%, gate {100 * c.gate:.0f}%).",
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
        f"- Success: final loss <= {c.rel_threshold} x initial **and** <= {c.abs_threshold}.",
        "",
        "## Absolute threshold",
        "",
        "Median MR-STFT loss of a render whose every knob is offset by ±delta (random signs) "
        "from the target:",
        "",
        "| delta (normalized) | median loss |",
        "|---|---|",
        *[f"| {d} | {v:.4f} |" for d, v in r.perturbation_floor.items()],
        "",
        f"The absolute threshold {c.abs_threshold} is below the loss of a render with every "
        "knob within ±0.01 (1% of its travel) of the target, so a passing trial is at least "
        "as close as that, i.e. audibly a match. It is also ~20x below the median initial "
        f"(all knobs at 0.5) loss of {_median(r.initial_loss):.3f}. The loss is this sensitive "
        "because its log-magnitude term weighs quiet bins (high harmonics, anti-alias stopband) "
        "as much as loud ones.",
        "",
        "## Losses",
        "",
        f"- Initial: median {_median(r.initial_loss):.4f}; final: median "
        f"{_median(r.final_loss):.4f}, max {max(r.final_loss):.4f}.",
        f"- Failed trials: {[i for i, s in enumerate(r.success) if not s] or 'none'}",
        "",
        "## Per-parameter normalized absolute error |u_hat - u|",
        "",
        "| knob | median | p90 |",
        "|---|---|---|",
        *[f"| `{n}` | {v['median']:.3f} | {v['p90']:.3f} |" for n, v in r.per_param.items()],
        "",
        "## Correlated errors (Spearman |r| >= 0.5 across trials)",
        "",
        "Strongly correlated signed errors mark trade-offs the loss cannot separate well "
        "(near non-identifiable pairs). With most errors ~0, the ranks are driven by the "
        "trials that did not converge exactly, so read these as directions of the residual "
        "error, not as global degeneracies:",
        "",
        *([f"- `{a}` vs `{b}`: r = {v:+.2f}" for a, b, v in r.correlated_pairs] or ["- none"]),
        "",
        "Expected near-degeneracies by construction:",
        "",
        "- `drive.gain_db` / `drive.asymmetry` / `drive.bias`: input gain, the negative-half "
        "ceiling and the operating point all set how hard each half-wave clips, so more gain "
        "can stand in for more asymmetry/bias.",
        "- `drive.level_db` vs the EQ gains, and `drive.tone_db` vs `eq.high_db`/`eq.low_db`: "
        "post-clip linear gain and tilt overlap the EQ's broad bands.",
        "- `drive.gain_db` vs `drive.level_db` is *not* degenerate here: the clipper makes the "
        "spectrum depend on the input gain, not just the output level.",
        "- `eq.mid_hz` is unidentifiable when `eq.mid_db` is near 0 dB (hence its large p90).",
        "",
    ]
    return "\n".join(lines)


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
