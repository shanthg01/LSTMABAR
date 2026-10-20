"""P6 stage 2 (task 6.3): ``L_param + L_audio`` through the pedalboard, from a stage-1
checkpoint. ``lstmabar train stage2 [--limit N] [--fixtures] [dotlist overrides]`` (Colab GPU,
``docs/colab-runbook.md`` section 4).

- **Init** (design §6 curriculum, "parameters only -> add the audio loss"): the predictor is
  loaded from ``stage2.init_checkpoint`` (``"{seed}"`` formatted with the run seed; default the
  bge-text stage-1 variant). Its run config's ``model`` section replaces ``cfg.model``, so the
  embedding inputs are the stage-1 ones. ``null`` trains from scratch with ``cfg.model``.
- **Data**: :class:`lstmabar.train.dataset.Stage1Dataset` (embeddings, labels, one caption
  sampled per item per epoch) joined with :class:`AudioCrops` (dry clip at the item's
  ``input_peak_dbfs``, stored wet). One random crop length per batch
  (``stage2.crop_seconds``) and a random crop start per item; ``stage2.pre_roll_seconds`` of
  dry before the crop is rendered and discarded (filter and envelope state; zero-padded at the
  clip start, where the stored render also started from rest). Crop starts, lengths and
  captions are drawn in the main process from the seeded generator, so DataLoader workers do
  no random draws and CPU runs are deterministic.
- **Loss**: ``L_param`` exactly as stage 1 (``train.loss.param_weight`` / ``gate_weight``,
  clamped grey-box items weighted by ``train.clamped_weight``) plus
  ``train.loss.audio_weight x ramp x L_audio``, ``L_audio`` = ``multi_resolution_stft_loss``
  (``eval.metrics.mrstft_fft_sizes``) between the **peak-normalized** render crop and the
  **peak-normalized** stored wet crop. The stored wets are the full renders peak-normalized to
  ``render.wet_peak_dbfs`` and the eval metric (``eval.metrics.mrstft``) peak-normalizes both
  signals, so the comparison is level-invariant in both places. The render is the predicted
  board with **soft gates** (``sigmoid`` of the gate logits, the pedalboard's wet/dry blend),
  which is also how ``ModelMethod`` renders, so gradients reach the knob and gate heads through
  the pedalboard. ``L_audio`` is unweighted per item (the clamp only corrupts the label, not
  the target audio). ``stage2.audio_ramp_steps`` ramps its weight linearly from 0.
- **L_clap** (drive-amount contrast, spike A): TODO, not implemented. It needs the CLAP audio
  tower on every render with gradients (``models/clap_diff.py`` has the differentiable
  resampler) and a drive-direction target; ``train.loss.clap_weight`` must stay 0.
- **Selection** (never gold, D8): evaluated at step 0 (the init checkpoint is a candidate) and
  every ``stage2.eval_every`` steps (default once per epoch). Score = the stage-1 parameter
  criterion on every (val item, caption) pair + ``stage2.select.mrstft_weight x`` val MR-STFT on
  a fixed subset (the first ``stage2.select.mrstft_items`` val items by id, first caption, the
  full clip rendered and compared as ``eval.metrics.mrstft`` does). Early stopping after
  ``train.patience`` evaluations without improvement.
- **Outputs** (``runs/<name>/<stamp>/``): ``config.yaml``, ``meta.json``, ``metrics.json``,
  ``checkpoint.pt`` (same format and meta keys as stage 1, so ``eval run`` scores it as
  ``p6_model`` unchanged) and ``summary.json``. ``stage2.export_checkpoint`` (optional,
  ``"{seed}"`` formatted) also copies the best checkpoint there, e.g. to a
  ``.../seed{seed}/checkpoint.pt`` template for ``eval.model.checkpoint``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lstmabar.data import levels
from lstmabar.dsp.losses import multi_resolution_stft_loss
from lstmabar.dsp.pedalboard import Pedalboard, default_pedalboard
from lstmabar.eval.baselines.text2fx import resolve_device
from lstmabar.eval.data import SynthItem, load_split
from lstmabar.models.predictor import ParamPredictor, load_checkpoint, save_checkpoint
from lstmabar.runs import create_run_dir, git_info
from lstmabar.seed import seed_everything
from lstmabar.train import stage1
from lstmabar.train.dataset import Batch, Stage1Dataset, load_tables

# ============================================================================ audio data


class AudioCrops(Dataset):
    """Dry/wet crops of synthetic items. Indexed by a key ``(item, u, n_crop, n_pre)``: crop
    length ``min(n_crop, clip length)``, start ``floor(u x (n - length))``, plus ``n_pre``
    samples of dry before it (zero-padded at the clip start). Returns float32
    ``(dry (n_pre + length,), wet (length,))``; no random draws here.

    Only the crop is decoded (FLAC seek); the input-level gain needs the stored dry clip's
    full peak, which is computed once per clip and cached (per worker process)."""

    def __init__(self, items: Sequence[SynthItem]) -> None:
        self.dry_paths = [str(i.dry_path) for i in items]
        self.wet_paths = [str(i.wet_path) for i in items]
        self.peaks = [float(i.item.input_peak_dbfs) for i in items]
        self.sample_rates = [int(i.sample_rate) for i in items]
        self._dry_peak: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.dry_paths)

    def _gain(self, i: int) -> float:
        """Linear gain taking the stored dry clip to the item's input peak (as
        ``levels.apply_input_level``)."""
        path = self.dry_paths[i]
        if path not in self._dry_peak:
            self._dry_peak[path] = levels.peak_dbfs(levels.read_audio(path)[0])
        p = self._dry_peak[path]
        return 1.0 if p <= levels.SILENCE_DBFS else float(levels.db_to_gain(self.peaks[i] - p))

    @staticmethod
    def _read(path: str, start: int, stop: int) -> np.ndarray:
        x, _ = sf.read(path, start=start, stop=stop, dtype="float64", always_2d=True)
        return x.mean(axis=1)

    def _frames(self, i: int) -> int:
        a, b = sf.info(self.dry_paths[i]), sf.info(self.wet_paths[i])
        if a.samplerate != self.sample_rates[i] or b.samplerate != a.samplerate:
            raise ValueError(
                f"item {i}: sample rates {a.samplerate}/{b.samplerate}, "
                f"expected {self.sample_rates[i]}"
            )
        return min(int(a.frames), int(b.frames))

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Full ``(dry at the item's input level, stored wet)``, trimmed to the same length."""
        dry, sr = levels.read_audio(self.dry_paths[i])
        wet, sr_w = levels.read_audio(self.wet_paths[i])
        if sr != self.sample_rates[i] or sr_w != sr:
            raise ValueError(f"item {i}: sample rates {sr}/{sr_w}, expected {self.sample_rates[i]}")
        dry = levels.apply_input_level(dry, self.peaks[i])  # the full clip's peak sets the gain
        n = min(dry.size, wet.size)
        return dry[:n], wet[:n]

    def __getitem__(self, key: tuple[int, float, int, int]) -> tuple[Tensor, Tensor]:
        i, u, n_crop, n_pre = key
        i = int(i)
        n = self._frames(i)
        length = min(int(n_crop), n)
        start = int(math.floor(float(u) * (n - length)))
        lo = start - int(n_pre)
        seg = self._read(self.dry_paths[i], max(lo, 0), start + length) * self._gain(i)
        if lo < 0:
            seg = np.concatenate([np.zeros(-lo), seg])
        wet = self._read(self.wet_paths[i], start, start + length)
        return (
            torch.from_numpy(seg.astype(np.float32)),
            torch.from_numpy(wet.astype(np.float32)),
        )


def collate_crops(rows: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Stack crops, trimmed at the end to the shortest crop of the batch (dry keeps its
    pre-roll, so dry and wet stay aligned)."""
    n_pre = rows[0][0].shape[0] - rows[0][1].shape[0]
    t = min(w.shape[0] for _, w in rows)
    dry = torch.stack([d[: n_pre + t] for d, _ in rows])
    wet = torch.stack([w[:t] for _, w in rows])
    return dry, wet


def peak_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """``(B, T)`` scaled to unit peak per row (differentiable; silent rows stay ~0)."""
    return x / x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)


# ============================================================================ helpers


def _to(b: Batch, device: torch.device) -> Batch:
    return Batch(
        **{
            f.name: (v.to(device) if isinstance(v := getattr(b, f.name), Tensor) else v)
            for f in dataclasses.fields(b)
        }
    )


def render(board: Pedalboard, model: ParamPredictor, out: dict[str, Tensor], dry: Tensor):
    """The predicted board (soft gates) on ``dry`` ``(B, T)``."""
    return board(dry, board.from_vector(model.board(out)))


def audio_loss(y: Tensor, wet: Tensor, fft_sizes: Sequence[int]) -> Tensor:
    return multi_resolution_stft_loss(
        peak_normalize(y), peak_normalize(wet), tuple(int(s) for s in fft_sizes)
    )


def _format_seed(template: str, seed: int) -> str:
    return str(template).replace("{seed}", str(seed))


def resolve_init(cfg) -> tuple[ParamPredictor | None, dict[str, Any] | None]:
    """The stage-1 model and checkpoint payload (or ``(None, None)`` from scratch); with a
    checkpoint, ``cfg.model`` is replaced (in place) by the checkpoint run's ``model``."""
    tmpl = cfg.stage2.get("init_checkpoint")
    if tmpl is None:
        return None, None
    path = Path(_format_seed(str(tmpl), int(cfg.seed)))
    if not path.exists():
        raise FileNotFoundError(
            f"stage2.init_checkpoint not found: {path.name} (seed {cfg.seed}); train stage 1 "
            "first or set stage2.init_checkpoint"
        )
    model, payload = load_checkpoint(path)
    meta = payload.get("meta") or {}
    run_model = (meta.get("config") or {}).get("model")
    if not run_model:
        raise ValueError(f"{path.name}: checkpoint meta has no run config (not a stage-1 run?)")
    cfg.model = OmegaConf.create(run_model)
    p = model.config
    if str(cfg.model.inputs) != p.inputs or str(cfg.model.mapping) != p.mapping:
        raise ValueError(f"{path.name}: run config disagrees with the checkpoint's predictor")
    return model, payload


# ============================================================================ evaluation


class ValAudio:
    """The fixed val subset for the selection MR-STFT: the first ``n`` items by id (first
    caption each), full clips held in memory."""

    def __init__(self, ds: Stage1Dataset, n: int) -> None:
        order = sorted(range(len(ds)), key=lambda i: ds.item_ids[i])[: max(0, int(n))]
        self.index = torch.tensor(order, dtype=torch.long)
        crops = AudioCrops([ds.items[i] for i in order])
        self.audio = [crops.load(j) for j in range(len(order))]

    def __len__(self) -> int:
        return len(self.audio)


@torch.no_grad()
def val_mrstft(
    model: ParamPredictor,
    board: Pedalboard,
    ds: Stage1Dataset,
    va: ValAudio,
    cfg,
    device: torch.device,
) -> float:
    """Mean per-item MR-STFT (eval-style: peak-normalized full render vs stored wet)."""
    if len(va) == 0:
        return float("nan")
    model.eval()
    fft = tuple(int(s) for s in cfg.eval.metrics.mrstft_fft_sizes)
    bs = max(1, int(cfg.stage2.select.batch_size))
    b = _to(ds._gather(va.index, None, None), device)  # first caption per item
    out = model(b.text, b.audio)
    vals: list[float] = []
    # group by length so each render batch is rectangular
    by_len: dict[int, list[int]] = {}
    for j, (d, _) in enumerate(va.audio):
        by_len.setdefault(d.size, []).append(j)
    for _, js in sorted(by_len.items()):
        for s in range(0, len(js), bs):
            chunk = js[s : s + bs]
            dry = torch.from_numpy(np.stack([va.audio[j][0] for j in chunk]).astype(np.float32))
            wet = torch.from_numpy(np.stack([va.audio[j][1] for j in chunk]).astype(np.float32))
            sub = {k: v[chunk] for k, v in out.items()}
            y = render(board, model, sub, dry.to(device))
            for r in range(len(chunk)):
                vals.append(float(audio_loss(y[r : r + 1], wet[r : r + 1].to(device), fft)))
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def evaluate(model, board, ds, va, cfg, device) -> dict[str, Any]:
    """Stage-1 val metrics (every (item, caption) pair) + the subset MR-STFT and the
    selection ``score``."""
    model.to("cpu")
    val = stage1.evaluate(model, ds, cfg)
    model.to(device)
    val["param_score"] = val.pop("score")
    val["mrstft"] = val_mrstft(model, board, ds, va, cfg, device)
    val["n_mrstft_items"] = len(va)
    w = float(cfg.stage2.select.mrstft_weight)
    m = val["mrstft"] if math.isfinite(val["mrstft"]) else math.inf
    val["score"] = val["param_score"] + (w * m if w else 0.0)
    return val


# ============================================================================ training


def epoch_batches(
    n_items: int,
    batch_size: int,
    gen: torch.Generator,
    sr: int,
    crop_seconds: Sequence[float],
    n_pre: int,
) -> list[list[tuple[int, float, int, int]]]:
    """One epoch of :class:`AudioCrops` keys: items shuffled, one crop length per batch."""
    lo, hi = (float(crop_seconds[0]), float(crop_seconds[-1]))
    order = torch.randperm(n_items, generator=gen)
    u = torch.rand(n_items, generator=gen, dtype=torch.float64)
    n_batches = -(-n_items // batch_size)
    lens = lo + (hi - lo) * torch.rand(n_batches, generator=gen, dtype=torch.float64)
    out = []
    for k, s in enumerate(range(0, n_items, batch_size)):
        n_crop = int(round(float(lens[k]) * sr))
        out.append([(int(i), float(u[i]), n_crop, n_pre) for i in order[s : s + batch_size]])
    return out


def _loader(crops: AudioCrops, batches, num_workers: int) -> Iterator:
    return iter(
        DataLoader(
            crops,
            batch_sampler=batches,
            num_workers=int(num_workers),
            collate_fn=collate_crops,
        )
    )


def train_model(
    cfg,
    train_ds: Stage1Dataset,
    val_ds: Stage1Dataset,
    run_dir: Path | None = None,
    init: ParamPredictor | None = None,
    init_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage-2 training with early stopping on the val score. Returns a summary (with
    ``"model"``); with ``run_dir`` writes ``metrics.json``, ``checkpoint.pt``, ``summary.json``."""
    seed = int(cfg.seed)
    seed_everything(seed)
    tc, s2 = cfg.train, cfg.stage2
    if float(tc.loss.get("clap_weight", 0.0) or 0.0) != 0.0:
        raise NotImplementedError("train.loss.clap_weight: L_clap is not implemented (TODO)")
    if tc.get("num_threads"):
        torch.set_num_threads(int(tc.num_threads))
    device = resolve_device(str(s2.device))
    if init is not None:
        model = init
    else:
        text_dim = train_ds.text.shape[1] if train_ds.text is not None else 0
        audio_dim = train_ds.audio.shape[1] if train_ds.audio is not None else 0
        model = stage1.build_model(cfg, text_dim, audio_dim)
    model.to(device)
    sr = int(cfg.audio.sample_rate)
    board = default_pedalboard(sr).to(device)  # oversampled Drive (4x)
    fft = tuple(int(s) for s in cfg.eval.metrics.mrstft_fft_sizes)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay)
    )
    bs = int(tc.batch_size)
    steps_per_epoch = train_ds.steps_per_epoch(bs)
    total = int(tc.epochs) * steps_per_epoch
    if s2.get("max_steps") is not None:
        total = min(total, int(s2.max_steps))
    warmup = int(round(float(tc.warmup_frac) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: stage1.lr_lambda(s, total, warmup, float(tc.min_lr_frac))
    )
    eval_every = int(s2.eval_every) if s2.get("eval_every") else steps_per_epoch
    ramp = int(s2.get("audio_ramp_steps") or 0)
    w_audio = float(tc.loss.audio_weight)
    grad_clip = float(tc.grad_clip) if tc.get("grad_clip") else 0.0
    n_pre = int(round(float(s2.pre_roll_seconds) * sr))
    crops = AudioCrops(train_ds.items)
    va = ValAudio(val_ds, int(s2.select.mrstft_items))
    gen = torch.Generator().manual_seed(seed)
    shuffle_audio = bool(cfg.model.get("shuffle_audio", False))
    patience = int(tc.patience)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    t0 = time.perf_counter()

    def do_eval(step: int, train_stats: dict | None) -> dict:
        val = evaluate(model, board, val_ds, va, cfg, device)
        row = {
            "step": step,
            "epoch": step / steps_per_epoch,
            "lr": float(sched.get_last_lr()[0]),
            "train": train_stats,
            "val": val,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        history.append(row)
        if run_dir is not None:
            (run_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return val

    def snapshot() -> dict[str, Tensor]:
        return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    val0 = do_eval(0, None)
    best = {"score": val0["score"], "step": 0, "val": val0}
    best_state = snapshot() if math.isfinite(val0["score"]) else None
    if best_state is None:
        best["score"] = math.inf
    bad = 0
    step = 0
    skipped = 0
    sums = {"loss": 0.0, "l_param": 0.0, "l_audio": 0.0}
    n_seen = 0
    step_times: list[float] = []
    stop = False
    while step < total and not stop:
        batches = epoch_batches(len(train_ds), bs, gen, sr, list(s2.crop_seconds), n_pre)
        train_ds.set_epoch(gen, permute_audio=shuffle_audio)
        cap, perm = train_ds._caption_choice, train_ds._audio_perm
        loader = _loader(crops, batches, int(s2.num_workers))
        for keys, (dry, wet) in zip(batches, loader, strict=True):
            if step >= total:
                break
            ts = time.perf_counter()
            model.train()
            idx = torch.tensor([k[0] for k in keys], dtype=torch.long)
            b = _to(train_ds._gather(idx, cap, perm), device)
            dry, wet = dry.to(device), wet.to(device)
            out = model(b.text, b.audio)
            l_param, _ = stage1._loss(model, out, b, cfg)
            y = render(board, model, out, dry)[:, n_pre:]
            l_audio = audio_loss(y, wet, fft)
            r = min(1.0, (step + 1) / ramp) if ramp > 0 else 1.0
            loss = l_param + w_audio * r * l_audio
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                sums["loss"] += loss.item() * len(keys)
                sums["l_param"] += l_param.item() * len(keys)
                sums["l_audio"] += l_audio.item() * len(keys)
                n_seen += len(keys)
            else:
                skipped += 1
            sched.step()
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - ts)
            if step % eval_every == 0 or step == total:
                stats = {k: v / max(n_seen, 1) for k, v in sums.items()}
                stats["skipped_nonfinite"] = skipped
                sums = dict.fromkeys(sums, 0.0)
                n_seen = 0
                val = do_eval(step, stats)
                if val["score"] < best["score"] - float(tc.min_delta):
                    best = {"score": val["score"], "step": step, "val": val}
                    best_state = snapshot()
                    bad = 0
                else:
                    bad += 1
                if bad >= patience:
                    stop = True
                    break
    if best_state is None:
        raise RuntimeError("stage2: no finite val score (NaN/inf loss?); check train.lr")
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    summary = {
        "best_step": best["step"],
        "best_epoch": best["step"] / steps_per_epoch,
        "steps_run": step,
        "steps_per_epoch": steps_per_epoch,
        "total_steps_planned": total,
        "val": best["val"],
        "val_init": val0,
        "n_train_items": len(train_ds),
        "n_val_items": len(val_ds),
        "skipped_nonfinite_steps": skipped,
        "device": device.type,
        "seconds_per_step": float(np.median(step_times)) if step_times else None,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if run_dir is not None:
        init_meta = (init_payload or {}).get("meta") or {}
        meta = {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "git": git_info(),
            "text_model": str(cfg.model.text_model) if model.config.inputs != "audio" else None,
            "audio_model": str(cfg.model.audio_model) if model.config.inputs != "text" else None,
            "epoch": summary["best_epoch"],
            "step": best["step"],
            "val": best["val"],
            "seed": seed,
            "stage": 2,
            "init": None
            if init_payload is None
            else {"git": init_meta.get("git"), "epoch": init_meta.get("epoch")},
        }
        ckpt = save_checkpoint(run_dir / "checkpoint.pt", model, meta)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        export = s2.get("export_checkpoint")
        if export:
            dst = Path(_format_seed(str(export), seed))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ckpt, dst)
    summary["model"] = model
    return summary


# ============================================================================ entry point


def load_data(cfg, limit: int | None = None) -> tuple[Stage1Dataset, Stage1Dataset]:
    """Train and val (``eval.select_split``, never gold/test) datasets for ``cfg.model``."""
    select = str(cfg.eval.select_split)
    if select in stage1.FORBIDDEN_SELECT_SPLITS:
        raise ValueError(f"eval.select_split must be a val split, not {select!r} (D8)")
    stage1.check_model_cfg(cfg)
    weight = float(cfg.train.clamped_weight)
    sets = []
    for split in ("train", select):
        items = load_split(cfg, split, limit)
        text, audio = load_tables(cfg, split)
        sets.append(Stage1Dataset(items, text, audio, clamped_weight=weight))
    return sets[0], sets[1]


def fixture_config(cfg, root: str | Path) -> DictConfig:
    """``cfg``'s model/train/stage2/eval.model on top of the eval fixture paths."""
    out = stage1.fixture_config(cfg, root)
    return OmegaConf.merge(out, OmegaConf.create({"stage2": cfg.stage2}))


def run(cfg, args) -> int:
    if not getattr(args, "fixtures", False):
        return _run(cfg, args)
    from lstmabar.eval.fixtures import make_eval_fixture

    # Fixture data, fake embeddings and a throwaway stage-1 checkpoint live in a temp dir;
    # the stage-2 run dir stays under paths.runs.
    with tempfile.TemporaryDirectory(prefix="lstmabar_stage2_fx_") as root:
        runs_root = cfg.paths.runs
        cfg = fixture_config(cfg, root)
        cfg.paths.runs = runs_root
        make_eval_fixture(cfg, seconds=0.5)
        stage1.write_fixture_embeddings(cfg)
        s1 = OmegaConf.merge(cfg, {"train": {"epochs": 2, "batch_size": 8}})
        tr, va = stage1.load_data(s1)
        stage1.train_model(s1, tr, va, run_dir=Path(root) / "stage1")
        cfg.stage2.init_checkpoint = str(Path(root) / "stage1" / "checkpoint.pt")
        if cfg.stage2.get("max_steps") is None:
            cfg.stage2.max_steps = 20
        cfg.stage2.num_workers = 0
        return _run(cfg, args)


def _run(cfg, args) -> int:
    limit = getattr(args, "limit", None)
    if limit is None and cfg.eval.get("max_items") is not None:
        limit = int(cfg.eval.max_items)
    seed_everything(int(cfg.seed))
    init, payload = resolve_init(cfg)
    train_ds, val_ds = load_data(cfg, limit)
    run_dir = create_run_dir(cfg, root=str(cfg.paths.runs))
    summary = train_model(cfg, train_ds, val_ds, run_dir, init=init, init_payload=payload)
    v, v0 = summary["val"], summary["val_init"]
    sps = summary["seconds_per_step"]
    print(
        f"stage2 ({summary['device']}): best step {summary['best_step']} of "
        f"{summary['steps_run']} ({summary['seconds']:.0f} s, "
        f"{(sps or float('nan')):.3f} s/step); val score {v0['score']:.4f} -> {v['score']:.4f} "
        f"(param L1 {v0['param_l1']:.4f} -> {v['param_l1']:.4f}, gate acc "
        f"{v0['gate_acc']:.3f} -> {v['gate_acc']:.3f}, MR-STFT on {v['n_mrstft_items']} items "
        f"{v0['mrstft']:.4f} -> {v['mrstft']:.4f})"
    )
    print(f"run dir: {run_dir.name} (under paths.runs/{cfg.name})")
    return 0


__all__ = [
    "AudioCrops",
    "ValAudio",
    "audio_loss",
    "collate_crops",
    "epoch_batches",
    "evaluate",
    "fixture_config",
    "load_data",
    "peak_normalize",
    "render",
    "resolve_init",
    "run",
    "train_model",
    "val_mrstft",
]
