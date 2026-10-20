"""Instruction text -> board knobs for the M3 demo (P5 task 5.6, D7). No gradio import.

A P5/P6 :class:`~lstmabar.eval.baselines.base.Method` (``demo.text_method``: the best
baseline on val once the selection names it; ``mean`` until then) predicts a normalized board
from the instruction (and the input clip, for audio methods). :class:`TextKnobSetter` builds and
fits it once, lazily, and caches it; the app writes its prediction into the generated sliders
via :func:`lstmabar.demo.render.normalized_to_controls`.

A demo query is never in the embedding cache, so embedding methods encode it on the fly with
:mod:`lstmabar.models.encoders` (kNN here, the P6 model inside ``ModelMethod``). Every failure
is turned into a :class:`TextMethodError` whose message is safe to show (no server paths);
``mean`` / ``noop`` work without any encoder.

No gold data is read here, and dataset captions are never shown (only the method's
``info()`` summary).
"""

from __future__ import annotations

import logging
import re
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from lstmabar.data.schema import FloatBoard
from lstmabar.eval.baselines.base import Method, build_method, complete_board
from lstmabar.eval.baselines.knn import KnnMethod
from lstmabar.eval.data import Query, gold_audio_key
from lstmabar.models.embed_cache import text_sha256

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "demo.yaml"

DEMO_METHODS: tuple[str, ...] = ("mean", "noop", "knn_clap", "knn_bge", "p6_model")
"""Methods the demo may use. The LLM baseline is left out on purpose: a public demo must not
spend API credit per click."""
DEFAULT_METHOD = "mean"
EMBEDDING_METHODS = frozenset({"knn_clap", "knn_bge", "p6_model"})
MAX_INSTRUCTION_CHARS = 500
MAX_REASON_CHARS = 300

ENCODERS_MISSING = (
    "on-the-fly text encoding (models.encoders) is not implemented yet; "
    "set demo.text_method to mean or noop"
)
NEEDS_CLAP_EXTRA = "encoding text needs the clap extra (uv sync --extra clap)"


class TextMethodError(Exception):
    """A user-facing failure of the text path; the message never contains a path."""


FILE_ERROR = "a required file could not be read (details in the server log)"

_PATH_RE = re.compile(
    # Windows drive and UNC paths may contain spaces: scrub up to the closing quote, or to
    # the end of the line when unquoted.
    r"(?:[A-Za-z]:[\\/]|\\\\)[^'\"`\n]*"
    # any token with a backslash (relative Windows paths)
    r"|[^\s'\"`]*\\[^\s'\"`]*"
    # absolute or relative POSIX-style paths: two or more separators, or one separator
    # and a file extension starting with a letter (keeps one-slash model ids such as
    # laion/clap-htsat-unfused and BAAI/bge-base-en-v1.5)
    r"|[^\s'\"`/]*(?:/[^\s'\"`/]+){2,}/?"
    r"|[^\s'\"`/]*/[^\s'\"`/]+\.[A-Za-z][A-Za-z0-9]{0,4}\b"
)


def scrub_paths(text: str) -> str:
    """Replace anything that looks like a filesystem path with ``<path>``."""
    return _PATH_RE.sub("<path>", str(text))


def _reason(e: BaseException) -> str:
    # OS-raised file errors (``filename`` set) carry paths in many shapes: say nothing
    # specific. Our own FileNotFoundErrors ("run `lstmabar eval embed` first") are kept.
    if isinstance(e, OSError) and (e.filename is not None or type(e) is not FileNotFoundError):
        return FILE_ERROR
    msg = scrub_paths(str(e)).strip() or type(e).__name__
    if len(msg) > MAX_REASON_CHARS:
        msg = msg[: MAX_REASON_CHARS - 1] + "…"
    return msg


# ============================================================================ demo queries


@dataclass(frozen=True)
class DemoQuery(Query):
    """A :class:`~lstmabar.eval.data.Query` whose dry input is held in memory (already at the
    inference level) instead of read from ``dry_path``."""

    audio: np.ndarray | None = field(default=None, repr=False, compare=False)

    def load_dry(self) -> np.ndarray:
        if self.audio is None:
            raise ValueError("the demo query has no input clip")
        return np.asarray(self.audio, dtype=np.float64)


def demo_query(text: str, clip: np.ndarray | None, sample_rate: int, peak_dbfs: float) -> DemoQuery:
    """The demo's single query: split ``demo`` (no cache table exists, so embedding methods
    encode it on the fly)."""
    key = f"demo-{text_sha256(text)[:16]}"
    return DemoQuery(
        query_id=key,
        unit_id="demo",
        split="demo",
        text=text,
        text_key=key,
        audio_key=gold_audio_key("demo", float(peak_dbfs)),
        clip_id="demo",
        input_peak_dbfs=float(peak_dbfs),
        sample_rate=int(sample_rate),
        dry_path=Path("demo-input-in-memory"),
        audio=clip,
    )


# ============================================================================ info summary

_SCALARS = (str, int, float, bool, type(None))


def _flatten(info: Mapping[str, Any], prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for k, v in info.items():
        key = f"{prefix}{k}"
        if isinstance(v, _SCALARS):
            out.append((key, v))
        elif isinstance(v, Mapping) and depth < 2:
            out.extend(_flatten(v, f"{key}.", depth + 1))
        # lists (e.g. per-k val scores) are left out of the one-line summary
    return out


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return scrub_paths(str(v)).replace("|", "/").replace("\n", " ")


def info_summary(info: Mapping[str, Any]) -> str:
    """``key: value`` pairs of a method's ``info()`` (scalars and nested dicts, no lists, no
    paths) on one line; ``name`` is shown separately."""
    pairs = [(k, v) for k, v in _flatten(info) if k != "name"]
    return ", ".join(f"{k}: {_fmt_value(v)}" for k, v in pairs)


@dataclass(frozen=True)
class TextPrediction:
    method: str
    board: FloatBoard  # complete normalized board with gates
    info: dict[str, Any]

    def summary_md(self) -> str:
        details = info_summary(self.info)
        line = f"**Knobs set from text** by `{self.method}`"
        return f"{line} ({details})" if details else line


# ============================================================================ the setter


def _select(cfg: Any, dotted: str, default: Any = None) -> Any:
    if not isinstance(cfg, DictConfig):
        return default
    v = OmegaConf.select(cfg, dotted, default=None)
    return default if v is None else v


def _load_train(cfg: Any):
    from lstmabar.eval.data import load_split

    return load_split(cfg, "train")


def _load_text_encoder(model: str):
    from lstmabar.models.encoders import load_text_encoder

    return load_text_encoder(model)


class TextKnobSetter:
    """Builds ``demo.text_method`` once (lazily, thread-safe), fits it on the train split if
    the method learns anything, and predicts one board per instruction.

    ``build`` / ``load_train`` / ``load_text_encoder`` are injectable for tests.
    """

    def __init__(
        self,
        cfg: Any,
        method: str | None = None,
        *,
        build: Callable[..., Method] = build_method,
        load_train: Callable[[Any], Sequence[Any]] = _load_train,
        load_text_encoder: Callable[[str], Any] = _load_text_encoder,
    ) -> None:
        self.cfg = cfg
        self.method_name = str(method or _select(cfg, "demo.text_method", DEFAULT_METHOD))
        self._build = build
        self._load_train = load_train
        self._load_encoder = load_text_encoder
        self._method: Method | None = None
        self._encoders: dict[str, Any] = {}
        self._error: TextMethodError | None = None  # a failed build, re-raised (no refits)
        self._lock = threading.Lock()  # building / fitting
        self._predict_lock = threading.Lock()  # lazy encoder / checkpoint loads in predict

    @property
    def peak_dbfs(self) -> float:
        return float(_select(self.cfg, "audio.inference_peak_dbfs", -15.0))

    def unavailable(self, reason: str) -> TextMethodError:
        prefix = "text model" if self.method_name in EMBEDDING_METHODS else "text method"
        return TextMethodError(f"{prefix} unavailable ({self.method_name}): {reason}")

    def method(self) -> Method:
        """The built and fitted method (built on first use, then cached; a failed build is
        cached too and re-raised, so clicks do not refit; restart the demo to retry)."""
        with self._lock:
            if self._method is not None:
                return self._method
            if self._error is not None:
                raise self._error
            try:
                self._method = self._build_method()
            except TextMethodError as e:
                self._error = e
                raise
            return self._method

    def _build_method(self) -> Method:
        name = self.method_name
        if name not in DEMO_METHODS:
            raise TextMethodError(
                f"text method unavailable: {name!r} is not offered in the demo "
                f"(demo.text_method: one of {', '.join(DEMO_METHODS)})"
            )
        if self.cfg is None:
            raise self.unavailable("no demo config was loaded")
        try:
            with warnings.catch_warnings():
                # the demo runs one fixed P6 checkpoint on purpose (not a D3 evaluation)
                warnings.filterwarnings("ignore", message="p6_model: fixed checkpoint")
                m = self._build(name, self.cfg, seed=int(_select(self.cfg, "seed", 0)))
            hparams = _select(self.cfg, "demo.text_hparams", None)
            if hparams:
                if isinstance(hparams, DictConfig):
                    hparams = OmegaConf.to_container(hparams, resolve=True)
                m.set_hparams(dict(hparams))
            if type(m).fit is not Method.fit:  # the method learns from train
                m.fit(self._load_train(self.cfg))
        except Exception as e:
            log.exception("could not build the demo text method %s", name)
            raise self.unavailable(_reason(e)) from e
        return m

    def warm_up(self) -> threading.Thread:
        """Build and fit the method in a background thread (fitting ``mean`` reads every train
        label, which takes a while on the full dataset); a click meanwhile waits for it.
        Failures are logged and reported again on the first click."""

        def run() -> None:
            try:
                self.method()
            except TextMethodError as e:
                log.warning("demo text method not ready: %s", e)

        t = threading.Thread(target=run, name="demo-text-warm-up", daemon=True)
        t.start()
        return t

    def _encode_text(self, model: str, text: str) -> np.ndarray:
        enc = self._encoders.get(model)
        if enc is None:
            try:
                enc = self._load_encoder(model)
            except NotImplementedError as e:
                raise self.unavailable(ENCODERS_MISSING) from e
            except ImportError as e:
                raise self.unavailable(NEEDS_CLAP_EXTRA) from e
            except Exception as e:
                log.exception("could not load the text encoder %s", model)
                raise self.unavailable(f"could not load {model}: {_reason(e)}") from e
            self._encoders[model] = enc
        return np.asarray(enc.encode_text([text]), dtype=np.float32)

    def predict(
        self,
        text: str | None,
        load_clip: Callable[[], np.ndarray] | None = None,
        sample_rate: int = 44100,
    ) -> TextPrediction:
        """Board for ``text``. ``load_clip`` gives the input clip at the inference level; it is
        only called for audio methods (``Method.uses_audio``)."""
        text = (text or "").strip()
        if not text:
            raise TextMethodError("Type an instruction first.")
        if len(text) > MAX_INSTRUCTION_CHARS:
            raise TextMethodError(
                f"The instruction is too long (max {MAX_INSTRUCTION_CHARS} characters)."
            )
        m = self.method()
        with self._predict_lock:  # encoders / the P6 checkpoint load lazily on first use
            board = self._predict_board(m, text, load_clip, sample_rate)
        try:
            info = dict(m.info())
        except Exception:  # metadata must never break the knobs
            log.exception("info() of %s failed", self.method_name)
            info = {"name": self.method_name}
        return TextPrediction(method=self.method_name, board=board, info=info)

    def _predict_board(
        self,
        m: Method,
        text: str,
        load_clip: Callable[[], np.ndarray] | None,
        sample_rate: int,
    ) -> FloatBoard:
        try:
            if isinstance(m, KnnMethod):
                boards = m.predict_vectors(self._encode_text(m.model, text))
            else:
                clip = None
                if m.uses_audio and load_clip is not None:
                    try:
                        clip = load_clip()
                    except ValueError as e:  # user-facing decode errors
                        raise TextMethodError(str(e)) from e
                query = demo_query(text, clip, sample_rate, self.peak_dbfs)
                boards = m.predict([query])
            return complete_board(boards[0])
        except TextMethodError:
            raise
        except NotImplementedError as e:
            raise self.unavailable(ENCODERS_MISSING) from e
        except Exception as e:
            log.exception("demo text method %s failed", self.method_name)
            raise self.unavailable(_reason(e)) from e


# ============================================================================ baseline vs model

BASELINE = "baseline"
MODEL = "model"
MODEL_METHOD = "p6_model"
DEFAULT_MODEL_CHECKPOINT = "runs/stage1_ab_bge_text/seed0/checkpoint.pt"
METHOD_LABELS = {
    "mean": "mean",
    "noop": "no-op",
    "knn_clap": "kNN-CLAP",
    "knn_bge": "kNN-bge",
    "p6_model": "P6 model",
}
MODEL_LABEL = "P6 model (bge-text variant, stage 1)"
MODEL_MISSING = (
    "The P6 model is not available here: its trained checkpoint is not part of this "
    "deployment (checkpoints are trained locally and not shipped). The baseline still works."
)
MODEL_NOTE = (
    "*Baseline vs model: on the LLM-written gold sets the P6 model (bge-text variant) is not "
    "distinguishable from the kNN-bge baseline (paired 2AFC differences -0.024 / +0.001 / "
    "-0.012, every 95 % CI includes 0), nor on the unseen pedal (MR-STFT -0.021, CI includes "
    "0); in distribution (synthetic test split) it is better (MR-STFT -0.130 "
    "[-0.173, -0.085]). Those are 5-seed results; the demo runs one trained seed. No real "
    "human test items yet. Details: reports/v2_results.md in the repository. The nearest "
    "pedal describes the predicted settings (a lookup), not an independent judgement.*"
)


def resolve_checkpoint(path: str | Path | None) -> Path | None:
    """An existing checkpoint file for ``demo.model_checkpoint`` (relative paths are tried
    against the working directory, then the repo root), or ``None``."""
    if path is None or not str(path).strip():
        return None
    p = Path(str(path))
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, REPO_ROOT / p]
    for c in candidates:
        if c.is_file():
            return c
    return None


def model_config(cfg: Any, checkpoint: Path) -> DictConfig:
    """``cfg`` with ``eval.model.checkpoint`` set to ``checkpoint`` and ``encode_missing``
    from ``demo.model_encode_missing`` (default true: demo queries are never cached)."""
    base = cfg if isinstance(cfg, DictConfig) else OmegaConf.create({})
    out = OmegaConf.merge(base, {})
    encode = bool(_select(base, "demo.model_encode_missing", True))
    OmegaConf.update(out, "eval.model.checkpoint", str(checkpoint), force_add=True)
    OmegaConf.update(out, "eval.model.encode_missing", encode, force_add=True)
    return out


def default_model_setter(cfg: Any) -> TextKnobSetter | None:
    """A :class:`TextKnobSetter` for the P6 model at ``demo.model_checkpoint``, or ``None``
    when no config is loaded or the checkpoint file does not exist (clean checkout, Space)."""
    if not isinstance(cfg, DictConfig):
        return None
    ckpt = resolve_checkpoint(_select(cfg, "demo.model_checkpoint", None))
    if ckpt is None:
        return None
    return TextKnobSetter(model_config(cfg, ckpt), MODEL_METHOD)


class TextMethods:
    """The two text paths of the M4 demo: the baseline (``demo.text_method``) and the P6
    model. :meth:`predict` routes by choice (:data:`BASELINE` / :data:`MODEL`); a missing
    model gives :data:`MODEL_MISSING` and never affects the baseline."""

    def __init__(self, baseline: TextKnobSetter, model: TextKnobSetter | None = None) -> None:
        self.baseline = baseline
        self.model = model

    @property
    def model_available(self) -> bool:
        return self.model is not None

    def label(self, choice: str) -> str:
        if choice == MODEL:
            return MODEL_LABEL
        name = self.baseline.method_name
        return f"{METHOD_LABELS.get(name, name)} (baseline)"

    def choices(self) -> list[tuple[str, str]]:
        """Radio choices ``(label, value)``; the model is listed even when it is missing (it
        then answers with :data:`MODEL_MISSING`)."""
        return [(self.label(BASELINE), BASELINE), (self.label(MODEL), MODEL)]

    def setter(self, choice: str | None) -> TextKnobSetter:
        if choice in (None, "", BASELINE):
            return self.baseline
        if choice == MODEL:
            if self.model is None:
                raise TextMethodError(MODEL_MISSING)
            return self.model
        raise TextMethodError(f"unknown text method choice {choice!r}")

    def predict(
        self,
        choice: str | None,
        text: str | None,
        load_clip: Callable[[], np.ndarray] | None = None,
        sample_rate: int = 44100,
    ) -> TextPrediction:
        return self.setter(choice).predict(text, load_clip, sample_rate=sample_rate)

    def warm_up(self) -> None:
        self.baseline.warm_up()
        if self.model is not None:
            self.model.warm_up()


# ============================================================================ attribution


def controls_match(
    expected_enabled: Sequence[bool],
    expected_sliders: Sequence[float],
    enabled_vals: Sequence[Any],
    slider_vals: Sequence[Any],
    steps: Sequence[float],
) -> bool:
    """Whether the current controls still hold what the text action set (within half a
    slider step, since the UI may snap values to its step)."""
    if len(expected_enabled) != len(enabled_vals) or len(expected_sliders) != len(slider_vals):
        return False
    if any(bool(a) != bool(b) for a, b in zip(expected_enabled, enabled_vals, strict=True)):
        return False
    for want, got, step in zip(expected_sliders, slider_vals, steps, strict=True):
        if got is None or abs(float(want) - float(got)) > 0.5 * float(step) + 1e-9:
            return False
    return True


__all__ = [
    "BASELINE",
    "DEFAULT_CONFIG",
    "DEFAULT_METHOD",
    "DEMO_METHODS",
    "MODEL",
    "MODEL_MISSING",
    "MODEL_NOTE",
    "DemoQuery",
    "TextKnobSetter",
    "TextMethodError",
    "TextMethods",
    "TextPrediction",
    "controls_match",
    "default_model_setter",
    "demo_query",
    "model_config",
    "resolve_checkpoint",
    "info_summary",
    "scrub_paths",
]
