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


# Windows drive paths, UNC paths and absolute POSIX paths with at least two components.
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"`]*"  # C:\... , C:/... , \\server\...
    r"|(?<![\w.~-])/(?:[^\s/'\"`]+/)+[^\s'\"`]*"  # /home/x/y
)


def scrub_paths(text: str) -> str:
    """Replace anything that looks like a filesystem path with ``<path>``."""
    return _PATH_RE.sub("<path>", str(text))


def _reason(e: BaseException) -> str:
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
        self._lock = threading.Lock()

    @property
    def peak_dbfs(self) -> float:
        return float(_select(self.cfg, "audio.inference_peak_dbfs", -15.0))

    def unavailable(self, reason: str) -> TextMethodError:
        prefix = "text model" if self.method_name in EMBEDDING_METHODS else "text method"
        return TextMethodError(f"{prefix} unavailable ({self.method_name}): {reason}")

    def method(self) -> Method:
        """The built and fitted method (built on first use, then cached)."""
        with self._lock:
            if self._method is not None:
                return self._method
            name = self.method_name
            if name not in DEMO_METHODS:
                raise TextMethodError(
                    f"text method unavailable: {name!r} is not offered in the demo "
                    f"(demo.text_method: one of {', '.join(DEMO_METHODS)})"
                )
            if self.cfg is None:
                raise self.unavailable("no demo config was loaded")
            try:
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
            self._method = m
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
            board = complete_board(boards[0])
        except TextMethodError:
            raise
        except NotImplementedError as e:
            raise self.unavailable(ENCODERS_MISSING) from e
        except Exception as e:
            log.exception("demo text method %s failed", self.method_name)
            raise self.unavailable(_reason(e)) from e
        try:
            info = dict(m.info())
        except Exception:  # metadata must never break the knobs
            log.exception("info() of %s failed", self.method_name)
            info = {"name": self.method_name}
        return TextPrediction(method=self.method_name, board=board, info=info)


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
    "DEFAULT_CONFIG",
    "DEFAULT_METHOD",
    "DEMO_METHODS",
    "DemoQuery",
    "TextKnobSetter",
    "TextMethodError",
    "TextPrediction",
    "controls_match",
    "demo_query",
    "info_summary",
    "scrub_paths",
]
