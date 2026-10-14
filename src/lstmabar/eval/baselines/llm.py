"""D4 LLM zero-shot baseline (LLM2Fx-style): instruction -> board parameters from Claude.

The model sees the board (blocks in chain order; per knob its name, unit, physical range,
taper and default, all generated from ``default_pedalboard().param_specs``; an on/off switch
per block) and the instruction, and answers with JSON of **physical** values plus an
``enabled`` boolean per block (structured outputs, ``output_config.format``). The reply is
parsed, validated, clamped to each knob's range, normalized with its ``ParamSpec`` and
completed with :func:`~lstmabar.eval.baselines.base.complete_board`. A reply that cannot be
parsed (or a refused / invalid request) gives the default board with every gate off and is
counted in ``info()["parse_failures"]``; per-field fixes are counted separately.

The prompt is written fresh. It has no examples at all (no few-shot), and never includes
caption templates, dataset captions or gold text (design §5). Its fingerprint
(:func:`prompt_hash`) is part of every cache key, so editing it without bumping
``eval.baselines.llm.prompt_version`` never reuses stale replies (a test pins the pair).

**Cost safety (the P4 R1 pattern, :mod:`lstmabar.data.captions`).** Replies are cached under
``data/cache/llm/zero_shot/`` keyed by (model, prompt version, prompt hash,
sha256(instruction)); unique instructions are deduplicated before submitting; requests go
through the Message Batches API (50 % price) with polling and resume of batches left in
flight. A separate :class:`~lstmabar.data.captions.SpendLedger`
(``data/cache/llm/zero_shot/ledger.jsonl``) records every batch; a submit is refused when
spent + in flight (worst case) + this batch's worst case (every request at ``max_tokens``)
exceeds ``eval.baselines.llm.max_usd``. One run at a time (``LedgerLock``).

Spending is opt-in: ``predict`` calls the API only with ``ANTHROPIC_API_KEY`` set **and**
``eval.baselines.llm.allow_spend=true``; otherwise uncached instructions raise
:class:`LlmCacheMiss` with the count and the estimated cost. The key is read by the SDK from
the environment and never logged. :func:`estimate` prices a run without spending
(``lstmabar eval run --estimate``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lstmabar.data.captions import (
    BudgetExceeded,
    LedgerLock,
    LedgerLocked,
    SpendLedger,
    SubmitUncertain,
    _definitely_rejected,
    _error_type,
    _usage_tokens,
    usd,
)
from lstmabar.data.layout import DataLayout
from lstmabar.data.schema import FloatBoard
from lstmabar.dsp.pedalboard import ENABLED, Pedalboard, default_pedalboard
from lstmabar.eval.baselines.base import Method, complete_board, default_board
from lstmabar.eval.data import Query

# ============================================================================ prompt


def _block_summary(block) -> str:
    """First docstring line of a block class, without reST cross-references."""
    doc = (type(block).__doc__ or "").strip().splitlines()
    line = doc[0].strip() if doc else ""
    line = re.sub(r"\s*\([^)]*:\w+:`[^`]*`[^)]*\)", "", line)
    return line


def _fmt(x: float) -> str:
    return f"{float(x):g}"


def board_description(board: Pedalboard | None = None) -> str:
    """The board's blocks in chain order with every knob's unit, range, taper and default."""
    board = board or default_pedalboard()
    lines: list[str] = []
    for i, name in enumerate(board.block_names, 1):
        block = board.blocks[name]
        summary = _block_summary(block)
        lines.append(f"{i}. {name}" + (f": {summary}" if summary else ""))
        lines.append(f'   - "{ENABLED}": true or false (default false = bypassed)')
        for s in block.param_specs:
            unit = s.unit or "unitless"
            lines.append(
                f'   - "{s.name}": {unit}, range {_fmt(s.min)} to {_fmt(s.max)}, '
                f"{s.taper} taper, default {_fmt(s.default)}"
            )
    return "\n".join(lines)


def system_prompt(board: Pedalboard | None = None) -> str:
    board = board or default_pedalboard()
    return (
        "You set the controls of a guitar effects board so that it produces the tone a "
        "guitarist asks for.\n\n"
        "A dry (clean, unprocessed) electric guitar signal passes through the blocks below, "
        "in this order. Every block has an on/off switch, "
        f'"{ENABLED}". A block that is off is bypassed and its knobs have no effect.\n\n'
        f"{board_description(board)}\n\n"
        "Reply with one JSON object that has one entry per block, each with "
        f'"{ENABLED}" and every knob of that block. Give knob values as physical values in '
        "the listed unit and inside the listed range (not normalized to 0-1). Switch a block "
        "on only if the request needs it, and leave knobs you have no reason to change at "
        "their defaults. If the request is vague or asks for something this board cannot "
        "do, give the closest setting the board can make."
    )


def user_message(instruction: str) -> str:
    return f"Requested tone:\n<request>\n{instruction}\n</request>"


def output_schema(board: Pedalboard | None = None) -> dict[str, Any]:
    """JSON schema of the reply: per block ``enabled`` (boolean) + its knobs (numbers)."""
    board = board or default_pedalboard()
    props: dict[str, Any] = {}
    for name in board.block_names:
        knobs = [s.name for s in board.blocks[name].param_specs]
        props[name] = {
            "type": "object",
            "properties": {ENABLED: {"type": "boolean"}} | {k: {"type": "number"} for k in knobs},
            "required": [ENABLED, *knobs],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": props,
        "required": list(board.block_names),
        "additionalProperties": False,
    }


def prompt_hash(board: Pedalboard | None = None) -> str:
    """Fingerprint of the system prompt, user template and output schema (cache keys)."""
    board = board or default_pedalboard()
    blob = "\n".join(
        [system_prompt(board), user_message("{instruction}"), json.dumps(output_schema(board))]
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def instruction_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_params(instruction: str, lc, board: Pedalboard | None = None) -> dict[str, Any]:
    """Message params for one instruction (``lc`` = ``eval.baselines.llm``)."""
    board = board or default_pedalboard()
    return {
        "model": str(lc.model),
        "max_tokens": int(lc.max_tokens),
        "thinking": {"type": "disabled"},
        "output_config": {
            "effort": str(lc.effort),
            "format": {"type": "json_schema", "schema": output_schema(board)},
        },
        "system": system_prompt(board),
        "messages": [{"role": "user", "content": user_message(instruction)}],
    }


# ============================================================================ parsing


@dataclass
class ParseStats:
    parse_failures: int = 0  # unparseable / missing / refused replies (whole board defaulted)
    blocks_defaulted: int = 0  # a block entry missing or malformed (gate off, defaults)
    knobs_defaulted: int = 0  # a knob missing or not a finite number (default)
    knobs_clamped: int = 0  # a knob outside its physical range (clamped)

    def add(self, other: ParseStats) -> None:
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(float(v))


def parse_reply(text: str | None, board: Pedalboard | None = None) -> tuple[FloatBoard, ParseStats]:
    """Reply text -> complete normalized board (+ what had to be fixed).

    Physical values are clamped to the knob's range and normalized with its ``ParamSpec``;
    ``enabled`` must be a boolean (``true`` -> 1, anything else -> gate off). Unknown keys
    are ignored. ``None`` or unparseable text -> :func:`default_board` with gates off.
    """
    board = board or default_pedalboard()
    st = ParseStats()
    try:
        data = json.loads(text) if text is not None else None
    except (TypeError, ValueError):
        data = None
    if not isinstance(data, dict) or not any(isinstance(data.get(b), dict) for b in board.blocks):
        st.parse_failures = 1
        return default_board(0.0), st
    out: dict[str, dict[str, float]] = {}
    for name in board.block_names:
        entry = data.get(name)
        if not isinstance(entry, dict):
            st.blocks_defaulted += 1
            out[name] = {ENABLED: 0.0}
            continue
        knobs: dict[str, float] = {ENABLED: 1.0 if entry.get(ENABLED) is True else 0.0}
        for s in board.blocks[name].param_specs:
            v = entry.get(s.name)
            if not _is_number(v):
                st.knobs_defaulted += 1
                continue
            v = float(v)
            if v < s.min or v > s.max:
                st.knobs_clamped += 1
                v = min(max(v, s.min), s.max)
            knobs[s.name] = float(s.normalize(torch.tensor(v, dtype=torch.float64)))
        out[name] = knobs
    return complete_board(out, enabled=0.0), st


# ============================================================================ cache


class ReplyCache:
    """Append-only JSONL of raw replies keyed by (model, prompt version, prompt hash,
    sha256(instruction)). ``text`` is ``None`` for a refused or invalid request (cached so it
    is never paid again). The instruction text itself is not stored."""

    def __init__(self, path: Path, model: str, prompt_version: str, phash: str) -> None:
        self.path = Path(path)
        self.model = model
        self.prompt_version = prompt_version
        self.phash = phash
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self._data[r["key"]] = r

    def key_of_sha(self, sha: str) -> str:
        return hashlib.sha256(
            f"{self.model}\n{self.prompt_version}\n{self.phash}\n{sha}".encode()
        ).hexdigest()

    def key(self, instruction: str) -> str:
        return self.key_of_sha(instruction_sha(instruction))

    def __contains__(self, instruction: str) -> bool:
        return self.key(instruction) in self._data

    def get(self, instruction: str) -> dict[str, Any] | None:
        return self._data.get(self.key(instruction))

    def put_many(self, entries: Mapping[str, Mapping[str, Any]]) -> None:
        """``{instruction sha256: {"text", "status", "stop_reason"}}``."""
        if not entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            for sha, e in entries.items():
                rec = {
                    "key": self.key_of_sha(sha),
                    "model": self.model,
                    "prompt_version": self.prompt_version,
                    "prompt_hash": self.phash,
                    "instruction_sha256": sha,
                    "text": e.get("text"),
                    "status": e.get("status", "ok"),
                    "stop_reason": e.get("stop_reason"),
                }
                self._data[rec["key"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


# ============================================================================ estimate


def _approx_tokens(params: Mapping[str, Any]) -> int:
    """Rough input-token count without the API (~3.5 characters per token)."""
    chars = len(params["system"]) + len(json.dumps(params["messages"]))
    chars += len(json.dumps(params["output_config"]))
    return math.ceil(chars / 3.5) + 20


def expected_output_tokens(board: Pedalboard | None = None) -> int:
    """Approximate tokens of a complete reply (the default board in physical units)."""
    board = board or default_pedalboard()
    reply = {
        n: {ENABLED: False} | {s.name: s.default for s in board.blocks[n].param_specs}
        for n in board.block_names
    }
    return math.ceil(len(json.dumps(reply, indent=1)) / 3.5) + 10


def count_input_tokens(client, params: Mapping[str, Any]) -> int:
    """Input tokens of one request via the free ``count_tokens`` endpoint."""
    r = client.messages.count_tokens(
        model=params["model"],
        system=params["system"],
        messages=params["messages"],
        thinking=params["thinking"],
        output_config=params["output_config"],
    )
    return int(r.input_tokens)


def estimate(texts: Sequence[str], cfg, client=None, layout: DataLayout | None = None) -> dict:
    """Price answering ``texts`` (deduplicated; cached ones are free). No spending.

    Input tokens: the longest uncached instruction's request via ``count_tokens`` when a
    client is given (else approximated) times the number of requests. Output: the expected
    reply size; worst case: every request at ``max_tokens``. Includes the ledger state and
    whether the cap check would pass.
    """
    lc = cfg.eval.baselines.llm
    layout = layout or DataLayout.from_config(cfg)
    board = default_pedalboard()
    phash = prompt_hash(board)
    cache = ReplyCache(
        _cache_dir(layout) / "replies.jsonl", str(lc.model), str(lc.prompt_version), phash
    )
    unique = list(dict.fromkeys(texts))
    todo = [t for t in unique if t not in cache]
    price = lc.price_per_mtok
    ledger = SpendLedger(_cache_dir(layout) / "ledger.jsonl")
    spent, flight = ledger.spent(), ledger.in_flight_worst()
    per_req, src = 0, "none"
    if todo:
        params = request_params(max(todo, key=len), lc, board)
        per_req, src = _approx_tokens(params), "approx"
        if client is not None:
            try:
                per_req, src = count_input_tokens(client, params), "count_tokens"
            except Exception as e:  # noqa: BLE001 - any API/network failure: fall back
                _log(f"count_tokens failed ({type(e).__name__}); input tokens approximated")
    n = len(todo)
    tin = per_req * n
    tout = expected_output_tokens(board) * n
    max_out = int(lc.max_tokens) * n
    est_usd = usd(tin, tout, price)
    worst = usd(tin, max_out, price)
    return {
        "model": str(lc.model),
        "prompt_version": str(lc.prompt_version),
        "prompt_hash": phash,
        "texts": len(texts),
        "unique_instructions": len(unique),
        "cached": len(unique) - n,
        "requests": n,
        "input_tokens": tin,
        "input_tokens_per_request": per_req,
        "input_source": src,
        "output_tokens": tout,
        "max_output_tokens": max_out,
        "usd": est_usd,
        "worst_usd": worst,
        "ledger_spent_usd": spent,
        "ledger_in_flight_worst_usd": flight,
        "max_usd": float(lc.max_usd),
        "within_cap": spent + flight + worst <= float(lc.max_usd),
    }


def estimate_lines(est: Mapping[str, Any]) -> list[str]:
    """Human-readable summary of :func:`estimate` (for ``eval run --estimate``)."""
    return [
        f"LLM zero-shot ({est['model']}, prompt {est['prompt_version']}): "
        f"{est['texts']} texts, {est['unique_instructions']} unique, {est['cached']} cached, "
        f"{est['requests']} requests to submit",
        f"input tokens: {est['input_tokens']} ({est['input_source']}); expected output "
        f"tokens: {est['output_tokens']}; max output tokens: {est['max_output_tokens']}",
        f"estimated cost ${est['usd']:.4f}; worst case ${est['worst_usd']:.4f}",
        f"ledger: spent ${est['ledger_spent_usd']:.4f}, in flight (worst) "
        f"${est['ledger_in_flight_worst_usd']:.4f}; cap ${est['max_usd']:.2f}: "
        + ("within cap" if est["within_cap"] else "WOULD EXCEED CAP (submit refused)"),
    ]


# ============================================================================ batches


class LlmCacheMiss(RuntimeError):
    """Uncached instructions and the API is not allowed (no key / spending not enabled)."""


def _log(msg: str) -> None:
    print(f"[llm_zero_shot] {msg}", file=sys.stderr, flush=True)


def _cache_dir(layout: DataLayout) -> Path:
    return layout.llm_cache / "zero_shot"


def default_client_factory():
    """Anthropic SDK client (``llm`` extra). The key is read from ``ANTHROPIC_API_KEY`` only."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise RuntimeError("the LLM baseline needs the llm extra: uv sync --extra llm") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@dataclass
class _Outcome:
    replies: dict[str, dict[str, Any]] = field(default_factory=dict)  # sha -> cache entry
    retry: list[str] = field(default_factory=list)  # shas (errored/expired/canceled/missing)
    failed: list[str] = field(default_factory=list)  # non-retryable (auth, billing): not cached
    input_tokens: int = 0
    output_tokens: int = 0


_INVALID = {"invalid_request_error", "invalid_request"}
_NON_RETRYABLE = {"authentication_error", "billing_error", "permission_error", "not_found_error"}


def _reply_entry(message) -> dict[str, Any]:
    stop = getattr(message, "stop_reason", None)
    text = next(
        (b.text for b in getattr(message, "content", []) if getattr(b, "type", None) == "text"),
        None,
    )
    if stop == "refusal":
        return {"text": None, "status": "refusal", "stop_reason": stop}
    if stop not in (None, "end_turn"):
        return {"text": None, "status": "malformed", "stop_reason": stop}
    return {"text": text, "status": "ok" if text is not None else "malformed", "stop_reason": stop}


def collect(client, batch_id: str, mapping: Mapping[str, str]) -> _Outcome:
    """Read a finished batch's results (any order) keyed by ``custom_id`` -> sha."""
    out = _Outcome()
    seen: set[str] = set()
    for res in client.messages.batches.results(batch_id):
        sha = mapping.get(res.custom_id)
        if sha is None:
            continue
        seen.add(res.custom_id)
        r = res.result
        etype = _error_type(r) if r.type == "errored" else ""
        if r.type == "succeeded":
            tin, tout = _usage_tokens(r.message)
            out.input_tokens += tin
            out.output_tokens += tout
            out.replies[sha] = _reply_entry(r.message)
        elif etype in _INVALID:
            out.replies[sha] = {"text": None, "status": "invalid_request", "stop_reason": None}
        elif etype in _NON_RETRYABLE:
            out.failed.append(sha)
        else:  # errored (server side, overloaded), expired, canceled
            out.retry.append(sha)
    out.retry.extend(sha for cid, sha in mapping.items() if cid not in seen)
    return out


class ZeroShotBatcher:
    """Submits uncached instructions under the spend cap and fills the reply cache."""

    def __init__(
        self,
        lc,
        layout: DataLayout,
        client,
        cache: ReplyCache,
        board: Pedalboard,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.lc = lc
        self.dir = _cache_dir(layout)
        self.client = client
        self.cache = cache
        self.board = board
        self.sleep = sleep
        self.ledger = SpendLedger(self.dir / "ledger.jsonl")
        self.input_tokens = 0
        self.output_tokens = 0

    def _batch_file(self, ref: str) -> Path:
        return self.dir / "batches" / f"{ref}.json"

    def _price(self, texts: Sequence[str]) -> tuple[float, float, int, int]:
        """(estimate, worst case, input tokens, output tokens) for ``texts``."""
        params = request_params(max(texts, key=len), self.lc, self.board)
        try:
            per_req = count_input_tokens(self.client, params)
        except Exception as e:  # noqa: BLE001 - count_tokens is advisory; approximate
            _log(f"count_tokens failed ({type(e).__name__}); input tokens approximated")
            per_req = _approx_tokens(params)
        tin = per_req * len(texts)
        tout = expected_output_tokens(self.board) * len(texts)
        price = self.lc.price_per_mtok
        worst = usd(tin, int(self.lc.max_tokens) * len(texts), price)
        return usd(tin, tout, price), worst, tin, tout

    def submit(self, texts: Sequence[str], attempt: int = 0) -> str:
        """Price, check the cap (worst case), record, create. Returns the ledger ref."""
        est, worst, tin, tout = self._price(texts)
        spent, flight = self.ledger.spent(), self.ledger.in_flight_worst()
        cap = float(self.lc.max_usd)
        if spent + flight + worst > cap:
            raise BudgetExceeded(
                f"refusing to submit: spent ${spent:.4f} + in flight (worst case) "
                f"${flight:.4f} + this batch's worst case ${worst:.4f} "
                f"(estimate ${est:.4f}) > cap ${cap:.2f}"
            )
        ref = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        requests, mapping = [], {}
        for i, t in enumerate(texts):
            cid = f"a{attempt}-{i:05d}"
            requests.append({"custom_id": cid, "params": request_params(t, self.lc, self.board)})
            mapping[cid] = instruction_sha(t)
        bf = self._batch_file(ref)
        bf.parent.mkdir(parents=True, exist_ok=True)
        bf.write_text(json.dumps({"ref": ref, "attempt": attempt, "mapping": mapping}), "utf-8")
        self.ledger.append(
            "submitted",
            ref,
            estimate_usd=est,
            worst_usd=worst,
            n_requests=len(texts),
            est_input_tokens=tin,
            est_output_tokens=tout,
            max_output_tokens=int(self.lc.max_tokens) * len(texts),
            model=str(self.lc.model),
            prompt_version=str(self.lc.prompt_version),
            prompt_hash=self.cache.phash,
            attempt=attempt,
        )
        try:
            # no SDK-level retries: a retried create after a lost response would make a
            # second (paid) batch that the ledger does not know about
            batch = self.client.with_options(max_retries=0).messages.batches.create(
                requests=requests
            )
        except Exception as e:
            if _definitely_rejected(e):
                self.ledger.append(
                    "ended",
                    ref,
                    cost_usd=0.0,
                    status="rejected",
                    error=type(e).__name__,
                    status_code=getattr(e, "status_code", None),
                )
                raise
            _log(
                f"batch create for ledger ref {ref} failed ambiguously ({type(e).__name__}); "
                "it stays in flight at its worst case. Check the Console or "
                "client.messages.batches.list() and close the entry by hand (SpendLedger)."
            )
            raise SubmitUncertain(ref) from e
        self.ledger.append("created", ref, batch_id=batch.id)
        _log(f"submitted batch {batch.id}: {len(texts)} requests, est ${est:.4f}")
        return ref

    def wait_and_collect(self, ref: str) -> _Outcome:
        st = self.ledger.state()[ref]
        info = json.loads(self._batch_file(ref).read_text("utf-8"))
        batch_id = st["batch_id"]
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                break
            self.sleep(float(self.lc.poll_seconds))
        out = collect(self.client, batch_id, info["mapping"])
        cost = usd(out.input_tokens, out.output_tokens, self.lc.price_per_mtok)
        # cache first: a crash after this re-collects the batch but never pays twice
        self.cache.put_many(out.replies)
        if out.failed:
            _log(f"{len(out.failed)} requests failed with non-retryable errors (not cached)")
        self.ledger.append(
            "ended",
            ref,
            batch_id=batch_id,
            input_tokens=out.input_tokens,
            output_tokens=out.output_tokens,
            cost_usd=cost,
            n_replies=len(out.replies),
            n_retry=len(out.retry),
            n_failed=len(out.failed),
            status="ended",
        )
        self.input_tokens += out.input_tokens
        self.output_tokens += out.output_tokens
        _log(f"batch {batch_id} ended: {len(out.replies)} replies, ${cost:.4f}")
        return out

    def resume_pending(self) -> None:
        """Collect batches left in flight by an earlier run."""
        for r in self.ledger.pending():
            if "batch_id" not in r or not self._batch_file(r["ref"]).exists():
                _log(
                    f"ledger ref {r['ref']} has no batch id (process died before creation, or "
                    "the create call failed ambiguously); it stays counted at its worst case "
                    "until closed by hand (see SpendLedger)"
                )
                continue
            self.wait_and_collect(r["ref"])

    def run(self, texts: Sequence[str]) -> None:
        try:
            with LedgerLock(self.ledger.path):
                self._run(texts)
        except LedgerLocked as e:
            raise LedgerLocked(
                f"{self.ledger.path.with_suffix('.lock').name} exists: another LLM zero-shot "
                "run is active (or one crashed; delete the lock file once no run is active)"
            ) from e

    def _run(self, texts: Sequence[str]) -> None:
        self.resume_pending()
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        attempt = 0
        while todo:
            out = self.wait_and_collect(self.submit(todo, attempt))
            failed = set(out.failed)
            todo = [t for t in todo if t not in self.cache and instruction_sha(t) not in failed]
            if todo and attempt >= int(self.lc.max_resubmits):
                _log(f"{len(todo)} requests still failing after {attempt} resubmits")
                break
            attempt += 1


# ============================================================================ method


class LlmZeroShot(Method):
    """D4 LLM zero-shot baseline (text only, no training data, no hyperparameters)."""

    name = "llm_zero_shot"
    uses_audio = False
    deterministic = True  # replies are cached once per instruction

    def __init__(
        self,
        cfg: Any = None,
        seed: int = 0,
        client_factory: Callable[[], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(cfg, seed)
        self.lc = cfg.eval.baselines.llm
        self.layout = DataLayout.from_config(cfg)
        self.board = default_pedalboard()
        self.phash = prompt_hash(self.board)
        self.cache = ReplyCache(
            _cache_dir(self.layout) / "replies.jsonl",
            str(self.lc.model),
            str(self.lc.prompt_version),
            self.phash,
        )
        self.client_factory = client_factory or default_client_factory
        self.sleep = sleep
        self.estimate_only = False  # the runner sets this for ``eval run --estimate``
        self.stats = ParseStats()
        self.n_predicted = 0
        self._client = None

    def _spend_allowed(self) -> bool:
        return (
            not self.estimate_only
            and bool(self.lc.get("allow_spend", False))
            and bool(os.environ.get("ANTHROPIC_API_KEY"))
        )

    def _client_or_none(self):
        if self._client is None and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                self._client = self.client_factory()
            except RuntimeError:
                return None
        return self._client

    def estimate(self, texts: Sequence[str]) -> dict:
        """See :func:`estimate` (uses ``count_tokens`` when a key is set)."""
        return estimate(texts, self.cfg, self._client_or_none(), self.layout)

    def predict(self, queries: Sequence[Query]) -> list[FloatBoard]:
        texts = [q.text for q in queries]
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            if not self._spend_allowed():
                est = estimate(missing, self.cfg, None, self.layout)
                why = (
                    "estimate mode"
                    if self.estimate_only
                    else "ANTHROPIC_API_KEY is not set"
                    if not os.environ.get("ANTHROPIC_API_KEY")
                    else "eval.baselines.llm.allow_spend is false"
                )
                raise LlmCacheMiss(
                    f"llm_zero_shot: {len(missing)} unique instructions have no cached reply "
                    f"({why}); submitting them would cost ~${est['usd']:.4f} "
                    f"(worst case ${est['worst_usd']:.4f}, cap ${est['max_usd']:.2f}). "
                    "Set ANTHROPIC_API_KEY and eval.baselines.llm.allow_spend=true to submit."
                )
            client = self._client_or_none() or self.client_factory()
            ZeroShotBatcher(self.lc, self.layout, client, self.cache, self.board, self.sleep).run(
                missing
            )
            still = [t for t in missing if t not in self.cache]
            if still:
                _log(f"{len(still)} instructions got no reply; they use the default board")
        boards: dict[str, FloatBoard] = {}
        for t in dict.fromkeys(texts):
            rec = self.cache.get(t)
            board, st = parse_reply(rec.get("text") if rec else None, self.board)
            self.stats.add(st)
            boards[t] = board
        self.n_predicted += len(queries)
        return [boards[t] for t in texts]

    def info(self) -> dict[str, Any]:
        ledger = SpendLedger(_cache_dir(self.layout) / "ledger.jsonl")
        return {
            "name": self.name,
            "model": str(self.lc.model),
            "prompt_version": str(self.lc.prompt_version),
            "prompt_hash": self.phash,
            "api": "message batches",
            "n_queries": self.n_predicted,
            "parse_failures": self.stats.parse_failures,
            "blocks_defaulted": self.stats.blocks_defaulted,
            "knobs_defaulted": self.stats.knobs_defaulted,
            "knobs_clamped": self.stats.knobs_clamped,
            "ledger_spent_usd": round(ledger.spent(), 6),
            "max_usd": float(self.lc.max_usd),
        }


__all__ = [
    "LlmCacheMiss",
    "LlmZeroShot",
    "ParseStats",
    "ReplyCache",
    "ZeroShotBatcher",
    "board_description",
    "estimate",
    "estimate_lines",
    "instruction_sha",
    "output_schema",
    "parse_reply",
    "prompt_hash",
    "request_params",
    "system_prompt",
    "user_message",
]
