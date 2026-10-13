"""Task 4.5: import friend-written gold instructions (``lstmabar data gold-import``).

``run(cfg, args)`` reads a CSV of responses (a Google Form export with the columns of
docs/gold-collection.md, read as utf-8-sig), validates every row, anonymises respondents to
``friend-NN`` and **appends** the accepted rows to ``gold.human_file`` as ``source: human``
items with unlabelled targets (every ``target`` field null until the owner labels it).

Guarantees:

* Rejections are printed as ``row N: <reason>`` (N = the spreadsheet row, header = row 1);
  never the row's content. Raw respondent names are never printed or written to the repo.
* Respondents map to ``friend-NN`` in first-seen order, stable across imports, through a
  local mapping (``gold.authors_file``, under the gitignored cache dir) keyed by the sha256
  of the normalised respondent. The raw name is not stored there either.
* The human file is append-only: the new items are appended as YAML text in the file's
  style, then the whole file is re-parsed and the old bytes and items are checked unchanged
  (on failure the original is restored and the error raised). Existing ``simulated_human``
  items are never touched.
* Rows whose instruction (whitespace- and case-normalised) already appears in any
  ``gold.gold_dir/instructions_*.yaml`` or earlier in the CSV are rejected, so re-running
  the same CSV appends nothing.
* ``--dry-run`` prints the plan (ids, author ids, counts) and writes nothing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SOURCE_HUMAN = "human"
AUTHOR_PREFIX = "friend-"
ID_PREFIX = "h"
MODES = ("example", "free")
# Column headers (exact). example_id may be absent from a free-mode-only form.
CSV_COLUMNS = (
    "timestamp",
    "respondent",
    "mode",
    "example_id",
    "instruction",
    "context",
    "own_words",
    "consent",
)
REQUIRED_COLUMNS = ("respondent", "mode", "instruction", "own_words", "consent")
TARGET_KEYS = (
    "in_scope",
    "family",
    "gain",
    "brightness",
    "mids",
    "compression",
    "clipping",
    "reference_pedal",
)
DEFAULT_MAX_CHARS = 1000

_EXAMPLE_ID = re.compile(r"ex\d\d")
_ITEM_ID = re.compile(rf"{ID_PREFIX}(\d+)")
_AUTHOR_ID = re.compile(rf"{AUTHOR_PREFIX}(\d+)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(
    r"(?:https?://|ftp://|www\.)\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|io|ly|gl|app|dev|be|me|tv|fm|co\.uk)\b(?:/\S*)?",
    re.IGNORECASE,
)
# Digit runs joined by single separators; judged by digit count in _looks_like_phone.
_PHONE_CANDIDATE = re.compile(r"\+?\(?\d(?:[\d]|[ .\-()]{1,2}(?=[\d(]))*\d")
_LOCAL_PHONE = re.compile(r"\d{3}[ .\-]\d{4}")
_HANDLE = re.compile(r"(?<![\w.])@[A-Za-z_][\w.]{2,}")


@dataclass(frozen=True)
class Submission:
    """One validated CSV row (``row`` = spreadsheet row number; header = 1)."""

    row: int
    respondent_key: str  # sha256 of the normalised respondent; the raw name is dropped
    mode: str
    example_id: str | None
    instruction: str
    context: str


@dataclass
class ImportPlan:
    items: list[dict[str, Any]] = field(default_factory=list)  # new items, file order
    rejected: list[tuple[int, str]] = field(default_factory=list)  # (row, reason)
    authors: dict[str, str] = field(default_factory=dict)  # key -> friend-NN (updated)
    new_authors: list[str] = field(default_factory=list)  # friend ids first assigned here


# --- normalisation -------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run (newlines included) to one space; strip the ends."""
    return " ".join(str(text).split())


def normalize_instruction(text: str) -> str:
    """Duplicate key: whitespace-normalised and casefolded."""
    return normalize_whitespace(text).casefold()


def respondent_key(respondent: str) -> str:
    """sha256 hex of the whitespace-normalised, casefolded respondent name/handle."""
    norm = normalize_whitespace(respondent).casefold()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def is_yes(value: str | None) -> bool:
    """``yes`` (any case), or a form option text that starts with the word "yes"."""
    v = normalize_whitespace(value or "").casefold()
    return v == "yes" or bool(re.match(r"yes\b", v))


# --- validation ----------------------------------------------------------------------


def _looks_like_phone(text: str) -> bool:
    for m in _PHONE_CANDIDATE.finditer(text):
        s = m.group(0)
        digits = sum(c.isdigit() for c in s)
        if digits >= 9 or (digits == 7 and _LOCAL_PHONE.fullmatch(s)):
            return True
    return False


def pii_reason(text: str) -> str | None:
    """Why ``text`` looks like it holds contact details (email, URL, phone, @handle), else
    None. A heuristic: the owner still reads imported text before committing it."""
    if _EMAIL.search(text):
        return "email address"
    if _URL.search(text):
        return "URL"
    if _looks_like_phone(text):
        return "phone number"
    if _HANDLE.search(text):
        return "social handle"
    return None


def _name_in_text(respondent: str, text: str) -> bool:
    """The respondent's full name (3+ chars), or any of its words of 4+ chars, is in ``text``."""
    norm = normalize_whitespace(respondent).casefold().lstrip("@")
    hay = normalize_whitespace(text).casefold()
    parts = [norm] if len(norm) >= 3 else []
    parts += [w for w in re.split(r"[^\w]+", norm) if len(w) >= 4]
    return any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", hay) for p in parts)


def validate_row(
    row: Mapping[str, str | None], row_number: int, *, max_chars: int = DEFAULT_MAX_CHARS
) -> tuple[Submission | None, str | None]:
    """``(Submission, None)`` if the row is acceptable (before duplicate checks), else
    ``(None, reason)``. Reasons never quote the row's content."""

    def get(key: str) -> str:
        return normalize_whitespace(row.get(key) or "")

    respondent = get("respondent")
    if not respondent:
        return None, "missing respondent"
    mode = get("mode").casefold()
    if mode not in MODES:
        return None, f"mode must be one of {', '.join(MODES)}"
    example_id = get("example_id").casefold()
    if mode == "example":
        if not example_id:
            return None, "missing example_id (required for mode=example)"
        if not _EXAMPLE_ID.fullmatch(example_id):
            return None, "invalid example_id (expected ex01, ex02, ...)"
    elif example_id:
        return None, "example_id given for mode=free"
    if not is_yes(row.get("own_words")):
        return None, "own_words is not yes"
    if not is_yes(row.get("consent")):
        return None, "consent is not yes"
    instruction = get("instruction")
    if not instruction:
        return None, "empty instruction"
    if len(instruction) > max_chars:
        return None, f"instruction longer than {max_chars} characters"
    context = get("context")
    if len(context) > max_chars:
        return None, f"context longer than {max_chars} characters"
    for name, text in (("instruction", instruction), ("context", context)):
        reason = pii_reason(text)
        if reason:
            return None, f"PII-like content in {name} ({reason})"
        if _name_in_text(respondent, text):
            return None, f"PII-like content in {name} (respondent's name)"
    return (
        Submission(
            row=row_number,
            respondent_key=respondent_key(respondent),
            mode=mode,
            example_id=example_id if mode == "example" else None,
            instruction=instruction,
            context=context,
        ),
        None,
    )


# --- reading -------------------------------------------------------------------------


def read_csv(path: str | Path) -> list[tuple[int, dict[str, str | None]]]:
    """``[(row_number, row)]`` from a utf-8(-sig) CSV; raises ValueError on missing columns."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = [(h or "").strip() for h in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
        reader.fieldnames = header
        return [(i, dict(r)) for i, r in enumerate(reader, start=2)]


def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Items of one gold YAML file (a list; an empty file is an empty list)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{Path(path).name}: expected a YAML list of items")
    return data


def existing_instructions(gold_dir: str | Path) -> set[str]:
    """Normalised instructions of every ``instructions_*.yaml`` in ``gold_dir``."""
    out: set[str] = set()
    for path in sorted(Path(gold_dir).glob("instructions_*.yaml")):
        for item in load_items(path):
            if isinstance(item, dict) and item.get("instruction"):
                out.add(normalize_instruction(item["instruction"]))
    return out


def load_authors(path: str | Path) -> dict[str, str]:
    """The local ``{sha256(respondent): friend-NN}`` mapping ({} if absent)."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return dict(data.get("authors", {}))


def save_authors(path: str | Path, authors: Mapping[str, str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = {
        "note": "sha256 of the normalised respondent -> pseudonym. Local only; never commit.",
        "authors": dict(sorted(authors.items(), key=lambda kv: kv[1])),
    }
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# --- planning ------------------------------------------------------------------------


def next_id_number(items: Iterable[Mapping[str, Any]]) -> int:
    """1 + the largest ``h###`` id number (1 for an empty file)."""
    nums = [int(m.group(1)) for it in items if (m := _ITEM_ID.fullmatch(str(it.get("id", ""))))]
    return max(nums, default=0) + 1


def _author_numbers(values: Iterable[str]) -> list[int]:
    return [int(m.group(1)) for v in values if (m := _AUTHOR_ID.fullmatch(str(v)))]


def new_item(item_id: str, author: str, sub: Submission) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "source": SOURCE_HUMAN,
        "author": author,
        "instruction": sub.instruction,
        "target": dict.fromkeys(TARGET_KEYS),
        "notes": sub.context,
    }
    if sub.example_id is not None:
        item["example_id"] = sub.example_id
    return item


def plan_import(
    rows: Sequence[tuple[int, Mapping[str, str | None]]],
    existing_items: Sequence[Mapping[str, Any]],
    known_instructions: set[str],
    authors: Mapping[str, str],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ImportPlan:
    """Validate, de-duplicate, anonymise and number ``rows``. Pure: writes nothing.

    ``known_instructions`` are normalised instructions already in the gold files;
    ``existing_items`` are the human file's items (for id and pseudonym continuation).
    """
    plan = ImportPlan(authors=dict(authors))
    seen = set(known_instructions)
    next_num = next_id_number(existing_items)
    used = _author_numbers(plan.authors.values()) + _author_numbers(
        str(it.get("author", "")) for it in existing_items
    )
    next_author = max(used, default=0) + 1
    for row_number, row in rows:
        sub, reason = validate_row(row, row_number, max_chars=max_chars)
        if sub is None:
            plan.rejected.append((row_number, reason or "invalid row"))
            continue
        key = normalize_instruction(sub.instruction)
        if key in seen:
            plan.rejected.append((row_number, "duplicate instruction"))
            continue
        seen.add(key)
        author = plan.authors.get(sub.respondent_key)
        if author is None:
            author = f"{AUTHOR_PREFIX}{next_author:02d}"
            next_author += 1
            plan.authors[sub.respondent_key] = author
            plan.new_authors.append(author)
        plan.items.append(new_item(f"{ID_PREFIX}{next_num:03d}", author, sub))
        next_num += 1
    return plan


# --- writing -------------------------------------------------------------------------


def _scalar(value: Any) -> str:
    """One YAML scalar, as safe_dump writes it in block style (quoted when needed)."""
    if value is None:
        return "null"
    text = yaml.safe_dump({"k": value}, allow_unicode=True, width=float("inf"), sort_keys=False)
    assert text.startswith("k: ") and text.count("\n") == 1, text
    return text[3:-1]


def format_items(items: Sequence[Mapping[str, Any]], newline: str = "\n") -> str:
    """YAML text for list items in the gold files' style (2-space keys, no blank lines)."""
    lines: list[str] = []
    for item in items:
        first = True
        for key, value in item.items():
            prefix = "- " if first else "  "
            first = False
            if isinstance(value, Mapping):
                lines.append(f"{prefix}{key}:")
                lines.extend(f"    {k}: {_scalar(v)}" for k, v in value.items())
            else:
                lines.append(f"{prefix}{key}: {_scalar(value)}")
    return "".join(line + newline for line in lines)


def append_items(path: str | Path, items: Sequence[Mapping[str, Any]]) -> None:
    """Append ``items`` to the YAML list file at ``path``, append-only and verified.

    The original bytes stay byte-identical (the text is appended); afterwards the file is
    re-parsed and must equal the old items followed by ``items``. On any failure the
    original bytes are restored and the error re-raised.
    """
    p = Path(path)
    original = p.read_bytes()
    old_items = load_items(p)
    newline = "\r\n" if b"\r\n" in original else "\n"
    text = format_items(items, newline)
    if original and not original.endswith(b"\n"):
        text = newline + text
    try:
        with open(p, "ab") as f:
            f.write(text.encode("utf-8"))
        after = p.read_bytes()
        if not after.startswith(original):
            raise RuntimeError(f"{p.name}: existing bytes changed during append")
        parsed = load_items(p)
        if parsed[: len(old_items)] != old_items:
            raise RuntimeError(f"{p.name}: existing items changed during append")
        if parsed[len(old_items) :] != [dict(it) for it in items]:
            raise RuntimeError(f"{p.name}: appended items did not round-trip")
    except BaseException:
        p.write_bytes(original)
        raise


# --- CLI step ------------------------------------------------------------------------


def _path(value: Any) -> Path:
    return Path(str(value))


def run(cfg, args) -> int:
    """``lstmabar data gold-import --input friends.csv [--dry-run]``."""
    gold = cfg.get("gold") or {}
    gold_dir = _path(gold.get("gold_dir", "data/gold"))
    human_file = _path(gold.get("human_file", gold_dir / "instructions_human.yaml"))
    authors_file = _path(
        gold.get("authors_file", _path(cfg.data.cache_dir) / "gold" / "authors.json")
    )
    max_chars = int(gold.get("max_instruction_chars", DEFAULT_MAX_CHARS))
    dry_run = bool(getattr(args, "dry_run", False))

    input_path = getattr(args, "input", None)
    if not input_path:
        print("gold-import: --input <responses.csv> is required")
        return 2
    if not Path(input_path).is_file():
        print("gold-import: input CSV not found")
        return 2
    try:
        rows = read_csv(input_path)
    except (ValueError, csv.Error, UnicodeDecodeError) as e:
        print(f"gold-import: cannot read the CSV: {e}")
        return 2

    existing = load_items(human_file)
    authors = load_authors(authors_file)
    if not authors and _author_numbers(str(it.get("author", "")) for it in existing):
        print(
            "warning: the human file already has friend-NN authors but the local author "
            "mapping is missing; returning friends will get new pseudonyms"
        )
    plan = plan_import(
        rows, existing, existing_instructions(gold_dir), authors, max_chars=max_chars
    )

    for row_number, reason in plan.rejected:
        print(f"row {row_number}: rejected: {reason}")
    verb = "would append" if dry_run else "appending"
    for item in plan.items:
        ex = f", example {item['example_id']}" if "example_id" in item else ""
        print(f"{verb} {item['id']} ({item['author']}{ex})")
    print(
        f"{len(rows)} rows: {len(plan.items)} accepted, {len(plan.rejected)} rejected; "
        f"{len(plan.new_authors)} new author(s), "
        f"{len({it['author'] for it in plan.items})} author(s) in this import"
    )
    if dry_run:
        print("dry run: nothing written")
        return 0
    if not plan.items:
        print("nothing to append")
        return 0
    if plan.new_authors:
        save_authors(authors_file, plan.authors)
    append_items(human_file, plan.items)
    print(
        f"appended {len(plan.items)} item(s) to {human_file.name} "
        f"({plan.items[0]['id']}..{plan.items[-1]['id']}); targets are unlabelled (null)"
    )
    return 0


__all__ = [
    "CSV_COLUMNS",
    "ImportPlan",
    "REQUIRED_COLUMNS",
    "Submission",
    "TARGET_KEYS",
    "append_items",
    "existing_instructions",
    "format_items",
    "is_yes",
    "load_authors",
    "load_items",
    "new_item",
    "next_id_number",
    "normalize_instruction",
    "normalize_whitespace",
    "pii_reason",
    "plan_import",
    "read_csv",
    "respondent_key",
    "run",
    "save_authors",
    "validate_row",
]
