"""Task 4.5: friend-written gold import (validation, anonymisation, append-only writes).

Every write goes to tmp copies; the real gold files are only read.
"""

import csv
import json
import shutil
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from lstmabar.cli import build_parser
from lstmabar.config import load_config
from lstmabar.data import gold_import as gi
from lstmabar.data.cli import DEFAULT_CONFIG, GOLD_CONFIG, STEPS, default_config_for

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_HUMAN = REPO_ROOT / "data" / "gold" / "instructions_human.yaml"

OTHER_GOLD = """\
# other gold file
- id: g001
  source: llm_draft
  instruction: "Make it sound like a cranked amp in a small room"
  target: {in_scope: true}
  notes: ""
"""


def _row(**kw) -> dict:
    base = {
        "timestamp": "2026/10/13 10:00:00",
        "respondent": "Alice Example",
        "mode": "free",
        "example_id": "",
        "instruction": "thick singing lead for a slow blues solo",
        "context": "",
        "own_words": "yes",
        "consent": "yes",
    }
    base.update(kw)
    return base


def _write_csv(path: Path, rows: list[dict], columns=gi.CSV_COLUMNS) -> Path:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return path


@pytest.fixture
def env(tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    human = gold / "instructions_human.yaml"
    shutil.copyfile(REAL_HUMAN, human)
    (gold / "instructions_llm_draft.yaml").write_text(OTHER_GOLD, encoding="utf-8")
    cfg = load_config(
        REPO_ROOT / GOLD_CONFIG,
        [
            f"gold.gold_dir={gold.as_posix()}",
            f"data.cache_dir={(tmp_path / 'cache').as_posix()}",
        ],
    )
    return Namespace(
        tmp=tmp_path, gold=gold, human=human, cfg=cfg, authors=Path(str(cfg.gold.authors_file))
    )


def _run(env, csv_path, dry_run=False) -> int:
    return gi.run(env.cfg, Namespace(input=str(csv_path), dry_run=dry_run))


def _reason(**kw) -> str | None:
    sub, reason = gi.validate_row(_row(**kw), 2)
    assert (sub is None) == (reason is not None)
    return reason


# --- validation ------------------------------------------------------------------------


def test_valid_rows_pass():
    assert _reason() is None
    assert _reason(mode="example", example_id="ex03") is None
    assert _reason(mode="Example", example_id=" EX12 ") is None
    assert _reason(own_words="Yes, written by me without AI tools", consent="YES") is None


@pytest.mark.parametrize(
    ("kw", "fragment"),
    [
        ({"respondent": "  "}, "missing respondent"),
        ({"mode": "both"}, "mode must be"),
        ({"mode": ""}, "mode must be"),
        ({"mode": "example", "example_id": ""}, "missing example_id"),
        ({"mode": "example", "example_id": "ex1"}, "invalid example_id"),
        ({"mode": "example", "example_id": "example01"}, "invalid example_id"),
        ({"mode": "free", "example_id": "ex01"}, "example_id given for mode=free"),
        ({"own_words": "no"}, "own_words"),
        ({"own_words": ""}, "own_words"),
        ({"consent": "no"}, "consent"),
        ({"consent": "yesterday"}, "consent"),
        ({"instruction": " \n "}, "empty instruction"),
        ({"instruction": "x" * 1001}, "longer than 1000"),
        ({"instruction": "mail me at a.b@example.org for the tone"}, "email address"),
        ({"instruction": "like this https://youtu.be/abc please"}, "URL"),
        ({"instruction": "like the one on www.example.net"}, "URL"),
        ({"instruction": "see example.com/tones"}, "URL"),
        ({"instruction": "call me on +44 7700 900123"}, "phone number"),
        ({"instruction": "text 555-123-4567 for details"}, "phone number"),
        ({"instruction": "call 555-1234"}, "phone number"),
        ({"instruction": "call (555) 123 4567"}, "phone number"),
        ({"instruction": "ig: @johnny_riffs has clips"}, "social handle"),
        ({"instruction": "ref at youtu.be/abc"}, "URL"),
        ({"context": "my site is example.com"}, "in context (URL)"),
        ({"instruction": "warm tone, thanks, Alice"}, "respondent's name"),
        ({"instruction": "warm tone -- alice example"}, "respondent's name"),
    ],
)
def test_validation_rejections(kw, fragment):
    reason = _reason(**kw)
    assert reason is not None and fragment in reason


def test_rejection_reasons_never_quote_content():
    for kw in (
        {"instruction": "mail me at secret@example.org"},
        {"instruction": "warm tone, Alice"},
        {"consent": "nope"},
    ):
        reason = _reason(**kw)
        assert "secret" not in reason and "Alice" not in reason and "nope" not in reason


def test_ordinary_tone_language_is_not_pii():
    for text in (
        "boost +2-3dB at 80Hz, cut 3.5k, 7-string in drop A, 1965-1972 Fuzz Face vibe",
        "Big Muff into a RAT, gain at 2 o'clock, like a '59 Les Paul through a Plexi",
        "J-pop cutting tone, Tele bridge, a bit of Dyna Comp. e.g. Nile Rodgers",
        "like a Marshall JCM800 2203 cranked",
    ):
        assert gi.pii_reason(text) is None, text
        assert _reason(instruction=text) is None, text


def test_short_respondent_names_do_not_block_ordinary_words():
    assert _reason(respondent="J", instruction="j-pop clean with sparkle") is None
    assert _reason(respondent="Al", instruction="all the gain please") is None


def test_missing_required_column(tmp_path):
    cols = [c for c in gi.CSV_COLUMNS if c != "consent"]
    path = _write_csv(tmp_path / "x.csv", [_row()], cols)
    with pytest.raises(ValueError, match="consent"):
        gi.read_csv(path)


def test_free_only_csv_without_example_id_column(env):
    cols = [c for c in gi.CSV_COLUMNS if c not in ("example_id", "timestamp", "context")]
    path = _write_csv(env.tmp / "f.csv", [_row()], cols)
    assert _run(env, path) == 0
    assert gi.load_items(env.human)[-1]["id"] == "h151"


def test_row_numbers_and_utf8_sig(tmp_path):
    path = _write_csv(tmp_path / "x.csv", [_row(), _row(instruction="other")])
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = gi.read_csv(path)
    assert [n for n, _ in rows] == [2, 3]
    assert "timestamp" in rows[0][1]  # BOM stripped from the first header


# --- import end to end -----------------------------------------------------------------


def _friends_csv(path: Path) -> Path:
    return _write_csv(
        path,
        [
            _row(respondent="Alice Example", mode="example", example_id="ex01"),
            _row(respondent="Bob Person", instruction="fuzzy  and\nsputtery, like a dying battery"),
            _row(
                respondent="alice   EXAMPLE",
                instruction="a bit more bite for rhythm",
                context="Strat, Deluxe Reverb",
            ),
            _row(respondent="Carol", consent="no", instruction="never stored"),
        ],
    )


def test_import_appends_human_items(env, capsys):
    before_bytes = env.human.read_bytes()
    before = gi.load_items(env.human)
    assert _run(env, _friends_csv(env.tmp / "a.csv")) == 0
    out = capsys.readouterr().out

    after_bytes = env.human.read_bytes()
    assert after_bytes.startswith(before_bytes)  # append-only, byte-identical prefix
    after = gi.load_items(env.human)
    assert after[: len(before)] == before
    assert all(it["source"] == "simulated_human" for it in after[: len(before)])

    new = after[len(before) :]
    assert [it["id"] for it in new] == ["h151", "h152", "h153"]
    assert [it["author"] for it in new] == ["friend-01", "friend-02", "friend-01"]
    assert all(it["source"] == "human" for it in new)
    assert all(it["target"] == dict.fromkeys(gi.TARGET_KEYS) for it in new)
    assert new[0]["example_id"] == "ex01"
    assert "example_id" not in new[1] and "example_id" not in new[2]
    assert new[1]["instruction"] == "fuzzy and sputtery, like a dying battery"
    assert new[1]["notes"] == "" and new[2]["notes"] == "Strat, Deluxe Reverb"
    assert list(new[0]) == [
        "id",
        "source",
        "author",
        "instruction",
        "target",
        "notes",
        "example_id",
    ]
    # same style as the file: 2-space keys, no blank lines
    tail = after_bytes[len(before_bytes) :].decode("utf-8")
    assert tail.startswith("- id: h151\n  source: human\n  author: friend-01\n")
    assert "\n\n" not in tail and "    in_scope: null\n" in tail

    assert "row 5: rejected: consent is not yes" in out
    for name in ("Alice", "alice", "Bob", "Carol"):
        assert name not in out
    assert "never stored" not in after_bytes.decode("utf-8")


def test_no_raw_names_written_and_ids_stable(env):
    _run(env, _friends_csv(env.tmp / "a.csv"))
    authors = json.loads(env.authors.read_text(encoding="utf-8"))
    text = env.authors.read_text(encoding="utf-8") + env.human.read_text(encoding="utf-8")
    for name in ("Alice", "alice", "Bob", "Example", "Person"):
        assert name not in text
    assert authors["authors"] == {
        gi.respondent_key("Alice Example"): "friend-01",
        gi.respondent_key("Bob Person"): "friend-02",
    }
    # second import: returning friends keep their ids, a new friend continues the numbering
    second = _write_csv(
        env.tmp / "b.csv",
        [
            _row(respondent="Dan New", instruction="glassy clean with a little squash"),
            _row(respondent="BOB PERSON", instruction="even more sputter"),
        ],
    )
    assert _run(env, second) == 0
    new = gi.load_items(env.human)[-2:]
    assert [(it["id"], it["author"]) for it in new] == [
        ("h154", "friend-03"),
        ("h155", "friend-02"),
    ]


def test_reimport_is_idempotent(env, capsys):
    path = _friends_csv(env.tmp / "a.csv")
    _run(env, path)
    snapshot = env.human.read_bytes()
    authors = env.authors.read_bytes()
    capsys.readouterr()
    assert _run(env, path) == 0
    out = capsys.readouterr().out
    assert env.human.read_bytes() == snapshot and env.authors.read_bytes() == authors
    assert out.count("duplicate instruction") == 3 and "nothing to append" in out


def test_dry_run_writes_nothing(env, capsys):
    snapshot = env.human.read_bytes()
    assert _run(env, _friends_csv(env.tmp / "a.csv"), dry_run=True) == 0
    out = capsys.readouterr().out
    assert env.human.read_bytes() == snapshot
    assert not env.authors.exists()
    assert "would append h151 (friend-01, example ex01)" in out
    assert "would append h153 (friend-01)" in out
    assert "3 accepted, 1 rejected" in out and "dry run: nothing written" in out


def test_duplicates_against_gold_files_and_earlier_rows(env, capsys):
    real_h001 = gi.load_items(env.human)[0]["instruction"]
    path = _write_csv(
        env.tmp / "d.csv",
        [
            _row(instruction="  MAKE it sound like a cranked amp\tin a small room "),  # other file
            _row(instruction=real_h001.upper()),  # human file
            _row(instruction="new one"),
            _row(respondent="Someone Else", instruction="New   One"),  # earlier in CSV
        ],
    )
    n_before = len(gi.load_items(env.human))
    assert _run(env, path) == 0
    out = capsys.readouterr().out
    for r in (2, 3, 5):
        assert f"row {r}: rejected: duplicate instruction" in out
    items = gi.load_items(env.human)
    assert len(items) == n_before + 1 and items[-1]["instruction"] == "new one"


def test_existing_instructions_reads_every_gold_file(env):
    known = gi.existing_instructions(env.gold)
    assert gi.normalize_instruction("make it sound like a cranked amp in a small room") in known
    assert len(known) >= 150


def test_append_failure_restores_original(env, monkeypatch):
    snapshot = env.human.read_bytes()
    items = [gi.new_item("h151", "friend-01", gi.validate_row(_row(), 2)[0])]

    def bad_format(items, newline="\n"):
        return "- id: h151\n  source: human\n bad: [unclosed\n"

    monkeypatch.setattr(gi, "format_items", bad_format)
    with pytest.raises(Exception):  # noqa: B017 - YAML parse error or our RuntimeError
        gi.append_items(env.human, items)
    assert env.human.read_bytes() == snapshot


def test_append_verifies_round_trip(env, monkeypatch):
    snapshot = env.human.read_bytes()
    items = [gi.new_item("h151", "friend-01", gi.validate_row(_row(), 2)[0])]
    real = gi.format_items
    monkeypatch.setattr(
        gi, "format_items", lambda it, nl="\n": real(it, nl).replace("h151", "h999")
    )
    with pytest.raises(RuntimeError, match="round-trip"):
        gi.append_items(env.human, items)
    assert env.human.read_bytes() == snapshot


def test_quoting_round_trips(tmp_path):
    path = tmp_path / "h.yaml"
    path.write_text("- id: h001\n  source: simulated_human\n", encoding="utf-8")
    tricky = [
        "null",
        "yes",
        "- starts with a dash",
        "key: value # not a comment",
        'it\'s "quoted" & {braced} [listed] @at `tick`',
        "Ünïcødé: 日本語のトーン",
        "123",
    ]
    items = [
        gi.new_item(f"h{i + 2:03d}", "friend-01", gi.validate_row(_row(instruction=t), 2)[0])
        for i, t in enumerate(tricky)
    ]
    gi.append_items(path, items)
    got = gi.load_items(path)
    assert [it["instruction"] for it in got[1:]] == tricky


def test_append_to_file_without_trailing_newline(tmp_path):
    path = tmp_path / "h.yaml"
    original = b"- id: h007\n  source: simulated_human"
    path.write_bytes(original)
    sub = gi.validate_row(_row(), 2)[0]
    plan = gi.plan_import([(2, _row())], gi.load_items(path), set(), {})
    assert plan.items[0]["id"] == "h008"
    gi.append_items(path, plan.items)
    assert path.read_bytes().startswith(original)
    assert gi.load_items(path)[1]["instruction"] == sub.instruction


def test_plan_continues_pseudonyms_from_file_when_mapping_lost():
    existing = [{"id": "h151", "source": "human", "author": "friend-04"}]
    plan = gi.plan_import([(2, _row())], existing, set(), {})
    assert plan.items[0]["author"] == "friend-05" and plan.items[0]["id"] == "h152"


# --- real files (read-only) and CLI ----------------------------------------------------


def test_real_human_file_has_only_simulated_items():
    items = yaml.safe_load(REAL_HUMAN.read_text(encoding="utf-8"))
    assert len(items) >= 150
    sim = [it for it in items if it["source"] == "simulated_human"]
    assert [it["id"] for it in sim[:150]] == [f"h{i:03d}" for i in range(1, 151)]
    # Real items may only be appended after h150 and only by the import (source: human).
    assert {it["source"] for it in items} <= {"simulated_human", "human"}
    assert all(it["source"] == "simulated_human" for it in items[:150])


def test_cli_step_registered_with_gold_config_default():
    assert STEPS["gold-import"] == (
        "gold_import",
        "run",
        "append friend-written gold instructions (4.5)",
    )
    args = build_parser().parse_args(["data", "gold-import", "--input", "f.csv", "--dry-run"])
    assert args.step == "gold-import" and args.input == "f.csv" and args.dry_run
    assert args.config is None
    assert default_config_for("gold-import") == GOLD_CONFIG == "configs/data/gold_v1.yaml"
    assert default_config_for("render") == DEFAULT_CONFIG
    assert build_parser().parse_args(["data", "render"]).dry_run is False


def test_gold_config_composes_v1():
    cfg = load_config(REPO_ROOT / GOLD_CONFIG)
    v1 = load_config(REPO_ROOT / DEFAULT_CONFIG)
    assert cfg.data == v1.data and cfg.render == v1.render
    assert cfg.gold.gold_dir == "data/gold"
    assert cfg.gold.human_file == "data/gold/instructions_human.yaml"
    assert cfg.gold.authors_file == "data/cache/gold/authors.json"
    assert cfg.gold.max_instruction_chars == 1000


def test_cmd_data_uses_gold_config_by_default(env, monkeypatch, capsys):
    from lstmabar.data import cli as data_cli

    seen = {}

    def fake_load(path, overrides):
        seen["path"] = path
        return env.cfg

    monkeypatch.setattr(data_cli, "load_config", fake_load)
    path = _friends_csv(env.tmp / "a.csv")
    args = build_parser().parse_args(["data", "gold-import", "--input", str(path), "--dry-run"])
    assert args.func(args) == 0
    assert seen["path"] == GOLD_CONFIG
    assert "dry run: nothing written" in capsys.readouterr().out


def test_run_requires_input(env, capsys):
    assert gi.run(env.cfg, Namespace(input=None, dry_run=False)) == 2
    assert gi.run(env.cfg, Namespace(input=str(env.tmp / "missing.csv"), dry_run=False)) == 2
    out = capsys.readouterr().out
    assert str(env.tmp) not in out
