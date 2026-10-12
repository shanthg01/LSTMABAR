"""Task 4.7: leakage tests and the freeze step (4.6) on a small synthetic frozen dataset.

The no-held-out-path rule for caption code is enforced by ``test_caption_hygiene.py`` (not
duplicated here). These tests build fixture dry clips (``make_fixture_dataset``), synthetic
wet items with real template captions, and run the freeze checks on them.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lstmabar.data import freeze, vocab_audit
from lstmabar.data.captions import synthetic_items, template_captions
from lstmabar.data.card import PENDING, render_data_card
from lstmabar.data.fixtures import fixture_config, make_fixture_dataset
from lstmabar.data.layout import DONE_MARKER, DataLayout
from lstmabar.data.levels import sha256_file, write_flac
from lstmabar.data.schema import Caption, DryClip, RenderItem, read_jsonl, write_jsonl
from lstmabar.data.vocab import allowed_names, item_caption_violations, strip_allowed_names

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = "rat"


def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(allow_partial=kw.get("allow_partial", False))


def _templates() -> dict[str, RenderItem]:
    """One synthetic item per role: a (non-holdout) pedal, a free board, the holdout."""
    pool = synthetic_items(300, holdout=HOLDOUT, seed=1)
    pick = {
        "pedal": next(i for i in pool if i.scenario == "pedal" and i.pedal_id != HOLDOUT),
        "free": next(i for i in pool if i.scenario == "eq"),
        "holdout": next(i for i in pool if i.pedal_id == HOLDOUT),
    }
    return pick


def build_dataset(root: Path, shards: int = 2):
    """Fixture clips + rendered-looking shards (items, captions, wet FLACs, DONE)."""
    cfg = fixture_config(root, [f"freeze.data_card={(root / 'DATA_CARD.md').as_posix()}"])
    clips = make_fixture_dataset(cfg, clips_per_performer=2, seconds=0.25)
    layout = DataLayout.from_config(cfg)
    tmpl = _templates()
    sr = int(cfg.audio.sample_rate)
    rng = np.random.default_rng(0)
    specs: list[tuple[DryClip, str, RenderItem]] = []
    for c in clips:
        specs.append((c, c.split, tmpl["pedal"]))
        specs.append((c, c.split, tmpl["free"]))
        if c.split == "test":
            specs.append((c, "test_unseen_pedal", tmpl["holdout"]))
    per = -(-len(specs) // shards)
    for s in range(shards):
        sdir = layout.shard_dir(s)
        items, caps = [], []
        for k, (clip, split, base) in enumerate(specs[s * per : (s + 1) * per]):
            item_id = f"{clip.clip_id}-r{s}{k}"
            rel = layout.wet_relpath(s, item_id)
            write_flac(layout.resolve_rendered(rel), 0.5 * rng.uniform(-1, 1, 2000), sr, "PCM_24")
            it = replace(
                base,
                item_id=item_id,
                clip_id=clip.clip_id,
                split=split,
                wet_path=rel,
                sha256=sha256_file(layout.resolve_rendered(rel)),
            )
            items.append(it)
            caps += template_captions(it, 1, 0.5, HOLDOUT)
            caps.append(
                Caption(
                    caption_id=f"{item_id}-c1",
                    item_id=item_id,
                    kind="paraphrase",
                    text="Give me a warm, smooth tone with a little grit",
                    model="test-model",
                    prompt_version="p1",
                )
            )
        write_jsonl(sdir / "items.jsonl", items)
        write_jsonl(sdir / "captions.jsonl", caps)
        (sdir / DONE_MARKER).write_text(json.dumps({"items": len(items)}) + "\n", "utf-8")
    return cfg, layout


@pytest.fixture
def frozen(tmp_path):
    return build_dataset(tmp_path)


def _rewrite(path: Path, cls, fn) -> None:
    write_jsonl(path, [fn(r) for r in read_jsonl(path, cls)])


def _fails(cfg, layout, match: str) -> None:
    with pytest.raises(freeze.FreezeError, match=match):
        freeze.verify(cfg, layout)


# ----------------------------------------------------------------------------- end to end


def test_freeze_end_to_end_is_deterministic(frozen):
    cfg, layout = frozen
    assert freeze.run(cfg, _args()) == 0
    index_1 = layout.index.read_bytes()
    card = Path(str(cfg.freeze.data_card))
    card_1 = card.read_text(encoding="utf-8")
    assert freeze.run(cfg, _args()) == 0
    assert layout.index.read_bytes() == index_1 and card.read_text(encoding="utf-8") == card_1

    ix = json.loads(index_1)
    items = [
        it
        for d in sorted(layout.wet_dir.glob("shard-*"))
        for it in read_jsonl(d / "items.jsonl", RenderItem)
    ]
    clips = list(read_jsonl(layout.dry_manifest, DryClip))
    lines = layout.file_hashes.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(items) + len(clips) == ix["files"]["count"]
    assert lines == sorted(lines, key=lambda line: line.split("  ", 1)[1])
    for line in lines:
        h, rel = line.split("  ", 1)
        assert sha256_file(layout.resolve_rendered(rel)) == h
    assert ix["items"]["total"] == len(items)
    assert ix["items"]["by_split"]["test_unseen_pedal"] == ix["items"]["by_pedal"][HOLDOUT]
    assert ix["holdout_pedal"] == HOLDOUT and ix["partial"] is False
    assert ix["captions"]["by_kind"] == {"paraphrase": len(items), "template": len(items)}
    assert set(ix["shards"]) == {"shard-00000", "shard-00001"}
    assert str(layout.rendered) not in index_1.decode()  # no local paths in committed files
    assert ix["dataset_sha256"] in card_1 and PENDING not in card_1
    assert f"{len(items)} rendered items" in card_1


def test_dataset_hash_changes_with_content(frozen):
    cfg, layout = frozen
    freeze.run(cfg, _args())
    before = json.loads(layout.index.read_text())["dataset_sha256"]
    path = layout.shard_dir(0) / "captions.jsonl"
    _rewrite(path, Caption, lambda c: replace(c, text=c.text + " please"))
    freeze.run(cfg, _args())
    assert json.loads(layout.index.read_text())["dataset_sha256"] != before


# ----------------------------------------------------------------------------- leakage


def test_no_recording_shared_across_splits(frozen):
    cfg, layout = frozen
    clips = list(read_jsonl(layout.dry_manifest, DryClip))
    by_rec: dict[str, set[str]] = {}
    for c in clips:
        by_rec.setdefault(c.recording_id, set()).add(c.split)
    assert all(len(s) == 1 for s in by_rec.values())
    # and the freeze catches a recording that leaks into a second split
    leaked = [
        replace(clips[0], clip_id="leak-0", split="test" if clips[0].split != "test" else "train")
    ]
    write_jsonl(layout.dry_manifest, clips + leaked)
    _fails(cfg, layout, "recording in several splits")


def test_captions_contain_no_numbers(frozen):
    cfg, layout = frozen
    items = {
        it.item_id: it
        for d in layout.wet_dir.glob("shard-*")
        for it in read_jsonl(d / "items.jsonl", RenderItem)
    }
    for d in layout.wet_dir.glob("shard-*"):
        for c in read_jsonl(d / "captions.jsonl", Caption):
            it = items[c.item_id]
            allowed = allowed_names(it.pedal_id, c.names_pedal, HOLDOUT, it.split)
            assert not re.search(r"\d", strip_allowed_names(c.text, allowed)), c.caption_id
            assert (
                item_caption_violations(c.text, it.pedal_id, c.names_pedal, HOLDOUT, it.split) == []
            )
    _rewrite(
        layout.shard_dir(0) / "captions.jsonl",
        Caption,
        lambda c: replace(c, text="boost the mids by 6 dB") if c.kind == "paraphrase" else c,
    )
    _fails(cfg, layout, "contains a digit")


def test_freeze_detects_leaked_holdout_name(frozen):
    cfg, layout = frozen
    path = layout.shard_dir(0) / "captions.jsonl"
    items = {it.item_id: it for it in read_jsonl(layout.shard_dir(0) / "items.jsonl", RenderItem)}
    victim = next(i for i, it in items.items() if it.split == "train")
    _rewrite(
        path,
        Caption,
        lambda c: (
            replace(c, text="Make it sound like a ProCo RAT")
            if c.item_id == victim and c.kind == "paraphrase"
            else c
        ),
    )
    _fails(cfg, layout, "names forbidden pedal")


def test_holdout_named_only_in_unseen_split(frozen):
    cfg, layout = frozen
    freeze.verify(cfg, layout)  # the fixture itself is clean
    path = layout.shard_dir(0) / "items.jsonl"
    first = next(read_jsonl(path, RenderItem)).item_id
    _rewrite(
        path,
        RenderItem,
        lambda it: replace(it, pedal_id=HOLDOUT, scenario="pedal") if it.item_id == first else it,
    )
    _fails(cfg, layout, "holdout pedal outside test_unseen_pedal")


def test_unseen_items_only_on_test_clips(frozen):
    cfg, layout = frozen
    path = layout.shard_dir(0) / "items.jsonl"
    victim = next(it for it in read_jsonl(path, RenderItem) if it.split == "train")
    _rewrite(
        path,
        RenderItem,
        lambda it: replace(it, split="test_unseen_pedal") if it.item_id == victim.item_id else it,
    )
    _fails(cfg, layout, "unseen-pedal item off test")


def test_freeze_detects_tampered_file(frozen):
    cfg, layout = frozen
    item = next(read_jsonl(layout.shard_dir(1) / "items.jsonl", RenderItem))
    path = layout.resolve_rendered(item.wet_path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    _fails(cfg, layout, "hash mismatch")
    assert freeze.run(cfg, _args()) == 1 and not layout.index.exists()  # nothing written


def test_freeze_detects_missing_and_stray_files(frozen):
    cfg, layout = frozen
    item = next(read_jsonl(layout.shard_dir(0) / "items.jsonl", RenderItem))
    layout.resolve_rendered(item.wet_path).rename(layout.shard_dir(0) / "stray.flac")
    with pytest.raises(freeze.FreezeError) as e:
        freeze.verify(cfg, layout)
    assert "missing file" in str(e.value) and "stray audio file" in str(e.value)


def test_partial_shards_need_allow_partial(frozen):
    cfg, layout = frozen
    (layout.shard_dir(1) / DONE_MARKER).unlink()
    _fails(cfg, layout, "unfinished shards")
    assert freeze.run(cfg, _args(allow_partial=True)) == 0
    ix = json.loads(layout.index.read_text())
    assert ix["partial"] is True and ix["skipped_shards"]
    assert "PARTIAL" in Path(str(cfg.freeze.data_card)).read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- data card


def test_committed_data_card_is_generated():
    """``DATA_CARD.md`` is the rendering of the committed index (or the pending card)."""
    from lstmabar.config import load_config

    cfg = load_config(REPO_ROOT / "configs" / "data" / "v1.yaml")
    layout = DataLayout.from_config(cfg, root=REPO_ROOT)
    index = json.loads(layout.index.read_text()) if layout.index.exists() else None
    committed = (REPO_ROOT / "DATA_CARD.md").read_text(encoding="utf-8")
    assert committed == render_data_card(index, layout.version)


# ----------------------------------------------------------------------------- vocab audit

_GOLD_LIKE = """
- id: a1
  source: human
  instruction: Make it warm and smooth with a little grit
  target: {}
- id: a2
  source: simulated_human
  instruction: Zesty quokka harmonics
  target: {}
"""


def test_vocab_audit_reports_aggregates_only(frozen, tmp_path):
    cfg, layout = frozen
    freeze.run(cfg, _args())
    gold = tmp_path / "goldlike"
    gold.mkdir()
    (gold / "set.yaml").write_text(_GOLD_LIKE, encoding="utf-8")
    report = tmp_path / "reports" / "vocab_overlap.md"
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(
        cfg, {"freeze": {"gold_dir": gold.as_posix(), "vocab_report": report.as_posix()}}
    )
    assert vocab_audit.run(cfg, _args()) == 0
    text = report.read_text(encoding="utf-8")
    assert "| human | 1 |" in text and "| simulated_human | 1 |" in text
    assert "llm_draft" in text.split("No items yet for:")[1]  # missing sources listed
    for leak in ("quokka", "Zesty", "grit", "warm and smooth"):
        assert leak not in text  # no instruction or caption text in the report
    assert vocab_audit.run(cfg, _args()) == 1  # one shot: refuses to overwrite


def test_vocab_audit_coverage_numbers():
    caps = [Caption("x-c0", "x", "template", "a warm smooth tone")]
    res = vocab_audit.audit([("human", "warm tone"), ("human", "zesty quokka")], caps)
    r = res["sources"]["human"]["all"]
    assert r["items"] == 2 and r["tokens"] == 4 and r["token_cov"] == pytest.approx(0.5)
    assert r["bigram_type_cov"] == pytest.approx(0.0)


def test_vocab_audit_requires_frozen_captions(frozen):
    cfg, layout = frozen
    with pytest.raises(FileNotFoundError):
        vocab_audit.load_frozen_captions(layout)
    freeze.run(cfg, _args())
    _rewrite(layout.shard_dir(0) / "captions.jsonl", Caption, lambda c: c)  # same content
    vocab_audit.load_frozen_captions(layout)
    _rewrite(layout.shard_dir(0) / "captions.jsonl", Caption, lambda c: replace(c, text="x"))
    with pytest.raises(RuntimeError, match="differs from the frozen dataset"):
        vocab_audit.load_frozen_captions(layout)


def test_caption_code_does_not_import_the_audit():
    src = (REPO_ROOT / "src" / "lstmabar" / "data" / "captions.py").read_text(encoding="utf-8")
    assert "vocab_audit" not in src


def test_cli_registers_freeze_steps():
    from lstmabar.cli import build_parser
    from lstmabar.data.cli import STEPS

    assert {"freeze", "vocab-audit"} <= set(STEPS)
    args = build_parser().parse_args(["data", "freeze", "--allow-partial"])
    assert args.step == "freeze" and args.allow_partial
