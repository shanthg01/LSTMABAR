"""D4 LLM zero-shot baseline: prompt, parsing, cache, spend cap. The SDK is always mocked."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lstmabar.data.captions import BudgetExceeded, SpendLedger, synthetic_items, template_captions
from lstmabar.dsp.pedalboard import ENABLED, default_pedalboard
from lstmabar.eval.baselines import llm
from lstmabar.eval.baselines.base import BOARD_LAYOUT, build_method, default_board
from lstmabar.eval.data import Query
from lstmabar.eval.fixtures import FIXTURE_GOLD, eval_fixture_config

BOARD = default_pedalboard()


@pytest.fixture
def cfg(tmp_path):
    return eval_fixture_config(tmp_path, ["eval.baselines.llm.poll_seconds=0"])


def _q(text: str, i: int = 0) -> Query:
    return Query(
        query_id=f"q{i}",
        unit_id=f"u{i}",
        split="test",
        text=text,
        text_key=f"q{i}",
        audio_key=f"u{i}",
        clip_id="c",
        input_peak_dbfs=-15.0,
        sample_rate=44100,
        dry_path=Path("unused.flac"),
    )


def _physical_reply(**overrides) -> dict:
    reply = {
        n: {ENABLED: False} | {s.name: s.default for s in BOARD.blocks[n].param_specs}
        for n in BOARD.block_names
    }
    for b, v in overrides.items():
        reply[b].update(v)
    return reply


# ---------------------------------------------------------------------------- fake SDK


class FakeBatches:
    def __init__(self, client):
        self.client = client
        self.created: list[list[dict]] = []
        self._batches: dict[str, list[dict]] = {}

    def create(self, requests):
        bid = f"msgbatch_{len(self.created)}"
        self.created.append(requests)
        self._batches[bid] = requests
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended")

    def results(self, bid):
        for r in self._batches[bid]:
            instr = r["params"]["messages"][0]["content"]
            text = self.client.reply(instr)
            msg = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1000, output_tokens=120),
            )
            yield SimpleNamespace(
                custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg)
            )


class FakeClient:
    def __init__(self, reply=None):
        self.reply = reply or (lambda instr: json.dumps(_physical_reply()))
        self.batches = FakeBatches(self)
        self.messages = SimpleNamespace(
            batches=self.batches,
            count_tokens=lambda **kw: SimpleNamespace(input_tokens=1000),
        )

    def with_options(self, **kw):
        return self


@pytest.fixture
def keyed(monkeypatch, cfg):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    cfg.eval.baselines.llm.allow_spend = True
    return cfg


# ---------------------------------------------------------------------------- prompt


def test_prompt_lists_every_knob_from_param_specs():
    prompt = llm.system_prompt()
    pos = [prompt.index(f"{i}. {n}") for i, n in enumerate(BOARD.block_names, 1)]
    assert pos == sorted(pos)  # chain order
    for name in BOARD.block_names:
        for s in BOARD.blocks[name].param_specs:
            line = next(ln for ln in prompt.splitlines() if f'"{s.name}"' in ln)
            assert f"range {s.min:g} to {s.max:g}" in line
            assert f"{s.taper} taper" in line and f"default {s.default:g}" in line
            assert (s.unit or "unitless") in line
    assert prompt.count(f'"{ENABLED}": true or false') == len(BOARD.block_names)
    schema = llm.output_schema()
    assert set(schema["required"]) == set(BOARD.block_names)
    for name in BOARD.block_names:
        props = schema["properties"][name]["properties"]
        assert set(props) == {ENABLED, *BOARD.blocks[name].param_names}


def test_prompt_has_no_dataset_or_gold_text():
    prompt = llm.system_prompt() + llm.user_message("")
    caps = [
        c.text
        for it in synthetic_items(60, seed=3)
        for c in template_captions(it, n=3, pedal_name_fraction=0.5, holdout="rat")
    ]
    assert caps
    low = prompt.lower()
    words = re.findall(r"[a-z]+", low)
    prompt_grams = {tuple(words[i : i + 4]) for i in range(len(words) - 3)}
    for t in caps:
        assert t.lower() not in low
        w = re.findall(r"[a-z]+", t.lower())
        shared = prompt_grams & {tuple(w[i : i + 4]) for i in range(len(w) - 3)}
        assert not shared, (t, shared)
    for _, text, _, _ in FIXTURE_GOLD:
        assert text.lower() not in low
    assert "example" not in low  # zero-shot: no few-shot examples


def test_prompt_version_pins_hash():
    # editing the prompt / schema needs a prompt_version bump (cache keys)
    from lstmabar.config import load_config
    from lstmabar.eval.fixtures import EVAL_CONFIG

    assert str(load_config(EVAL_CONFIG).eval.baselines.llm.prompt_version) == "z1"
    assert llm.prompt_hash() == "9abb4f272c8edcd2"


def test_request_params(cfg):
    p = llm.request_params("more grit", cfg.eval.baselines.llm)
    assert p["model"] == "claude-haiku-5-5"
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert "more grit" in p["messages"][0]["content"]
    assert "more grit" not in p["system"]


# ---------------------------------------------------------------------------- parsing


def test_parse_normalizes_clamps_and_gates():
    reply = _physical_reply(
        drive={ENABLED: True, "gain_db": 30.0, "pre_hpf_hz": 99999.0, "tone_db": "loud"},
        eq={ENABLED: True, "high_db": -50.0},
    )
    board, st = llm.parse_reply(json.dumps(reply))
    specs = {(b, s.name): s for b in BOARD.block_names for s in BOARD.blocks[b].param_specs}
    assert board["drive"][ENABLED] == 1.0 and board["eq"][ENABLED] == 1.0
    assert board["compressor"][ENABLED] == 0.0
    assert board["drive"]["gain_db"] == pytest.approx(0.5)
    assert board["drive"]["pre_hpf_hz"] == pytest.approx(1.0)  # clamped to max
    assert board["eq"]["high_db"] == pytest.approx(0.0)  # clamped to min
    assert board["drive"]["tone_db"] == pytest.approx(specs["drive", "tone_db"].default_normalized)
    s = specs["compressor", "ratio"]  # log taper default round-trips
    assert board["compressor"]["ratio"] == pytest.approx(s.default_normalized)
    assert float(s.denormalize(torch.tensor(board["compressor"]["ratio"]))) == pytest.approx(4.0)
    assert (st.parse_failures, st.knobs_clamped, st.knobs_defaulted) == (0, 2, 1)
    assert sorted((b, k) for b in board for k in board[b]) == sorted(BOARD_LAYOUT)


@pytest.mark.parametrize("text", [None, "", "not json", "[1, 2]", '{"foo": {}}', "null"])
def test_parse_failure_gives_default_board_gates_off(text):
    board, st = llm.parse_reply(text)
    assert board == default_board(0.0)
    assert st.parse_failures == 1


def test_parse_partial_reply_defaults_missing_blocks():
    board, st = llm.parse_reply(json.dumps({"drive": {ENABLED: True, "gain_db": 60}}))
    assert st.parse_failures == 0 and st.blocks_defaulted == 2
    assert board["drive"][ENABLED] == 1.0 and board["drive"]["gain_db"] == 1.0
    assert board["eq"] == default_board(0.0)["eq"]
    # a non-boolean gate counts as off
    board, _ = llm.parse_reply(json.dumps({"drive": {ENABLED: 1, "gain_db": 60}}))
    assert board["drive"][ENABLED] == 0.0


# ---------------------------------------------------------------------------- predict / cache


def test_registry_builds_method(cfg):
    m = build_method("llm_zero_shot", cfg)
    assert isinstance(m, llm.LlmZeroShot) and not m.uses_audio


def test_predict_without_key_raises_with_estimate(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    m = llm.LlmZeroShot(cfg, client_factory=lambda: pytest.fail("no client without a key"))
    with pytest.raises(llm.LlmCacheMiss, match=r"2 unique instructions.*\$"):
        m.predict([_q("a"), _q("b", 1), _q("a", 2)])


def test_predict_needs_allow_spend(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    m = llm.LlmZeroShot(cfg, client_factory=lambda: FakeClient())
    with pytest.raises(llm.LlmCacheMiss, match="allow_spend"):
        m.predict([_q("a")])
    m.estimate_only = True
    cfg.eval.baselines.llm.allow_spend = True
    with pytest.raises(llm.LlmCacheMiss, match="estimate mode"):
        m.predict([_q("a")])


def test_predict_submits_deduped_then_hits_cache(keyed, tmp_path):
    def reply(instr):
        if "bright" in instr:
            return json.dumps(_physical_reply(eq={ENABLED: True, "high_db": 6.0}))
        if "broken" in instr:
            return "{oops"
        return json.dumps(_physical_reply(drive={ENABLED: True}))

    client = FakeClient(reply)
    m = llm.LlmZeroShot(keyed, client_factory=lambda: client, sleep=lambda s: None)
    qs = [_q("bright", 0), _q("dirty", 1), _q("bright", 2), _q("broken", 3)]
    out = m.predict(qs)
    assert len(client.batches.created) == 1
    assert len(client.batches.created[0]) == 3  # deduped
    assert out[0] == out[2] and out[0]["eq"][ENABLED] == 1.0
    assert out[0]["eq"]["high_db"] == pytest.approx(0.75)
    assert out[1]["drive"][ENABLED] == 1.0
    assert out[3] == default_board(0.0)
    info = m.info()
    assert info["parse_failures"] == 1 and info["n_queries"] == 4
    assert info["ledger_spent_usd"] == pytest.approx(3 * (1000 * 0.05 + 120 * 0.25) / 1e6)
    # cache hit: a fresh instance makes no API call, also without a key
    cache_dir = tmp_path / "cache" / "llm" / "zero_shot"
    assert (cache_dir / "replies.jsonl").exists() and (cache_dir / "ledger.jsonl").exists()
    assert not (cache_dir / "ledger.lock").exists()
    raw = (cache_dir / "replies.jsonl").read_text("utf-8")
    assert "bright" not in raw and "test-key" not in raw  # instruction text / key not stored
    m2 = llm.LlmZeroShot(keyed, client_factory=lambda: pytest.fail("cache miss"))
    assert m2.predict(qs) == out
    est = llm.estimate(["bright", "dirty", "new one"], keyed)
    assert (est["cached"], est["requests"]) == (2, 1)


def test_cap_refuses_submit(keyed):
    keyed.eval.baselines.llm.max_usd = 0.0001  # one request's worst case is ~$0.00015
    client = FakeClient()
    m = llm.LlmZeroShot(keyed, client_factory=lambda: client, sleep=lambda s: None)
    with pytest.raises(BudgetExceeded, match="refusing to submit"):
        m.predict([_q("a")])
    assert client.batches.created == []
    est = m.estimate(["a"])
    assert not est["within_cap"] and est["worst_usd"] > est["max_usd"]


def test_cap_counts_spent_and_in_flight(keyed, tmp_path):
    ledger = SpendLedger(tmp_path / "cache" / "llm" / "zero_shot" / "ledger.jsonl")
    ledger.append("submitted", "old", estimate_usd=0.5, worst_usd=0.99995)  # in flight
    client = FakeClient()
    m = llm.LlmZeroShot(keyed, client_factory=lambda: client, sleep=lambda s: None)
    with pytest.raises(BudgetExceeded):
        m.predict([_q("a")])
    assert client.batches.created == []


def test_retryable_errors_are_resubmitted(keyed):
    client = FakeClient()
    calls = {"n": 0}
    orig = client.batches.results

    def flaky(bid):
        calls["n"] += 1
        for res in orig(bid):
            if calls["n"] == 1:  # first batch: every request errors (overloaded)
                err = SimpleNamespace(error=SimpleNamespace(type="overloaded_error"))
                yield SimpleNamespace(
                    custom_id=res.custom_id, result=SimpleNamespace(type="errored", error=err)
                )
            else:
                yield res

    client.batches.results = flaky
    m = llm.LlmZeroShot(keyed, client_factory=lambda: client, sleep=lambda s: None)
    out = m.predict([_q("a"), _q("b", 1)])
    assert len(client.batches.created) == 2
    assert client.batches.created[1][0]["custom_id"].startswith("a1-")
    assert m.info()["parse_failures"] == 0 and len(out) == 2


def test_resume_collects_batch_left_in_flight(keyed):
    client = FakeClient()
    m = llm.LlmZeroShot(keyed, client_factory=lambda: client, sleep=lambda s: None)
    b = llm.ZeroShotBatcher(m.lc, m.layout, client, m.cache, m.board, sleep=lambda s: None)
    b.submit(["a"])  # created, never collected (the process "died")
    m2 = llm.LlmZeroShot(keyed, client_factory=lambda: client, sleep=lambda s: None)
    m2.predict([_q("a")])
    assert len(client.batches.created) == 1  # resumed, not resubmitted
    assert b.ledger.pending() == []


def test_estimate_dedupes_and_prices(cfg):
    est = llm.estimate(["x", "y", "x", "a longer instruction"], cfg)
    assert est["unique_instructions"] == 3 and est["requests"] == 3
    assert est["input_source"] == "approx"
    lc = cfg.eval.baselines.llm
    assert est["max_output_tokens"] == 3 * int(lc.max_tokens)
    assert est["worst_usd"] >= est["usd"] > 0
    assert est["within_cap"]
    assert any("worst case" in line for line in llm.estimate_lines(est))
    est2 = llm.estimate(["x"], cfg, client=FakeClient())
    assert est2["input_source"] == "count_tokens" and est2["input_tokens"] == 1000
