"""P4 task 4.3: caption templates, paraphrase batching, spend guards. No network: the SDK
client is a fake injected through ``client_factory`` / ``Paraphraser(client=...)``."""

import argparse
import json
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lstmabar.data import captions as cap
from lstmabar.data.fixtures import fixture_config
from lstmabar.data.layout import DataLayout
from lstmabar.data.schema import Caption, RenderItem, read_jsonl, write_jsonl
from lstmabar.data.vocab import (
    PEDAL_ALIASES,
    caption_violations,
    forbidden_names,
    holdout_names,
    item_caption_violations,
    mentions,
)

HOLDOUT = "rat"


# --------------------------------------------------------------------------- fakes


def _message(text, usage=(1000, 900), stop="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
    )


def default_rewrite(text):
    return f"I really want {text.lower().rstrip('?')} for this one"


class FakeClient:
    """Mimics the parts of ``anthropic.Anthropic`` the caption code uses.

    ``outcome(custom_id, attempt_calls)`` -> "succeeded" | "errored" | "invalid" | "expired".
    ``rewrite(template)`` -> paraphrase text (or None to omit the entry).
    """

    def __init__(
        self,
        rewrite=default_rewrite,
        outcome=None,
        shuffle=True,
        raw=None,
        create_error=None,
        count_error=None,
    ):
        self.rewrite = rewrite
        self.create_error = create_error  # exception raised by batches.create
        self.count_error = count_error  # exception raised by count_tokens
        self._opts = None
        self.outcome = outcome or (lambda cid, n: "succeeded")
        self.shuffle = shuffle
        self.raw = raw  # override the reply text entirely
        self.created = []  # list of request lists
        self.count_calls = 0
        self.messages = SimpleNamespace(
            count_tokens=self._count,
            batches=SimpleNamespace(
                create=self._create, retrieve=self._retrieve, results=self._results
            ),
        )

    def with_options(self, **kw):
        self._opts = kw
        return self

    def _count(self, **kw):
        assert set(kw) <= {"model", "system", "messages", "thinking", "output_config"}
        self.count_calls += 1
        if self.count_error is not None:
            raise self.count_error
        return SimpleNamespace(input_tokens=1000)

    def _create(self, requests):
        assert self._opts == {"max_retries": 0}, "batches.create must not use SDK retries"
        self._opts = None
        if self.create_error is not None:
            raise self.create_error
        self.created.append(list(requests))
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}", processing_status="in_progress")

    def _retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def _results(self, batch_id):
        reqs = self.created[int(batch_id.split("_")[1]) - 1]
        out = []
        for r in reqs:
            params = r["params"]
            assert params["thinking"] == {"type": "disabled"}
            assert params["output_config"]["effort"] == "low"
            assert params["output_config"]["format"]["type"] == "json_schema"
            src = json.loads(params["messages"][0]["content"])["captions"]
            kind = self.outcome(r["custom_id"], len(self.created))
            if kind == "succeeded":
                if self.raw is not None:
                    text = self.raw
                else:
                    entries = [
                        {"id": e["id"], "text": t}
                        for e in src
                        if (t := self.rewrite(e["text"])) is not None
                    ]
                    text = json.dumps({"captions": entries[::-1]})  # entries in any order
                res = SimpleNamespace(type="succeeded", message=_message(text))
            elif kind in ("errored", "invalid") or kind.endswith("_error"):
                etype = {"invalid": "invalid_request_error", "errored": "api_error"}.get(kind, kind)
                res = SimpleNamespace(
                    type="errored",
                    error=SimpleNamespace(type="error", error=SimpleNamespace(type=etype)),
                )
            else:
                res = SimpleNamespace(type=kind)
            out.append(SimpleNamespace(custom_id=r["custom_id"], result=res))
        if self.shuffle:
            random.Random(0).shuffle(out)
        return iter(out)


@pytest.fixture
def cfg(tmp_path):
    return fixture_config(tmp_path, ["captions.poll_seconds=0"])


def _para(cfg, client):
    return cap.Paraphraser(cfg.captions, DataLayout.from_config(cfg), client, 0.0, lambda s: None)


def _item(**kw) -> RenderItem:
    base = dict(
        item_id="c-r0",
        clip_id="c",
        split="train",
        seed=1,
        scenario="pedal",
        engine="greybox",
        pedal_id="ts808",
        pedal_knobs={"drive": 0.5},
        board_params={
            "compressor": {"threshold_db": 0.5, "ratio": 0.5, "makeup_db": 0.0, "enabled": 0.0},
            "drive": {"gain_db": 0.5, "tone_db": 0.5, "softness": 0.5, "enabled": 1.0},
            "eq": {"low_db": 0.5, "mid_db": 0.5, "high_db": 0.5, "enabled": 1.0},
        },
        board_params_derived=None,
        label_source="derived",
        clamped=False,
        clamp_notes=[],
        input_peak_dbfs=-12.0,
        wet_gain_db=0.0,
        wet_path="v1/wet/shard-00000/c-r0.flac",
        voiced_fraction=0.9,
    )
    base.update(kw)
    return RenderItem(**base)


def _board(item, block, **knobs):
    b = {k: dict(v) for k, v in item.board_params.items()}
    b[block].update(knobs)
    return b


# --------------------------------------------------------------------------- KB / DSP mirrors


def test_kb_and_knob_mirrors_match_sources():
    from lstmabar.dsp.pedalboard import default_pedalboard
    from lstmabar.physics.kb import load_kb

    kb = load_kb()
    assert {k: p.family for k, p in kb.items()} == cap.FAMILY
    assert {k: tuple(p.descriptors) for k, p in kb.items()} == cap.KB_DESCRIPTORS
    assert set(kb) == set(PEDAL_ALIASES)
    board = default_pedalboard()
    specs = {name: {s.name: s for s in blk.param_specs} for name, blk in board.blocks.items()}
    for block, knobs in cap.KNOB_RANGES.items():
        for knob, (lo, hi, taper) in knobs.items():
            s = specs[block][knob]
            assert (s.min, s.max, s.taper) == (lo, hi, taper), (block, knob)


# --------------------------------------------------------------------------- templates


def _rules(c: Caption, item: RenderItem):
    allowed = list(PEDAL_ALIASES[item.pedal_id]) if c.names_pedal else []
    forbidden = holdout_names(HOLDOUT) if item.split != "test_unseen_pedal" else []
    return allowed, forbidden


def test_templates_never_contain_numbers_or_leaks():
    items = cap.synthetic_items(4000, HOLDOUT, seed=1)
    # adversarial extremes: every descriptor missing, unvoiced, knobs at the ends
    rng = random.Random(0)
    for i in range(500):
        it = items[i]
        board = {b: {k: rng.choice([0.0, 1.0]) for k in v} for b, v in it.board_params.items()}
        items.append(
            replace(
                it,
                item_id=f"edge-{i}",
                board_params=board,
                voiced_fraction=rng.choice([None, 0.0, 1.0]),
                descriptors_wet={k: None for k in it.descriptors_wet},
                descriptors_dry={},
            )
        )
    n = 0
    for it in items:
        for c in cap.template_captions(it, 2, 0.25, HOLDOUT):
            allowed, forbidden = _rules(c, it)
            assert caption_violations(c.text, allowed, forbidden) == [], c.text
            v = item_caption_violations(c.text, it.pedal_id, c.names_pedal, HOLDOUT, it.split)
            assert v == [], c.text
            # unnamed captions never mention any pedal name
            if not c.names_pedal:
                assert not mentions(c.text, [n for ns in PEDAL_ALIASES.values() for n in ns])
            n += 1
    assert n == 2 * len(items)


def test_template_variety():
    items = cap.synthetic_items(2000, HOLDOUT, seed=2)
    caps = [cap.template_captions(it, 1, 0.25, HOLDOUT)[0] for it in items]
    texts = {c.text for c in caps}
    frames = {c.template_id for c in caps}
    assert len(texts) > 0.9 * len(caps)
    assert len(frames) >= 25


def test_pedal_name_fraction():
    items = [
        it
        for it in cap.synthetic_items(6000, HOLDOUT, seed=3)
        if it.scenario == "pedal" and it.pedal_id != HOLDOUT
    ]
    named = [cap.template_captions(it, 1, 0.25, HOLDOUT)[0].names_pedal for it in items]
    assert abs(sum(named) / len(named) - 0.25) < 0.03
    free = [it for it in cap.synthetic_items(500, HOLDOUT, seed=3) if it.scenario != "pedal"]
    assert not any(cap.template_captions(it, 1, 1.0, HOLDOUT)[0].names_pedal for it in free)


def test_holdout_never_named_outside_unseen_split():
    names = holdout_names(HOLDOUT)
    for split in ("train", "val", "test"):
        for s in range(300):
            it = _item(item_id=f"h-{split}-{s}", seed=s, pedal_id=HOLDOUT, split=split)
            for c in cap.template_captions(it, 2, 1.0, HOLDOUT):
                assert not c.names_pedal
                assert not mentions(c.text, names), c.text
    # in test_unseen_pedal it may be named (only via aliases)
    named = [
        cap.template_captions(
            _item(item_id=f"u-{s}", seed=s, pedal_id=HOLDOUT, split="test_unseen_pedal"),
            1,
            1.0,
            HOLDOUT,
        )[0]
        for s in range(50)
    ]
    assert all(c.names_pedal and mentions(c.text, names) for c in named)


def test_templates_deterministic():
    items = cap.synthetic_items(200, HOLDOUT, seed=4)
    a = [cap.template_captions(it, 2, 0.25, HOLDOUT) for it in items]
    b = [cap.template_captions(it, 2, 0.25, HOLDOUT) for it in items]
    assert a == b
    other = [cap.template_captions(replace(it, seed=it.seed + 1), 1)[0].text for it in items]
    assert sum(x != y[0].text for x, y in zip(other, a, strict=True)) > 150
    assert [c.caption_id for c in a[0]] == [f"{items[0].item_id}-c0", f"{items[0].item_id}-c1"]


def test_bins_from_knobs_and_descriptors():
    it = _item(
        board_params=_board(_item(), "drive", gain_db=0.9, tone_db=0.0)
        | {"eq": {"low_db": 0.5, "mid_db": 0.0, "high_db": 0.5, "enabled": 1.0}},
        descriptors_wet={"centroid_hz": 2000.0, "harmonic_energy_fraction": 0.4, "hnr_db": 2.0},
        descriptors_dry={"harmonic_energy_fraction": 0.05, "hnr_db": 20.0},
    )
    a = cap.attributes(it)
    assert (a.family, a.drive, a.treble, a.mids, a.character, a.noise) == (
        "overdrive",
        "heavy",
        "dark",
        "scooped",
        "fuzzy",
        "noisy",
    )
    assert a.treble_strong and a.mids_strong
    # knob says dark, centroid says bright -> no treble word
    it2 = replace(it, descriptors_wet={"centroid_hz": 4000.0})
    assert cap.attributes(it2).treble is None
    # compressor: threshold -60 dB, ratio 20 at a -12 dBFS peak -> heavy
    comp = _item(
        scenario="comp",
        pedal_id=None,
        board_params={
            "compressor": {"threshold_db": 0.0, "ratio": 1.0, "makeup_db": 0.0, "enabled": 1.0},
            "drive": {"gain_db": 0.5, "enabled": 0.0},
            "eq": {"enabled": 0.0},
        },
    )
    assert cap.attributes(comp).comp == "heavy"
    assert cap.attributes(comp).family is None


def test_none_descriptors_and_low_voicing_are_not_captioned():
    w = {"centroid_hz": None, "harmonic_energy_fraction": 0.4, "odd_even_db": 20.0, "hnr_db": 0.0}
    for vf in (None, 0.1):
        a = cap.attributes(_item(voiced_fraction=vf, descriptors_wet=w))
        assert a.character is None and a.noise is None and a.treble is None
    a = cap.attributes(
        _item(voiced_fraction=0.9, descriptors_wet={k: None for k in w}, descriptors_dry={})
    )
    assert a.character is None and a.noise is None and a.treble is None
    # spectral descriptor still used when unvoiced
    a = cap.attributes(_item(voiced_fraction=0.0, descriptors_wet={"centroid_hz": 500.0}))
    assert a.treble == "dark"


def test_bypass_and_clean_phrasings():
    bypass = _item(scenario="bypass", pedal_id=None, board_params={"drive": {"enabled": 0.0}})
    t = cap.template_captions(bypass)[0]
    assert t.template_id.startswith("bypass")
    eq = _item(
        scenario="eq",
        pedal_id=None,
        board_params={
            "drive": {"gain_db": 1.0, "enabled": 0.0},
            "eq": {"low_db": 0.5, "mid_db": 0.5, "high_db": 1.0, "enabled": 1.0},
        },
    )
    texts = [cap.template_captions(replace(eq, item_id=f"e{i}"))[0].text for i in range(40)]
    assert all("distort" not in x and "overdriv" not in x and "fuzz" not in x for x in texts)
    assert any("clean" in x for x in texts)


# --------------------------------------------------------------------------- estimate / ledger


def test_estimate_math(cfg):
    layout = DataLayout.from_config(cfg)
    texts = [f"template {chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(60)]
    client = FakeClient()
    est = cap.estimate(texts, cfg.captions, layout, client)
    assert est.n_requests == 3 and client.count_calls == 1
    assert est.input_tokens == 3000 and est.input_source == "count_tokens"
    assert est.output_tokens == 60 * 45 and est.output_source == "config"
    assert est.usd == pytest.approx((3000 * 0.05 + 2700 * 0.25) / 1e6)
    approx = cap.estimate(texts, cfg.captions, layout, None)
    assert approx.input_source == "approx" and approx.input_tokens > 0


def test_ledger_cap_refuses_including_in_flight(cfg):
    layout = DataLayout.from_config(cfg)
    ledger = cap.SpendLedger(layout.ledger)
    ledger.append("submitted", "old-a", estimate_usd=1.0)
    ledger.append("ended", "old-a", cost_usd=1.0)
    ledger.append("submitted", "old-b", estimate_usd=1.5)  # in flight, no batch id
    assert ledger.spent() == pytest.approx(1.0)
    assert ledger.in_flight() == pytest.approx(1.5)
    client = FakeClient()
    para = _para(cfg, client)
    with pytest.raises(cap.BudgetExceeded):
        para.submit(["a bright clean tone"])  # any positive estimate crosses 2.50
    assert client.created == []
    # without the in-flight entry the same submit fits
    layout.ledger.unlink()
    ledger.append("submitted", "old-a", estimate_usd=1.0)
    ledger.append("ended", "old-a", cost_usd=1.0)
    para.submit(["a bright clean tone"])
    assert len(client.created) == 1


def test_ledger_records_actual_cost_and_cache_avoids_resubmission(cfg):
    client = FakeClient()
    para = _para(cfg, client)
    texts = [f"make it warm number {chr(97 + i)}" for i in range(30)]
    para.run(texts)
    assert len(client.created) == 1 and len(client.created[0]) == 2  # 25 per request
    ledger = cap.SpendLedger(DataLayout.from_config(cfg).ledger)
    assert ledger.in_flight() == 0
    assert ledger.spent() == pytest.approx((2000 * 0.05 + 1800 * 0.25) / 1e6)
    for t in texts:
        assert para.cache.get(t) == default_rewrite(t)
    # a fresh paraphraser (new process) reads the cache: nothing resubmitted
    client2 = FakeClient()
    _para(cfg, client2).run(texts)
    assert client2.created == []


def test_results_mapped_by_custom_id_in_any_order(cfg):
    client = FakeClient(shuffle=True)
    para = _para(cfg, client)
    texts = [f"give me tone {chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(80)]
    para.run(texts)
    assert len(client.created[0]) == 4
    for t in texts:
        assert para.cache.get(t) == default_rewrite(t)


def test_errored_requests_resubmitted_at_most_max_resubmits(cfg):
    # first batch: everything errors (server side); second: expires; then give up
    client = FakeClient(outcome=lambda cid, n: "errored" if n == 1 else "expired")
    para = _para(cfg, client)
    para.run(["a gritty overdrive"])
    assert len(client.created) == 1 + int(cfg.captions.max_resubmits)
    assert "a gritty overdrive" not in para.cache  # failed requests are not cached
    # non-retryable errors are not resubmitted
    client = FakeClient(outcome=lambda cid, n: "invalid")
    _para(cfg, client).run(["a fuzzy tone"])
    assert len(client.created) == 1


def test_pending_batches_resumed_not_resubmitted(cfg):
    client = FakeClient()
    para = _para(cfg, client)
    ref = para.submit(["a crunchy distortion"])  # created but never collected (crash)
    assert cap.SpendLedger(para.ledger.path).in_flight() > 0
    para2 = _para(cfg, client)
    para2.run(["a crunchy distortion"])
    assert len(client.created) == 1  # collected the pending batch instead of resubmitting
    assert para2.cache.get("a crunchy distortion") == default_rewrite("a crunchy distortion")
    assert ref in para2.ledger.state() and para2.ledger.in_flight() == 0


# --------------------------------------------------------------------------- post-filter


@pytest.mark.parametrize(
    "bad",
    [
        "crank the gain to 7",
        "about ten dB more treble",
        "make it sound like a Tube Screamer",  # adds a name the template did not have
        "like a Klon",
        "a RAT tone",  # holdout
    ],
)
def test_violating_paraphrase_falls_back_to_template(cfg, bad):
    it = _item(item_id="v-r0", seed=7)
    c0 = _nonamed(cfg)
    tpl = cap.template_captions(it, 1, 0.0, HOLDOUT)[0]
    client = FakeClient(rewrite=lambda t: bad)
    para = _para(c0, client)
    para.run([tpl.text])
    stats = {}
    caps = cap.build_captions([it], c0, para.cache, stats)
    assert [c.kind for c in caps] == ["template"]
    assert stats["rejected"] == 1


def _nonamed(cfg):
    from omegaconf import OmegaConf

    return OmegaConf.merge(cfg, {"captions": {"pedal_name_fraction": 0.0}})


def test_malformed_reply_falls_back_and_is_not_repaid(cfg):
    it = _item(item_id="m-r0", seed=8)
    c0 = _nonamed(cfg)
    tpl = cap.template_captions(it, 1, 0.0, HOLDOUT)[0]
    client = FakeClient(raw="not json at all")
    para = _para(c0, client)
    para.run([tpl.text])
    stats = {}
    caps = cap.build_captions([it], c0, para.cache, stats)
    assert [c.kind for c in caps] == ["template"] and stats["malformed"] == 1
    client2 = FakeClient()
    _para(c0, client2).run([tpl.text])
    assert client2.created == []


def test_paraphrase_keeps_allowed_name(cfg):
    it = _item(item_id="n-r0", seed=9)
    tpl = cap.template_captions(it, 1, 1.0, HOLDOUT)[0]
    assert tpl.names_pedal
    client = FakeClient()
    from omegaconf import OmegaConf

    c1 = OmegaConf.merge(cfg, {"captions": {"pedal_name_fraction": 1.0}})
    para = _para(c1, client)
    para.run([tpl.text])
    caps = cap.build_captions([it], c1, para.cache)
    assert [c.kind for c in caps] == ["template", "paraphrase"]
    p = caps[1]
    assert p.caption_id == "n-r0-c1" and p.names_pedal and p.model == "claude-haiku-5-5"
    assert p.prompt_version == str(c1.captions.prompt_version)


# --------------------------------------------------------------------------- run()


def _args(**kw):
    base = dict(estimate=False, pilot=False, limit=None, fixtures=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _write_shards(cfg, n_shards=2, per=30):
    layout = DataLayout.from_config(cfg)
    items = cap.synthetic_items(n_shards * per, HOLDOUT, seed=11)
    for s in range(n_shards):
        d = layout.shard_dir(s)
        write_jsonl(d / "items.jsonl", items[s * per : (s + 1) * per])
        (d / "DONE").write_text("", encoding="utf-8")
    return layout, items


def test_run_writes_captions_and_is_idempotent(cfg):
    layout, items = _write_shards(cfg)
    client = FakeClient()
    assert cap.run(cfg, _args(), client_factory=lambda: client) == 0
    files = [layout.shard_dir(s) / "captions.jsonl" for s in range(2)]
    first = [f.read_text(encoding="utf-8") for f in files]
    caps = [c for f in files for c in read_jsonl(f, Caption)]
    assert {c.item_id for c in caps} == {it.item_id for it in items}
    assert sum(c.kind == "template" for c in caps) == len(items)
    assert sum(c.kind == "paraphrase" for c in caps) == len(items)
    n_batches = len(client.created)
    assert cap.run(cfg, _args(), client_factory=lambda: client) == 0
    assert len(client.created) == n_batches  # cache: no new spend
    assert [f.read_text(encoding="utf-8") for f in files] == first


def test_run_estimate_spends_nothing(cfg, capsys):
    _write_shards(cfg)
    client = FakeClient()
    assert cap.run(cfg, _args(estimate=True), client_factory=lambda: client) == 0
    assert client.created == [] and client.count_calls == 1
    out = capsys.readouterr().out
    assert "estimated cost" in out and "cap $2.50" in out
    assert not cap.SpendLedger(DataLayout.from_config(cfg).ledger).events()


def test_run_estimate_without_key_uses_approx(cfg, capsys):
    def no_key():
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    assert cap.run(cfg, _args(estimate=True, fixtures=True, limit=500), client_factory=no_key) == 0
    assert "(approx)" in capsys.readouterr().out


def test_run_refuses_over_cap(cfg):
    _write_shards(cfg)
    cap.SpendLedger(DataLayout.from_config(cfg).ledger).append("submitted", "x", estimate_usd=2.5)
    client = FakeClient()
    assert cap.run(cfg, _args(), client_factory=lambda: client) == 2
    assert client.created == []


def test_pilot_records_measured_output_tokens(cfg):
    layout, _ = _write_shards(cfg, 2, 80)
    client = FakeClient()
    assert cap.run(cfg, _args(pilot=True), client_factory=lambda: client) == 0
    n_sent = sum(
        len(json.loads(r["params"]["messages"][0]["content"])["captions"])
        for r in client.created[0]
    )
    assert n_sent <= int(cfg.captions.pilot_items)
    per, src = cap.output_per_caption(layout, cfg.captions)
    assert src == "pilot"
    assert per == pytest.approx(900 * len(client.created[0]) / n_sent)
    assert not (layout.shard_dir(0) / "captions.jsonl").exists()


def test_templates_only_when_paraphrases_disabled(cfg):
    from omegaconf import OmegaConf

    layout, items = _write_shards(cfg, 1, 10)
    c0 = OmegaConf.merge(cfg, {"captions": {"paraphrases_per_item": 0}})

    def boom():
        raise AssertionError("no client needed")

    assert cap.run(c0, _args(), client_factory=boom) == 0
    caps = list(read_jsonl(layout.shard_dir(0) / "captions.jsonl", Caption))
    assert len(caps) == len(items) and all(c.kind == "template" for c in caps)


def test_request_ids_carry_no_digits_and_schema_is_strict(cfg):
    params, ids = cap.request_params(["x"] * 30, cfg.captions)
    assert all(k.isalpha() for k in ids) and len(set(ids)) == 30
    assert params["model"] == "claude-haiku-5-5"
    schema = params["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert "never" in cap.SYSTEM_PROMPT.lower()


# --------------------------------------------------------------------------- review fixes


class _Status(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_ambiguous_create_failure_stays_in_flight(cfg):
    client = FakeClient(create_error=TimeoutError("lost response"))
    para = _para(cfg, client)
    with pytest.raises(cap.SubmitUncertain):
        para.submit(["a warm overdrive"])
    ledger = cap.SpendLedger(para.ledger.path)
    assert ledger.in_flight() > 0 and ledger.spent() == 0
    for err in (_Status(500), _Status(408)):
        client.create_error = err
        with pytest.raises(cap.SubmitUncertain):
            para.submit(["a warm overdrive"])
    assert len(ledger.pending()) == 3


def test_definite_4xx_create_failure_closes_entry_at_zero(cfg):
    client = FakeClient(create_error=_Status(400))
    para = _para(cfg, client)
    with pytest.raises(_Status):
        para.submit(["a warm overdrive"])
    ledger = cap.SpendLedger(para.ledger.path)
    assert ledger.in_flight() == 0 and ledger.spent() == 0
    assert [r["status"] for r in ledger.state().values()] == ["rejected"]


def test_run_returns_nonzero_on_ambiguous_submit(cfg):
    _write_shards(cfg, 1, 10)
    client = FakeClient(create_error=ConnectionError("reset"))
    assert cap.run(cfg, _args(), client_factory=lambda: client) == 3
    assert cap.SpendLedger(DataLayout.from_config(cfg).ledger).in_flight() > 0


def test_cap_uses_worst_case_not_estimate(cfg):
    layout = DataLayout.from_config(cfg)
    texts = [f"tone {chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(50)]
    client = FakeClient()
    est = cap.estimate(texts, cfg.captions, layout, client)
    assert est.max_output_tokens == 2 * cap.max_tokens_for(25)
    assert est.worst_usd == pytest.approx((2000 * 0.05 + est.max_output_tokens * 0.25) / 1e6)
    assert est.worst_usd > est.usd
    # spent leaves room for the estimate but not for the worst case
    room = (est.usd + est.worst_usd) / 2
    ledger = cap.SpendLedger(layout.ledger)
    ledger.append("submitted", "old", estimate_usd=0.0)
    ledger.append("ended", "old", cost_usd=float(cfg.captions.max_usd) - room)
    with pytest.raises(cap.BudgetExceeded):
        _para(cfg, client).submit(texts)
    assert client.created == []
    # in-flight entries count at their worst case
    layout.ledger.unlink()
    ledger.append("submitted", "x", estimate_usd=0.1, worst_usd=float(cfg.captions.max_usd))
    assert ledger.in_flight() == pytest.approx(0.1)
    with pytest.raises(cap.BudgetExceeded):
        _para(cfg, client).submit(texts[:1])


def test_cache_written_before_ended(cfg):
    client = FakeClient()
    para = _para(cfg, client)
    order = []
    orig_put, orig_append = para.cache.put_many, para.ledger.append

    def put(*a, **k):
        order.append("cache")
        return orig_put(*a, **k)

    def append(ev, *a, **k):
        order.append(ev)
        return orig_append(ev, *a, **k)

    para.cache.put_many = put
    para.ledger.append = append
    para.run(["a dark fuzz"])
    assert order.index("cache") < order.index("ended")


def test_non_retryable_errors(cfg):
    for etype in ("authentication_error", "billing_error", "permission_error"):
        client = FakeClient(outcome=lambda cid, n, e=etype: e)
        para = _para(cfg, client)
        para.run([f"a fuzzy tone {etype}"])
        assert len(client.created) == 1  # not resubmitted
        assert f"a fuzzy tone {etype}" not in para.cache  # logged, not cached
    client = FakeClient(outcome=lambda cid, n: "invalid")
    para = _para(cfg, client)
    para.run(["an invalid one"])
    assert "an invalid one" in para.cache and para.cache.get("an invalid one") is None
    client2 = FakeClient()
    _para(cfg, client2).run(["an invalid one"])
    assert client2.created == []  # cached as failed: never resubmitted


def test_resume_keeps_resubmit_count(cfg):
    max_re = int(cfg.captions.max_resubmits)
    # an earlier run submitted its last allowed attempt and crashed: errors are final
    client = FakeClient(outcome=lambda cid, n: "errored")
    _para(cfg, client).submit(["a gritty tone"], attempt=max_re)
    _para(cfg, client).run(["a gritty tone"])
    assert len(client.created) == 1  # collected, not resubmitted again
    # an attempt-0 batch that errors gets exactly max_resubmits more tries after resuming
    client = FakeClient(outcome=lambda cid, n: "errored")
    _para(cfg, client).submit(["a crunchy tone"], attempt=0)
    _para(cfg, client).run(["a crunchy tone"])
    assert len(client.created) == 1 + max_re
    assert client.created[-1][0]["custom_id"].startswith(f"a{max_re}-")


def test_ledger_lock_refuses_concurrent_run(cfg):
    layout = DataLayout.from_config(cfg)
    client = FakeClient()
    with cap.LedgerLock(layout.ledger):
        with pytest.raises(cap.LedgerLocked):
            _para(cfg, client).run(["a clean tone"])
        _write_shards(cfg, 1, 5)
        assert cap.run(cfg, _args(), client_factory=lambda: client) == 3
    assert client.created == []
    _para(cfg, client).run(["a clean tone"])  # lock released
    assert len(client.created) == 1


def test_estimate_falls_back_when_count_tokens_fails(cfg):
    client = FakeClient(count_error=ConnectionError("no network"))
    est = cap.estimate(["a bright tone"], cfg.captions, DataLayout.from_config(cfg), client)
    assert est.input_source == "approx" and est.input_tokens > 0


def test_prompt_hash_pinned_to_version(cfg):
    # editing SYSTEM_PROMPT or OUTPUT_SCHEMA must come with a captions.prompt_version bump
    # (and a new pin here); the hash is also part of every cache key
    pins = {"p1": "d9294614b618db3a"}
    assert pins[str(cfg.captions.prompt_version)] == cap.PROMPT_HASH
    assert cap.cache_key("t", "m", "p1") != cap.cache_key("t", "m", "p2")


def test_kb_character_dropped_when_measured():
    it = _item(
        pedal_id="ts808",
        descriptors_wet={"harmonic_energy_fraction": 0.12, "odd_even_db": 12.0},
        descriptors_dry={"harmonic_energy_fraction": 0.0},
    )
    a = cap.attributes(it)
    assert a.character == "gritty"
    assert "smooth" not in a.kb


def test_modifiers_not_doubled():
    for it in cap.synthetic_items(3000, HOLDOUT, seed=12):
        text = cap.template_captions(it, 1, 0.25, HOLDOUT)[0].text
        assert " and with " not in text, text
        assert "make my guitar into" not in text


def test_vocab_forbidden_names_shared_rule():
    f = forbidden_names("ds1", False, HOLDOUT, "train")
    assert "DS-1" in f and "RAT" in f and "Blues Driver" in f
    f = forbidden_names("ds1", True, HOLDOUT, "train")
    assert "DS-1" not in f and "Tube Screamer" in f
    assert "RAT" not in forbidden_names(HOLDOUT, True, HOLDOUT, "test_unseen_pedal")
    assert "RAT" in forbidden_names(HOLDOUT, True, HOLDOUT, "test")
    assert item_caption_violations("a Boss DS-1 tone", "ds1", True, HOLDOUT, "train") == []
    assert item_caption_violations("a Boss tone", "ds1", False, HOLDOUT, "train")


def test_cli_accepts_overrides_after_flags(monkeypatch):
    from lstmabar import cli
    from lstmabar.data import cli as data_cli

    seen = {}

    def fake_cmd(args):
        seen["args"] = args
        return 0

    real_build = cli.build_parser

    def build():
        parser = real_build()
        data = parser._subparsers._group_actions[0].choices["data"]
        data.set_defaults(func=fake_cmd)
        return parser

    monkeypatch.setattr(cli, "build_parser", build)
    assert data_cli.cmd_data is not fake_cmd
    assert cli.main(["data", "captions", "--estimate", "a.b=1", "--limit", "3", "c=2"]) == 0
    a = seen["args"]
    assert a.estimate and a.limit == 3 and a.overrides == ["a.b=1", "c=2"]
    with pytest.raises(SystemExit):
        cli.main(["data", "captions", "--bogus"])
