"""M3 text -> knobs logic (P5 5.6, D7): gradio-free parts of the demo."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lstmabar.audio import synth_riff
from lstmabar.data.levels import peak_dbfs
from lstmabar.demo.render import (
    iter_specs,
    normalized_to_controls,
    physical_from_slider,
    prepare_inference_input,
    slider_config,
)
from lstmabar.demo.text import (
    DEMO_METHODS,
    DemoQuery,
    TextKnobSetter,
    TextMethodError,
    controls_match,
    demo_query,
    info_summary,
    scrub_paths,
)
from lstmabar.dsp.pedalboard import ENABLED, default_pedalboard
from lstmabar.eval.baselines.base import Method, complete_board, default_board
from tests.demo.fakes import FakeBoard

SR = 44100
CFG = OmegaConf.create({"seed": 0, "audio": {"inference_peak_dbfs": -15.0}})


class FakeMethod(Method):
    """Records calls; predicts a fixed board (on the real board's names)."""

    name = "fake"

    def __init__(self, board=None, uses_audio=False):
        super().__init__(None)
        self.board = board or {"drive": {"gain_db": 0.75, ENABLED: 1.0}, "eq": {ENABLED: 0.0}}
        self.uses_audio = uses_audio
        self.fits, self.queries, self.hparams = 0, [], []

    def fit(self, train):
        self.fits += 1

    def set_hparams(self, hparams):
        self.hparams.append(hparams)

    def predict(self, queries):
        self.queries.extend(queries)
        return [complete_board(self.board, enabled=1.0) for _ in queries]

    def info(self):
        return {"name": self.name, "k": 3, "cache": r"C:\Users\me\data\cache", "scores": [1, 2]}


def _setter(method, cfg=CFG, name="mean", **kw):
    built = []

    def build(n, c, seed=0):
        built.append(n)
        return method

    s = TextKnobSetter(cfg, name, build=build, load_train=lambda c: ["item"], **kw)
    return s, built


# --- normalized board -> controls ---------------------------------------------------------------


def test_normalized_to_controls_round_trips_on_the_real_board():
    board = default_pedalboard(SR)
    rng = np.random.default_rng(0)
    fb = default_board(1.0)
    for knobs in fb.values():
        for k in knobs:
            knobs[k] = float(rng.uniform())
    fb["eq"][ENABLED] = 0.2
    enabled, sliders = normalized_to_controls(board, fb)
    assert enabled == [fb[b][ENABLED] >= 0.5 for b in board.block_names]
    for (name, spec), s in zip(iter_specs(board), sliders, strict=True):
        phys = physical_from_slider(spec, s)
        u = float(spec.normalize(torch.tensor(phys, dtype=torch.float64)))
        assert u == pytest.approx(fb[name][spec.name], abs=1e-6), (name, spec.name)
        cfg = slider_config(spec)
        assert cfg["minimum"] <= s <= cfg["maximum"]


def test_normalized_to_controls_keeps_missing_knobs_and_gates():
    board = FakeBoard()  # drive.gain_db, eq.cutoff_hz, eq.mix, level.gain_db
    enabled, sliders = normalized_to_controls(
        board, {"drive": {"gain_db": 0.75, ENABLED: 1.0}, "eq": {ENABLED: 0.0, "mix": 2.0}}
    )
    assert enabled == [True, False, None]
    assert sliders == [pytest.approx(12.0), None, 1.0, None]  # out-of-range mix clamped


# --- inference level (R3) -------------------------------------------------------------------------


def test_prepare_inference_input_normalizes_riff_and_upload():
    riff = prepare_inference_input(None, "power_chords", SR, 15.0, -15.0, riff_seconds=0.5)
    assert riff.dtype == np.float32 and peak_dbfs(riff) == pytest.approx(-15.0, abs=1e-4)
    loud = (np.random.default_rng(0).uniform(-1, 1, SR) * 32767).astype(np.int16)
    clip = prepare_inference_input((SR, loud), "power_chords", SR, 15.0, -9.0)
    assert peak_dbfs(clip) == pytest.approx(-9.0, abs=1e-4)
    silent = prepare_inference_input((SR, np.zeros(100, np.int16)), "x", SR, 15.0)
    assert not np.any(silent)
    with pytest.raises(ValueError, match="empty"):
        prepare_inference_input((SR, np.zeros(0, np.int16)), "x", SR, 15.0)


# --- the setter -----------------------------------------------------------------------------------


def test_setter_builds_and_fits_once_and_predicts():
    m = FakeMethod()
    s, built = _setter(m)
    p1 = s.predict("more crunch")
    p2 = s.predict("darker please")
    assert built == ["mean"] and m.fits == 1
    assert p1.method == "mean" and p1.board["drive"]["gain_db"] == 0.75
    assert p1.board["eq"][ENABLED] == 0.0 and set(p1.board) == set(p2.board)
    assert [q.text for q in m.queries] == ["more crunch", "darker please"]
    assert all(isinstance(q, DemoQuery) and q.split == "demo" for q in m.queries)
    md = p1.summary_md()
    assert "`mean`" in md and "k: 3" in md and "Users" not in md and "scores" not in md


def test_setter_applies_hparams_and_passes_the_clip_to_audio_methods():
    cfg = OmegaConf.merge(CFG, {"demo": {"text_hparams": {"k": 1}}})
    m = FakeMethod(uses_audio=True)
    s, _ = _setter(m, cfg=cfg)
    clip = synth_riff("single_notes", seconds=0.2, sample_rate=SR)
    calls = []
    s.predict("brighter", lambda: calls.append(1) or clip, sample_rate=SR)
    assert m.hparams == [{"k": 1}] and calls == [1]
    q = m.queries[0]
    assert q.input_peak_dbfs == -15.0 and q.sample_rate == SR
    np.testing.assert_allclose(q.load_dry(), clip)


def test_text_only_methods_do_not_decode_the_clip():
    m = FakeMethod()
    s, _ = _setter(m)
    s.predict("brighter", lambda: pytest.fail("decoded the clip for a text-only method"))


def test_default_method_comes_from_config():
    assert TextKnobSetter(CFG).method_name == "mean"
    cfg = OmegaConf.merge(CFG, {"demo": {"text_method": "knn_bge"}})
    assert TextKnobSetter(cfg).method_name == "knn_bge"
    assert {"mean", "noop", "knn_clap", "knn_bge", "p6_model"} == set(DEMO_METHODS)


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_instruction(text):
    s, built = _setter(FakeMethod())
    with pytest.raises(TextMethodError, match="Type an instruction"):
        s.predict(text)
    assert built == []  # nothing built for an empty box


def test_error_messages_have_no_paths():
    def boom(c):
        raise FileNotFoundError(r"no rendered items under C:\Users\me\secret\data\wet")

    s = TextKnobSetter(CFG, "mean", build=lambda n, c, seed=0: FakeMethod(), load_train=boom)
    with pytest.raises(TextMethodError) as e:
        s.predict("x")
    msg = str(e.value)
    assert msg.startswith("text method unavailable (mean)") and "secret" not in msg
    assert "<path>" in msg

    with pytest.raises(TextMethodError, match="not offered"):
        TextKnobSetter(CFG, "llm_zero_shot").predict("x")
    with pytest.raises(TextMethodError, match="no demo config"):
        TextKnobSetter(None, "mean").predict("x")

    class Broken(FakeMethod):
        def predict(self, queries):
            raise RuntimeError("failed reading /home/me/secret/x.npz")

    s, _ = _setter(Broken())
    with pytest.raises(TextMethodError) as e:
        s.predict("x")
    assert "secret" not in str(e.value)


def test_noop_and_mean_need_no_encoder(tmp_path):
    from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture

    cfg = eval_fixture_config(tmp_path)
    make_eval_fixture(cfg, seconds=0.5)

    def no_encoder(model):
        raise AssertionError("encoder loaded")

    noop = TextKnobSetter(cfg, "noop", load_text_encoder=no_encoder).predict("anything")
    assert all(knobs[ENABLED] == 0.0 for knobs in noop.board.values())
    mean = TextKnobSetter(cfg, "mean", load_text_encoder=no_encoder).predict("anything")
    assert set(mean.board) == set(default_pedalboard().block_names)
    assert "train_items" in mean.summary_md()


def _knn_cache(cfg, model):
    """Random train caption vectors in the embedding cache layout."""
    from lstmabar.eval.data import load_split
    from lstmabar.models.embed_cache import EmbeddingTable, cache_path, text_sha256

    train = load_split(cfg, "train")
    caps = [c for it in train for c in it.captions]
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(caps), 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    EmbeddingTable(
        ids=[c.caption_id for c in caps],
        vectors=vecs,
        sha256=[text_sha256(c.text) for c in caps],
        model=model,
        kind="text",
    ).save(cache_path(cfg.eval.embeddings.dir, model, "train", "text"))
    return train, caps, vecs


def test_knn_encodes_the_query_on_the_fly(tmp_path):
    from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture

    cfg = eval_fixture_config(tmp_path, ["demo.text_hparams.k=1"])
    make_eval_fixture(cfg, seconds=0.5)
    model = str(cfg.eval.baselines.knn.clap_model)
    train, caps, vecs = _knn_cache(cfg, model)
    target = 3  # the query embeds exactly like train caption 3

    class Enc:
        name, dim = model, 8

        def encode_text(self, texts, batch_size=32):
            assert texts == ["make it fuzzier"]
            return vecs[[target]]

    loaded = []
    s = TextKnobSetter(cfg, "knn_clap", load_text_encoder=lambda m: loaded.append(m) or Enc())
    pred = s.predict("make it fuzzier")
    s.predict("make it fuzzier")
    owner = next(it for it in train if any(c is caps[target] for c in it.captions))
    want = complete_board(owner.label)
    for b, knobs in want.items():
        assert pred.board[b][ENABLED] == (1.0 if knobs[ENABLED] >= 0.5 else 0.0)
        for k, v in knobs.items():
            if k != ENABLED:
                assert pred.board[b][k] == pytest.approx(v)
    assert loaded == [model]  # the encoder is cached too
    assert "k: 1" in pred.summary_md() and "knn_clap" in pred.summary_md()


def test_knn_without_encoders_says_text_model_unavailable(tmp_path):
    from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture

    cfg = eval_fixture_config(tmp_path)
    make_eval_fixture(cfg, seconds=0.5)
    _knn_cache(cfg, str(cfg.eval.baselines.knn.bge_model))

    def not_impl(model):
        raise NotImplementedError

    s = TextKnobSetter(cfg, "knn_bge", load_text_encoder=not_impl)
    with pytest.raises(TextMethodError, match="text model unavailable .*not implemented"):
        s.predict("darker")

    def no_extra(model):
        raise ImportError("No module named 'transformers'")

    s = TextKnobSetter(cfg, "knn_bge", load_text_encoder=no_extra)
    with pytest.raises(TextMethodError, match="clap extra"):
        s.predict("darker")


def test_knn_without_cache_and_p6_without_checkpoint_fail_cleanly(tmp_path):
    from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture

    cfg = eval_fixture_config(tmp_path)
    make_eval_fixture(cfg, seconds=0.5)
    with pytest.raises(TextMethodError, match="text model unavailable .*eval embed") as e:
        TextKnobSetter(cfg, "knn_clap").predict("darker")
    assert str(tmp_path) not in str(e.value)
    with pytest.raises(TextMethodError, match="text model unavailable .*checkpoint") as e:
        TextKnobSetter(cfg, "p6_model").predict("darker")
    assert str(tmp_path) not in str(e.value)


# --- helpers ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        r"cannot open C:\Users\me\x.npz",
        "cannot open C:/Users/me/x.npz",
        r"share \\server\share\x",
        "read /home/me/data/x.npz failed",
    ],
)
def test_scrub_paths(text):
    out = scrub_paths(text)
    assert "me" not in out.replace("cannot", "") and "<path>" in out


def test_scrub_keeps_model_ids():
    s = "model laion/clap-htsat-unfused, run `lstmabar eval embed` first"
    assert scrub_paths(s) == s


def test_info_summary_flattens_scalars_only():
    info = {
        "name": "knn_clap",
        "k": 5,
        "selection": {"criterion": "runner", "chosen": {"k": 5}, "scores": [{"k": 1}]},
        "where": "/home/me/cache/x.npz",
    }
    out = info_summary(info)
    assert "selection.chosen.k: 5" in out and "selection.criterion: runner" in out
    assert "scores" not in out and "home" not in out and "name" not in out


def test_demo_query_is_not_a_cache_key():
    q = demo_query("brighter", None, SR, -15.0)
    assert q.split == "demo" and q.text_key.startswith("demo-") and q.audio_key == "demo@-15.0"
    with pytest.raises(ValueError, match="no input clip"):
        q.load_dry()


def test_controls_match_within_half_a_step():
    assert controls_match([True], [1.0, 0.5], [True], [1.004, 0.5], [0.01, 0.001])
    assert not controls_match([True], [1.0, 0.5], [False], [1.0, 0.5], [0.01, 0.001])
    assert not controls_match([True], [1.0, 0.5], [True], [1.02, 0.5], [0.01, 0.001])
    assert not controls_match([True], [1.0, 0.5], [True], [1.0, None], [0.01, 0.001])


def test_warm_up_builds_once_in_the_background():
    m = FakeMethod()
    s, built = _setter(m)
    s.warm_up().join(timeout=10)
    s.predict("x")
    assert built == ["mean"] and m.fits == 1
    bad = TextKnobSetter(None, "mean")
    bad.warm_up().join(timeout=10)  # logged, not raised
    with pytest.raises(TextMethodError, match="no demo config"):
        bad.predict("x")
