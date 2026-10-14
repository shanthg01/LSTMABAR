"""P5 metrics (task 5.1): known-answer tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lstmabar.audio import synth_riff
from lstmabar.data.gold import expand, load_targets, region_from_dict, sample
from lstmabar.eval import metrics as M
from lstmabar.eval.baselines.base import complete_board, default_board
from lstmabar.eval.fixtures import TARGETS_FILE


@pytest.fixture(scope="module")
def targets():
    return load_targets(TARGETS_FILE)


def _board(**gates):
    b = default_board()
    for k, v in gates.items():
        b[k]["enabled"] = float(v)
    return b


def test_param_l1_masked_by_label_gates():
    label = _board(drive=1, eq=0, compressor=0)
    pred = complete_board(
        {"drive": {"gain_db": 1.0, "enabled": 0.0}, "eq": {"high_db": 0.0, "enabled": 1.0}}
    )
    out = M.param_l1(pred, label)
    assert not any(k.startswith(("eq.", "compressor.")) for k in out)
    dg = abs(1.0 - label["drive"]["gain_db"])
    assert out["drive.gain_db"] == pytest.approx(dg)
    n_drive = len(label["drive"]) - 1
    assert out["mean"] == pytest.approx(dg / n_drive)
    assert math.isnan(M.param_l1(pred, _board())["mean"])


def test_gate_f1_hand_cases():
    labels = [_board(drive=1), _board(drive=1), _board(drive=0), _board(drive=0)]
    preds = [_board(drive=1), _board(drive=0), _board(drive=1), _board(drive=0)]
    f = M.gate_f1(preds, labels)
    # drive: tp=1 fp=1 fn=1 -> 2/4
    assert f["drive"] == pytest.approx(0.5)
    # eq/compressor: never on in labels or preds -> NaN, ignored by the macro
    assert math.isnan(f["eq"]) and math.isnan(f["compressor"])
    assert f["macro"] == pytest.approx(0.5)
    assert M.gate_f1(labels, labels)["drive"] == 1.0
    assert M.gate_correct(preds[1], labels[1]) == pytest.approx(2 / 3)
    assert M.gate_correct(labels[0], labels[0]) == 1.0
    with pytest.raises(ValueError):
        M.gate_f1(preds[:1], labels)


def test_region_distance_zero_inside_free_region(targets):
    rng = np.random.default_rng(0)
    for rd in (
        {"kind": "free", "gain": "none", "brightness": "bright"},
        {
            "kind": "free",
            "gain": "high",
            "family": "distortion",
            "clipping": "hard",
            "brightness": "dark",
            "compression": "heavy",
        },
    ):
        region = region_from_dict(rd, "t")
        for _ in range(3):
            _, board = sample(expand(region, targets), rng)
            assert M.region_distance(board, region, targets) == pytest.approx(0.0, abs=1e-9)


def test_region_distance_outside_and_gates(targets):
    region = region_from_dict({"kind": "free", "gain": "none", "brightness": "bright"}, "t")
    _, inside = sample(expand(region, targets), np.random.default_rng(1))
    dark = region_from_dict({"kind": "free", "gain": "none", "brightness": "dark"}, "t")
    _, outside = sample(expand(dark, targets), np.random.default_rng(1))
    assert M.region_distance(outside, region, targets) > 0.05
    on = {k: dict(v) for k, v in inside.items()}
    on["drive"]["enabled"] = 1.0  # a gate mismatch on a block the region disables
    assert M.region_distance(on, region, targets) > 0.0
    # knobs the region does not constrain (drive voicing of a disabled block) are skipped
    off = {k: dict(v) for k, v in inside.items()}
    off["drive"]["gain_db"] = 1.0 - off["drive"]["gain_db"]
    assert M.region_distance(off, region, targets) == pytest.approx(0.0, abs=1e-9)


def test_region_distance_pedal_nearest_reference(targets):
    region = region_from_dict({"kind": "pedal", "pedal": "ts808", "gain": "medium"}, "t")
    a = _board(drive=1, eq=1)
    b = {k: dict(v) for k, v in a.items()}
    b["drive"]["gain_db"] = 0.9
    assert M.region_distance(a, region, targets, [b, a]) == 0.0
    assert M.region_distance(a, region, targets, [b]) > 0.0
    assert math.isnan(M.region_distance(a, region, targets, []))
    # output level knobs (tables.fixed) are skipped
    c = {k: dict(v) for k, v in a.items()}
    c["drive"]["level_db"] = 1.0 - c["drive"]["level_db"]
    assert M.region_distance(a, region, targets, [c]) == 0.0


def test_mrstft_identity_and_scale_invariance():
    x = synth_riff("single_notes", 0.5, 44100, seed=1)
    assert M.mrstft(x, x) == pytest.approx(0.0, abs=1e-5)
    assert M.mrstft(0.1 * x, x) == pytest.approx(0.0, abs=1e-5)  # peak-normalized
    assert M.mrstft(x, np.concatenate([x, np.zeros(1000)])) == pytest.approx(0.0, abs=1e-5)
    y = np.tanh(8 * x)
    assert M.mrstft(y, x) > 0.05
    assert math.isnan(M.mrstft(np.full_like(x, np.nan), x))


def test_descriptor_errors_nan_aware():
    a = {"centroid_hz": 2000.0, "odd_even_db": 3.0, "hnr_db": 10.0}
    b = {"centroid_hz": 1000.0, "odd_even_db": 1.0, "hnr_db": float("nan")}
    arch = {"sine": 0.5, "noise": 0.5}
    arch2 = {"sine": 0.25, "noise": 0.75}
    e = M.descriptor_errors(a, b, arch, arch2)
    assert set(e) == set(M.DESCRIPTOR_TERMS)
    assert e["centroid_oct"] == pytest.approx(1.0)
    assert e["odd_even_db"] == pytest.approx(2.0)
    assert math.isnan(e["hnr_db"])
    assert e["archetype_l1"] == pytest.approx(0.5)
    e2 = M.descriptor_errors(a, b, arch, arch2, harmonic=False)
    assert e2["centroid_oct"] == pytest.approx(1.0)
    assert all(math.isnan(e2[k]) for k in ("odd_even_db", "hnr_db", "archetype_l1"))
    e3 = M.descriptor_errors(a, {"centroid_hz": None}, None, arch2)
    assert all(math.isnan(v) for v in e3.values())
    # 1 octave / 1 + 2 dB / 6 -> mean
    assert M.descriptor_distance(a, b) == pytest.approx((1.0 + 2.0 / 6.0) / 2)
    assert M.descriptor_distance(a, b, harmonic=False) == pytest.approx(1.0)
    assert math.isnan(M.descriptor_distance({}, {}))
    assert M.descriptor_distance(a, a) == 0.0
    # archetype L1 (0.5, unscaled) joins when both vectors are given
    assert M.descriptor_distance(a, b, True, arch, arch2) == pytest.approx(
        (1.0 + 2.0 / 6.0 + 0.5) / 3
    )
    assert M.descriptor_distance(a, b, False, arch, arch2) == pytest.approx(1.0)


def test_analyse_render_signal_descriptors(tmp_path):
    from lstmabar.eval.fixtures import eval_fixture_config

    cfg = eval_fixture_config(tmp_path)
    x = synth_riff("single_notes", 0.5, 44100, seed=2)
    desc, arch = M.analyse_render(x, cfg, track=None, harmonic=True)
    assert desc["centroid_hz"] > 0
    assert math.isnan(desc["hnr_db"]) and all(math.isnan(v) for v in arch.values())


def test_two_afc_and_ties():
    assert M.two_afc(0.0, [1.0, 2.0]) == 1.0
    assert M.two_afc(1.0, [1.0]) == 0.5
    assert M.two_afc(3.0, [1.0, 5.0]) == 0.5
    assert M.two_afc(1.0, [float("nan"), 0.5]) == 0.0
    assert math.isnan(M.two_afc(1.0, [float("nan")]))
    assert math.isnan(M.two_afc(float("nan"), [1.0]))
    # target vs itself: d(target, target) = 0 beats every distinct distractor
    x = synth_riff("single_notes", 0.5, 44100, seed=3)
    d = np.tanh(10 * x)
    assert M.two_afc(M.mrstft(x, x), [M.mrstft(x, d)]) == 1.0


def test_clap_score_cosine():
    t = np.array([[1.0, 0.0], [1.0, 1.0]])
    a = np.array([[2.0, 0.0], [-1.0, -1.0]])
    np.testing.assert_allclose(M.clap_score(t, a), [1.0, -1.0])
    with pytest.raises(ValueError):
        M.clap_score(t, a[:1])
