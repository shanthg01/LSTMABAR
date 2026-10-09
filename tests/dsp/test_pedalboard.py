import time

import pytest
import torch

from lstmabar.dsp import Pedalboard, blocks, default_pedalboard
from lstmabar.dsp.blocks import EQ3, Drive, Gain
from tests.dsp.signals import SR, riff


@pytest.fixture(scope="module")
def board():
    return default_pedalboard(SR)


def test_default_board_order_and_attributes(board):
    assert board.block_names == ["compressor", "drive", "eq"]
    assert list(board.blocks.keys()) == board.block_names
    assert board.sample_rate == SR
    assert isinstance(board.blocks["drive"], blocks.Drive)


def test_layout_is_stable_and_ends_each_block_with_enabled(board):
    layout = board.layout()
    assert layout == default_pedalboard(SR).layout()
    expected = []
    for name, block in board.blocks.items():
        expected += [(name, k) for k in block.param_names] + [(name, "enabled")]
    assert layout == expected
    assert len(layout) == 5 + 1 + 7 + 1 + 4 + 1


def test_defaults_render_finite_audio(board):
    x = riff(2, 0.5)
    y = board(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.isfinite(y).all() and y.abs().max() > 1e-3
    torch.testing.assert_close(board(x, board.default_params(2)), y)


def test_float64(board):
    x = riff(1, 0.3).double()
    y = board(x, board.default_params(1, dtype=torch.float64))
    assert y.dtype == torch.float64 and torch.isfinite(y).all()


def test_unknown_names_raise(board):
    x = riff(1, 0.3)
    with pytest.raises(KeyError):
        board(x, {"reverb": {}})
    with pytest.raises(KeyError):
        board(x, {"drive": {"nope": torch.zeros(1)}})


def test_gate_semantics(board):
    x = riff(2, 0.5)
    b = 2
    p = board.default_params(b)
    for name in board.block_names:
        p[name]["enabled"] = torch.zeros(b)
    assert torch.equal(board(x, p), x)  # all bypassed: exact

    drive = board.blocks["drive"]
    knobs = {"gain_db": torch.full((b,), 0.7)}
    wet = drive(x, knobs)
    only = lambda g: {  # noqa: E731
        "compressor": {"enabled": torch.zeros(b)},
        "eq": {"enabled": torch.zeros(b)},
        "drive": {**knobs, "enabled": g},
    }
    assert torch.equal(board(x, only(torch.zeros(b))), x)
    torch.testing.assert_close(board(x, only(torch.ones(b))), wet)
    torch.testing.assert_close(board(x, only(torch.full((b,), 0.5))), 0.5 * wet + 0.5 * x)
    # per-example gates: row 0 bypassed exactly, row 1 wet
    mixed = board(x, only(torch.tensor([0.0, 1.0])))
    assert torch.equal(mixed[0], x[0])
    torch.testing.assert_close(mixed[1], wet[1])


def test_missing_blocks_and_knobs_use_defaults(board):
    x = riff(1, 0.5)
    full = board.default_params(1)
    full["drive"]["gain_db"] = torch.tensor([0.9])
    torch.testing.assert_close(
        board(x, {"drive": {"gain_db": torch.tensor([0.9])}}), board(x, full)
    )


def test_vector_roundtrip(board):
    g = torch.Generator().manual_seed(0)
    p = board.random_params(4, generator=g)
    v = board.to_vector(p)
    assert v.shape == (4, len(board.layout()))
    q = board.from_vector(v)
    assert q.keys() == p.keys()
    for name in p:
        assert q[name].keys() == p[name].keys()
        for k in p[name]:
            assert torch.equal(q[name][k], p[name][k])
    torch.testing.assert_close(board.to_vector(q), v)


def test_to_vector_fills_defaults(board):
    v = board.to_vector({"eq": {"low_db": torch.tensor([0.25, 0.75])}})
    d = board.to_vector(board.default_params(2))
    j = board.layout().index(("eq", "low_db"))
    assert torch.equal(v[:, j], torch.tensor([0.25, 0.75]))
    mask = torch.ones(v.shape[1], dtype=torch.bool)
    mask[j] = False
    torch.testing.assert_close(v[:, mask], d[:, mask])


def test_random_params_reproducible_and_in_range(board):
    a = board.random_params(64, generator=torch.Generator().manual_seed(3), p_enabled=0.7)
    b = board.random_params(64, generator=torch.Generator().manual_seed(3), p_enabled=0.7)
    c = board.random_params(64, generator=torch.Generator().manual_seed(4), p_enabled=0.7)
    va, vb, vc = board.to_vector(a), board.to_vector(b), board.to_vector(c)
    assert torch.equal(va, vb) and not torch.equal(va, vc)
    assert va.min() >= 0 and va.max() <= 1 and va.dtype == torch.float32
    for name in board.block_names:
        gate = a[name]["enabled"]
        assert set(gate.unique().tolist()) <= {0.0, 1.0}
    gates = torch.stack([a[n]["enabled"] for n in board.block_names])
    assert 0.5 < gates.mean() < 0.9
    off = board.random_params(8, generator=torch.Generator().manual_seed(0), p_enabled=0.0)
    assert all((off[n]["enabled"] == 0).all() for n in board.block_names)


def test_describe_returns_physical_units(board):
    d = board.describe(board.default_params(1))
    assert d["drive"]["pre_hpf_hz"] == pytest.approx(720.0, rel=1e-4)
    assert d["drive"]["level_db"] == pytest.approx(-12.0, abs=1e-4)
    assert d["compressor"]["ratio"] == pytest.approx(4.0, rel=1e-4)
    assert d["eq"]["mid_hz"] == pytest.approx(800.0, rel=1e-4)
    assert d["eq"]["enabled"] == 1.0
    assert all(isinstance(v, float) for v in d["drive"].values())
    p = board.default_params(3)
    p["eq"]["low_db"] = torch.tensor([0.0, 0.5, 1.0])
    d3 = board.describe(p)
    assert d3["eq"]["low_db"] == pytest.approx([-12.0, 0.0, 12.0], abs=1e-4)
    assert len(d3["drive"]["gain_db"]) == 3


def test_batch_independence(board):
    x = riff(3, 0.5)
    p = board.random_params(3, generator=torch.Generator().manual_seed(5), p_enabled=1.0)
    y = board(x, p)
    for i in range(3):
        pi = {n: {k: v[i : i + 1] for k, v in kn.items()} for n, kn in p.items()}
        torch.testing.assert_close(y[i : i + 1], board(x[i : i + 1], pi), atol=1e-5, rtol=1e-4)


def test_every_board_param_gets_gradient(board):
    x = riff(2, 0.5)
    g = torch.Generator().manual_seed(2)
    v = (0.2 + 0.6 * torch.rand(2, len(board.layout()), generator=g)).requires_grad_()
    target = riff(2, 0.5) * 0.2
    (board(x, board.from_vector(v)) - target).pow(2).mean().backward()
    assert torch.isfinite(v.grad).all()
    zero = [board.layout()[j] for j in range(v.shape[1]) if (v.grad[:, j] == 0).any()]
    assert not zero, zero


def test_custom_board_validation():
    with pytest.raises(ValueError):
        Pedalboard([("a", Gain(SR)), ("a", Gain(SR))], SR)
    with pytest.raises(ValueError):
        Pedalboard([("a", Gain(22050))], SR)
    b = Pedalboard([("pre", Gain(SR)), ("drive", Drive(SR, oversample=2)), ("eq", EQ3(SR))], SR)
    assert b.block_names == ["pre", "drive", "eq"]
    x = riff(1, 0.3)
    torch.testing.assert_close(
        b(x, {"pre": {"enabled": torch.zeros(1)}}), b.blocks["eq"](b.blocks["drive"](x))
    )


def test_forward_backward_timing(board):
    """Reported speed (not a strict gate): B=8 x 3 s forward+backward on CPU."""
    x = riff(8, 3.0)
    v = board.to_vector(board.random_params(8, generator=torch.Generator().manual_seed(0)))
    v.requires_grad_()
    t0 = time.perf_counter()
    board(x, board.from_vector(v)).pow(2).mean().backward()
    elapsed = time.perf_counter() - t0
    print(f"default_pedalboard B=8 x 3 s fwd+bwd: {elapsed:.2f} s")
    assert elapsed < 30.0
