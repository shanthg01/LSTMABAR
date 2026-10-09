import pytest
import torch

from lstmabar.dsp.base import EffectBlock, ParamSpec


def test_linear_spec_roundtrip():
    s = ParamSpec("gain_db", -12.0, 12.0, 0.0, "dB")
    u = torch.tensor([0.0, 0.25, 0.5, 1.0])
    v = s.denormalize(u)
    assert torch.allclose(v, torch.tensor([-12.0, -6.0, 0.0, 12.0]))
    assert torch.allclose(s.normalize(v), u)
    assert s.default_normalized == pytest.approx(0.5)


def test_log_spec_roundtrip_and_midpoint_is_geometric_mean():
    s = ParamSpec("freq_hz", 100.0, 10000.0, 1000.0, "Hz", taper="log")
    assert float(s.denormalize(torch.tensor(0.5))) == pytest.approx(1000.0, rel=1e-5)
    v = torch.tensor([100.0, 316.2278, 10000.0])
    assert torch.allclose(s.denormalize(s.normalize(v)), v, rtol=1e-4)


def test_denormalize_clamps_out_of_range():
    s = ParamSpec("x", 0.0, 1.0, 0.5)
    assert torch.equal(s.denormalize(torch.tensor([-1.0, 2.0])), torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="a", min=1.0, max=1.0, default=1.0),
        dict(name="a", min=0.0, max=1.0, default=2.0),
        dict(name="a", min=0.0, max=1.0, default=0.5, taper="log"),
    ],
)
def test_invalid_specs_raise(kwargs):
    with pytest.raises(ValueError):
        ParamSpec(**kwargs)


class _Gain(EffectBlock):
    param_specs = (ParamSpec("gain_db", -24.0, 24.0, 0.0, "dB"),)

    def process(self, x, p):
        return x * 10 ** (p["gain_db"][:, None] / 20)


def test_block_uses_defaults_and_denormalizes():
    block = _Gain()
    x = torch.ones(2, 8)
    assert torch.allclose(block(x), x)
    y = block(x, {"gain_db": torch.tensor([0.5 + 6 / 48, 0.5])})
    assert torch.allclose(y[0], x[0] * 10 ** (6 / 20))
    assert torch.allclose(y[1], x[1])


def test_block_rejects_unknown_params_and_bad_shapes():
    block = _Gain()
    with pytest.raises(KeyError):
        block(torch.ones(1, 4), {"nope": torch.zeros(1)})
    with pytest.raises(ValueError):
        block(torch.ones(4))


def test_block_is_differentiable_wrt_normalized_params():
    block = _Gain()
    u = torch.full((1,), 0.6, requires_grad=True)
    block(torch.ones(1, 4), {"gain_db": u}).sum().backward()
    assert u.grad is not None and u.grad.abs().item() > 0
