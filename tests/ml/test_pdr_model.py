"""Regression checks for the retained complex PDR checkpoint family."""
import math

import pytest
import torch

from ml.baselines import gs_multiplane, normalize_amplitude
from ml.config import Optics
from ml.pdr_model import ComplexConvBlock, MultiPlaneUNet, build_model


@pytest.mark.parametrize("mode", ["raw", "backprop", "imgphase"])
@pytest.mark.parametrize("batch", [1, 3])
def test_pdr_all_input_modes_support_batches_and_physics_start(mode, batch):
    model = MultiPlaneUNet(base=4, input_mode=mode, optics=Optics(size=32)).eval()
    x = torch.rand(batch, 4, 32, 32)
    with torch.no_grad():
        features, base = model.features(x)
        assert features.shape[1] == model.features.channels
        phase = model(x)
        if base is None:
            base = torch.angle(model.prop.back(normalize_amplitude(x) * model.features.psi).sum(1))
    assert phase.shape == (batch, 32, 32)
    assert torch.isfinite(phase).all()
    assert phase.abs().max() <= math.pi
    assert torch.angle(torch.exp(1j * (phase - base))).abs().max() < 1e-4


def test_pdr_unroll_executes_physical_stages():
    model = MultiPlaneUNet(base=4, optics=Optics(size=32), unroll=2, stage_base=4).eval()
    x = torch.rand(2, 4, 32, 32)
    with torch.no_grad():
        _, base = model.features(x)
        expected = gs_multiplane(x, model.prop, 2, init_phase=base)
        actual = model(x)
    assert len(model.stages) == 2
    assert torch.angle(torch.exp(1j * (actual - expected))).abs().max() < 1e-3


def test_complex_normalization_stays_bounded_with_finite_gradients():
    torch.manual_seed(91)
    block = ComplexConvBlock(2, 8)
    x = torch.complex(torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)).requires_grad_()
    out = block(x)
    assert out.abs().max() < 10
    out.abs().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(block.conv_real.weight.grad).all()


def test_pdr_blank_images_have_finite_gradients():
    model = MultiPlaneUNet(base=4, optics=Optics(size=32))
    x = torch.zeros(2, 4, 32, 32)
    phase = model(x)
    phase.square().mean().backward()
    assert torch.isfinite(phase).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_pdr_config_and_unroll_zero_checkpoint_shapes_are_preserved():
    model = MultiPlaneUNet(base=4, input_mode="imgphase", optics=Optics(size=32))
    state = model.state_dict()
    assert state["unet.branch_high.enc.0.conv_real.weight"].shape == (4, 8, 3, 3)
    assert state["unet.head.weight"].shape == (2, 8, 1, 1)
    assert not any(key.startswith("stages.") for key in state)
    restored = build_model(model.config(), Optics(size=32))
    restored.load_state_dict(state, strict=True)
    assert restored.config()["architecture"] == "pdr"


@pytest.mark.parametrize("kwargs", [
    {"base": 1}, {"unroll": -1}, {"stage_base": 0},
    {"input_mode": "unknown"}, {"optics": Optics(size=48)}, {"num_planes": 3},
])
def test_pdr_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MultiPlaneUNet(**kwargs)
