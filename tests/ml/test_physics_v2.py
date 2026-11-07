"""Scalable spectral model: physical starts, training, precision and checkpoints."""
import copy
import math

import pytest
import torch

from ml.baselines import gs_multiplane, normalize_amplitude
from ml.config import PHYSICS_VERSION, Optics
from ml.model import build_model, load_model
from ml.physics_model import FourierMixing, PhysicsUnrolledNetV2


@pytest.mark.parametrize("mode", ["raw", "backprop", "imgphase"])
def test_v2_trajectory_starts_as_reference_gs(mode):
    net = PhysicsUnrolledNetV2(base=4, stage_base=4, optics=Optics(size=32),
                               input_mode=mode, unroll=2, spectral_modes=3).eval()
    x = torch.rand(2, 4, 32, 32)
    with torch.inference_mode():
        phases = net(x, return_stages=True)
        final = net(x)
        _, initial = net.features(x)
        if initial is None:
            initial = torch.angle(net.prop.back(normalize_amplitude(x) * net.features.psi).sum(1))
        assert len(phases) == 3
        for index, phase in enumerate(phases):
            reference = gs_multiplane(x, net.prop, index, init_phase=initial)
            difference = torch.angle(torch.polar(torch.ones_like(phase), phase - reference))
            assert difference.abs().max() < 2e-4
            assert phase.shape == (2, 32, 32)
            assert phase.dtype == torch.float32
            assert phase.abs().max() <= math.pi + 1e-6
        torch.testing.assert_close(final, phases[-1], rtol=0, atol=0)


def test_v2_gradients_reach_every_spectral_and_attention_branch():
    torch.manual_seed(4)
    net = PhysicsUnrolledNetV2(base=4, stage_base=4, optics=Optics(size=32), unroll=2, spectral_modes=3)
    x = torch.rand(2, 4, 32, 32)
    target = normalize_amplitude(x)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        trajectory = net(x, return_stages=True)
        loss = sum((net.prop(phase) - target).square().mean() for phase in trajectory)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        optimizer.step()
    for learner in [net.unet, *(stage.net for stage in net.stages)]:
        assert learner.enc[0].conv1.weight.grad.abs().sum() > 0
        assert learner.context.spectral.filters.grad.abs().sum() > 0
        assert learner.context.attention.in_proj_weight.grad.abs().sum() > 0


def test_activation_checkpointing_preserves_outputs_and_gradients():
    torch.manual_seed(8)
    ordinary = PhysicsUnrolledNetV2(base=4, stage_base=4, optics=Optics(size=32), unroll=2, spectral_modes=3)
    # Exercise all branches rather than comparing the zero output heads alone.
    with torch.no_grad():
        for learner in [ordinary.unet, *(stage.net for stage in ordinary.stages)]:
            learner.head.weight.normal_(std=1e-3)
    checkpointed = copy.deepcopy(ordinary)
    checkpointed.set_activation_checkpointing(True)
    assert checkpointed.config() == ordinary.config()
    x = torch.rand(2, 4, 32, 32)
    result = ordinary(x, return_stages=True)
    recomputed = checkpointed(x, return_stages=True)
    for a, b in zip(result, recomputed):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    sum(ordinary.prop(p).square().mean() for p in result).backward()
    sum(checkpointed.prop(p).square().mean() for p in recomputed).backward()
    for a, b in zip(ordinary.parameters(), checkpointed.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)


def test_bfloat16_convolutions_leave_phase_and_physics_in_float32():
    net = PhysicsUnrolledNetV2(base=4, stage_base=4, optics=Optics(size=32), unroll=1, spectral_modes=3)
    x = torch.rand(2, 4, 32, 32)
    x[0] = 0
    x[1, 1] = 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        phases = net(x, return_stages=True)
    assert all(phase.dtype == torch.float32 and torch.isfinite(phase).all() for phase in phases)
    sum((net.prop(phase) - normalize_amplitude(x)).square().mean() for phase in phases).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@pytest.mark.parametrize("shape", [(2, 4, 1, 1), (2, 4, 3, 5), (2, 4, 8, 8)])
def test_fourier_filters_handle_capped_and_disjoint_frequency_bands(shape):
    layer = FourierMixing(4, modes=8)
    x = torch.rand(shape, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(x)
    assert output.shape == x.shape
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(layer.filters.grad).all()
    assert layer.filters.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_v2_strict_checkpoint_roundtrip_and_runtime_memory_setting(tmp_path):
    optics = Optics(size=32)
    config = dict(architecture="physics-unrolled", model_version=2, base=4, stage_base=4,
                   input_mode="imgphase", unroll=1, spectral_modes=3)
    net = build_model(config, optics).eval()
    net.set_activation_checkpointing(True)
    checkpoint = dict(physics_version=PHYSICS_VERSION, optics=optics.as_dict(),
                       model=net.config(), net=net.state_dict())
    path = tmp_path / "spectral.pt"
    torch.save(checkpoint, path)
    restored, metadata = load_model(path)
    assert metadata["model"] == config
    assert restored.activation_checkpointing is False
    x = torch.rand(2, 4, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(net(x), restored(x), atol=0, rtol=0)
    checkpoint["net"].pop("unet.context.spectral.filters")
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path)


@pytest.mark.parametrize("options", [{"spectral_modes": 0}, {"unroll": -1}, {"base": 1}])
def test_v2_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        PhysicsUnrolledNetV2(**options)
