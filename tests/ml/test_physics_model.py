"""Physics starts, trainability, checkpoint compatibility and deployed PNGs."""
import math

import numpy as np
import pytest
import torch

from ml.baselines import gs_multiplane, normalize_amplitude
from ml.config import DEFAULT_OPTICS, PHYSICS_VERSION, Optics
from ml.model import MultiPlaneUNet, build_model, load_model
from ml.physics_model import PhysicsUnrolledNet
from ml.pipeline import run


@pytest.mark.parametrize("mode", ["raw", "backprop", "imgphase"])
@pytest.mark.parametrize("unroll", [0, 2])
def test_untrained_model_is_reference_gs(mode, unroll):
    net = PhysicsUnrolledNet(base=4, stage_base=4, optics=Optics(size=32),
                             input_mode=mode, unroll=unroll).eval()
    x = torch.rand(3, 4, 32, 32)
    with torch.no_grad():
        phase = net(x)
        _, initial = net.features(x)
        if initial is None:
            initial = torch.angle(net.prop.back(normalize_amplitude(x) * net.features.psi).sum(1))
        expected = gs_multiplane(x, net.prop, unroll, init_phase=initial)
    error = torch.angle(torch.polar(torch.ones_like(phase), phase - expected))
    assert error.abs().max() < 2e-4
    assert phase.shape == (3, 32, 32)
    assert phase.abs().max() <= math.pi + 1e-6


def test_gradients_reach_initializer_global_context_and_every_projection():
    torch.manual_seed(7)
    net = PhysicsUnrolledNet(base=4, stage_base=4, optics=Optics(size=32), unroll=2)
    x = torch.rand(2, 4, 32, 32)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (net.prop(net(x)) - normalize_amplitude(x)).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        optimizer.step()
    for learner in [net.unet, *(s.net for s in net.stages)]:
        assert learner.enc[0].net[0].weight.grad.abs().sum() > 0
        assert learner.context.attention.in_proj_weight.grad.abs().sum() > 0
    assert net.stages[0].plane_weights[-1].weight.grad.abs().sum() > 0
    assert net.stages[0].relaxation.grad.abs() > 0


def test_blank_and_mixed_blank_planes_have_finite_gradients():
    net = PhysicsUnrolledNet(base=4, stage_base=4, optics=Optics(size=32), unroll=1)
    x = torch.rand(2, 4, 32, 32)
    x[0] = 0
    x[1, 0] = 0
    phase = net(x)
    net.prop(phase).square().mean().backward()
    assert torch.isfinite(phase).all()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@pytest.mark.parametrize("architecture", ["unet", "pdr", "physics-unrolled"])
def test_checkpoint_round_trip_and_strict_parameter_loading(tmp_path, architecture):
    optics = Optics(size=32)
    config = dict(architecture=architecture, base=4, stage_base=4, unroll=0, input_mode="imgphase")
    model = build_model(config, optics).eval()
    x = torch.rand(2, 4, 32, 32)
    checkpoint = dict(physics_version=PHYSICS_VERSION, optics=optics.as_dict(),
                      model=model.config(), net=model.state_dict())
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    restored, meta = load_model(path)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
    assert meta["model"]["architecture"] == architecture
    checkpoint["net"].pop(next(iter(checkpoint["net"])))
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path)


@pytest.mark.parametrize("architecture", ["unet", "pdr"])
def test_old_checkpoint_family_is_detected(tmp_path, architecture):
    config = dict(architecture=architecture, base=4, input_mode="backprop", unroll=0, stage_base=4)
    model = build_model(config, Optics(size=32))
    old_config = {k: config[k] for k in ("base", "input_mode", "unroll", "stage_base")}
    path = tmp_path / "legacy.pt"
    torch.save(dict(physics_version=PHYSICS_VERSION, optics=model.prop.optics.as_dict(),
                    model=old_config, net=model.state_dict()), path)
    restored, meta = load_model(path)
    assert meta["model"]["architecture"] == architecture
    assert restored.config()["architecture"] == architecture


def test_original_app_checkpoint_parameter_layout_unchanged():
    from app.model import MultiPlaneUNet as OriginalUNet
    old = OriginalUNet(base=4, optics=Optics(size=32), unroll=1, stage_base=4)
    new = MultiPlaneUNet(base=4, optics=Optics(size=32), unroll=1, stage_base=4)
    new.load_state_dict(old.state_dict(), strict=True)


def test_model_runs_through_quantized_submission_pipeline():
    net = PhysicsUnrolledNet(base=4, stage_base=4, unroll=1).eval()
    sources = np.random.default_rng(4).integers(0, 256, (4, 128, 128), dtype=np.uint8)
    result = run(sources, "net", model=net, device="cpu")
    assert result.doe_u8.shape == (128, 128)
    assert result.recon_u8.dtype == np.uint8
    from ml.pipeline import reconstruct
    recon, _, _ = reconstruct(result.doe_u8, sources, DEFAULT_OPTICS)
    np.testing.assert_array_equal(result.recon_u8, recon)


def test_invalid_shapes_and_unknown_architectures_fail_early():
    net = PhysicsUnrolledNet(base=4, stage_base=4, optics=Optics(size=32), unroll=0)
    with pytest.raises(ValueError, match="amps must"):
        net(torch.rand(1, 3, 32, 32))
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model({"architecture": "typo"})
    with pytest.raises(ValueError, match="model_version"):
        build_model({"architecture": "physics-unrolled", "model_version": 2})
