"""Physics teachers preserve student graphs and phase-invariant supervision."""
import math

import pytest
import torch

from ml.config import Optics
from ml.fresnel import FresnelSingleFFT
from ml.losses import multiplane_loss
from ml.teacher import phase_distillation_loss, synthesize_teacher


def test_distillation_ignores_global_phase_and_wrap_boundary():
    phase = torch.rand(3, 16, 16, requires_grad=True)
    shifted = phase.detach() + torch.tensor([0.9, -2.0, 2 * math.pi])[:, None, None]
    loss = phase_distillation_loss(phase, shifted)
    assert loss.abs().max() < 1e-6
    loss.sum().backward()
    assert torch.isfinite(phase.grad).all()


def test_distillation_masks_black_targets_and_has_useful_gradients():
    student = torch.rand(2, 16, 16, requires_grad=True)
    target = torch.ones(2, 4, 16, 16)
    target[0] = 0
    loss = phase_distillation_loss(student, torch.zeros_like(student), target)
    assert loss[0] == 0
    assert loss[1] > 0
    loss.sum().backward()
    assert student.grad[0].abs().sum() == 0
    assert student.grad[1].abs().sum() > 0


def test_teacher_is_deterministic_detached_and_never_worsens_initial_physics_loss():
    torch.manual_seed(10)
    prop = FresnelSingleFFT(Optics(size=16))
    targets = torch.rand(2, 4, 16, 16)
    initial = torch.rand(2, 16, 16, requires_grad=True)
    kwargs = dict(gs_iters=2, adam_steps=3, seed=4, restarts=2, init_phase=initial)
    first = synthesize_teacher(targets, prop, **kwargs)
    second = synthesize_teacher(targets, prop, **kwargs)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not first.requires_grad
    assert initial.grad is None
    before = multiplane_loss(prop(initial.detach()), targets)[0]
    after = multiplane_loss(prop(first), targets)[0]
    assert torch.all(after <= before + 1e-6)
    assert first.abs().max() <= math.pi + 1e-6


def test_teacher_black_targets_and_disabled_steps_are_finite():
    prop = FresnelSingleFFT(Optics(size=16))
    target = torch.zeros(2, 4, 16, 16)
    with torch.no_grad():
        teacher = synthesize_teacher(target, prop, gs_iters=0, adam_steps=2)
    assert torch.isfinite(teacher).all()
    with pytest.raises(ValueError, match="iterations"):
        synthesize_teacher(target, prop, restarts=0)


def test_teacher_preserves_live_student_graph_under_autocast():
    torch.manual_seed(5)
    prop = FresnelSingleFFT(Optics(size=16))
    targets = torch.rand(2, 4, 16, 16)
    student = torch.nn.Linear(16, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        phase = student(torch.rand(2, 16, 16))
        teacher = synthesize_teacher(targets, prop, init_phase=phase, gs_iters=1, adam_steps=2)
    assert teacher.dtype == torch.float32
    assert all(parameter.grad is None for parameter in student.parameters())
    physical = multiplane_loss(prop(phase.float()), targets)[0]
    hint = phase_distillation_loss(phase, teacher, targets)
    (physical + 0.1 * hint).mean().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in student.parameters())
    assert student.weight.grad.abs().sum() > 0
