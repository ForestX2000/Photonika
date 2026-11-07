"""Small on-demand physics teachers and phase-invariant distillation targets.

The teacher refines a detached student proposal with GS/Adam and keeps the
lowest physics loss separately for each example. It never owns network weights
and cannot accumulate gradients into the student's graph.
"""
from __future__ import annotations

import torch

from .baselines import gs_multiplane, random_phase
from .fresnel import FresnelSingleFFT
from .losses import multiplane_loss


def phase_distillation_loss(student: torch.Tensor, teacher: torch.Tensor,
                            target: torch.Tensor | None = None) -> torch.Tensor:
    """Circular phase loss (B,), invariant to a per-example global phase shift.

    Global phase leaves every propagated amplitude unchanged. Thus a raw L1/L2
    difference between phase angles would penalize physically equivalent DOEs.
    All-black targets are omitted when supplied because they constrain no phase.
    """
    if student.ndim != 3 or teacher.shape != student.shape:
        raise ValueError("student and teacher phase must have the same (B,H,W) shape")
    difference = student.float() - teacher.detach().float()
    phasor = torch.polar(torch.ones_like(difference), difference)
    loss = 1.0 - phasor.mean((-2, -1)).abs().clamp(max=1.0)
    if target is not None:
        if target.shape[0] != student.shape[0]:
            raise ValueError("target batch does not match phase batch")
        active = target.detach().float().square().flatten(1).mean(1) > 1e-12
        loss = loss * active.to(loss.dtype)
    return loss


def synthesize_teacher(target: torch.Tensor, prop: FresnelSingleFFT, *, gs_iters: int = 8,
                       adam_steps: int = 8, seed: int = 0, restarts: int = 1,
                       init_phase: torch.Tensor | None = None, lr: float = 0.03,
                       loss_kwargs: dict | None = None) -> torch.Tensor:
    """Return detached best phases, keeping the initial proposal as a candidate.

    The first restart uses init_phase when provided; subsequent restarts have
    deterministic independent random starts. Selection is per example, using
    the same unquantized physics objective that is passed through loss_kwargs.
    """
    if target.ndim != 4 or target.shape[0] < 1:
        raise ValueError("target must be nonempty (B,P,H,W)")
    if gs_iters < 0 or adam_steps < 0 or restarts < 1 or lr <= 0:
        raise ValueError("iterations must be nonnegative, restarts and lr positive")
    b, p, h, w = target.shape
    if (p, h, w) != (prop.num_planes, prop.optics.size, prop.optics.size):
        raise ValueError("target shape does not match teacher optics")
    if init_phase is not None and init_phase.shape != (b, h, w):
        raise ValueError("init_phase must be (B,H,W)")
    kwargs = loss_kwargs or {}
    # Explicitly disable autocast only here; trainable CNNs can still use BF16.
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(target.device.type, enabled=False):
        targets = target.detach().clone().float()
        best_phase = torch.zeros((b, h, w), device=target.device)
        best_loss = torch.full((b,), float("inf"), device=target.device)

        @torch.no_grad()
        def consider(phase):
            nonlocal best_phase, best_loss
            losses = multiplane_loss(prop(phase), targets, **kwargs)[0]
            improves = torch.isfinite(losses) & (losses < best_loss)
            best_phase = torch.where(improves[:, None, None], phase.detach(), best_phase)
            best_loss = torch.where(improves, losses, best_loss)

        for restart in range(restarts):
            initial = (init_phase.detach().clone().float() if restart == 0 and init_phase is not None
                       else random_phase((b, h, w), seed + restart, target.device))
            consider(initial)
            phase = gs_multiplane(targets, prop, gs_iters, init_phase=initial)
            consider(phase)
            if adam_steps:
                phase = phase.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([phase], lr=lr)
                for _ in range(adam_steps):
                    optimizer.zero_grad(set_to_none=True)
                    losses = multiplane_loss(prop(phase), targets, **kwargs)[0]
                    if not torch.isfinite(losses).all():
                        break
                    gradient, = torch.autograd.grad(losses.sum(), phase)
                    if not torch.isfinite(gradient).all():
                        break
                    phase.grad = gradient
                    optimizer.step()
                    consider(phase)
        if not torch.isfinite(best_loss).all():
            raise RuntimeError("teacher failed to produce a finite phase candidate")
        return torch.atan2(best_phase.sin(), best_phase.cos()).detach()
