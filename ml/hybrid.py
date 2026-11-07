"""Numerical refinement starts exclusively from the network's synthesized DOE."""
from __future__ import annotations

import math

import numpy as np
import torch

from .baselines import gs_multiplane
from .fresnel import phase_to_uint8
from .losses import multiplane_loss


def refine_network_phase(initial: torch.Tensor, target: torch.Tensor, prop, *, gs_steps: int = 25,
                         adam_steps: int = 100, lr: float = 0.03,
                         check_every: int = 25) -> tuple[torch.Tensor, dict]:
    """Keep the best actual 8-bit DOE, including the unchanged neural proposal.

    Every candidate descends from ``initial``. There are no random restarts or
    classical initializations. Targets are the known synthesis inputs, so this
    per-input objective is available during deployment as well as validation.
    CPU jury scoring and all optimization are included in the pipeline timing.
    """
    if min(gs_steps, adam_steps) < 0 or check_every < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("invalid hybrid refinement budget or learning rate")
    if initial.shape != (len(target), *target.shape[-2:]):
        raise ValueError("initial phase shape must match target batch/spatial shape")
    if not torch.isfinite(initial).all():
        raise ValueError("network phase must be finite")
    from .pipeline import reconstruct
    from .training import quantize_phase_ste

    sources = (target.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    best = initial.detach().clone().float()
    best_scores = np.full(len(best), -np.inf)
    selected = ["network"] * len(best)
    checks = 0

    @torch.no_grad()
    def consider(candidate, label):
        nonlocal best, best_scores, checks
        phases = candidate.detach().double().cpu().numpy()
        scores = []
        for phase, source in zip(phases, sources):
            _, _, rows = reconstruct(phase_to_uint8(phase), source, prop.optics)
            scores.append(float(np.mean([r["pcc"] + r["ssim"] for r in rows])))
        scores = np.asarray(scores)
        if not np.isfinite(scores).all():
            raise RuntimeError("non-finite quantized hybrid score")
        improves = scores > best_scores
        mask = torch.from_numpy(improves).to(best.device)
        best = torch.where(mask[:, None, None], candidate.detach(), best)
        best_scores = np.where(improves, scores, best_scores)
        for i, improved in enumerate(improves):
            if improved:
                selected[i] = label
        checks += 1

    with torch.inference_mode(False), torch.enable_grad():
        consider(best, "network")
        initial_scores = best_scores.copy()
        if gs_steps:
            projected = gs_multiplane(target, prop, gs_steps, init_phase=best)
            consider(projected, "network+gs")
        phase = best.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([phase], lr=lr) if adam_steps else None
        completed = 0
        for step in range(adam_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = multiplane_loss(prop(quantize_phase_ste(phase)), target)[0].sum()
            if not torch.isfinite(loss):
                break
            gradient, = torch.autograd.grad(loss, phase)
            if not torch.isfinite(gradient).all():
                break
            phase.grad = gradient
            optimizer.step()
            completed = step + 1
            if completed % check_every == 0 or completed == adam_steps:
                consider(torch.atan2(phase.sin(), phase.cos()), f"network+adam:{completed}")
        return best.detach(), dict(initialization="network", random_restarts=0,
                                  selected_candidate=selected, quantized_checks=checks,
                                  initial_pcc_ssim=initial_scores.tolist(),
                                  final_pcc_ssim=best_scores.tolist(), completed_adam_steps=completed)
