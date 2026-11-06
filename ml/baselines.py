"""Итерационные базовые методы для 4 плоскостей: многоплоскостный GS и Adam по фазе.

Это и страховка (готовый ответ без нейросети), и эталон для сравнения с сетью.
Оба метода работают с батчем (B, P, N, N) на CPU и GPU.
"""
from __future__ import annotations

import math

import torch

from .fresnel import FresnelSingleFFT
from .losses import multiplane_loss


def normalize_amplitude(target: torch.Tensor) -> torch.Tensor:
    """Масштабирует амплитуды так, чтобы mean(A²) = 1 в каждой плоскости.

    При единичной амплитуде на ДОЭ и ortho-FFT средний квадрат модуля поля равен 1,
    поэтому такая цель согласована по энергии с тем, что ДОЭ может дать.
    """
    rms = target.float().square().mean((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
    return target.float() / rms


def random_phase(shape, seed: int, device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.rand(shape, generator=gen) * 2 * math.pi - math.pi).to(device)


@torch.no_grad()
def gs_multiplane(target: torch.Tensor, prop: FresnelSingleFFT, n_iter: int = 200,
                  init_phase: torch.Tensor | None = None, seed: int = 0,
                  plane_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Многоплоскостный Gerchberg–Saxton с усреднением обратных полей.

    target: (B, P, N, N) амплитуды ≥ 0. Возвращает фазу (B, N, N) в [−π, π].
    Итерация: поле во всех плоскостях → замена модуля на целевой при сохранении фазы →
    обратное распространение каждой плоскости в плоскость ДОЭ → взвешенная сумма →
    оставляем только фазу (ДОЭ фазовый).
    """
    amp = normalize_amplitude(target)
    b, p, n, _ = amp.shape
    phase = random_phase((b, n, n), seed, amp.device) if init_phase is None else init_phase.float().clone()
    w = None if plane_weights is None else plane_weights.to(amp.device).float().view(1, p, 1, 1)
    for _ in range(n_iter):
        field = prop.field(phase)
        field = amp * torch.sgn(field)          # sgn(z) = z/|z|: фаза поля, модуль цели
        back = prop.back(field)
        if w is not None:
            back = back * w
        phase = torch.angle(back.sum(1))
    return phase


def adam_phase(target: torch.Tensor, prop: FresnelSingleFFT, n_steps: int = 400,
               init_phase: torch.Tensor | None = None, lr: float = 0.05, seed: int = 0,
               loss_kwargs: dict | None = None) -> torch.Tensor:
    """Градиентная оптимизация фазы ДОЭ (Adam) по той же функции потерь, что и у сети."""
    b, p, n, _ = target.shape
    start = random_phase((b, n, n), seed, target.device) if init_phase is None else init_phase.float()
    phase = start.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    kwargs = loss_kwargs or {}
    with torch.enable_grad():
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            loss, _ = multiplane_loss(prop(phase), target.float(), **kwargs)
            loss.sum().backward()
            opt.step()
    return torch.angle(torch.polar(torch.ones_like(phase), phase.detach()))
