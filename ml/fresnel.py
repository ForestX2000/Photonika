"""Дифракция Френеля с одним преобразованием Фурье (формула из презентации организаторов).

    K(x, y) = exp(jkz)/(jλz) · exp(jπ/(λz)·(x²+y²)) · FFT[ H(u, v) · exp(jπ/(λz)·(u²+v²)) ]

Внешние множители имеют единичный модуль или постоянны по плоскости, поэтому для
модуля поля |K| они не нужны:

    |K_i| ∝ |FFT( exp(jφ) · Q_i )|,   Q_i(u, v) = exp(jπ/(λ z_i) · (u² + v²)).

Координаты (u, v) отсчитываются от центра ДОЭ с шагом Δx, результат FFT
центрируется (fftshift). FFT нормирован как "ortho": энергия сохраняется,
и при единичной амплитуде на ДОЭ средний квадрат модуля в каждой плоскости равен 1.

Модуль numpy — эталонный «восстановитель» для формата сдачи.
Модуль torch — то же самое, но дифференцируемое (для обучения и Adam-оптимизации).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import DEFAULT_OPTICS, Optics

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Общие вспомогательные функции
# ---------------------------------------------------------------------------
def centered_coords(size: int, pitch: float) -> np.ndarray:
    """Координаты пикселей вдоль оси, ноль в пикселе size//2 (как у fftshift)."""
    return (np.arange(size) - size // 2) * pitch


def chirp(size: int, pitch: float, wavelength: float, z: float) -> np.ndarray:
    """Квадратичная фаза Q(u, v) = exp(jπ/(λz)·(u²+v²)), complex128."""
    c = centered_coords(size, pitch)
    r2 = c[None, :] ** 2 + c[:, None] ** 2
    return np.exp(1j * np.pi / (wavelength * z) * r2)


def chirps(optics: Optics = DEFAULT_OPTICS, distances: Sequence[float] | None = None) -> np.ndarray:
    zs = optics.distances if distances is None else tuple(distances)
    return np.stack([chirp(optics.size, optics.pitch, optics.wavelength, z) for z in zs])


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Привести фазу к диапазону [−π, π]."""
    return np.angle(np.exp(1j * np.asarray(phase, dtype=np.float64)))


def phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    """Фаза → 8 бит по условию: 0 ↔ −π, 255 ↔ +π."""
    p = (wrap_phase(phase) + np.pi) / TWO_PI * 255.0
    return np.clip(np.round(p), 0, 255).astype(np.uint8)


def uint8_to_phase(code: np.ndarray) -> np.ndarray:
    """8 бит → фаза в радианах: 0 → −π, 255 → +π."""
    return np.asarray(code, dtype=np.float64) / 255.0 * TWO_PI - np.pi


# ---------------------------------------------------------------------------
# numpy: эталонный расчёт
# ---------------------------------------------------------------------------
def fresnel_field_np(phase: np.ndarray, optics: Optics = DEFAULT_OPTICS,
                     distances: Sequence[float] | None = None) -> np.ndarray:
    """Комплексное поле (P, N, N) в плоскостях восстановления для фазового ДОЭ (N, N)."""
    phase = np.asarray(phase, dtype=np.float64)
    if phase.shape != (optics.size, optics.size):
        raise ValueError(f"phase must be {(optics.size, optics.size)}, got {phase.shape}")
    h = np.exp(1j * phase)
    q = chirps(optics, distances)
    return np.fft.fftshift(np.fft.fft2(h[None] * q, norm="ortho"), axes=(-2, -1))


def fresnel_modulus_np(phase: np.ndarray, optics: Optics = DEFAULT_OPTICS,
                       distances: Sequence[float] | None = None) -> np.ndarray:
    """Модуль поля |K| (P, N, N) — то, что сохраняется в восстановленные PNG."""
    return np.abs(fresnel_field_np(phase, optics, distances))


def fresnel_direct_np(phase: np.ndarray, optics: Optics, z: float) -> np.ndarray:
    """Прямое суммирование по формуле (O(N⁴)), только для проверки на малых N.

    K(x, y) ∝ Σ_{u,v} H(u,v)·Q(u,v)·exp(−2πj(ux + vy)/(λz)),
    x, y — центрированная сетка с шагом λz/(NΔx). Возвращает модуль, нормированный как ortho.
    """
    n = optics.size
    u = centered_coords(n, optics.pitch)
    x = centered_coords(n, optics.output_pitch(z))
    hq = np.exp(1j * np.asarray(phase, dtype=np.float64)) * chirp(n, optics.pitch, optics.wavelength, z)
    kern = np.exp(-2j * np.pi * np.outer(x, u) / (optics.wavelength * z))  # (x, u)
    field = kern @ hq @ kern.T
    return np.abs(field) / n


# ---------------------------------------------------------------------------
# torch: дифференцируемая версия
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
except ImportError:  # numpy-часть работает без torch
    torch = None
    nn = None


if torch is not None:

    class FresnelSingleFFT(nn.Module):
        """Дифференцируемый расчёт |K_i| для одной фазовой маски на P расстояниях.

        forward(phase): phase (B, N, N) или (B, 1, N, N), вещественная, радианы
                        → модули (B, P, N, N), float32.
        field(phase):   комплексные поля (B, P, N, N).
        back(field):    сопряжённая операция (обратное распространение в плоскость ДОЭ),
                        (B, P, N, N) → (B, P, N, N). back(field(φ)) == exp(jφ) в каждой плоскости.
        """

        def __init__(self, optics: Optics = DEFAULT_OPTICS, distances: Sequence[float] | None = None):
            super().__init__()
            self.optics = optics
            self.distances = tuple(optics.distances if distances is None else distances)
            q = torch.from_numpy(chirps(optics, self.distances)).to(torch.complex64)
            # complex-буфер не приводится .half()/.float(), а переезжает на устройство вместе с модулем
            self.register_buffer("q", q, persistent=False)

        @property
        def num_planes(self) -> int:
            return len(self.distances)

        def field(self, phase: "torch.Tensor") -> "torch.Tensor":
            if phase.dim() == 4:
                phase = phase[:, 0]
            if phase.dim() != 3:
                raise ValueError("phase must be (B, N, N) or (B, 1, N, N)")
            phase = phase.float()
            h = torch.polar(torch.ones_like(phase), phase)            # exp(jφ), complex64
            u = torch.fft.fft2(h[:, None] * self.q, norm="ortho")       # (B, P, N, N)
            return torch.fft.fftshift(u, dim=(-2, -1))

        def forward(self, phase: "torch.Tensor") -> "torch.Tensor":
            return self.field(phase).abs()

        def back(self, field: "torch.Tensor") -> "torch.Tensor":
            u = torch.fft.ifftshift(field, dim=(-2, -1))
            return torch.fft.ifft2(u, norm="ortho") * self.q.conj()
