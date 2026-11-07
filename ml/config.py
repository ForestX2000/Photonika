"""Параметры оптической системы из условий кейса.

Все величины в СИ (метры). Значения по умолчанию взяты из условия:
ДОЭ 128×128, λ = 532 нм, пиксель ПВМС 10 мкм, расстояния 24,1 / 25,3 / 26,5 / 27,7 мм.
i-е входное изображение восстанавливается на i-м расстоянии.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

WAVELENGTH = 532e-9
PIXEL_PITCH = 10e-6
SIZE = 128
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
NUM_PLANES = len(DISTANCES)

# Версия постановки: меняется, если меняется физика или формат входа сети.
# Чекпоинт с другой версией загружать нельзя.
PHYSICS_VERSION = "fresnel-single-fft-4plane-v1"


@dataclass(frozen=True)
class Optics:
    wavelength: float = WAVELENGTH
    pitch: float = PIXEL_PITCH
    size: int = SIZE
    distances: tuple[float, ...] = DISTANCES

    def __post_init__(self) -> None:
        if self.wavelength <= 0 or self.pitch <= 0 or self.size < 8:
            raise ValueError("wavelength, pitch must be positive; size >= 8")
        if not self.distances or any(z <= 0 for z in self.distances):
            raise ValueError("distances must be positive")

    @property
    def critical_distance(self) -> float:
        """z_c = N·Δx²/λ. При z ≥ z_c квадратичная фаза дискретизирована без алиасинга."""
        return self.size * self.pitch ** 2 / self.wavelength

    def output_pitch(self, z: float) -> float:
        """Шаг пикселя в плоскости восстановления для формулы с одним FFT: λz/(NΔx)."""
        return self.wavelength * z / (self.size * self.pitch)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["distances"] = list(self.distances)
        return d


DEFAULT_OPTICS = Optics()
