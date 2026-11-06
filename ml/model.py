"""Нейросеть «4 изображения → фаза ДОЭ».

Связь «плоскость ДОЭ ↔ плоскость изображения» — это FFT, она глобальна, и свёрточной
сети трудно выучить её с нуля. Поэтому (как в DeepCGH, Eybposh et al., 2020) сеть
получает на вход не только сами изображения, но и «физические» каналы: каждая цель
обратно распространяется в плоскость ДОЭ с фиксированной псевдослучайной фазой ψ_i:

    B_i = IFFT(A_i · e^{jψ_i}) · Q_i*,    S = Σ_i B_i.

∠S — это ровно одна итерация многоплоскостного GS. Сеть предсказывает поправку к ней
в виде комплексного числа (a + jb), итоговая фаза φ = ∠((a + jb) · e^{j∠S}) ∈ [−π, π]
(через atan2, без разрыва на границе ±π). Голова инициализирована в (1, 0), поэтому
необученная сеть сразу выдаёт ∠S, а обучение только улучшает этот старт.

Опция unroll = K (развёрнутая сеть, «learned GS»): после U-Net выполняется K стадий
«физический шаг GS (4 FFT + 4 IFFT) → маленькая CNN-поправка». При инициализации это
в точности K итераций GS от выхода U-Net; обучение подстраивает поправки. Это по-прежнему
один проход сети без итераций во время инференса — время фиксировано и мало.

Режимы входа (input_mode):
    raw       — U-Net в плоскости ДОЭ, вход — только 4 изображения (4 канала), старт с нулевой фазы;
    backprop  — U-Net в плоскости ДОЭ, вход — 4 изображения + Re/Im B_i (8) + cos/sin ∠S (2) = 14 каналов;
    imgphase  — U-Net в плоскости изображений (как DeepCGH): по 4 изображениям и cos/sin ψ_i (12 каналов)
                предсказывает фазу поля ψ'_i в каждой из 4 плоскостей, затем
                φ = ∠Σ_i IFFT(A_i·e^{jψ'_i})·Q_i*. Признаки и цели здесь совпадают пространственно,
                свёрткам это удобнее, чем шумоподобная фаза в плоскости ДОЭ. Старт тоже ∠S.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import normalize_amplitude
from .config import DEFAULT_OPTICS, NUM_PLANES, Optics
from .fresnel import FresnelSingleFFT

INPUT_MODES = ("raw", "backprop", "imgphase")


def _groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """Две свёртки 3×3 с GroupNorm и SiLU. Паддинг циклический: FFT периодичен."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(_groups(c_out), c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(_groups(c_out), c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """U-Net с levels понижениями разрешения. Выход — комплексная поправка (a, b),
    при инициализации ровно (1, 0); gain ускоряет рост поправки при обучении."""

    def __init__(self, c_in: int, base: int, levels: int = 4, gain: float = 10.0, pairs: int = 1):
        super().__init__()
        self.pairs = pairs
        widths = [base * min(2 ** i, 8) for i in range(levels + 1)]
        self.enc = nn.ModuleList([ConvBlock(c_in, widths[0])] +
                                 [ConvBlock(widths[i], widths[i + 1]) for i in range(levels)])
        self.dec = nn.ModuleList([ConvBlock(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(levels))])
        self.head = nn.Conv2d(widths[0], 2 * pairs, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gain = gain

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for block in self.dec:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        ab = self.head(x).float() * self.gain
        if self.pairs > 1:  # (B, pairs, N, N) для каждой из a, b
            ab = ab.view(ab.shape[0], self.pairs, 2, *ab.shape[-2:])
            return 1.0 + ab[:, :, 0], ab[:, :, 1]
        return 1.0 + ab[:, 0], ab[:, 1]


def rotate(a: torch.Tensor, b: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """∠((a + jb)·e^{jφ}) через atan2 — результат в [−π, π]."""
    c, s = torch.cos(phase), torch.sin(phase)
    return torch.atan2(a * s + b * c, a * c - b * s)


class PhysicsFeatures(nn.Module):
    """Строит входные каналы сети и базовую фазу ∠S."""

    def __init__(self, prop: FresnelSingleFFT, mode: str = "backprop", seed: int = 1234):
        super().__init__()
        if mode not in INPUT_MODES:
            raise ValueError(f"input mode must be one of {INPUT_MODES}")
        self.prop = prop
        self.mode = mode
        p, n = prop.num_planes, prop.optics.size
        gen = torch.Generator().manual_seed(seed)
        psi = torch.rand((p, n, n), generator=gen) * 2 * math.pi
        self.register_buffer("psi", torch.polar(torch.ones_like(psi), psi), persistent=False)

    @property
    def channels(self) -> int:
        p = self.prop.num_planes
        return {"raw": p, "backprop": p + 2 * p + 2, "imgphase": p + 2 * p}[self.mode]

    def forward(self, amps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = amps.float() * 2.0 - 1.0
        if self.mode == "raw":
            return x, torch.zeros_like(x[:, 0])
        if self.mode == "imgphase":
            psi = self.psi.expand(x.shape[0], -1, -1, -1)
            return torch.cat([x, psi.real, psi.imag], dim=1), None
        amp = normalize_amplitude(amps)
        back = self.prop.back(amp * self.psi)                  # (B, P, N, N), complex
        s = back.sum(1)
        unit = torch.sgn(s)
        feats = torch.cat([x, back.real, back.imag, unit.real[:, None], unit.imag[:, None]], dim=1)
        return feats, torch.angle(s)


class GSStage(nn.Module):
    """Один шаг многоплоскостного GS + обучаемая поправка к получившейся фазе."""

    STAGE_CH = 6  # cos/sin текущей фазы, cos/sin фазы после шага GS, |S| и средняя ошибка модуля

    def __init__(self, prop: FresnelSingleFFT, base: int = 16, levels: int = 3, gain: float = 10.0):
        super().__init__()
        self.prop = prop
        self.net = UNet(self.STAGE_CH, base, levels, gain)

    def forward(self, phase: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
        field = self.prop.field(phase)
        err = (field.abs() - amp).abs().mean(1, keepdim=True)    # где цель воспроизводится хуже
        s = self.prop.back(amp * torch.sgn(field)).sum(1)
        gs_phase = torch.angle(s)
        mag = s.abs()[:, None]
        mag = mag / mag.mean((-2, -1), keepdim=True).clamp_min(1e-6)
        feats = torch.cat([torch.cos(phase)[:, None], torch.sin(phase)[:, None],
                           torch.cos(gs_phase)[:, None], torch.sin(gs_phase)[:, None], mag, err], dim=1)
        a, b = self.net(feats)
        return rotate(a, b, gs_phase)


class MultiPlaneUNet(nn.Module):
    """U-Net на 4 уровня (128 → 64 → 32 → 16 → 8) [+ unroll стадий GS]. Выход — фаза (B, N, N) в [−π, π]."""

    def __init__(self, base: int = 32, input_mode: str = "backprop", optics: Optics = DEFAULT_OPTICS,
                 num_planes: int = NUM_PLANES, unroll: int = 0, stage_base: int = 16):
        super().__init__()
        if optics.size % 16:
            raise ValueError("size must be divisible by 16 (4 pooling levels)")
        if len(optics.distances) != num_planes:
            raise ValueError("optics.distances must match num_planes")
        self.base, self.input_mode, self.unroll, self.stage_base = base, input_mode, unroll, stage_base
        prop = FresnelSingleFFT(optics)
        self.features = PhysicsFeatures(prop, input_mode)
        self.prop = prop
        pairs = num_planes if input_mode == "imgphase" else 1
        self.unet = UNet(self.features.channels, base, levels=4, pairs=pairs)
        self.stages = nn.ModuleList([GSStage(prop, stage_base) for _ in range(unroll)])

    def forward(self, amps: torch.Tensor) -> torch.Tensor:
        """amps: (B, P, N, N) — исходные изображения в [0, 1]. Возвращает фазу (B, N, N)."""
        x, base_phase = self.features(amps)
        a, b = self.unet(x)
        amp = normalize_amplitude(amps)
        if self.input_mode == "imgphase":
            psi = rotate(a, b, torch.angle(self.features.psi))          # (B, P, N, N)
            phase = torch.angle(self.prop.back(amp * torch.polar(torch.ones_like(psi), psi)).sum(1))
        else:
            phase = rotate(a, b, base_phase)
        if self.stages:
            for stage in self.stages:
                phase = stage(phase, amp)
        return phase

    def config(self) -> dict:
        return {"base": self.base, "input_mode": self.input_mode, "unroll": self.unroll,
                "stage_base": self.stage_base}


def build_model(config: dict, optics: Optics = DEFAULT_OPTICS) -> MultiPlaneUNet:
    return MultiPlaneUNet(base=int(config["base"]), input_mode=config["input_mode"], optics=optics,
                          unroll=int(config.get("unroll", 0)), stage_base=int(config.get("stage_base", 16)))


def load_model(path, device="cpu") -> tuple[MultiPlaneUNet, dict]:
    """Загрузка весов с проверкой версии физики и оптики."""
    from .config import PHYSICS_VERSION

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("physics_version") != PHYSICS_VERSION:
        raise ValueError(f"checkpoint physics {ckpt.get('physics_version')!r} != {PHYSICS_VERSION!r}")
    optics = Optics(**{k: (tuple(v) if k == "distances" else v) for k, v in ckpt["optics"].items()})
    model = build_model(ckpt["model"], optics)
    model.load_state_dict(ckpt["net"])
    return model.to(device).eval(), ckpt
