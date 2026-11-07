"""Fixed-depth, physics-aware hologram synthesis with learned GS projections.

Attention at the U-Net bottleneck supplies global context. Every refinement
stage operates on complex residuals backpropagated into DOE coordinates.
Zero heads, uniform plane weights and unit relaxation reproduce GS exactly
before training. This is a project-specific architecture, not a published SOTA
implementation or a claim of improved metrics without a trained benchmark.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .baselines import normalize_amplitude
from .config import DEFAULT_OPTICS, NUM_PLANES, Optics
from .fresnel import FresnelSingleFFT
from .legacy_model import ConvBlock, INPUT_MODES, PhysicsFeatures


def unit_field(phase: torch.Tensor) -> torch.Tensor:
    return torch.polar(torch.ones_like(phase), phase)


def wrap(phase: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phase), torch.cos(phase))


class GlobalContext(nn.Module):
    """Global token mixing only at the smallest spatial resolution."""

    def __init__(self, channels: int):
        super().__init__()
        heads = next(h for h in (8, 4, 2, 1) if channels % h == 0)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(),
                                nn.Linear(channels * 2, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        mixed, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + mixed
        tokens = tokens + self.ff(self.ff_norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class PhaseUNet(nn.Module):
    """Circular convolutions, global bottleneck, and bounded phase corrections."""

    def __init__(self, channels: int, base: int, levels: int, outputs: int):
        super().__init__()
        widths = [base * min(2 ** i, 8) for i in range(levels + 1)]
        self.enc = nn.ModuleList([ConvBlock(channels, widths[0])] +
                                 [ConvBlock(widths[i], widths[i + 1]) for i in range(levels)])
        self.context = GlobalContext(widths[-1])
        self.dec = nn.ModuleList([ConvBlock(widths[i + 1] + widths[i], widths[i])
                                 for i in reversed(range(levels))])
        self.head = nn.Conv2d(base, outputs, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        x = self.context(skips.pop())
        for block in self.dec:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return math.pi * torch.tanh(self.head(x).float())


class LearnedProjection(nn.Module):
    """Weighted GS, learned relaxation and a phase correction in DOE coordinates."""

    def __init__(self, prop: FresnelSingleFFT, base: int):
        super().__init__()
        self.prop = prop
        p = prop.num_planes
        self.net = PhaseUNet(5 + 3 * p, base, levels=3, outputs=1)
        self.plane_weights = nn.Sequential(nn.Linear(p, base), nn.SiLU(), nn.Linear(base, p))
        nn.init.zeros_(self.plane_weights[-1].weight)
        nn.init.zeros_(self.plane_weights[-1].bias)
        self.relaxation = nn.Parameter(torch.zeros(()))

    def forward(self, phase: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
        field = self.prop.field(phase)
        amplitude = field.abs()
        projected = self.prop.back(amp * torch.sgn(field))
        errors = (amplitude - amp).square().mean((-2, -1))
        weights = torch.softmax(self.plane_weights(torch.log1p(errors)), dim=1)
        summed = (projected * weights[:, :, None, None]).sum(1)
        gs_phase = torch.angle(summed)
        current = unit_field(phase)
        # A complex residual per plane, rotated relative to the current DOE.
        residual = (current[:, None] - projected) * current.conj()[:, None]
        residual = residual / (1.0 + residual.abs().mean((-2, -1), keepdim=True))
        mag = summed.abs()[:, None]
        mag = torch.log1p(mag / mag.mean((-2, -1), keepdim=True).clamp_min(1e-6))
        features = torch.cat([current.real[:, None], current.imag[:, None],
                              torch.cos(gs_phase)[:, None], torch.sin(gs_phase)[:, None], mag,
                              residual.real, residual.imag,
                              torch.log1p(errors)[:, :, None, None].expand_as(amp)], dim=1)
        delta = self.net(features)[:, 0]
        step = torch.angle(unit_field(gs_phase - phase))
        relaxed = phase + (2 * self.relaxation.sigmoid()) * step
        return wrap(relaxed + delta)


class PhysicsUnrolledNet(nn.Module):
    def __init__(self, base: int = 32, input_mode: str = "imgphase", optics: Optics = DEFAULT_OPTICS,
                 num_planes: int = NUM_PLANES, unroll: int = 4, stage_base: int = 16):
        super().__init__()
        if base < 2 or stage_base < 2:
            raise ValueError("base and stage_base must be at least 2")
        if optics.size % 16:
            raise ValueError("size must be divisible by 16")
        if len(optics.distances) != num_planes:
            raise ValueError("optics.distances must match num_planes")
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input mode must be one of {INPUT_MODES}")
        if unroll < 0:
            raise ValueError("unroll must be nonnegative")
        self.base, self.input_mode, self.unroll, self.stage_base = base, input_mode, unroll, stage_base
        self.prop = FresnelSingleFFT(optics)
        self.features = PhysicsFeatures(self.prop, input_mode)
        self.unet = PhaseUNet(self.features.channels, base, levels=4,
                              outputs=num_planes if input_mode == "imgphase" else 1)
        self.stages = nn.ModuleList([LearnedProjection(self.prop, stage_base) for _ in range(unroll)])

    def forward(self, amps: torch.Tensor) -> torch.Tensor:
        expected = (self.prop.num_planes, self.prop.optics.size, self.prop.optics.size)
        if amps.ndim != 4 or tuple(amps.shape[1:]) != expected or amps.shape[0] < 1:
            raise ValueError(f"amps must be (B, {expected[0]}, {expected[1]}, {expected[2]}) with B >= 1")
        amps = amps.float()
        x, initial = self.features(amps)
        correction = self.unet(x)
        amp = normalize_amplitude(amps)
        if self.input_mode == "imgphase":
            psi = self.features.psi * unit_field(correction)
            phase = torch.angle(self.prop.back(amp * psi).sum(1))
        else:
            phase = wrap(initial + correction[:, 0])
        for stage in self.stages:
            phase = stage(phase, amp)
        return phase

    def config(self) -> dict:
        return dict(architecture="physics-unrolled", model_version=1, base=self.base,
                    input_mode=self.input_mode, unroll=self.unroll, stage_base=self.stage_base)
