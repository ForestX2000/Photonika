"""Compatibility implementation of the dual-resolution complex-valued model.

Existing PDR checkpoints retain their parameter names and shapes. Quality must
be measured on held-out images; this architecture does not imply a fixed SSIM.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .baselines import normalize_amplitude
from .config import DEFAULT_OPTICS, NUM_PLANES, Optics
from .fresnel import FresnelSingleFFT
from .legacy_model import GSStage

INPUT_MODES = ("raw", "backprop", "imgphase")

def _groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0: return g
    return 1

class ComplexConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv_real = nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="circular", bias=False)
        self.conv_imag = nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="circular", bias=False)
        self.gn = nn.GroupNorm(_groups(c_out), c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x.real, x.imag
        out_real = self.conv_real(real) - self.conv_imag(imag)
        out_imag = self.conv_real(imag) + self.conv_imag(real)
        out = torch.complex(out_real, out_imag)

        mag = torch.abs(out)
        # A normalized magnitude may be negative. Dividing by its clipped
        # value amplified half of the activations by 1000. Normalize the
        # actual magnitude and use a positive learned magnitude instead.
        out = out / mag.clamp_min(1e-6) * F.softplus(self.gn(mag))

        return torch.complex(F.silu(out.real), F.silu(out.imag))

class ComplexUNetBranch(nn.Module):
    """Одна ветка U-Net (используется для high-res и low-res)."""
    def __init__(self, c_in: int, base: int, levels: int = 4):
        super().__init__()
        widths = [base * min(2 ** i, 8) for i in range(levels + 1)]
        self.enc = nn.ModuleList([ComplexConvBlock(c_in, widths[0])] +
                                 [ComplexConvBlock(widths[i], widths[i + 1]) for i in range(levels)])
        self.dec = nn.ModuleList([ComplexConvBlock(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(levels))])
        self.out_ch = widths[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            if i > 0:
                x = torch.complex(F.max_pool2d(x.real, 2), F.max_pool2d(x.imag, 2))
            x = block(x)
            skips.append(x)
        x = skips.pop()
        for block in self.dec:
            skip = skips.pop()
            x = torch.complex(
                F.interpolate(x.real, size=skip.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(x.imag, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            )
            x = block(torch.cat([x, skip], dim=1))
        return x

class PDRNet(nn.Module):
    """Phase Dual-Resolution Network: две ветки (128 и 64) + слияние."""
    def __init__(self, c_in: int, base: int = 32, levels: int = 4, gain: float = 10.0):
        super().__init__()
        # Ветка 1: полное разрешение
        self.branch_high = ComplexUNetBranch(c_in, base, levels)
        # Ветка 2: низкое разрешение (base//2)
        self.branch_low = ComplexUNetBranch(c_in, base // 2, levels)

        ch_high = self.branch_high.out_ch
        ch_low = self.branch_low.out_ch

        # Слияние: high (ch_high) + low (ch_low) -> conv -> out (ch_high)
        self.fuse = nn.Conv2d(ch_high * 2 + ch_low * 2, ch_high * 2, 1)
        self.head = nn.Conv2d(ch_high * 2, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ветка 1: полное разрешение
        out_high = self.branch_high(x)  # (B, C, H, W) complex

        # Ветка 2: downsampled (H/2, W/2)
        x_low = torch.complex(F.avg_pool2d(x.real, 2), F.avg_pool2d(x.imag, 2))
        out_low = self.branch_low(x_low)  # (B, C_low, H/2, W/2) complex

        # Upsample low до full
        out_low_up = torch.complex(
            F.interpolate(out_low.real, size=out_high.shape[-2:], mode="bilinear", align_corners=False),
            F.interpolate(out_low.imag, size=out_high.shape[-2:], mode="bilinear", align_corners=False)
        )

        # Слияние: (high.real, high.imag, low.real, low.imag) -> fuse -> (real, imag)
        fused = self.fuse(torch.cat([out_high.real, out_high.imag, out_low_up.real, out_low_up.imag], dim=1))
        ab = self.head(fused) * self.gain
        return torch.complex(1.0 + ab[:, 0], ab[:, 1])

class PhysicsFeatures(nn.Module):
    def __init__(self, prop: FresnelSingleFFT, mode: str = "backprop", seed: int = 1234):
        super().__init__()
        if mode not in INPUT_MODES:
            raise ValueError(f"input mode must be one of {INPUT_MODES}")
        self.prop, self.mode = prop, mode
        p, n = prop.num_planes, prop.optics.size
        gen = torch.Generator().manual_seed(seed)
        psi = torch.rand((p, n, n), generator=gen) * 2 * math.pi
        self.register_buffer("psi", torch.polar(torch.ones_like(psi), psi), persistent=False)

    @property
    def channels(self) -> int:
        p = self.prop.num_planes
        return {"raw": p, "backprop": p + p + 1, "imgphase": p + p}[self.mode]

    def forward(self, amps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = amps.float() * 2.0 - 1.0
        if self.mode == "raw":
            return torch.complex(x, torch.zeros_like(x)), torch.zeros_like(x[:, 0])
        if self.mode == "imgphase":
            psi = self.psi.expand(x.shape[0], -1, -1, -1)
            return torch.cat([torch.complex(x, torch.zeros_like(x)), psi], dim=1), None

        amp = normalize_amplitude(amps)
        back = self.prop.back(amp * self.psi)  # (B, P, N, N) complex
        s = back.sum(1)                        # (B, N, N) complex
        # Защита от sgn(0) = 0
        s_safe = torch.where(torch.abs(s) < 1e-6, torch.ones_like(s) * 1e-6, s)
        unit = torch.sgn(s_safe)               # (B, N, N) complex

        feats = torch.cat([torch.complex(x, torch.zeros_like(x)), back, unit.unsqueeze(1)], dim=1)
        return feats, torch.angle(s)

def rotate(a, b, phase):
    c, s = torch.cos(phase), torch.sin(phase)
    return torch.atan2(a * s + b * c, a * c - b * s)

class MultiPlaneUNet(nn.Module):
    def __init__(self, base: int = 32, input_mode: str = "backprop", optics: Optics = DEFAULT_OPTICS,
                 num_planes: int = NUM_PLANES, unroll: int = 0, stage_base: int = 16):
        super().__init__()
        if isinstance(base, bool) or not isinstance(base, int) or base < 4:
            raise ValueError("base must be an integer >= 4")
        if optics.size % 32:
            raise ValueError("PDR size must be divisible by 32 (dual-resolution pooling)")
        if len(optics.distances) != num_planes:
            raise ValueError("optics.distances must match num_planes")
        if isinstance(unroll, bool) or not isinstance(unroll, int) or unroll < 0:
            raise ValueError("unroll must be a nonnegative integer")
        if isinstance(stage_base, bool) or not isinstance(stage_base, int) or stage_base < 2:
            raise ValueError("stage_base must be an integer >= 2")
        self.base, self.input_mode, self.unroll, self.stage_base = base, input_mode, unroll, stage_base
        prop = FresnelSingleFFT(optics)
        self.features = PhysicsFeatures(prop, input_mode)
        self.prop = prop
        # PDRNet вместо обычного U-Net
        self.unet = PDRNet(self.features.channels, base, levels=4)
        self.stages = nn.ModuleList([GSStage(prop, stage_base) for _ in range(unroll)])

    def forward(self, amps: torch.Tensor) -> torch.Tensor:
        x, base_phase = self.features(amps)
        ab = self.unet(x)  # (B, N, N) complex
        amp = normalize_amplitude(amps)
        if self.input_mode == "imgphase":
            # The legacy PDR head predicts one shared phase correction. Make
            # its plane axis explicit so any batch size broadcasts correctly.
            psi = rotate(ab.real[:, None], ab.imag[:, None], torch.angle(self.features.psi))
            phase = torch.angle(self.prop.back(amp * torch.polar(torch.ones_like(psi), psi)).sum(1))
        else:
            phase = rotate(ab.real, ab.imag, base_phase)
        for stage in self.stages:
            phase = stage(phase, amp)
        return phase

    def config(self) -> dict:
        return {"architecture": "pdr", "base": self.base, "input_mode": self.input_mode, "unroll": self.unroll,
                "stage_base": self.stage_base, "complex": True, "pdr": True}

def build_model(config: dict, optics: Optics = DEFAULT_OPTICS) -> MultiPlaneUNet:
    return MultiPlaneUNet(base=int(config["base"]), input_mode=config["input_mode"], optics=optics,
                          unroll=int(config.get("unroll", 0)), stage_base=int(config.get("stage_base", 16)))

def load_model(path, device="cpu"):
    from .config import PHYSICS_VERSION
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("physics_version") != PHYSICS_VERSION:
        raise ValueError("checkpoint physics mismatch")
    optics = Optics(**{k: (tuple(v) if k == "distances" else v) for k, v in ckpt["optics"].items()})
    model = build_model(ckpt["model"], optics)
    model.load_state_dict(ckpt["net"])
    return model.to(device).eval(), ckpt
