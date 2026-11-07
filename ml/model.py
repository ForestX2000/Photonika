"""Model dispatch and strict, architecture-aware checkpoint loading.

The original U-Net remains available as MultiPlaneUNet. New training uses the
physics-unrolled family; checkpoints select their own architecture explicitly.
"""
from __future__ import annotations

import torch
from torch import nn

from .config import DEFAULT_OPTICS, PHYSICS_VERSION, Optics
from .legacy_model import INPUT_MODES, MultiPlaneUNet as LegacyUNet

ARCHITECTURES = ("unet", "pdr", "physics-unrolled")


class MultiPlaneUNet(LegacyUNet):
    """Original real-valued U-Net, with unchanged parameter names and shapes."""

    def config(self) -> dict:
        return {"architecture": "unet", **super().config()}


def build_model(config: dict, optics: Optics = DEFAULT_OPTICS) -> nn.Module:
    architecture = config.get("architecture")
    if architecture is None:
        architecture = "pdr" if config.get("pdr") or config.get("complex") else "unet"
    kwargs = dict(base=int(config.get("base", 32)), input_mode=config.get("input_mode", "imgphase"),
                  optics=optics, num_planes=len(optics.distances),
                  unroll=int(config.get("unroll", 4 if architecture == "physics-unrolled" else 0)),
                  stage_base=int(config.get("stage_base", 16)))
    if architecture == "unet":
        return MultiPlaneUNet(**kwargs)
    if architecture == "pdr":
        from .pdr_model import MultiPlaneUNet as PDRNet
        return PDRNet(**kwargs)
    if architecture == "physics-unrolled":
        from .physics_model import PhysicsUnrolledNet, PhysicsUnrolledNetV2
        version = config.get("model_version", 1)
        if version == 1:
            return PhysicsUnrolledNet(**kwargs)
        if version == 2:
            return PhysicsUnrolledNetV2(**kwargs, spectral_modes=int(config.get("spectral_modes", 8)))
        raise ValueError("Unsupported physics-unrolled model_version")
    raise ValueError(f"Unknown architecture {architecture!r}; expected one of {ARCHITECTURES}")


def load_model(path, device="cpu") -> tuple[nn.Module, dict]:
    """Load old and new checkpoints strictly; never accept partial random weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("physics_version") != PHYSICS_VERSION:
        raise ValueError(f"checkpoint physics {ckpt.get('physics_version')!r} != {PHYSICS_VERSION!r}")
    optics = Optics(**{k: tuple(v) if k == "distances" else v for k, v in ckpt["optics"].items()})
    config = dict(ckpt["model"])
    if "architecture" not in config:
        # An older trainer omitted the PDR flags even for complex checkpoints.
        if any(k.startswith("unet.branch_high.") for k in ckpt["net"]):
            config["architecture"] = "pdr"
    model = build_model(config, optics)
    model.load_state_dict(ckpt["net"], strict=True)
    ckpt = {**ckpt, "model": model.config()}
    return model.to(device).eval(), ckpt
