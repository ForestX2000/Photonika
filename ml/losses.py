"""Дифференцируемые метрики и функция потерь для многоплоскостного синтеза."""
from __future__ import annotations
import torch
import torch.nn.functional as F
import torch.nn as nn

EPS = 1e-8

def fit_scale(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = (pred * target).sum((-2, -1), keepdim=True)
    den = (pred * pred).sum((-2, -1), keepdim=True).clamp_min(EPS)
    return num / den

def pcc_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean((-2, -1), keepdim=True)
    b = b - b.mean((-2, -1), keepdim=True)
    num = (a * b).sum((-2, -1))
    den = ((a * a).sum((-2, -1)) * (b * b).sum((-2, -1))).clamp_min(EPS).sqrt()
    return num / den

def ssim_t(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0, win: int = 7,
           k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    shape = a.shape
    a = a.reshape(-1, 1, *shape[-2:]).float()
    b = b.reshape(-1, 1, *shape[-2:]).float()
    np_ = win * win
    cov_norm = np_ / (np_ - 1)

    def mean(x):
        return F.avg_pool2d(x, win, stride=1)

    ux, uy = mean(a), mean(b)
    uxx, uyy, uxy = mean(a * a), mean(b * b), mean(a * b)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return s.mean((-3, -2, -1)).reshape(shape[:-2])

def ms_ssim_t(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0,
              weights: list[float] | None = None) -> torch.Tensor:
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    ssims = []
    for i, w in enumerate(weights):
        if i > 0:
            if a.shape[-1] >= 2 and a.shape[-2] >= 2:
                a = F.avg_pool2d(a, 2, 2)
                b = F.avg_pool2d(b, 2, 2)
            else:
                break
        ssims.append(ssim_t(a, b, data_range))
    if not ssims:
        return ssim_t(a, b, data_range)
    w = weights[:len(ssims)]
    w = [wi / sum(w) for wi in w]
    return sum(wi * s for wi, s in zip(w, ssims))

def max_normalize(x: torch.Tensor) -> torch.Tensor:
    peak = x.amax((-2, -1), keepdim=True)
    return x / peak.clamp_min(1e-3)  # Защита от деления на ноль

def multiplane_loss(modulus: torch.Tensor, target: torch.Tensor, *, w_mse: float = 1.0,
                    w_pcc: float = 1.0, w_ssim: float = 1.0, worst: float = 0.5,
                    temperature: float = 0.1, ssim_norm: str = "max",
                    ssim_global: float = 0.5) -> tuple[torch.Tensor, dict]:
    modulus = modulus.float()
    target = target.float()
    s = fit_scale(modulus, target)
    scaled = s * modulus
    mse = (scaled - target).square().mean((-2, -1)) / target.square().mean((-2, -1)).clamp_min(EPS)
    pcc = pcc_t(modulus, target)
    
    if ssim_norm == "max":
        mn, tn = max_normalize(modulus), max_normalize(target)
        ssim = ms_ssim_t(mn, tn)
    else:
        ssim = ssim_t(scaled.clamp(0.0, 1.0), target)
    
    per_plane = w_mse * mse + w_pcc * (1.0 - pcc) + w_ssim * (1.0 - ssim)      # (B, P)
    weights = torch.softmax(per_plane.detach() / temperature, dim=-1)
    loss = (1.0 - worst) * per_plane.mean(-1) + worst * (weights * per_plane).sum(-1)
    diag = {
        "mse": float(mse.mean().detach()),
        "pcc": float(pcc.mean().detach()),
        "ssim": float(ssim.mean().detach()),
        "pcc_min_plane": float(pcc.detach().mean(0).min()),
    }
    return loss, diag

def plane_metrics(modulus: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, t = modulus.float(), target.float()
    return pcc_t(m, t), ssim_t(max_normalize(m), max_normalize(t))