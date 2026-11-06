"""Дифференцируемые метрики и функция потерь для многоплоскостного синтеза.

SSIM здесь повторяет skimage.metrics.structural_similarity (окно 7×7, равномерное,
выборочная ковариация, усреднение только по окнам целиком внутри изображения),
поэтому сеть оптимизирует ровно ту величину, по которой её будут оценивать.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def fit_scale(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Оптимальный масштаб s = <pred, target> / <pred, pred> по последним двум осям."""
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
    """SSIM как в skimage (по умолчанию). a, b: (..., H, W). Возвращает (...)."""
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


def ssim_global_t(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0,
                  k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    """ИСС со слайда 3: одно окно на всё изображение (средние и дисперсии всего кадра)."""
    a, b = a.float(), b.float()
    ma, mb = a.mean((-2, -1)), b.mean((-2, -1))
    va = (a - ma[..., None, None]).square().mean((-2, -1))
    vb = (b - mb[..., None, None]).square().mean((-2, -1))
    cov = ((a - ma[..., None, None]) * (b - mb[..., None, None])).mean((-2, -1))
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    return (2 * ma * mb + c1) * (2 * cov + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


def max_normalize(x: torch.Tensor) -> torch.Tensor:
    """Нормировка на 1 делением на максимум по последним двум осям (протокол жюри)."""
    return x / x.amax((-2, -1), keepdim=True).clamp_min(EPS)


def multiplane_loss(modulus: torch.Tensor, target: torch.Tensor, *, w_mse: float = 1.0,
                    w_pcc: float = 1.0, w_ssim: float = 1.0, worst: float = 0.5,
                    temperature: float = 0.1, ssim_norm: str = "max",
                    ssim_global: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Потери по всем плоскостям.

    modulus, target: (B, P, N, N); target — амплитуда исходного изображения в [0, 1].
    Для каждой плоскости:
        L_i = w_mse·MSE(s_i·|K_i|, A_i)/mean(A_i²) + w_pcc·(1 − PCC_i) + w_ssim·(1 − SSIM_i),
    где s_i — оптимальный масштаб (MSE оценивает форму, а не яркость). SSIM считается так же,
    как у жюри: |K_i| и A_i делятся каждый на свой максимум (ssim_norm="max"); градиент через
    максимум прижимает яркие спеклы, которые иначе «затемняют» всё изображение.
    SSIM-член = (1 − ssim_global)·локальный SSIM (skimage) + ssim_global·глобальный ИСС со слайда 3. Один локальный
    SSIM оптимизатор «обманывает»: делает картинку почти чёрной с одной яркой точкой — на чёрном
    фоне окна совпадают. Глобальный ИСС через яркостную составляющую это штрафует.
    ssim_norm="lsq" — прежний вариант: clip(s_i·|K_i|, 0, 1) против A_i, только локальный SSIM.
    Плоскости складываются как (1 − worst)·среднее + worst·soft-max-взвешенная сумма:
    худшая плоскость получает больший вес и не «умирает».

    Возвращает (loss на каждый пример (B,), словарь средних по батчу диагностик).
    """
    modulus = modulus.float()
    target = target.float()
    s = fit_scale(modulus, target)
    scaled = s * modulus
    mse = (scaled - target).square().mean((-2, -1)) / target.square().mean((-2, -1)).clamp_min(EPS)
    pcc = pcc_t(modulus, target)
    if ssim_norm == "max":
        mn, tn = max_normalize(modulus), max_normalize(target)
        ssim = (1.0 - ssim_global) * ssim_t(mn, tn) + ssim_global * ssim_global_t(mn, tn)
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
    """Быстрые PCC и SSIM (B, P) без квантования, по протоколу жюри — для валидации при обучении."""
    m, t = modulus.float(), target.float()
    return pcc_t(m, t), ssim_t(max_normalize(m), max_normalize(t))
