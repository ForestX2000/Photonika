"""Метрики качества и перевод модуля поля в 8 бит.

Протокол сравнения (ответ организаторов):
  1) сравниваются амплитуды (модуль поля |K|), а не интенсивности;
  2) каждое изображение перед сравнением нормируется на 1 (делится на свой максимум).

Метрики считаются по 8-битным изображениям (как в формате сдачи), после нормировки на максимум:

* PCC — коэффициент корреляции Пирсона, формула со слайда 3 (от нормировки не зависит);
* SSIM — skimage.metrics.structural_similarity, data_range = 1
  (окно 7×7, равномерное, K1 = 0,01, K2 = 0,03 — значения skimage по умолчанию);
* SSIM_global — формула ИСС со слайда 3, посчитанная по всему изображению одним окном,
  c1 = 0,01², c2 = 0,03². Приводим дополнительно, пока не уточнено, какой SSIM у жюри.

Модуль поля сохраняется в 0–255 делением на максимум (mode="max") — тогда сохранённый PNG,
нормированный жюри на 1, совпадает с тем, по чему считаем мы.
"""
from __future__ import annotations

import numpy as np

NORM_MODES = ("max", "lsq", "p99.5")
DEFAULT_NORM = "max"


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Коэффициент корреляции Пирсона (КК со слайда 3)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """SSIM из scikit-image (стандартная локальная версия)."""
    from skimage.metrics import structural_similarity

    return float(structural_similarity(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                                       data_range=data_range))


def ssim_global(a: np.ndarray, b: np.ndarray, data_range: float = 255.0,
                k1: float = 0.01, k2: float = 0.03) -> float:
    """ИСС со слайда 3: одно окно на всё изображение."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float((2 * ma * mb + c1) * (2 * cov + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2)))


def modulus_to_uint8(modulus: np.ndarray, target_u8: np.ndarray | None = None,
                     mode: str = DEFAULT_NORM) -> np.ndarray:
    """Модуль поля |K| (N, N) → 8-битное изображение.

    max    — деление на максимум (протокол организаторов, по умолчанию).
    lsq    — масштаб по МНК к исходному: s = <|K|, A> / <|K|, |K|>, затем clip(s·|K|, 0, 255);
             светлее, но с насыщением ярких пикселей. Требует target_u8. Для диагностики.
    p99.5  — деление на 99,5-й перцентиль с насыщением самых ярких 0,5 % пикселей.
    """
    m = np.asarray(modulus, dtype=np.float64)
    if mode == "lsq":
        if target_u8 is None:
            raise ValueError("lsq normalisation needs the source image")
        t = np.asarray(target_u8, dtype=np.float64)
        denom = (m * m).sum()
        scaled = m * ((m * t).sum() / denom) if denom > 0 else m
    elif mode == "max":
        peak = m.max()
        scaled = m / peak * 255.0 if peak > 0 else m
    elif mode == "p99.5":
        ref = np.percentile(m, 99.5)
        scaled = m / ref * 255.0 if ref > 0 else m
    else:
        raise ValueError(f"unknown normalisation {mode!r}; choose from {NORM_MODES}")
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def normalize_max(img: np.ndarray) -> np.ndarray:
    """Нормировка на 1: деление на максимум (протокол организаторов)."""
    a = np.asarray(img, dtype=np.float64)
    peak = a.max()
    return a / peak if peak > 0 else a


def pair_metrics(source_u8: np.ndarray, recon_u8: np.ndarray) -> dict:
    """PCC и SSIM пары «исходное — восстановленное», оба изображения нормированы на свой максимум."""
    a, b = normalize_max(source_u8), normalize_max(recon_u8)
    return {
        "pcc": pcc(a, b),
        "ssim": ssim(a, b, data_range=1.0),
        "ssim_global": ssim_global(a, b, data_range=1.0),
    }


def summarize(rows: list[dict]) -> dict:
    """Среднее по плоскостям для каждого ключа метрик."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
