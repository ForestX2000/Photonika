#!/usr/bin/env python3
"""Независимая проверка результата по PNG-файлам (не импортирует код проекта).

    python tools/verify_submission.py --doe results/doe.png --inputs 1.png 2.png 3.png 4.png \
        [--recon results/recon_1.png ... results/recon_4.png]

Что делает:
  1. читает doe.png (0 → −π, 255 → +π);
  2. для каждого z ∈ {24,1; 25,3; 26,5; 27,7} мм считает модуль поля по формуле Френеля
     с одним FFT: |FFT(exp(jφ)·exp(jπ(u²+v²)/(λz)))|, λ = 532 нм, Δx = 10 мкм;
  3. сохраняемое восстановление — модуль, делённый на максимум (0–255);
  4. по протоколу организаторов сравнивает амплитуды, каждое изображение нормировано на 1
     (делится на свой максимум), и печатает PCC (Пирсон), SSIM (skimage, data_range = 1) и
     глобальный ИСС со слайда 3 — для каждой пары; если переданы присланные recon_*.png —
     метрики и по ним.
Зависимости: numpy, Pillow, scikit-image.
"""
import argparse

import numpy as np
from PIL import Image

LAMBDA, PITCH, N = 532e-9, 10e-6, 128
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)


def read_u8(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"), dtype=np.uint8)
    if a.shape != (N, N):
        raise SystemExit(f"{path}: expected {N}x{N}, got {a.shape}")
    return a


def modulus(phase, z):
    c = (np.arange(N) - N // 2) * PITCH
    q = np.exp(1j * np.pi / (LAMBDA * z) * (c[None, :] ** 2 + c[:, None] ** 2))
    return np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phase) * q, norm="ortho")))


def to_u8(m):
    return np.clip(np.round(m / m.max() * 255), 0, 255).astype(np.uint8)


def norm1(a):
    a = a.astype(np.float64)
    return a / a.max() if a.max() > 0 else a


def pcc(a, b):
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(norm1(a), norm1(b), data_range=1.0))


def ssim_global(a, b, c1=0.01 ** 2, c2=0.03 ** 2):
    a, b = norm1(a), norm1(b)
    ma, mb, cov = a.mean(), b.mean(), ((a - a.mean()) * (b - b.mean())).mean()
    return float((2 * ma * mb + c1) * (2 * cov + c2) / ((ma ** 2 + mb ** 2 + c1) * (a.var() + b.var() + c2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doe", required=True)
    ap.add_argument("--inputs", nargs=4, required=True)
    ap.add_argument("--recon", nargs=4, default=None)
    args = ap.parse_args()
    phase = read_u8(args.doe).astype(np.float64) / 255.0 * 2 * np.pi - np.pi
    sources = [read_u8(p) for p in args.inputs]
    given = [read_u8(p) for p in args.recon] if args.recon else None
    rows = []
    for i, (z, src) in enumerate(zip(DISTANCES, sources)):
        rec = to_u8(modulus(phase, z))
        row = [pcc(src, rec), ssim(src, rec), ssim_global(src, rec)]
        if given:
            row += [pcc(src, given[i]), ssim(src, given[i])]
        rows.append(row)
        extra = f" | присланный recon: PCC {row[3]:.4f} SSIM {row[4]:.4f}" if given else ""
        print(f"z = {z * 1e3:.1f} мм: PCC {row[0]:.4f} SSIM {row[1]:.4f} SSIM_global {row[2]:.4f}{extra}")
    m = np.mean(rows, axis=0)
    print(f"среднее: PCC {m[0]:.4f} SSIM {m[1]:.4f} SSIM_global {m[2]:.4f}"
          + (f" | присланные: PCC {m[3]:.4f} SSIM {m[4]:.4f}" if given else ""))


if __name__ == "__main__":
    main()
