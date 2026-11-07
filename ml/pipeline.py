"""Конвейер в формате сдачи.

    4 PNG → метод синтеза → фаза → doe.png (0 = −π, 255 = +π)
          → doe.png перечитывается с диска → Френель (numpy) на 4 расстояниях
          → recon_1..4.png (модуль поля, 8 бит) → PCC / SSIM по 8-битным изображениям.

Метрики всегда считаются после 8-битного квантования ДОЭ и восстановлений —
ровно так, как их получит жюри из сданных файлов.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DEFAULT_OPTICS, PHYSICS_VERSION, Optics
from .data import load_png, save_png
from .fresnel import fresnel_modulus_np, phase_to_uint8, uint8_to_phase
from .metrics import DEFAULT_NORM, modulus_to_uint8, pair_metrics, summarize

METHODS = ("net", "net+gs", "net+adam", "net+gs+adam", "gs", "gs+adam", "adam")
METHOD_HELP = {
    "net": "один проход нейросети",
    "net+gs": "нейросеть + K итераций GS (быстрое доуточнение)",
    "net+adam": "нейросеть + K шагов Adam (доуточнение)",
    "net+gs+adam": "синтез нейросетью → GS → Adam; выбор лучшего 8-битного ДОЭ",
    "gs": "многоплоскостный Gerchberg–Saxton",
    "gs+adam": "GS, затем Adam по фазе (лучший итерационный эталон)",
    "adam": "Adam по фазе со случайного старта",
}


@dataclass
class Result:
    method: str
    params: dict
    phase: np.ndarray
    doe_u8: np.ndarray
    sources_u8: np.ndarray
    recon_u8: np.ndarray
    modulus: np.ndarray
    rows: list[dict]
    summary: dict
    time_ms: float
    device: str
    extra: dict = field(default_factory=dict)

    def to_json(self, optics: Optics = DEFAULT_OPTICS, norm: str = DEFAULT_NORM) -> dict:
        return {
            "method": self.method,
            "method_description": METHOD_HELP.get(self.method, ""),
            "params": self.params,
            "device": self.device,
            "synthesis_time_ms": self.time_ms,
            "physics_version": PHYSICS_VERSION,
            "optics": optics.as_dict(),
            "recon_normalisation": norm,
            "planes": [
                {"index": i + 1, "distance_mm": round(z * 1e3, 4), **row}
                for i, (z, row) in enumerate(zip(optics.distances, self.rows))
            ],
            "mean": self.summary,
            **self.extra,
        }


def _sync(device) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def synthesize_phase(sources_u8: np.ndarray, method: str = "gs+adam", *, model=None, device: str = "cpu",
                     optics: Optics = DEFAULT_OPTICS, gs_iters: int = 200, adam_steps: int = 400,
                     refine_steps: int = 30, lr: float = 0.05, refine_lr: float = 0.03,
                     seed: int = 0, prop=None) -> tuple[np.ndarray, float, dict]:
    """Фаза ДОЭ (N, N) в радианах, время синтеза в мс (с синхронизацией CUDA), параметры."""
    import torch

    from .baselines import adam_phase, gs_multiplane
    from .fresnel import FresnelSingleFFT

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    sources_u8 = np.asarray(sources_u8)
    if sources_u8.shape != (len(optics.distances), optics.size, optics.size):
        raise ValueError(f"expected {len(optics.distances)} images {optics.size}×{optics.size}")
    if method.startswith("net") and model is None:
        raise ValueError("method 'net' needs trained weights (--weights)")
    prop = prop or FresnelSingleFFT(optics).to(device)
    if method.startswith("net"):
        if prop.optics != optics:
            raise ValueError("propagator optics do not match requested reconstruction optics")
        if model.prop.optics != prop.optics or model.prop.distances != prop.distances:
            raise ValueError("network checkpoint optics do not match reconstruction optics")
    target = torch.from_numpy(sources_u8.astype(np.float32) / 255.0)[None].to(device)
    params: dict = {"seed": seed}
    _sync(device)
    t0 = time.perf_counter()
    if method.startswith("net"):
        with torch.inference_mode():
            phase = model(target)
        if method == "net+gs":
            phase = gs_multiplane(target, prop, refine_steps, init_phase=phase.clone())
            params.update(refine_steps=refine_steps)
        elif method == "net+adam":
            phase = adam_phase(target, prop, refine_steps, init_phase=phase.clone(), lr=refine_lr)
            params.update(refine_steps=refine_steps, refine_lr=refine_lr)
        elif method == "net+gs+adam":
            from .hybrid import refine_network_phase
            phase, diagnostics = refine_network_phase(phase.clone(), target, prop,
                                                      gs_steps=refine_steps, adam_steps=adam_steps, lr=refine_lr)
            params.update(refine_steps=refine_steps, adam_steps=adam_steps, refine_lr=refine_lr,
                          neural_primary=True, refinement=diagnostics)
    elif method == "gs":
        phase = gs_multiplane(target, prop, gs_iters, seed=seed)
        params.update(gs_iters=gs_iters)
    elif method == "gs+adam":
        phase = gs_multiplane(target, prop, gs_iters, seed=seed)
        phase = adam_phase(target, prop, adam_steps, init_phase=phase, lr=lr)
        params.update(gs_iters=gs_iters, adam_steps=adam_steps, lr=lr)
    else:
        phase = adam_phase(target, prop, adam_steps, lr=lr, seed=seed)
        params.update(adam_steps=adam_steps, lr=lr)
    _sync(device)
    elapsed = (time.perf_counter() - t0) * 1e3
    return phase[0].detach().double().cpu().numpy(), elapsed, params


def reconstruct(doe_u8: np.ndarray, sources_u8: np.ndarray, optics: Optics = DEFAULT_OPTICS,
                norm: str = DEFAULT_NORM) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Официальное восстановление из 8-битного ДОЭ: (recon_u8 (P,N,N), модули, метрики по плоскостям)."""
    modulus = fresnel_modulus_np(uint8_to_phase(doe_u8), optics)
    recon = np.stack([modulus_to_uint8(m, s, norm) for m, s in zip(modulus, sources_u8)])
    rows = [pair_metrics(s, r) for s, r in zip(sources_u8, recon)]
    return recon, modulus, rows


def run(sources_u8: np.ndarray, method: str = "gs+adam", *, norm: str = DEFAULT_NORM,
        optics: Optics = DEFAULT_OPTICS, **kwargs) -> Result:
    phase, ms, params = synthesize_phase(sources_u8, method, optics=optics, **kwargs)
    doe = phase_to_uint8(phase)
    recon, modulus, rows = reconstruct(doe, sources_u8, optics, norm)
    return Result(method, params, phase, doe, np.asarray(sources_u8), recon, modulus, rows,
                  summarize(rows), ms, str(kwargs.get("device", "cpu")))


def write_outputs(result: Result, out_dir: Path, optics: Optics = DEFAULT_OPTICS,
                  norm: str = DEFAULT_NORM, preview: bool = True) -> Result:
    """Сохраняет doe.png, перечитывает его, заново восстанавливает и пишет recon_*.png и метрики.

    Возвращает результат, пересчитанный по перечитанному файлу (он и идёт в отчёт).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png(out_dir / "doe.png", result.doe_u8)
    doe_reloaded = load_png(out_dir / "doe.png", optics.size)
    if not np.array_equal(doe_reloaded, result.doe_u8):
        raise RuntimeError("doe.png round-trip mismatch")
    recon, modulus, rows = reconstruct(doe_reloaded, result.sources_u8, optics, norm)
    for i, (src, rec) in enumerate(zip(result.sources_u8, recon), 1):
        save_png(out_dir / f"recon_{i}.png", rec)
        save_png(out_dir / f"source_{i}.png", src)
    final = Result(result.method, result.params, result.phase, doe_reloaded, result.sources_u8, recon, modulus,
                   rows, summarize(rows), result.time_ms, result.device, result.extra)
    report = final.to_json(optics, norm)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["plane", "distance_mm", "pcc", "ssim", "ssim_global"])
        for p in report["planes"]:
            w.writerow([p["index"], p["distance_mm"], f"{p['pcc']:.6f}", f"{p['ssim']:.6f}", f"{p['ssim_global']:.6f}"])
        m = report["mean"]
        w.writerow(["mean", "", f"{m['pcc']:.6f}", f"{m['ssim']:.6f}", f"{m['ssim_global']:.6f}"])
    if preview:
        save_preview(final, out_dir / "preview.png", optics)
    return final


def save_preview(result: Result, path: Path, optics: Optics = DEFAULT_OPTICS) -> None:
    """Картинка «исходные / ДОЭ / восстановления» для отчёта."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = len(result.sources_u8)
    fig, axes = plt.subplots(2, p + 1, figsize=(2.6 * (p + 1), 5.6))
    axes[0, 0].imshow(result.doe_u8, cmap="twilight", vmin=0, vmax=255)
    axes[0, 0].set_title("ДОЭ (фаза)")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 0.5, f"{result.method}\n{result.time_ms:.1f} мс\n"
                    f"PCC {result.summary['pcc']:.3f}\nSSIM {result.summary['ssim']:.3f}",
                    fontsize=11, va="center")
    for i in range(p):
        z = optics.distances[i] * 1e3
        axes[0, i + 1].imshow(result.sources_u8[i], cmap="gray", vmin=0, vmax=255)
        axes[0, i + 1].set_title(f"исходное {i + 1}")
        axes[1, i + 1].imshow(result.recon_u8[i], cmap="gray", vmin=0, vmax=255)
        axes[1, i + 1].set_title(f"z={z:.1f} мм · PCC {result.rows[i]['pcc']:.2f}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_sources(paths: list[Path], optics: Optics = DEFAULT_OPTICS) -> np.ndarray:
    if len(paths) != len(optics.distances):
        raise ValueError(f"need exactly {len(optics.distances)} input images, got {len(paths)}")
    return np.stack([load_png(Path(p), optics.size) for p in paths])
