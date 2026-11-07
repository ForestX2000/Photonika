"""Сервисный слой: синтез ДОЭ, кэш результатов, рендер PNG для фронтенда.

Вся физика и метрики — из ml/ (тот же конвейер, что у CLI: doe.png → перечитывание →
Френель → recon → метрики по протоколу организаторов). Здесь только упаковка для HTTP.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ml.config import DEFAULT_OPTICS, PHYSICS_VERSION
from ml.data import fit_square
from ml.fresnel import fresnel_modulus_np, uint8_to_phase
from ml.metrics import DEFAULT_NORM
from ml.pipeline import METHOD_HELP, METHODS, Result, reconstruct, run

ROOT = Path(__file__).resolve().parent.parent
TESTSETS = ROOT / "data" / "testsets"
DEFAULT_WEIGHTS = ROOT / "data" / "weights" / "doe_multiplane.pt"
OPTICS = DEFAULT_OPTICS
N = OPTICS.size

METHOD_TITLES = {
    "net+gs": "Нейросеть + GS",
    "net": "Нейросеть",
    "net+adam": "Нейросеть + Adam",
    "gs+adam": "GS + Adam",
    "gs": "Gerchberg–Saxton",
    "adam": "Adam",
}
TESTSET_TITLES = {
    "photos_classic": "Фото: камера, астронавт, монеты, Луна",
    "text_binary": "Текст и бинарные изображения",
    "mephi_letters": "Буквы «МИФИ»",
    "digits": "Цифры «2026»",
}
LIMITS = {"gs_iters": (1, 1000), "adam_steps": (1, 1000), "refine_steps": (0, 500)}
MAX_UPLOAD = 15 * 1024 * 1024


class UserError(ValueError):
    """Ошибка во входных данных — показывается пользователю как есть."""


# ---------------------------------------------------------------------------
# Изображения
# ---------------------------------------------------------------------------
def png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    mode = "L" if img.ndim == 2 else "RGB"
    Image.fromarray(np.ascontiguousarray(img), mode=mode).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def data_url(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode("ascii")


def phase_colors(doe_u8: np.ndarray) -> np.ndarray:
    """Фаза ДОЭ в циклической палитре twilight (−π и +π одного цвета)."""
    import matplotlib

    return (matplotlib.colormaps["twilight"](doe_u8 / 255.0)[..., :3] * 255).astype(np.uint8)


def laser_colors(modulus: np.ndarray) -> np.ndarray:
    """Модуль поля в «лазерной» зелёной палитре (532 нм): чёрный → зелёный → белый."""
    ref = np.percentile(modulus, 99.5)
    v = np.clip(modulus / ref, 0.0, 1.0) if ref > 0 else np.zeros_like(modulus)
    hi = np.clip(2.0 * v - 1.0, 0.0, 1.0)
    rgb = np.stack([0.15 * v + 0.85 * hi ** 1.5, v ** 0.8, 0.35 * v + 0.65 * hi ** 1.5], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def decode_image(raw: bytes, name: str = "image") -> np.ndarray:
    """Загруженный файл → одноканальное 8-битное 128×128 (RGB → яркость, другой размер → центр + масштаб)."""
    if not raw:
        raise UserError(f"{name}: пустой файл")
    if len(raw) > MAX_UPLOAD:
        raise UserError(f"{name}: файл больше {MAX_UPLOAD // (1024 * 1024)} МБ")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if im.mode in ("I;16", "I", "F"):
                arr = np.asarray(im, dtype=np.float64)
                arr = arr / max(arr.max(), 1e-12) * 255
                im = Image.fromarray(arr.astype(np.uint8))
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
                im = Image.alpha_composite(bg, im)
            arr = np.asarray(im.convert("L"), dtype=np.uint8)
    except UserError:
        raise
    except Exception as exc:  # битый файл, не изображение
        raise UserError(f"{name}: не удалось прочитать изображение ({exc.__class__.__name__})") from None
    return arr if arr.shape == (N, N) else fit_square(arr, N)


# ---------------------------------------------------------------------------
# Сервис
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    result: Result
    created: float
    compare: dict | None = None


class SynthesisService:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: str = "auto", cache_size: int = 32):
        import torch

        from ml.fresnel import FresnelSingleFFT

        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.prop = FresnelSingleFFT(OPTICS).to(self.device)
        self.lock = threading.Lock()          # torch-вычисления по одному: память GPU и честный замер времени
        self.cache: OrderedDict[str, Entry] = OrderedDict()
        self.cache_size = cache_size
        self.weights = Path(weights)
        self.model, self.model_info = None, None
        self._model_loaded = False

    def _ensure_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True
        self.reload_model()

    # -- модель ----------------------------------------------------------------
    def reload_model(self) -> None:
        if not self.weights.is_file():
            self.model, self.model_info = None, None
            return
        from ml.model import load_model

        try:
            model, ckpt = load_model(self.weights, self.device)
            if model.prop.optics != OPTICS:
                raise ValueError("checkpoint optics do not match server optics")
        except Exception as exc:  # несовместимые веса не должны ронять сервер
            self.model, self.model_info = None, {"error": str(exc)}
            return
        self.model = model
        self.model_info = {"config": ckpt.get("model"), "step": ckpt.get("step"),
                           "validation_pcc": (ckpt.get("validation") or {}).get("pcc")}

    @property
    def has_net(self) -> bool:
        self._ensure_model()
        return self.model is not None

    def methods(self) -> list[dict]:
        order = ["net+gs", "net", "net+adam", "gs+adam", "gs", "adam"]
        return [{"id": m, "title": METHOD_TITLES[m], "description": METHOD_HELP[m],
                 "available": self.has_net or not m.startswith("net")} for m in order if m in METHODS]

    def default_method(self) -> str:
        return "net+gs" if self.has_net else "gs+adam"

    def info(self) -> dict:
        self._ensure_model()
        import torch

        return {
            "physics_version": PHYSICS_VERSION,
            "size": N,
            "wavelength_nm": OPTICS.wavelength * 1e9,
            "pitch_um": OPTICS.pitch * 1e6,
            "distances_mm": [round(z * 1e3, 3) for z in OPTICS.distances],
            "critical_distance_mm": round(OPTICS.critical_distance * 1e3, 3),
            "output_pitch_um": [round(OPTICS.output_pitch(z) * 1e6, 3) for z in OPTICS.distances],
            "device": self.device,
            "gpu": torch.cuda.get_device_name(0) if self.device.startswith("cuda") else None,
            "weights": self.has_net,
            "model": self.model_info,
            "methods": self.methods(),
            "default_method": self.default_method(),
            "limits": LIMITS,
            "protocol": "амплитуды; каждое изображение нормируется на максимум (= 1)",
        }

    # -- тестовые наборы ----------------------------------------------------------
    def testsets(self) -> list[dict]:
        if not TESTSETS.is_dir():
            return []
        out = []
        for d in sorted(TESTSETS.iterdir(), key=lambda p: (p.name.startswith("random"), p.name)):
            if d.is_dir() and all((d / f"{i}.png").is_file() for i in range(1, 5)):
                out.append({"name": d.name, "title": self._testset_title(d.name),
                            "images": [f"/api/testsets/{d.name}/{i}.png" for i in range(1, 5)]})
        return out

    @staticmethod
    def _testset_title(name: str) -> str:
        if name in TESTSET_TITLES:
            return TESTSET_TITLES[name]
        if name.startswith("random_"):
            _, kind, k = name.split("_")
            kinds = {"letters": "буквы", "photos": "фото", "mixed": "смешанный"}
            return f"Случайный набор: {kinds.get(kind, kind)} №{int(k) + 1}"
        return name

    def testset_path(self, name: str, index: int) -> Path:
        if not name.replace("_", "").isalnum() or index not in (1, 2, 3, 4):
            raise UserError("неизвестный тестовый набор")
        path = TESTSETS / name / f"{index}.png"
        if not path.is_file():
            raise UserError(f"нет тестового набора {name!r}")
        return path

    def load_testset(self, name: str) -> np.ndarray:
        return np.stack([decode_image(self.testset_path(name, i).read_bytes(), f"{name}/{i}.png")
                         for i in range(1, 5)])

    # -- синтез -------------------------------------------------------------------
    def _params(self, gs_iters: int, adam_steps: int, refine_steps: int) -> dict:
        values = {"gs_iters": gs_iters, "adam_steps": adam_steps, "refine_steps": refine_steps}
        for key, value in values.items():
            lo, hi = LIMITS[key]
            if not lo <= int(value) <= hi:
                raise UserError(f"{key} должно быть от {lo} до {hi}")
        return {k: int(v) for k, v in values.items()}

    def synthesize(self, sources: np.ndarray, method: str, gs_iters: int = 200, adam_steps: int = 400,
                   refine_steps: int = 25, seed: int = 0) -> tuple[str, Result]:
        if method not in METHODS:
            raise UserError(f"неизвестный метод {method!r}")
        if method.startswith("net") and not self.has_net:
            raise UserError("веса нейросети не загружены на сервере — выберите GS или GS + Adam")
        params = self._params(gs_iters, adam_steps, refine_steps)
        with self.lock:
            if self.device.startswith("cuda"):  # прогрев: первый вызов CUDA включает инициализацию
                run(sources, method, norm=DEFAULT_NORM, model=self.model, device=self.device, prop=self.prop,
                    seed=seed, **params)
            result = run(sources, method, norm=DEFAULT_NORM, model=self.model, device=self.device,
                         prop=self.prop, seed=seed, **params)
        rid = uuid.uuid4().hex[:12]
        self.cache[rid] = Entry(result, time.time())
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return rid, result

    def get(self, rid: str) -> Entry:
        entry = self.cache.get(rid)
        if entry is None:
            raise KeyError(rid)
        self.cache.move_to_end(rid)
        return entry

    def compare(self, rid: str, gs_iters: int = 50) -> dict:
        entry = self.get(rid)
        self._params(gs_iters, 1, 0)
        with self.lock:
            run(entry.result.sources_u8, "gs", device=self.device, prop=self.prop, gs_iters=gs_iters)  # прогрев
            ref = run(entry.result.sources_u8, "gs", device=self.device, prop=self.prop, gs_iters=gs_iters)
        entry.compare = {"method": "gs", "gs_iters": gs_iters, "time_ms": ref.time_ms, "mean": ref.summary,
                         "planes": [{"pcc": r["pcc"], "ssim": r["ssim"]} for r in ref.rows],
                         "recon": [data_url(r) for r in ref.recon_u8]}
        return entry.compare

    def slice_png(self, rid: str, z_mm: float) -> bytes:
        if not 15.0 <= z_mm <= 40.0:
            raise UserError("z должно быть в диапазоне 15–40 мм")
        entry = self.get(rid)
        mod = fresnel_modulus_np(uint8_to_phase(entry.result.doe_u8), OPTICS, [z_mm * 1e-3])[0]
        return png_bytes(laser_colors(mod))

    def zip_bytes(self, rid: str) -> bytes:
        entry = self.get(rid)
        res = entry.result
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("doe.png", png_bytes(res.doe_u8))
            for i, (s, r) in enumerate(zip(res.sources_u8, res.recon_u8), 1):
                z.writestr(f"recon_{i}.png", png_bytes(r))
                z.writestr(f"source_{i}.png", png_bytes(s))
            report = res.to_json(OPTICS, DEFAULT_NORM)
            z.writestr("metrics.json", json.dumps(report, indent=2, ensure_ascii=False))
            lines = ["plane,distance_mm,pcc,ssim,ssim_global"]
            lines += [f"{p['index']},{p['distance_mm']},{p['pcc']:.6f},{p['ssim']:.6f},{p['ssim_global']:.6f}"
                      for p in report["planes"]]
            m = report["mean"]
            lines.append(f"mean,,{m['pcc']:.6f},{m['ssim']:.6f},{m['ssim_global']:.6f}")
            z.writestr("metrics.csv", "\n".join(lines) + "\n")
        return buf.getvalue()

    # -- ответ для фронтенда ---------------------------------------------------------
    def to_payload(self, rid: str, res: Result) -> dict:
        # Проверка: восстановление из сохранённого 8-битного ДОЭ совпадает с тем, что в результате
        recon, _, _ = reconstruct(res.doe_u8, res.sources_u8, OPTICS, DEFAULT_NORM)
        assert np.array_equal(recon, res.recon_u8)
        return {
            "id": rid,
            "method": res.method,
            "method_title": METHOD_TITLES.get(res.method, res.method),
            "params": res.params,
            "device": res.device,
            "time_ms": res.time_ms,
            "mean": res.summary,
            "doe": data_url(res.doe_u8),
            "doe_color": data_url(phase_colors(res.doe_u8)),
            "planes": [
                {"index": i + 1, "distance_mm": round(OPTICS.distances[i] * 1e3, 3), **row,
                 "source": data_url(res.sources_u8[i]), "recon": data_url(res.recon_u8[i]),
                 "recon_laser": data_url(laser_colors(res.modulus[i]))}
                for i, row in enumerate(res.rows)
            ],
        }
