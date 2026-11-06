"""Данные: пул одноканальных 8-битных изображений 128×128 и сборка четвёрок.

Категории пула:
    letters  — символы и короткие строки из TTF-шрифтов (кириллица, латиница, цифры);
    shapes   — геометрические фигуры разной яркости;
    photos   — фото в оттенках серого: встроенные изображения scikit-image и (необязательно)
               папка data/photos (например, DIV2K: python -m ml.data fetch-div2k);
    textures — 1/f-шум (статистика, похожая на естественные изображения).

Четвёрки для обучения собираются случайно на лету (иногда из одной категории, иногда
смешанные), с аугментациями: инверсия, гамма. Фиксированные val/test-четвёрки
генерируются по отдельным seed и в обучение не попадают. Тестовые фото (camera,
astronaut, coins, moon, text, page, horse) из обучающего пула исключены.

CLI:
    python -m ml.data build-pool --count 20000      # собрать и закэшировать пул
    python -m ml.data make-testsets                 # PNG-наборы в data/testsets/
    python -m ml.data fetch-div2k                   # скачать фото DIV2K (≈450 МБ)
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE = 128
CATEGORIES = ("letters", "shapes", "photos", "textures")
# Организаторы: «изображения могут быть разнообразные, но не совсем простые» — упор на фото и текстуры.
POOL_MIX = {"letters": 0.25, "shapes": 0.15, "photos": 0.4, "textures": 0.2}
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CYRILLIC = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
ALPHABET = CYRILLIC + LATIN + DIGITS

# Изображения scikit-image, зарезервированные под тестовые наборы.
TEST_PHOTOS = ("camera", "astronaut", "coins", "moon", "text", "page", "horse")
TRAIN_PHOTOS = ("brick", "grass", "gravel", "coffee", "chelsea", "rocket", "clock",
                "immunohistochemistry", "hubble_deep_field", "logo", "colorwheel",
                "microaneurysms", "retina", "checkerboard", "cell", "motorcycle_left",
                "eagle", "lily", "kidney", "human_mitosis", "skin")

FONT_CANDIDATES = (
    "arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "cour.ttf", "courbd.ttf",
    "calibri.ttf", "calibrib.ttf", "verdana.ttf", "verdanab.ttf", "tahoma.ttf", "georgia.ttf",
    "segoeui.ttf", "seguisb.ttf", "consola.ttf", "trebuc.ttf", "cambria.ttc",
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf",
    "LiberationSans-Regular.ttf", "LiberationSerif-Regular.ttf", "LiberationMono-Regular.ttf",
)
FONT_DIRS = (
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
    Path("/Library/Fonts"), Path("/System/Library/Fonts"),
)


# ---------------------------------------------------------------------------
# Источники
# ---------------------------------------------------------------------------
_FONTS: list[str] | None = None


def find_fonts() -> list[str]:
    global _FONTS
    if _FONTS is None:
        found = []
        for d in FONT_DIRS:
            if not d.is_dir():
                continue
            for name in FONT_CANDIDATES:
                hits = [d / name] if (d / name).is_file() else list(d.rglob(name))[:1]
                found.extend(str(h) for h in hits if h.is_file())
        _FONTS = sorted(set(found))
    return _FONTS


def _font(rng: np.random.Generator, size: int):
    fonts = find_fonts()
    if fonts:
        return ImageFont.truetype(fonts[int(rng.integers(len(fonts)))], size)
    return ImageFont.load_default(size)


def render_text(rng: np.random.Generator, text: str | None = None, size: int = SIZE) -> np.ndarray:
    """Символы или короткая строка: случайный шрифт, размер, положение, яркость, наклон."""
    if text is None:
        k = int(rng.choice([1, 1, 1, 2, 2, 3, 4]))
        text = "".join(ALPHABET[int(i)] for i in rng.integers(len(ALPHABET), size=k))
    big = Image.new("L", (size * 2, size * 2), 0)
    draw = ImageDraw.Draw(big)
    target_h = rng.uniform(0.35, 0.9) * size * 2 / max(1, 0.55 * len(text) if len(text) > 2 else 1)
    font = _font(rng, max(12, int(target_h)))
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    if w > size * 1.9:  # не влезает — уменьшаем
        font = _font(rng, max(12, int(target_h * size * 1.8 / w)))
        box = draw.textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
    cx = size + rng.uniform(-0.25, 0.25) * max(0, size * 2 - w)
    cy = size + rng.uniform(-0.25, 0.25) * max(0, size * 2 - h)
    draw.text((cx - w / 2 - box[0], cy - h / 2 - box[1]), text, fill=int(rng.integers(150, 256)), font=font)
    if rng.random() < 0.5:
        big = big.rotate(float(rng.uniform(-25, 25)), resample=Image.BILINEAR)
    img = big.resize((size, size), Image.LANCZOS)
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5))))
    return np.asarray(img, dtype=np.uint8)


def render_shapes(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    img = Image.new("L", (size, size), int(rng.integers(0, 40)) if rng.random() < 0.3 else 0)
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(1, 7))):
        kind = rng.choice(["ellipse", "rect", "poly", "line", "ring"])
        x0, y0 = rng.uniform(0.05, 0.75, 2) * size
        w, h = rng.uniform(0.1, 0.5, 2) * size
        fill = int(rng.integers(60, 256))
        box = [x0, y0, min(size - 1, x0 + w), min(size - 1, y0 + h)]
        if kind == "ellipse":
            draw.ellipse(box, fill=fill)
        elif kind == "rect":
            draw.rectangle(box, fill=fill)
        elif kind == "poly":
            pts = [tuple(p) for p in rng.uniform(0.05, 0.95, (int(rng.integers(3, 7)), 2)) * size]
            draw.polygon(pts, fill=fill)
        elif kind == "line":
            draw.line(box, fill=fill, width=int(rng.integers(2, 9)))
        else:
            draw.ellipse(box, outline=fill, width=int(rng.integers(2, 8)))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.5))))
    return np.asarray(img, dtype=np.uint8)


def render_texture(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """1/f^β шум, β ∈ [1,5; 3]: от «облаков» до мелкой фактуры."""
    beta = rng.uniform(1.5, 3.0)
    f = np.fft.fftfreq(size)
    r = np.sqrt(f[None, :] ** 2 + f[:, None] ** 2)
    r[0, 0] = 1.0
    spec = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))) / r ** (beta / 2)
    spec[0, 0] = 0
    x = np.real(np.fft.ifft2(spec))
    x = (x - x.min()) / (np.ptp(x) + 1e-12)
    x = x ** rng.uniform(0.6, 1.6)
    return np.round(x * 255).astype(np.uint8)


def _to_gray_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    arr = arr.astype(np.float64)
    if arr.max() <= 1.0 + 1e-9 and arr.dtype != np.uint8:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_skimage(name: str) -> np.ndarray | None:
    """Встроенное изображение scikit-image (без скачивания), в оттенках серого."""
    try:
        from skimage import data as skdata

        fn = getattr(skdata, name, None)
        if fn is None:
            return None
        img = fn()
        if isinstance(img, tuple):
            img = img[0]
        return _to_gray_u8(img)
    except Exception:
        return None


def fit_square(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Центральный квадрат, масштабированный до size×size."""
    h, w = img.shape
    s = min(h, w)
    crop = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.LANCZOS), dtype=np.uint8)


def random_crop(img: np.ndarray, rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    h, w = img.shape
    s = int(min(h, w) * rng.uniform(0.25, 1.0))
    s = max(min(s, h, w), min(size, h, w))
    y = int(rng.integers(0, h - s + 1))
    x = int(rng.integers(0, w - s + 1))
    out = np.asarray(Image.fromarray(img[y:y + s, x:x + s]).resize((size, size), Image.LANCZOS), dtype=np.uint8)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.2:
        out = np.rot90(out, int(rng.integers(1, 4)))
    return np.ascontiguousarray(out)


def load_photo_sources(photo_dir: Path | None = None, max_files: int = 2000) -> list[np.ndarray]:
    photos = [p for p in (load_skimage(n) for n in TRAIN_PHOTOS) if p is not None and min(p.shape) >= 64]
    photo_dir = photo_dir or DATA_DIR / "photos"
    if photo_dir.is_dir():
        files = sorted(f for f in photo_dir.rglob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        for f in files[:max_files]:
            try:
                with Image.open(f) as im:
                    im = im.convert("L")
                    if max(im.size) > 768:  # экономим память: исходники DIV2K ≈2K
                        im.thumbnail((768, 768), Image.LANCZOS)
                    photos.append(np.asarray(im, dtype=np.uint8))
            except Exception as exc:  # битый файл не должен ломать сборку пула
                print(f"[data] skip {f}: {exc}", file=sys.stderr)
    return photos


# ---------------------------------------------------------------------------
# Пул
# ---------------------------------------------------------------------------
def generate(category: str, rng: np.random.Generator, photos: list[np.ndarray]) -> np.ndarray:
    if category == "letters":
        return render_text(rng)
    if category == "shapes":
        return render_shapes(rng)
    if category == "textures":
        return render_texture(rng)
    if category == "photos":
        if not photos:
            return render_texture(rng)
        return random_crop(photos[int(rng.integers(len(photos)))], rng)
    raise ValueError(category)


def build_pool(count: int, seed: int = 0, photo_dir: Path | None = None,
               verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    photos = load_photo_sources(photo_dir)
    if verbose:
        print(f"[data] fonts: {len(find_fonts())}, photo sources: {len(photos)}", flush=True)
    probs = np.array([POOL_MIX[c] for c in CATEGORIES])
    cats = rng.choice(len(CATEGORIES), size=count, p=probs / probs.sum()).astype(np.uint8)
    images = np.empty((count, SIZE, SIZE), dtype=np.uint8)
    for i in range(count):
        images[i] = generate(CATEGORIES[cats[i]], rng, photos)
        if verbose and (i + 1) % 5000 == 0:
            print(f"[data] {i + 1}/{count}", flush=True)
    return images, cats


def load_or_build_pool(count: int, seed: int = 0, cache_dir: Path | None = None,
                       photo_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(cache_dir or DATA_DIR / "pool")
    cache = cache_dir / f"pool_{count}_seed{seed}.npz"
    if cache.is_file():
        with np.load(cache) as z:
            return z["images"], z["cats"]
    images, cats = build_pool(count, seed, photo_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez_compressed(tmp, images=images, cats=cats)
    os.replace(tmp, cache)
    return images, cats


class QuadSampler:
    """Случайные четвёрки из пула: (B, 4, N, N) float32 в [0, 1]."""

    def __init__(self, images: np.ndarray, cats: np.ndarray, seed: int = 0, same_category: float = 0.5,
                 invert: float = 0.15, gamma: float = 0.3):
        self.images = images
        self.by_cat = [np.flatnonzero(cats == c) for c in range(len(CATEGORIES))]
        self.by_cat = [ids for ids in self.by_cat if len(ids)]
        self.rng = np.random.default_rng(seed)
        self.same_category, self.invert, self.gamma = same_category, invert, gamma

    def sample(self, batch: int, planes: int = 4) -> np.ndarray:
        rng = self.rng
        ids = np.empty((batch, planes), dtype=np.int64)
        for b in range(batch):
            if rng.random() < self.same_category:
                pool = self.by_cat[int(rng.integers(len(self.by_cat)))]
                ids[b] = pool[rng.integers(len(pool), size=planes)]
            else:
                ids[b] = rng.integers(len(self.images), size=planes)
        x = self.images[ids].astype(np.float32) / 255.0
        flip = rng.random((batch, planes)) < self.invert
        x[flip] = 1.0 - x[flip]
        g = np.where(rng.random((batch, planes)) < self.gamma, rng.uniform(0.6, 1.6, (batch, planes)), 1.0)
        x = x ** g[..., None, None].astype(np.float32)
        return x


# ---------------------------------------------------------------------------
# Фиксированные наборы
# ---------------------------------------------------------------------------
SPLIT_SEEDS = {"val": 1_000_003, "test": 2_000_003}
FIXED_KINDS = ("letters", "photos", "mixed")


def fixed_quads(split: str, per_kind: int = 16, photo_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Детерминированные четвёрки {kind: (K, 4, N, N) uint8} для val/test.

    Фото для val/test берутся случайными кропами из тех же источников, что и обучение,
    но с другим seed; для честной проверки на «невиданных» фото см. наборы
    data/testsets/photos_classic (эти изображения в обучение не входят).
    """
    rng = np.random.default_rng(SPLIT_SEEDS[split])
    photos = load_photo_sources(photo_dir)
    out = {}
    for kind in FIXED_KINDS:
        quads = np.empty((per_kind, 4, SIZE, SIZE), dtype=np.uint8)
        for k in range(per_kind):
            for p in range(4):
                cat = kind if kind != "mixed" else CATEGORIES[int(rng.integers(len(CATEGORIES)))]
                quads[k, p] = generate(cat, rng, photos)
        out[kind] = quads
    return out


def classic_testsets() -> dict[str, np.ndarray]:
    """Наборы для отчёта и демо (в обучение не входят), {name: (4, N, N) uint8}."""
    sets = {}
    photos = [load_skimage(n) for n in ("camera", "astronaut", "coins", "moon")]
    if all(p is not None for p in photos):
        sets["photos_classic"] = np.stack([fit_square(p) for p in photos])
    text = [load_skimage(n) for n in ("text", "page", "horse")]
    if all(t is not None for t in text):
        rng = np.random.default_rng(7)
        blobs = render_shapes(rng)
        sets["text_binary"] = np.stack([fit_square(t) for t in text] + [blobs])
    rng = np.random.default_rng(11)
    sets["mephi_letters"] = np.stack([_letter(ch, rng) for ch in "МИФИ"])
    rng = np.random.default_rng(13)
    sets["digits"] = np.stack([_letter(ch, rng) for ch in "2026"])
    return sets


def _letter(ch: str, rng: np.random.Generator) -> np.ndarray:
    """Крупная белая буква по центру на чёрном фоне."""
    img = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    fonts = find_fonts()
    bold = [f for f in fonts if "bd" in Path(f).name.lower() or "bold" in f.lower()]
    path = (bold or fonts or [None])[0]
    font = ImageFont.truetype(path, 96) if path else ImageFont.load_default(96)
    box = draw.textbbox((0, 0), ch, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.text(((SIZE - w) / 2 - box[0], (SIZE - h) / 2 - box[1]), ch, fill=255, font=font)
    return np.asarray(img, dtype=np.uint8)


def save_png(path: Path, img: np.ndarray) -> None:
    img = np.asarray(img)
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError("expected a 2-D uint8 image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(path)


def load_png(path: Path, size: int = SIZE) -> np.ndarray:
    """8-битное одноканальное изображение size×size. RGB приводится к яркости,
    другой размер — центральный квадрат с масштабированием (с предупреждением)."""
    with Image.open(path) as im:
        if im.mode == "I;16" or im.mode == "I":
            arr = np.asarray(im, dtype=np.float64)
            arr = arr / max(arr.max(), 1) * 255
            im = Image.fromarray(arr.astype(np.uint8))
        img = np.asarray(im.convert("L"), dtype=np.uint8)
    if img.shape != (size, size):
        print(f"[data] {path}: size {img.shape} → {size}×{size} (center crop + resize)", file=sys.stderr)
        img = fit_square(img, size)
    return img


def make_testsets(out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or DATA_DIR / "testsets")
    written = []
    for name, imgs in classic_testsets().items():
        for i, img in enumerate(imgs, 1):
            save_png(out_dir / name / f"{i}.png", img)
        written.append(out_dir / name)
    test = fixed_quads("test", per_kind=2)
    for kind, quads in test.items():
        for k, quad in enumerate(quads):
            for i, img in enumerate(quad, 1):
                save_png(out_dir / f"random_{kind}_{k}" / f"{i}.png", img)
            written.append(out_dir / f"random_{kind}_{k}")
    return written


DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


def fetch_div2k(dest: Path | None = None, url: str = DIV2K_URL) -> Path:
    """Скачать DIV2K valid HR (100 фото, ≈450 МБ) в data/photos/div2k."""
    import urllib.request

    dest = Path(dest or DATA_DIR / "photos" / "div2k")
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "DIV2K_valid_HR.zip"
    if not archive.is_file():
        tmp = archive.with_suffix(".part")
        print(f"[data] downloading {url} → {archive}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done % (50 << 20) < (1 << 20):
                    print(f"[data] {done >> 20}/{total >> 20} MB", flush=True)
        os.replace(tmp, archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member}")
        z.extractall(dest)
    print(f"[data] photos ready in {dest}; rebuild the pool to use them", flush=True)
    return dest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Датасет для многоплоскостного синтеза ДОЭ")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-pool")
    b.add_argument("--count", type=int, default=20000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--photo-dir", type=Path, default=None)
    t = sub.add_parser("make-testsets")
    t.add_argument("--out", type=Path, default=None)
    sub.add_parser("fetch-div2k")
    s = sub.add_parser("preview")
    s.add_argument("--out", type=Path, default=DATA_DIR / "pool_preview.png")
    args = ap.parse_args(argv)
    if args.cmd == "build-pool":
        images, cats = load_or_build_pool(args.count, args.seed, photo_dir=args.photo_dir)
        counts = {c: int((cats == i).sum()) for i, c in enumerate(CATEGORIES)}
        print(f"[data] pool {images.shape}, categories {counts}")
    elif args.cmd == "make-testsets":
        for p in make_testsets(args.out):
            print(p)
    elif args.cmd == "fetch-div2k":
        fetch_div2k()
    elif args.cmd == "preview":
        rng = np.random.default_rng(0)
        photos = load_photo_sources()
        tiles = [generate(CATEGORIES[i % 4], rng, photos) for i in range(32)]
        grid = np.vstack([np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(4)])
        save_png(args.out, grid)
        print(args.out)


if __name__ == "__main__":
    main()
