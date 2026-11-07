"""Картинки для первого экрана сайта из настоящего результата синтеза.

    python tools/make_web_assets.py

photos_classic → GS-200 + Adam-400 → frontend/public/images/doe_phase.png (фаза в палитре twilight),
frontend/public/images/plane_{1..4}.png (модуль поля на 4 расстояниях в «лазерной» палитре).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ml.data import classic_testsets  # noqa: E402
from ml.pipeline import run  # noqa: E402
from backend.service import laser_colors, phase_colors  # noqa: E402


def main() -> None:
    out = ROOT / "frontend" / "public" / "images"
    out.mkdir(parents=True, exist_ok=True)
    res = run(classic_testsets()["photos_classic"], "gs+adam", device="cpu")
    Image.fromarray(phase_colors(res.doe_u8)).save(out / "doe_phase.png", optimize=True)
    for i, mod in enumerate(res.modulus, 1):
        Image.fromarray(laser_colors(mod)).save(out / f"plane_{i}.png", optimize=True)
    print(f"PCC {np.mean([r['pcc'] for r in res.rows]):.3f} → {out}")


if __name__ == "__main__":
    main()
