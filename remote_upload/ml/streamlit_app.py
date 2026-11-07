"""Streamlit-демо: 4 изображения → фазовый ДОЭ → 4 восстановления на 24,1–27,7 мм.

Запуск из корня репозитория:
    streamlit run ml/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # streamlit запускает файл как скрипт
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from ml.config import DEFAULT_OPTICS  # noqa: E402
from ml.data import fit_square  # noqa: E402
from ml.fresnel import FresnelSingleFFT, fresnel_modulus_np, uint8_to_phase  # noqa: E402
from ml.metrics import NORM_MODES, modulus_to_uint8  # noqa: E402
from ml.pipeline import METHOD_HELP, Result, run  # noqa: E402

OPTICS = DEFAULT_OPTICS
WEIGHTS = ROOT / "data" / "weights" / "doe_multiplane.pt"
TESTSETS = ROOT / "data" / "testsets"

st.set_page_config(page_title="Photonika — синтез ДОЭ", layout="wide")
css = Path(__file__).resolve().parent / "assets" / "style.css"
if css.is_file():
    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(max_entries=2)
def load_net(path: str, mtime: int, device: str):
    from ml.model import load_model

    model, ckpt = load_model(path, device)
    return model, {k: ckpt.get(k) for k in ("model", "step", "best_score", "validation")}


@st.cache_resource(max_entries=2)
def get_prop(device: str):
    return FresnelSingleFFT(OPTICS).to(device)


def phase_rgb(doe_u8: np.ndarray) -> np.ndarray:
    import matplotlib

    return (matplotlib.colormaps["twilight"](doe_u8 / 255.0)[..., :3] * 255).astype(np.uint8)


def to_u8_image(file) -> np.ndarray:
    with Image.open(file) as im:
        arr = np.asarray(im.convert("L"), dtype=np.uint8)
    return arr if arr.shape == (OPTICS.size, OPTICS.size) else fit_square(arr, OPTICS.size)


def zip_result(res: Result, norm: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def png(name, arr):
            b = io.BytesIO()
            Image.fromarray(arr, mode="L").save(b, format="PNG")
            z.writestr(name, b.getvalue())
        png("doe.png", res.doe_u8)
        for i, (s, r) in enumerate(zip(res.sources_u8, res.recon_u8), 1):
            png(f"recon_{i}.png", r)
            png(f"source_{i}.png", s)
        z.writestr("metrics.json", json.dumps(res.to_json(OPTICS, norm), indent=2, ensure_ascii=False))
    return buf.getvalue()


# ---------------------------------------------------------------------------
st.title("Нейросетевой синтез фазового ДОЭ: 4 изображения на 4 расстояниях")
st.markdown(
    "Один фазовый ДОЭ 128×128 формирует 4 разных изображения на расстояниях "
    "**24,1 / 25,3 / 26,5 / 27,7 мм** (λ = 532 нм, пиксель ПВМС 10 мкм). "
    "Восстановление — дифракция Френеля с одним FFT; метрики считаются по 8-битным PNG."
)

with st.sidebar:
    st.header("Метод")
    device = st.selectbox("Устройство", (["cuda"] if torch.cuda.is_available() else []) + ["cpu"])
    has_net = WEIGHTS.is_file()
    methods = (["net+gs", "net", "net+adam"] if has_net else []) + ["gs+adam", "gs", "adam"]
    method = st.selectbox("Метод синтеза", methods, format_func=lambda m: f"{m} — {METHOD_HELP[m]}")
    if not has_net:
        st.info("Весов сети нет (data/weights/doe_multiplane.pt) — доступны итерационные методы.")
    gs_iters = st.slider("Итераций GS", 10, 500, 200, 10)
    adam_steps = st.slider("Шагов Adam", 0, 1000, 400, 50)
    refine_steps = st.slider("Шагов доуточнения после сети (GS или Adam)", 0, 200, 25, 5)
    norm = st.selectbox("Нормировка модуля в 0–255", NORM_MODES,
                        help="max — деление на максимум (протокол организаторов: каждое изображение "
                             "нормируется на 1); lsq — масштаб по МНК к исходному; p99.5 — на 99,5-й перцентиль. "
                             "Метрики всегда считаются после нормировки обоих изображений на максимум.")
    model = None
    if has_net:
        try:
            model, info = load_net(str(WEIGHTS), WEIGHTS.stat().st_mtime_ns, device)
            st.caption(f"Сеть: {info['model']}, шаг {info['step']}")
        except Exception as exc:  # несовместимые веса не должны ломать демо
            st.warning(f"Веса не загрузились: {exc}")

st.header("1. Исходные изображения")
source_mode = st.radio("Откуда взять 4 картинки", ["Тестовый набор", "Загрузить PNG"], horizontal=True)
sources = None
if source_mode == "Тестовый набор":
    names = sorted(p.name for p in TESTSETS.iterdir() if p.is_dir()) if TESTSETS.is_dir() else []
    if not names:
        st.warning("Нет data/testsets — выполните python -m ml.cli make-testsets")
    else:
        name = st.selectbox("Набор", names, index=names.index("photos_classic") if "photos_classic" in names else 0)
        sources = np.stack([to_u8_image(TESTSETS / name / f"{i}.png") for i in range(1, 5)])
else:
    files = st.file_uploader("4 PNG по порядку расстояний (имена сортируются)", type=["png", "jpg", "jpeg", "bmp"],
                             accept_multiple_files=True)
    if files:
        files = sorted(files, key=lambda f: f.name)
        if len(files) != 4:
            st.error(f"Нужно ровно 4 изображения, загружено {len(files)}")
        else:
            sources = np.stack([to_u8_image(f) for f in files])

if sources is not None:
    cols = st.columns(4)
    for i, (c, img) in enumerate(zip(cols, sources)):
        c.image(img, caption=f"{i + 1}: z = {OPTICS.distances[i] * 1e3:.1f} мм", width="stretch")

st.header("2. Синтез ДОЭ")
if sources is not None and st.button("Синтезировать", type="primary"):
    with st.spinner("Считаю…"):
        st.session_state["result"] = run(sources, method, norm=norm, model=model, device=device,
                                         prop=get_prop(device), gs_iters=gs_iters, adam_steps=adam_steps,
                                         refine_steps=refine_steps)
        st.session_state["norm"] = norm
        st.session_state.pop("gs_ref", None)

res: Result | None = st.session_state.get("result")
if res is not None:
    norm_used = st.session_state.get("norm", norm)
    left, right = st.columns([1, 2])
    with left:
        st.image(phase_rgb(res.doe_u8), caption="ДОЭ: фаза от −π до π (цветовая шкала twilight)",
                 width="stretch")
        st.metric("Время синтеза", f"{res.time_ms:.1f} мс")
        st.metric("PCC (среднее)", f"{res.summary['pcc']:.3f}")
        st.metric("SSIM (среднее)", f"{res.summary['ssim']:.3f}")
        st.download_button("Скачать doe.png, recon_*.png, metrics.json", zip_result(res, norm_used),
                           file_name="doe_result.zip", mime="application/zip")
    with right:
        cols = st.columns(4)
        for i, c in enumerate(cols):
            r = res.rows[i]
            c.image(res.recon_u8[i], caption=f"z = {OPTICS.distances[i] * 1e3:.1f} мм", width="stretch")
            c.caption(f"PCC {r['pcc']:.3f} · SSIM {r['ssim']:.3f}")
        st.dataframe(
            [{"плоскость": i + 1, "z, мм": OPTICS.distances[i] * 1e3, "PCC": r["pcc"], "SSIM": r["ssim"],
              "SSIM (глоб., слайд 3)": r["ssim_global"]} for i, r in enumerate(res.rows)],
            hide_index=True, width="stretch")

    st.header("3. Сравнение по времени с итерационным методом")
    if st.button("Посчитать GS-50 на тех же картинках"):
        with st.spinner("GS-50…"):
            st.session_state["gs_ref"] = run(res.sources_u8, "gs", norm=norm_used, device=device,
                                             prop=get_prop(device), gs_iters=50)
    ref: Result | None = st.session_state.get("gs_ref")
    if ref is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{res.method}: время", f"{res.time_ms:.1f} мс")
        c2.metric("GS-50: время", f"{ref.time_ms:.1f} мс",
                  delta=f"{ref.time_ms / max(res.time_ms, 1e-6):.1f}× медленнее" if ref.time_ms > res.time_ms else None)
        c3.metric("PCC: метод / GS-50", f"{res.summary['pcc']:.3f} / {ref.summary['pcc']:.3f}")

    st.header("4. 3D: картина вдоль оси z")
    st.markdown("Двигайте z: изображения по очереди фокусируются на своих расстояниях.")
    z_mm = st.slider("z, мм", 23.0, 29.0, float(OPTICS.distances[0] * 1e3), 0.1)
    mod = fresnel_modulus_np(uint8_to_phase(res.doe_u8), OPTICS, [z_mm * 1e-3])[0]
    nearest = int(np.argmin([abs(z_mm - d * 1e3) for d in OPTICS.distances]))
    img = modulus_to_uint8(mod, mode="p99.5")
    c1, c2 = st.columns([2, 1])
    c1.image(img, caption=f"|K| при z = {z_mm:.1f} мм (нормировка p99.5)", width="stretch")
    c2.write(f"Ближайшая плоскость: **{nearest + 1}** (z = {OPTICS.distances[nearest] * 1e3:.1f} мм)")
    c2.image(res.sources_u8[nearest], caption="её исходное изображение", width="stretch")
    if z_mm * 1e-3 < OPTICS.critical_distance:
        c2.caption(f"z < N·Δx²/λ = {OPTICS.critical_distance * 1e3:.2f} мм: квадратичная фаза "
                   "дискретизирована с алиасингом, картинка менее достоверна.")

st.markdown("---")
st.caption("Photonika · хакатон НИЯУ МИФИ «Разработка нейросетевого метода синтеза ДОЭ»")
