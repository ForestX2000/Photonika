"""HTTP API Photonika (FastAPI) + раздача фронтенда из web/.

    python -m server                         # http://127.0.0.1:8000
    python -m server --host 0.0.0.0 --port 8000 --device cuda

API (все ответы JSON, кроме PNG/ZIP):
    GET  /api/health
    GET  /api/info                               физика, устройство, методы, наличие весов
    GET  /api/testsets                           список тестовых наборов
    GET  /api/testsets/{name}/{i}.png            изображение набора (i = 1..4)
    POST /api/synthesize                         multipart: method, gs_iters, adam_steps, refine_steps,
                                                 testset ИЛИ image_1..image_4 → ДОЭ, 4 восстановления, метрики
    POST /api/results/{id}/compare?gs_iters=50   то же для GS-N (сравнение по времени и качеству)
    GET  /api/results/{id}/slice.png?z_mm=26.5   модуль поля на произвольном z (для 3D-просмотра)
    GET  /api/results/{id}/download.zip          doe.png, recon_1..4.png, source_1..4.png, metrics.json/csv
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .service import DEFAULT_WEIGHTS, SynthesisService, UserError, decode_image

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def create_app(weights: Path | None = None, device: str | None = None) -> FastAPI:
    weights = Path(weights or os.environ.get("PHOTONIKA_WEIGHTS", DEFAULT_WEIGHTS))
    device = device or os.environ.get("PHOTONIKA_DEVICE", "auto")
    service = SynthesisService(weights, device)
    app = FastAPI(title="Photonika API", version="1.0",
                  description="Синтез фазового ДОЭ под 4 изображения на 4 расстояниях")
    app.state.service = service

    # Фронтенд может работать на отдельном dev-сервере (например, Vite на :5173)
    origins = [o for o in os.environ.get("PHOTONIKA_CORS", "").split(",") if o]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(UserError)
    async def user_error(_: Request, exc: UserError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    def entry_or_404(rid: str):
        try:
            return service.get(rid)
        except KeyError:
            raise HTTPException(404, "Результат устарел или не найден — запустите синтез заново") from None

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/info")
    def info():
        return service.info()

    @app.get("/api/testsets")
    def testsets():
        return service.testsets()

    @app.get("/api/testsets/{name}/{index}.png")
    def testset_image(name: str, index: int):
        return FileResponse(service.testset_path(name, index), media_type="image/png",
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/api/synthesize")
    def synthesize(method: str = Form(None), gs_iters: int = Form(200), adam_steps: int = Form(400),
                   refine_steps: int = Form(25), testset: str | None = Form(None),
                   image_1: UploadFile | None = File(None), image_2: UploadFile | None = File(None),
                   image_3: UploadFile | None = File(None), image_4: UploadFile | None = File(None)):
        uploads = [image_1, image_2, image_3, image_4]
        if testset:
            sources = service.load_testset(testset)
        elif all(u is not None for u in uploads):
            import numpy as np

            sources = np.stack([decode_image(u.file.read(), u.filename or f"image_{i}")
                                for i, u in enumerate(uploads, 1)])
        else:
            raise UserError("Нужно 4 изображения (image_1..image_4) или тестовый набор (testset)")
        rid, result = service.synthesize(sources, method or service.default_method(), gs_iters, adam_steps,
                                         refine_steps)
        return service.to_payload(rid, result)

    @app.post("/api/results/{rid}/compare")
    def compare(rid: str, gs_iters: int = Query(50)):
        entry_or_404(rid)
        return service.compare(rid, gs_iters)

    @app.get("/api/results/{rid}/slice.png")
    def slice_png(rid: str, z_mm: float = Query(...)):
        entry_or_404(rid)
        return Response(service.slice_png(rid, z_mm), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=600"})

    @app.get("/api/results/{rid}/download.zip")
    def download(rid: str):
        entry_or_404(rid)
        return Response(service.zip_bytes(rid), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="photonika_{rid}.zip"'})

    if WEB.is_dir():
        app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    else:
        @app.get("/")
        def root():
            return {"message": "Photonika API. Фронтенд не найден (папка web/). Документация API: /docs"}

    return app


app = create_app()
