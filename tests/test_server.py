"""Тесты HTTP API (FastAPI TestClient): синтез по тестовому набору и по загрузке, 3D-срез, архив, ошибки."""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.config import DEFAULT_OPTICS, PHYSICS_VERSION  # noqa: E402
from app.model import MultiPlaneUNet  # noqa: E402
from server.main import create_app  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
needs_testsets = pytest.mark.skipif(not (ROOT / "data/testsets/mephi_letters").is_dir(),
                                    reason="run `python -m app.cli make-testsets` first")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    missing = tmp_path_factory.mktemp("w") / "none.pt"
    return TestClient(create_app(weights=missing, device="cpu"))


@pytest.fixture(scope="module")
def net_client(tmp_path_factory):
    path = tmp_path_factory.mktemp("w") / "tiny.pt"
    net = MultiPlaneUNet(base=8)
    torch.save({"physics_version": PHYSICS_VERSION, "optics": DEFAULT_OPTICS.as_dict(), "model": net.config(),
                "net": net.state_dict(), "step": 0}, path)
    return TestClient(create_app(weights=path, device="cpu"))


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_info_and_testsets(client):
    info = client.get("/api/info").json()
    assert info["distances_mm"] == [24.1, 25.3, 26.5, 27.7]
    assert info["weights"] is False and info["default_method"] == "gs+adam"
    assert {m["id"]: m["available"] for m in info["methods"]}["net+gs"] is False
    assert client.get("/api/health").json() == {"status": "ok"}


@needs_testsets
def test_synthesize_testset_slice_compare_zip(client):
    sets = client.get("/api/testsets").json()
    assert any(s["name"] == "mephi_letters" for s in sets)
    assert client.get(sets[0]["images"][0]).headers["content-type"] == "image/png"
    r = client.post("/api/synthesize", data={"testset": "mephi_letters", "method": "gs", "gs_iters": 15})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["planes"]) == 4 and body["planes"][0]["recon"].startswith("data:image/png;base64,")
    assert 0.3 < body["mean"]["pcc"] <= 1.0
    rid = body["id"]

    sl = client.get(f"/api/results/{rid}/slice.png", params={"z_mm": 26.5})
    assert sl.status_code == 200 and sl.content[:4] == b"\x89PNG"
    assert client.get(f"/api/results/{rid}/slice.png", params={"z_mm": 99}).status_code == 400

    cmp = client.post(f"/api/results/{rid}/compare", params={"gs_iters": 5}).json()
    assert cmp["gs_iters"] == 5 and "time_ms" in cmp

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/results/{rid}/download.zip").content))
    names = set(z.namelist())
    assert {"doe.png", "metrics.json", "metrics.csv"} | {f"recon_{i}.png" for i in range(1, 5)} <= names
    with Image.open(io.BytesIO(z.read("doe.png"))) as im:
        assert im.mode == "L" and im.size == (128, 128)
    report = json.loads(z.read("metrics.json"))
    assert report["mean"]["pcc"] == pytest.approx(body["mean"]["pcc"])


def test_upload_rgb_and_other_size(client):
    rng = np.random.default_rng(0)
    files = {f"image_{i}": (f"{i}.png", _png(rng.integers(0, 256, (200, 160, 3), dtype=np.uint8)), "image/png")
             for i in range(1, 5)}
    r = client.post("/api/synthesize", data={"method": "gs", "gs_iters": 5}, files=files)
    assert r.status_code == 200, r.text
    assert len(r.json()["planes"]) == 4


def test_errors(client):
    r = client.post("/api/synthesize", data={"method": "gs"},
                    files={"image_1": ("a.png", _png(np.zeros((128, 128), np.uint8)), "image/png")})
    assert r.status_code == 400 and "4 изображения" in r.json()["detail"]
    files = {f"image_{i}": (f"{i}.png", b"not an image", "image/png") for i in range(1, 5)}
    assert client.post("/api/synthesize", data={"method": "gs"}, files=files).status_code == 400
    assert client.post("/api/synthesize", data={"testset": "mephi_letters", "method": "net"}).status_code == 400
    assert client.post("/api/synthesize", data={"testset": "mephi_letters", "method": "gs",
                                                "gs_iters": 5000}).status_code == 400
    assert client.post("/api/synthesize", data={"testset": "../../etc", "method": "gs"}).status_code == 400
    assert client.get("/api/results/deadbeef/slice.png", params={"z_mm": 25}).status_code == 404


@needs_testsets
def test_network_methods_when_weights_present(net_client):
    info = net_client.get("/api/info").json()
    assert info["weights"] is True and info["default_method"] == "net+gs"
    r = net_client.post("/api/synthesize", data={"testset": "mephi_letters", "method": "net+gs", "refine_steps": 5})
    assert r.status_code == 200, r.text
    assert r.json()["method"] == "net+gs"


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
