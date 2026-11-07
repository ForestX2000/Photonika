"""A checkpoint must use the same optics as its reconstruction and API report."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from ml.config import DEFAULT_OPTICS, PHYSICS_VERSION, Optics
from ml.fresnel import FresnelSingleFFT
from ml.model import MultiPlaneUNet
from ml.pipeline import synthesize_phase


@pytest.mark.parametrize("method", ["net", "net+gs", "net+adam"])
@pytest.mark.parametrize("changes", [
    {"wavelength": 633e-9},
    {"pitch": 12e-6},
    {"distances": (0.0242, 0.0253, 0.0265, 0.0277)},
])
def test_pipeline_rejects_checkpoint_using_different_physics(method, changes):
    requested = Optics(size=32)
    model = MultiPlaneUNet(base=2, optics=replace(requested, **changes))
    sources = np.zeros((4, 32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="network checkpoint optics"):
        synthesize_phase(sources, method, model=model, optics=requested)


def test_pipeline_rejects_supplied_propagator_using_different_physics():
    optics = Optics(size=32)
    wrong_optics = replace(optics, wavelength=633e-9)
    model = MultiPlaneUNet(base=2, optics=wrong_optics)
    sources = np.zeros((4, 32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="propagator optics"):
        synthesize_phase(sources, "net", model=model, optics=optics,
                         prop=FresnelSingleFFT(wrong_optics))


def test_pipeline_rejects_propagator_with_overridden_distances():
    optics = Optics(size=32)
    model = MultiPlaneUNet(base=2, optics=optics)
    sources = np.zeros((4, 32, 32), dtype=np.uint8)
    prop = FresnelSingleFFT(optics, distances=(0.0242, 0.0253, 0.0265, 0.0277))
    with pytest.raises(ValueError, match="network checkpoint optics"):
        synthesize_phase(sources, "net", model=model, optics=optics, prop=prop)


def test_pipeline_accepts_matching_custom_optics():
    optics = Optics(size=32, wavelength=633e-9)
    model = MultiPlaneUNet(base=2, optics=optics).eval()
    sources = np.ones((4, 32, 32), dtype=np.uint8)
    phase, _, _ = synthesize_phase(sources, "net", model=model, optics=optics)
    assert phase.shape == (32, 32)
    assert np.isfinite(phase).all()


def test_api_marks_checkpoint_with_different_optics_unavailable(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from backend.main import create_app

    optics = replace(DEFAULT_OPTICS, wavelength=633e-9)
    model = MultiPlaneUNet(base=2, optics=optics)
    path = tmp_path / "wrong-optics.pt"
    torch.save(dict(physics_version=PHYSICS_VERSION, optics=optics.as_dict(),
                    model=model.config(), net=model.state_dict()), path)
    client = TestClient(create_app(weights=path, device="cpu"))
    response = client.get("/api/info")
    assert response.status_code == 200
    info = response.json()
    assert info["weights"] is False
    assert info["default_method"] == "gs+adam"
    assert "optics" in info["model"]["error"]
    assert all(not method["available"] for method in info["methods"] if method["id"].startswith("net"))
    assert info["wavelength_nm"] == pytest.approx(532.0)
