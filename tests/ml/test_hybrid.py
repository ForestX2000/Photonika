"""Hybrid synthesis must start at the network and retain the best exported DOE."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from ml.config import PHYSICS_VERSION, Optics
from ml.fresnel import FresnelSingleFFT
from ml.model import build_model
from ml.pipeline import run
from tools.benchmark_models import file_sha256, validate_neural_profiles
from tools.synthesize_a6000 import synthesize


def test_network_precedes_all_numerics_and_quantized_result_does_not_regress(monkeypatch):
    import ml.baselines
    import ml.hybrid

    prop = FresnelSingleFFT(Optics(size=32))
    calls = []
    initial = torch.rand(1, 32, 32) * 4 - 2

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prop = prop
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            calls.append("network")
            return initial + self.bias

    network = Network().eval()
    original_gs = ml.hybrid.gs_multiplane

    def gs(target, propagator, steps, init_phase, **kwargs):
        calls.append("gs")
        torch.testing.assert_close(init_phase, initial)
        return original_gs(target, propagator, steps, init_phase=init_phase, **kwargs)

    original_adam = torch.optim.Adam

    def adam(*args, **kwargs):
        calls.append("adam")
        return original_adam(*args, **kwargs)

    monkeypatch.setattr(ml.hybrid, "gs_multiplane", gs)
    monkeypatch.setattr(torch.optim, "Adam", adam)
    monkeypatch.setattr(ml.baselines, "random_phase", lambda *a, **k: pytest.fail("hybrid used a random classical start"))
    source = np.random.default_rng(2).integers(0, 256, (4, 32, 32), dtype=np.uint8)
    baseline = run(source, "net", optics=prop.optics, model=network)
    calls.clear()
    result = run(source, "net+gs+adam", optics=prop.optics, model=network, refine_steps=2, adam_steps=3)
    assert calls == ["network", "gs", "adam"]
    assert result.summary["pcc"] + result.summary["ssim"] >= baseline.summary["pcc"] + baseline.summary["ssim"] - 1e-10
    assert result.params["neural_primary"] is True
    assert result.params["refinement"]["random_restarts"] == 0
    assert result.params["refinement"]["quantized_checks"] == 3
    assert network.bias.grad is None


def test_hybrid_cannot_run_without_a_network():
    with pytest.raises(ValueError, match="trained weights"):
        run(np.zeros((4, 128, 128), dtype=np.uint8), "net+gs+adam")


@pytest.mark.parametrize("profile", [
    {"label": "bad", "method": "gs+adam", "params": {}},
    {"label": "bad", "method": "net+gs+adam", "params": {"adam_steps": -1}},
    {"label": "bad", "method": "net", "params": {"model": "replace"}},
])
def test_inference_profiles_cannot_replace_neural_synthesis(profile):
    with pytest.raises(ValueError):
        validate_neural_profiles([profile])


def test_portable_validated_preset_produces_submission_files(tmp_path):
    optics = Optics(size=32)
    model = build_model(dict(base=4, stage_base=4, unroll=0, input_mode="imgphase", architecture="unet"), optics)
    weights = tmp_path / "chosen.pt"
    torch.save(dict(physics_version=PHYSICS_VERSION, optics=optics.as_dict(), model=model.config(), net=model.state_dict()), weights)
    inputs = []
    for i, pixels in enumerate(np.random.default_rng(9).integers(0, 256, (4, 32, 32), dtype=np.uint8)):
        path = tmp_path / f"{i}.png"
        Image.fromarray(pixels).save(path)
        inputs.append(path)
    preset = tmp_path / "inference.json"
    config = dict(schema_version=1, neural_primary=True, selected_on="validation", weights="chosen.pt",
                  weights_sha256=file_sha256(weights),
                  profile={"label": "test-hybrid", "method": "net+gs+adam", "params": {"refine_steps": 1, "adam_steps": 1}})
    preset.write_text(json.dumps(config), encoding="utf-8")
    result = synthesize(preset, inputs, tmp_path / "out", device="cpu")
    assert result.method == "net+gs+adam"
    assert (tmp_path / "out/doe.png").is_file()
    report = json.loads((tmp_path / "out/metrics.json").read_text(encoding="utf-8"))
    assert report["inference_policy"]["selected_on"] == "validation"
    weights.write_bytes(b"unrelated weights")
    with pytest.raises(ValueError, match="do not match"):
        synthesize(preset, inputs, tmp_path / "out2", device="cpu")
