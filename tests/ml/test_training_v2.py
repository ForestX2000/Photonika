"""One real spectral-model update and resume through the complete trainer."""
import json

import numpy as np
import torch

from ml import training
from ml.config import Optics
from ml.model import build_model, load_model
from ml.fresnel import phase_to_uint8, uint8_to_phase


def test_training_quantization_matches_submission_codes_with_finite_gradient():
    phase = torch.linspace(-20.123, 20.321, 16384).reshape(1, 128, 128).requires_grad_(True)
    quantized = training.quantize_phase_ste(phase)
    reference = uint8_to_phase(phase_to_uint8(phase.detach().numpy()[0]))
    # The endpoints -pi/+pi are physically equivalent for a phase-only DOE.
    difference = np.angle(np.exp(1j * (quantized.detach().numpy()[0] - reference)))
    assert np.abs(difference).max() < 2e-6
    quantized.square().mean().backward()
    assert torch.isfinite(phase.grad).all()
    assert phase.grad.abs().sum() > 0


def test_real_v2_training_resumes_ema_optimizer_and_accumulated_samples(tmp_path, monkeypatch):
    optics = Optics(size=32)
    images = np.random.default_rng(5).integers(10, 256, (12, 32, 32), dtype=np.uint8)
    categories = np.arange(12) % 4
    val = {kind: images[:4][None] for kind in training.FIXED_KINDS}
    monkeypatch.setattr(training, "DEFAULT_OPTICS", optics)
    monkeypatch.setattr(training, "build_model", lambda config: build_model(config, optics))
    monkeypatch.setattr(training, "load_or_build_pool", lambda *a, **kw: (images, categories))
    monkeypatch.setattr(training, "fixed_quads", lambda *a, **kw: val)
    monkeypatch.setattr("ml.data.classic_testsets", lambda: {})
    options = dict(minutes=1, device="cpu", architecture="physics-unrolled", model_version=2,
                   base=4, stage_base=4, unroll=1, spectral_modes=3, batch="1", grad_accum=2,
                   pool_size=12, val_per_kind=1, output=tmp_path / "run", weights=tmp_path / "best.pt",
                   activation_checkpointing=True, quantization_aware=True, deep_supervision=0.1,
                   ema_decay=0.95, teacher_weight=0.05, teacher_every=1, teacher_gs_iters=1,
                   teacher_adam_steps=1)
    training.train(**options, max_steps=1)
    first = torch.load(options["output"] / "last.pt", weights_only=True)
    assert first["step"] == 1
    assert first["training"]["grad_accum"] == 2
    assert first["model"]["model_version"] == 2
    assert first["model"]["spectral_modes"] == 3
    assert first["ema"].keys() == first["net"].keys()
    assert any(not torch.equal(first["ema"][key], value) for key, value in first["net"].items())
    assert all(torch.isfinite(value).all() for value in first["net"].values())
    assert all(float(state["step"]) == 1 for state in first["optimizer"]["state"].values())

    training.train(**options, max_steps=2, resume=True)
    resumed = torch.load(options["output"] / "last.pt", weights_only=True)
    assert resumed["step"] == 2
    assert all(float(state["step"]) == 2 for state in resumed["optimizer"]["state"].values())
    for key, average in resumed["ema"].items():
        expected = first["ema"][key].lerp(resumed["net"][key], 0.75)
        torch.testing.assert_close(average, expected, atol=1e-7, rtol=1e-6)
    sampler = training.QuadSampler(images, categories)
    sampler.rng.bit_generator.state = first["sampler_rng"]
    sampler.sample(1)
    sampler.sample(1)
    assert sampler.rng.bit_generator.state == resumed["sampler_rng"]
    report = json.loads((options["output"] / "config.json").read_text(encoding="utf-8"))
    assert report["effective_batch"] == 2
    assert report["activation_checkpointing"] is True
    model, best = load_model(options["weights"])
    assert best["weight_source"] == "ema"
    with torch.inference_mode():
        assert torch.isfinite(model(torch.from_numpy(images[:4][None] / 255.0).float())).all()
