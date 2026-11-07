"""Validation and resumed training must preserve the meaning of reported metrics."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ml import training
from ml.data import QuadSampler
from ml.losses import max_normalize, multiplane_loss, ssim_global_t, ssim_t


def test_validation_disables_gradients_and_weights_every_example(monkeypatch):
    class Net(nn.Module):
        def forward(self, x):
            assert not torch.is_grad_enabled()
            assert not self.training
            return x

    def loss_fn(pred, target, **kwargs):
        return target.mean((1, 2, 3)), {}

    monkeypatch.setattr(training, "multiplane_loss", loss_fn)
    monkeypatch.setattr(training, "plane_metrics", lambda p, t: (t.mean((-2, -1)), t.mean((-2, -1))))
    val = {"a": torch.tensor([0.0, 0.0, 1.0])[:, None, None, None].expand(-1, 4, 8, 8),
           "b": torch.ones(2, 4, 8, 8)}
    result = training.evaluate(Net(), nn.Identity(), val, 2, torch.device("cpu"), {})
    assert result["a"]["loss"] == pytest.approx(1 / 3)
    for name in ("loss", "pcc", "ssim"):
        assert result[name] == pytest.approx(3 / 5)


def test_rng_checkpoint_restores_sample_stream_and_is_weights_only_safe(tmp_path):
    images = np.random.default_rng(4).integers(0, 256, (12, 8, 8), dtype=np.uint8)
    sampler = QuadSampler(images, np.arange(12) % 4, seed=7)
    state = training.capture_rng(sampler, torch.device("cpu"))
    path = tmp_path / "rng.pt"
    training.atomic_save(path, state)
    expected_images = sampler.sample(3)
    expected_random = torch.rand(5)
    expected_numpy = np.random.rand(5)
    sampler.sample(8)
    torch.rand(20)
    np.random.rand(20)
    training.restore_rng(torch.load(path, weights_only=True), sampler, torch.device("cpu"))
    np.testing.assert_array_equal(sampler.sample(3), expected_images)
    torch.testing.assert_close(torch.rand(5), expected_random)
    np.testing.assert_array_equal(np.random.rand(5), expected_numpy)


def test_ssim_global_weight_changes_the_actual_objective():
    gen = torch.Generator().manual_seed(10)
    a, b = torch.rand(2, 4, 32, 32, generator=gen), torch.rand(2, 4, 32, 32, generator=gen)
    a, b = max_normalize(a), max_normalize(b)
    local, _ = multiplane_loss(a, b, w_mse=0, w_pcc=0, worst=0, ssim_global=0)
    global_, _ = multiplane_loss(a, b, w_mse=0, w_pcc=0, worst=0, ssim_global=1)
    torch.testing.assert_close(local, (1 - ssim_t(a, b)).mean(1))
    torch.testing.assert_close(global_, (1 - ssim_global_t(a, b)).mean(1))
    assert not torch.allclose(local, global_)


def test_resume_restores_adam_moments_and_sampler_state(tmp_path, monkeypatch):
    config = {"architecture": "physics-unrolled", "base": 4, "input_mode": "raw",
              "unroll": 1, "stage_base": 4}

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.5))
            self.features = SimpleNamespace(channels=4)

        def forward(self, x):
            return self.weight * x.mean(1)

        def config(self):
            return config

    class Prop(nn.Module):
        def forward(self, phase):
            return phase[:, None].expand(-1, 4, -1, -1)

    def load(path, device):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        net = TinyNet()
        net.load_state_dict(checkpoint["net"])
        return net, checkpoint

    images = np.random.default_rng(0).integers(20, 256, (12, 8, 8), dtype=np.uint8)
    monkeypatch.setattr(training, "build_model", lambda cfg: TinyNet())
    monkeypatch.setattr(training, "load_model", load)
    monkeypatch.setattr(training, "FresnelSingleFFT", lambda opts: Prop())
    monkeypatch.setattr(training, "load_or_build_pool", lambda *a, **kw: (images, np.arange(12) % 4))
    monkeypatch.setattr(training, "fixed_quads", lambda *a: {k: images[:4][None] for k in training.FIXED_KINDS})
    monkeypatch.setattr("ml.data.classic_testsets", lambda: {})
    out, weights = tmp_path / "run", tmp_path / "weights.pt"
    options = dict(minutes=1, device="cpu", base=4, input_mode="raw", unroll=1, stage_base=4,
                   batch="1", pool_size=12, val_per_kind=1, output=out, weights=weights)
    training.train(**options, max_steps=1)
    first = torch.load(out / "last.pt", weights_only=True)
    assert first["step"] == 1
    first["schedule_progress"] = 0.9
    torch.save(first, out / "last.pt")
    training.train(**options, max_steps=2, resume=True)
    resumed = torch.load(out / "last.pt", weights_only=True)
    assert resumed["step"] == 2
    assert float(next(iter(resumed["optimizer"]["state"].values()))["step"]) == 2
    assert resumed["schedule_progress"] >= 0.9
    assert resumed["optimizer"]["param_groups"][0]["lr"] < 1e-6
    reference = QuadSampler(images, np.arange(12) % 4)
    reference.rng.bit_generator.state = first["sampler_rng"]
    reference.sample(1)
    assert resumed["sampler_rng"] == reference.rng.bit_generator.state


def test_incompatible_init_metadata_is_rejected():
    metadata = {"physics_version": training.PHYSICS_VERSION,
                "optics": training.DEFAULT_OPTICS.as_dict(), "model": {"architecture": "physics-unrolled"}}
    checkpoint = dict(metadata, model={"architecture": "unet"})
    with pytest.raises(ValueError, match="model"):
        training.check_checkpoint(checkpoint, metadata)


@pytest.mark.parametrize("changed", [{"seed": 99}, {"best_score": 0.9}, {"step": 11}])
def test_unrelated_best_weights_cannot_be_used_for_resume(changed):
    metadata = {"physics_version": training.PHYSICS_VERSION,
                "optics": training.DEFAULT_OPTICS.as_dict(),
                "model": {"architecture": "physics-unrolled"},
                "seed": 42, "loss": {"ssim_global": 0.5}, "pool_size": 100}
    checkpoint = dict(metadata, best_score=0.8, step=5)
    training.check_best_checkpoint(checkpoint, metadata, 0.8, 10)
    checkpoint.update(changed)
    with pytest.raises(ValueError, match="seed|score|step"):
        training.check_best_checkpoint(checkpoint, metadata, 0.8, 10)
