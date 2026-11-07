"""Тесты метрик: PCC по формуле, SSIM torch == skimage, нормировка в 8 бит."""
import numpy as np
import pytest
import torch

from app.losses import multiplane_loss, pcc_t, ssim_global_t, ssim_t
from app.metrics import modulus_to_uint8, pair_metrics, pcc, ssim, ssim_global


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (128, 128)).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-60, 60, (128, 128)), 0, 255).astype(np.uint8)
    return a, b


def test_pcc_matches_numpy():
    a, b = _pair()
    assert pcc(a, b) == pytest.approx(np.corrcoef(a.ravel(), b.ravel())[0, 1], abs=1e-12)
    assert pcc(a, a) == pytest.approx(1.0)
    assert pcc(a, np.zeros_like(a)) == 0.0


def test_torch_ssim_matches_skimage():
    a, b = _pair(1)
    ref = ssim(a, b)
    got = float(ssim_t(torch.from_numpy(a / 255.0).float(), torch.from_numpy(b / 255.0).float()))
    assert got == pytest.approx(ref, abs=2e-4)
    assert float(pcc_t(torch.from_numpy(a / 1.0), torch.from_numpy(b / 1.0))) == pytest.approx(pcc(a, b), abs=1e-6)


def test_ssim_identity():
    a, _ = _pair(2)
    assert ssim(a, a) == pytest.approx(1.0)
    assert ssim_global(a, a) == pytest.approx(1.0)


def test_modulus_normalisation_modes():
    a, _ = _pair(3)
    m = a.astype(float) * 0.013          # тот же рисунок в другом масштабе
    np.testing.assert_array_equal(modulus_to_uint8(m, a, "lsq"), a)
    assert modulus_to_uint8(m, mode="max").max() == 255
    assert modulus_to_uint8(m, mode="p99.5").dtype == np.uint8
    with pytest.raises(ValueError):
        modulus_to_uint8(m, mode="lsq")


def test_loss_is_zero_for_perfect_reconstruction():
    t = torch.rand(2, 4, 32, 32)
    loss, diag = multiplane_loss(t * 3.0, t)       # масштаб не важен
    assert torch.all(loss < 1e-3)
    assert diag["pcc"] == pytest.approx(1.0, abs=1e-5)


def test_jury_protocol_normalises_each_image_to_one():
    """Организаторы: сравниваются амплитуды, каждое изображение нормируется на 1 (делится на максимум)."""
    a, b = _pair(4)
    a = (a // 2) * 2                                   # чётные значения: a/2 точно целое
    half = (a // 2).astype(np.uint8)                   # то же изображение в 2 раза темнее
    m1, m2 = pair_metrics(a, b), pair_metrics(half, b)
    for k in ("pcc", "ssim", "ssim_global"):
        assert m1[k] == pytest.approx(m2[k], abs=1e-9)
    assert pair_metrics(a, a)["ssim"] == pytest.approx(1.0)


def test_torch_global_ssim_matches_numpy():
    a, b = _pair(5)
    ref = ssim_global(a / 255.0, b / 255.0, data_range=1.0)
    got = float(ssim_global_t(torch.from_numpy(a / 255.0), torch.from_numpy(b / 255.0)))
    assert got == pytest.approx(ref, abs=1e-6)
