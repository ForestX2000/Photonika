"""Тесты физики: формула, согласие numpy/torch, энергия, PNG-кодирование фазы, фокусировка."""
import numpy as np
import pytest
import torch

from ml.config import DEFAULT_OPTICS, Optics
from ml.fresnel import (FresnelSingleFFT, fresnel_direct_np, fresnel_modulus_np, phase_to_uint8,
                         uint8_to_phase, wrap_phase)


def test_physical_constants_from_task():
    o = DEFAULT_OPTICS
    assert o.size == 128 and o.wavelength == 532e-9 and o.pitch == 10e-6
    assert o.distances == (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
    # критическое расстояние N·Δx²/λ = 24,06 мм — все z из условия чуть больше него
    assert o.critical_distance == pytest.approx(24.06e-3, abs=0.01e-3)
    assert all(z > o.critical_distance for z in o.distances)
    assert o.output_pitch(24.1e-3) == pytest.approx(10.02e-6, rel=1e-3)
    assert o.output_pitch(27.7e-3) == pytest.approx(11.51e-6, rel=1e-3)


def test_fft_matches_direct_formula_small_grid():
    # N = 16, z берём ≥ критического для этой сетки
    o = Optics(size=16, distances=(3.2e-3,))
    rng = np.random.default_rng(0)
    phase = rng.uniform(-np.pi, np.pi, (16, 16))
    fast = fresnel_modulus_np(phase, o)[0]
    direct = fresnel_direct_np(phase, o, 3.2e-3)
    np.testing.assert_allclose(fast, direct, atol=1e-10)


def test_torch_matches_numpy():
    rng = np.random.default_rng(1)
    phase = rng.uniform(-np.pi, np.pi, (128, 128))
    ref = fresnel_modulus_np(phase)
    got = FresnelSingleFFT()(torch.from_numpy(phase)[None]).numpy()[0]
    np.testing.assert_allclose(got, ref, atol=2e-4)


def test_energy_is_conserved():
    rng = np.random.default_rng(2)
    phase = rng.uniform(-np.pi, np.pi, (128, 128))
    m = fresnel_modulus_np(phase)
    # ortho-FFT, единичная амплитуда: Σ|K|² = N² в каждой плоскости
    np.testing.assert_allclose((m ** 2).sum((-2, -1)), 128 * 128, rtol=1e-10)


def test_back_is_inverse_of_field():
    prop = FresnelSingleFFT()
    phase = torch.rand(2, 128, 128) * 6.28 - 3.14
    back = prop.back(prop.field(phase))
    expected = torch.polar(torch.ones_like(phase), phase)[:, None].expand_as(back)
    assert torch.allclose(back, expected, atol=1e-4)


def test_phase_png_roundtrip():
    rng = np.random.default_rng(3)
    phase = rng.uniform(-np.pi, np.pi, (128, 128))
    code = phase_to_uint8(phase)
    assert code.dtype == np.uint8
    assert phase_to_uint8(np.array([-np.pi]))[0] in (0, 255)   # −π и +π — одна и та же фаза
    assert phase_to_uint8(np.array([np.pi]))[0] == 255
    assert phase_to_uint8(np.array([-np.pi + 1e-9]))[0] == 0
    err = np.abs(wrap_phase(uint8_to_phase(code) - phase))
    assert err.max() <= np.pi / 255 + 1e-9
    np.testing.assert_array_equal(phase_to_uint8(uint8_to_phase(code)), code)


def test_lens_focuses_at_its_own_distance():
    """Фаза линзы −π/(λz)·r² компенсирует Q(z): в плоскости z свет собирается в точку."""
    o = DEFAULT_OPTICS
    c = (np.arange(128) - 64) * o.pitch
    r2 = c[None] ** 2 + c[:, None] ** 2
    for i, z in enumerate(o.distances):
        lens = -np.pi / (o.wavelength * z) * r2
        m = fresnel_modulus_np(lens)
        peaks = [m[j].max() for j in range(4)]
        assert int(np.argmax(peaks)) == i
        assert np.unravel_index(m[i].argmax(), m[i].shape) == (64, 64)
