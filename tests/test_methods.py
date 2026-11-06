"""Тесты методов синтеза, модели и конвейера в формате сдачи."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from app.baselines import adam_phase, gs_multiplane
from app.data import classic_testsets, load_png
from app.fresnel import FresnelSingleFFT
from app.losses import plane_metrics
from app.model import MultiPlaneUNet
from app.pipeline import run, write_outputs


@pytest.fixture(scope="module")
def letters():
    return classic_testsets()["mephi_letters"]


def test_gs_focuses_each_image_on_its_own_plane():
    # 4 разных фото (в «МИФИ» буква И повторяется, поэтому для этой проверки он не годится)
    photos = classic_testsets()["photos_classic"]
    prop = FresnelSingleFFT()
    target = torch.from_numpy(photos / 255.0).float()[None]
    phase = gs_multiplane(target, prop, 60, seed=0)
    mod = prop(phase)[0]
    # матрица PCC «плоскость i против цели j»: диагональ должна доминировать
    c = np.array([[float(plane_metrics(mod[i][None, None], target[0, j][None, None])[0]) for j in range(4)]
                  for i in range(4)])
    off = c[~np.eye(4, dtype=bool)]
    assert np.mean(np.diag(c)) > off.mean() + 0.2
    for i in range(3):  # плоскость 4 (Луна, низкий контраст) страдает от помех сильнее всех
        assert c[i, i] == c[i].max()


def test_adam_improves_on_gs(letters):
    prop = FresnelSingleFFT()
    target = torch.from_numpy(letters / 255.0).float()[None]
    gs = gs_multiplane(target, prop, 30, seed=0)
    refined = adam_phase(target, prop, 60, init_phase=gs, lr=0.05)
    from app.losses import multiplane_loss
    # Adam оптимизирует функцию потерь (PCC + SSIM по протоколу жюри + MSE) — она должна упасть
    l0 = multiplane_loss(prop(gs), target)[0].mean()
    l1 = multiplane_loss(prop(refined), target)[0].mean()
    assert l1 < l0
    assert refined.abs().max() <= np.pi + 1e-5


@pytest.mark.parametrize("mode", ["raw", "backprop", "imgphase"])
def test_model_shapes_and_start_point(mode):
    net = MultiPlaneUNet(base=8, input_mode=mode).eval()
    x = torch.rand(2, 4, 128, 128)
    with torch.no_grad():
        phase = net(x)
        assert phase.shape == (2, 128, 128)
        assert phase.abs().max() <= np.pi + 1e-6
        # необученная сеть выдаёт ровно базовую фазу (∠S или 0)
        _, base = net.features(x)
        if base is None:  # imgphase: старт тоже ∠S, считаем его напрямую
            from app.baselines import normalize_amplitude
            base = torch.angle(net.prop.back(normalize_amplitude(x) * net.features.psi).sum(1))
        diff = torch.angle(torch.polar(torch.ones_like(phase), phase - base))
        assert diff.abs().max() < 1e-4


def test_unrolled_model_starts_as_gs():
    """Необученная развёрнутая сеть == K итераций многоплоскостного GS от ∠S."""
    net = MultiPlaneUNet(base=8, unroll=3, stage_base=8).eval()
    x = torch.rand(1, 4, 128, 128)
    with torch.no_grad():
        phase = net(x)
        _, base = net.features(x)
        gs = gs_multiplane(x, FresnelSingleFFT(), 3, init_phase=base)
    diff = torch.angle(torch.polar(torch.ones_like(phase), phase - gs))
    assert diff.abs().max() < 1e-3


def test_model_gradient_flows():
    net = MultiPlaneUNet(base=8, unroll=1, stage_base=8)
    prop = FresnelSingleFFT()
    x = torch.rand(2, 4, 128, 128)
    from app.losses import multiplane_loss
    for _ in range(2):  # после первого шага голова уже не нулевая
        loss, _ = multiplane_loss(prop(net(x)), x)
        loss.mean().backward()
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p -= 1e-3 * p.grad
                    p.grad = None
    loss, _ = multiplane_loss(prop(net(x)), x)
    loss.mean().backward()
    assert net.unet.enc[0].net[0].weight.grad.abs().sum() > 0
    assert net.stages[0].net.enc[0].net[0].weight.grad.abs().sum() > 0


def test_pipeline_writes_submission_format(tmp_path, letters):
    res = run(letters, "gs", gs_iters=20, device="cpu")
    final = write_outputs(res, tmp_path)
    for name in ["doe.png"] + [f"recon_{i}.png" for i in range(1, 5)]:
        with Image.open(tmp_path / name) as im:
            assert im.mode == "L" and im.size == (128, 128)
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(report["planes"]) == 4
    assert [p["distance_mm"] for p in report["planes"]] == [24.1, 25.3, 26.5, 27.7]
    assert report["mean"]["pcc"] == pytest.approx(final.summary["pcc"])
    np.testing.assert_array_equal(load_png(tmp_path / "doe.png"), final.doe_u8)
    assert (tmp_path / "metrics.csv").is_file() and (tmp_path / "preview.png").is_file()


@pytest.mark.parametrize("method", ["net", "net+gs", "net+adam"])
def test_pipeline_with_network(method, letters):
    net = MultiPlaneUNet(base=8).eval()
    res = run(letters, method, model=net, refine_steps=5, device="cpu")
    assert res.doe_u8.shape == (128, 128) and res.recon_u8.shape == (4, 128, 128)
    assert -1.0 <= res.summary["pcc"] <= 1.0
    if method == "net+gs":  # необученная сеть = ∠S, дальше 5 итераций GS не ухудшают старт
        base = run(letters, "net", model=net, device="cpu")
        assert res.summary["pcc"] > base.summary["pcc"]
