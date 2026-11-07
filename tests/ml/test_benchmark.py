"""Reproducibility and accounting checks for the quantized model benchmark."""
from types import SimpleNamespace

import numpy as np
import pytest

from ml.config import DEFAULT_OPTICS
from tools.benchmark_models import benchmark, summarize


def test_summary_distinguishes_worst_plane_from_worst_pair():
    records = [
        {"planes": [{"pcc": .2, "ssim": .1}, {"pcc": .8, "ssim": .7}],
         "synthesis_time_ms": 2., "pipeline_time_ms": 3.},
        {"planes": [{"pcc": .8, "ssim": .7}, {"pcc": .8, "ssim": .9}],
         "synthesis_time_ms": 4., "pipeline_time_ms": 5.},
    ]
    result = summarize(records)
    assert result["pcc"] == pytest.approx(.65)
    assert result["min_plane_mean_pcc"] == pytest.approx(.5)
    assert result["worst_pair_pcc"] == pytest.approx(.2)
    assert result["synthesis_time_ms_median"] == 3.


def test_variants_share_inputs_and_exclude_warmup(monkeypatch, tmp_path):
    import ml.model
    import ml.pipeline
    import tools.benchmark_models as tool

    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"benchmark fixture")
    quads = {"photos": np.arange(2 * 4 * 8 * 8, dtype=np.uint8).reshape(2, 4, 8, 8)}
    monkeypatch.setattr(tool, "fixed_quads", lambda split, count: quads)
    metadata = {"optics": DEFAULT_OPTICS.as_dict(), "model": {"architecture": "fixture"}, "step": 3}
    monkeypatch.setattr(ml.model, "load_model", lambda path, device: (object(), metadata))
    calls = []

    def fake_run(source, method, **kwargs):
        calls.append((id(source), method, kwargs["seed"]))
        # Warmup timing would dominate if it accidentally enters summaries.
        duration = 10000. if len(calls) in (1, 6, 11) else 2.
        return SimpleNamespace(rows=[{"pcc": .25, "ssim": .125}] * 4, time_ms=duration)

    monkeypatch.setattr(ml.pipeline, "run", fake_run)
    report = benchmark([path], device="cpu", count=2, seeds=(4, 7), gs_iters=2, refine_steps=1,
                       classic=False)
    assert len(report["variants"]) == 3
    assert len(calls) == 3 * (1 + 2 * 2)
    assert report["protocol"]["distinct_quads"] == 2
    assert report["protocol"]["synthesis_seeds"] == [4, 7]
    assert len(report["protocol"]["input_sha256"]["photos"]) == 64
    for variant in report["variants"]:
        assert variant["summary"]["runs"] == 4
        assert variant["summary"]["synthesis_time_ms_median"] == 2.
        assert variant["summary"]["pcc"] == .25
    # Every method receives the exact same source arrays and seed sequence.
    assert [(s, seed) for s, _, seed in calls[1:5]] == [(s, seed) for s, _, seed in calls[6:10]]
    assert [(s, seed) for s, _, seed in calls[1:5]] == [(s, seed) for s, _, seed in calls[11:15]]


def test_classic_photo_sources_are_separately_reported(monkeypatch, tmp_path):
    import ml.model
    import ml.pipeline
    import tools.benchmark_models as tool

    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(tool, "fixed_quads", lambda split, count:
                        {"photos": np.zeros((2, 4, 8, 8), dtype=np.uint8)})
    monkeypatch.setattr(tool, "classic_testsets", lambda:
                        {"photos_classic": np.ones((4, 8, 8), dtype=np.uint8)})
    metadata = {"optics": DEFAULT_OPTICS.as_dict(), "model": {"architecture": "fixture"}}
    monkeypatch.setattr(ml.model, "load_model", lambda path, device: (object(), metadata))
    monkeypatch.setattr(ml.pipeline, "run", lambda source, method, **kwargs:
                        SimpleNamespace(rows=[{"pcc": float(source.mean()), "ssim": .1}] * 4,
                                        time_ms=1.))
    report = benchmark([path], device="cpu", count=2, gs_iters=1, refine_steps=0)
    assert report["protocol"]["classic_case_names"] == ["photos_classic"]
    assert report["protocol"]["generated_distinct_quads"] == 2
    assert report["protocol"]["distinct_quads"] == 3
    for variant in report["variants"]:
        assert variant["by_population"]["generated"]["pcc"] == 0.
        assert variant["by_population"]["classic"]["pcc"] == 1.
        assert variant["by_kind"]["classic/photos_classic"]["runs"] == 1


def test_strong_baseline_receives_matching_gs_budget(monkeypatch, tmp_path):
    import ml.model
    import ml.pipeline
    import tools.benchmark_models as tool

    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(tool, "fixed_quads", lambda split, count:
                        {"photos": np.zeros((1, 4, 8, 8), dtype=np.uint8)})
    metadata = {"optics": DEFAULT_OPTICS.as_dict(), "model": {"architecture": "fixture"}}
    monkeypatch.setattr(ml.model, "load_model", lambda path, device: (object(), metadata))
    calls = []

    def fake_run(source, method, **kwargs):
        calls.append((method, kwargs))
        return SimpleNamespace(rows=[{"pcc": .2, "ssim": .1}] * 4, time_ms=1.)

    monkeypatch.setattr(ml.pipeline, "run", fake_run)
    report = benchmark([path], device="cpu", count=1, gs_iters=17, adam_steps=23,
                       refine_steps=0, classic=False)
    assert [variant["method"] for variant in report["variants"]] == ["gs", "gs+adam", "net"]
    baseline = report["variants"][1]
    assert baseline["params"] == {"gs_iters": 17, "adam_steps": 23}
    for method, kwargs in calls:
        if method.startswith("gs"):
            assert kwargs["gs_iters"] == 17
        if method == "gs+adam":
            assert kwargs["adam_steps"] == 23


def test_source_protocol_requests_held_out_source_split_and_records_manifest(monkeypatch, tmp_path):
    import ml.data
    import ml.model
    import ml.pipeline
    import tools.benchmark_models as tool

    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"source split fixture")
    calls = []

    def quads(split, count, photo_split):
        calls.append((split, photo_split))
        return {"photos": np.zeros((1, 4, 8, 8), dtype=np.uint8)}

    manifest = [{"id": "held-out", "sha256": "sample", "split": "test"}]
    monkeypatch.setattr(tool, "fixed_quads", quads)
    monkeypatch.setattr(ml.data, "photo_source_manifest", lambda: manifest)
    metadata = {"optics": DEFAULT_OPTICS.as_dict(), "model": {"architecture": "fixture"}}
    monkeypatch.setattr(ml.model, "load_model", lambda *a: (object(), metadata))
    monkeypatch.setattr(ml.pipeline, "run", lambda *a, **kw:
                        SimpleNamespace(rows=[{"pcc": .2, "ssim": .1}] * 4, time_ms=1.))
    report = benchmark([path], device="cpu", count=1, split="test", photo_split="source",
                       gs_iters=1, refine_steps=0, classic=False)
    assert calls == [("test", "test")]
    assert report["protocol"]["photo_sources"]["hash"] == ml.data.photo_manifest_hash(manifest)
    assert report["protocol"]["photo_sources"]["sources"] == manifest
    assert report["protocol"]["split_scope"] == "content-disjoint photograph sources"


@pytest.mark.parametrize("kwargs", [{"count": 0}, {"gs_iters": 0},
                                    {"refine_steps": -1}, {"adam_steps": -1},
                                    {"seeds": ()}, {"seeds": (1, 1)}])
def test_invalid_budgets_fail_before_loading_data(tmp_path, kwargs):
    with pytest.raises(ValueError):
        benchmark([tmp_path / "missing.pt"], device="cpu", **kwargs)
