"""Experiment selection must be resumable and keep test data out of selection."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import a6000


def test_dry_run_needs_no_gpu_or_output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(a6000, "preflight", lambda *a: pytest.fail("preflight in dry run"))
    out = tmp_path / "new"
    plan = a6000.run(output=out, weights=tmp_path / "weights.pt", dry_run=True)
    assert not out.exists()
    assert plan["common_training"]["model_version"] == 2
    assert plan["common_training"]["photo_split"] == "source"
    assert plan["common_training"]["preview"] is False
    assert plan["pilot_validation"]["split"] == "val"
    assert plan["final_test"]["split"] == "test"
    assert plan["final_test"]["adam_steps"] == 200
    assert len(plan["candidates"]) == 3
    assert plan["candidates"][-1]["base"] == 64
    assert plan["nominal_main_training_minutes"] == 120
    assert plan["common_training"]["threads"] == 12
    assert plan["common_training"]["pool_in_ram"] is True
    assert plan["common_training"]["vram_limit_gib"] == 40
    assert plan["common_training"]["pool_size"] == 100000
    assert plan["selection_validation"]["neural_profiles"][-1]["method"] == "net+gs+adam"


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError, match="minutes"):
        a6000.make_plan(0)


def test_hybrid_policy_is_selected_on_validation_and_classical_variants_are_excluded():
    report = {"variants": [
        {"method": "gs+adam", "summary": {"pcc": 1., "ssim": 1.}, "checkpoint": {"path": "forbidden.pt"}},
        {"method": "net", "summary": {"pcc": .5, "ssim": .2}, "checkpoint": {"path": "small.pt"}},
        {"method": "net+gs+adam", "params": {"refine_steps": 25, "adam_steps": 50},
         "summary": {"pcc": .7, "ssim": .3, "synthesis_time_ms_median": 40.}, "checkpoint": {"path": "large.pt"}},
    ]}
    chosen = a6000.select_neural_variant(report)
    assert chosen["method"] == "net+gs+adam"
    assert chosen["checkpoint"]["path"] == "large.pt"
    assert chosen["params"]["adam_steps"] == 50


def test_low_ssd_space_is_rejected_and_budget_fits_the_reported_disk(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(a6000.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**30))
    with pytest.raises(RuntimeError, match="free SSD"):
        a6000.check_storage(tmp_path, 100000)
    monkeypatch.setattr(a6000.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * 2**30))
    budget = a6000.check_storage(tmp_path, 100000)
    assert budget["pool_bytes"] == 100000 * 128 * 128
    assert 8 * 2**30 < budget["required_bytes"] < 30 * 2**30


def fake_backend(monkeypatch, *, main_score=0.7, fail_large=False):
    calls = {"train": [], "benchmark": [], "pool": []}
    monkeypatch.setattr(a6000, "preflight", lambda d, p: {"device": "cpu", "gpu": {"name": "fake"}})
    monkeypatch.setattr(a6000, "check_storage", lambda *a: {"free_bytes": 20 * 2**30})
    monkeypatch.setattr(a6000, "prepare_pool", lambda count: calls["pool"].append(count))
    monkeypatch.setattr(a6000, "dataset_info", lambda: {"version": "fake-v1", "sha256": "source-hash",
                        "counts": {"train": 80, "val": 10, "test": 10}, "sources": []})

    def training(config, log, stop, deadline):
        calls["train"].append(config)
        if fail_large and config["base"] >= 48:
            raise RuntimeError("synthetic OOM")
        path = Path(config["weights"])
        path.write_text(json.dumps({"name": path.parent.name, "base": config["base"]}))
        (path.parent / "last.pt").write_text("resume")

    def benchmark(paths, *, device, parameters):
        calls["benchmark"].append((list(paths), parameters))
        variants = []
        for path in paths:
            name = Path(path).parent.name
            score = main_score if name == "main" else 0.65 if "capacity" in name else 0.6 if "quality" in name else 0.5
            variants.append({"method": "net", "checkpoint": {"path": str(Path(path).resolve())},
                             "summary": {"pcc": score, "ssim": 0.1}})
        return {"variants": variants, "split": parameters["split"]}

    monkeypatch.setattr(a6000, "run_training", training)
    monkeypatch.setattr(a6000, "run_benchmark", benchmark)
    return calls


def test_selects_on_validation_then_tests_once_and_resumes_without_retraining(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    options = dict(minutes=30, device="cpu", precision="fp32", skip_fetch=True,
                   output=tmp_path / "run", weights=tmp_path / "export.pt")
    result = a6000.run(**options)
    assert result["status"] == "completed"
    assert result["pilot_winner"] == "capacity_b64_u8"
    assert len(calls["train"]) == 4
    assert Path(calls["train"][-1]["init_weights"]).parts[-2:] == ("capacity_b64_u8", "best.pt")
    assert [p[1]["split"] for p in calls["benchmark"]] == ["val", "val", "val", "val", "test"]
    assert len(calls["benchmark"][-2][0]) == 4
    assert result["data"]["counts"]["train"] == 80
    assert result["quality_comparison"]["status"] == "not_available"
    a6000.run(**options, resume=True)
    assert len(calls["train"]) == 4
    assert len(calls["benchmark"]) == 5
    assert calls["pool"] == [100000]


def test_evaluation_thread_override_keeps_existing_resume_identity(tmp_path, monkeypatch):
    import torch

    calls = fake_backend(monkeypatch)
    original = a6000.run_benchmark

    def interrupted(paths, *, device, parameters):
        if parameters["split"] == "test":
            raise InterruptedError("stopped final evaluation")
        return original(paths, device=device, parameters=parameters)

    monkeypatch.setattr(a6000, "run_benchmark", interrupted)
    options = dict(minutes=30, device="cpu", precision="fp32", skip_fetch=True,
                   output=tmp_path / "run", weights=tmp_path / "export.pt")
    with pytest.raises(InterruptedError):
        a6000.run(**options, eval_threads=3)

    def resumed(paths, *, device, parameters):
        assert torch.get_num_threads() == 2
        assert parameters["cache_dir"] == options["output"] / "benchmark_cache" / "test"
        assert parameters["stop_file"] == options["output"] / "STOP"
        assert parameters["deadline"] > 0
        return original(paths, device=device, parameters=parameters)

    monkeypatch.setattr(a6000, "run_benchmark", resumed)
    result = a6000.run(**options, resume=True, eval_threads=2)
    assert result["status"] == "completed"
    assert len(calls["train"]) == 4


def test_main_training_is_not_assumed_better_than_pilot(tmp_path, monkeypatch):
    fake_backend(monkeypatch, main_score=0.1)
    result = a6000.run(minutes=30, device="cpu", precision="fp32", skip_fetch=True,
                       output=tmp_path / "run", weights=tmp_path / "export.pt")
    assert "capacity_b64_u8" in result["export"]["selected_from"]


def test_selected_hybrid_policy_is_exported_before_test_and_reused_unchanged(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    original = a6000.run_benchmark

    def benchmark(paths, *, device, parameters):
        report = original(paths, device=device, parameters=parameters)
        if parameters["split"] == "val" and parameters.get("neural_profiles"):
            report["variants"].append({"method": "net+gs+adam", "label": "validated-hybrid",
                "params": {"refine_steps": 3, "adam_steps": 5},
                "checkpoint": {"path": str(paths[-1].resolve())},
                "summary": {"pcc": .8, "ssim": .3, "synthesis_time_ms_median": 20.}})
        return report

    monkeypatch.setattr(a6000, "run_benchmark", benchmark)
    output = tmp_path / "run"
    result = a6000.run(minutes=30, device="cpu", precision="fp32", skip_fetch=True,
                       output=output, weights=tmp_path / "export.pt")
    preset = json.loads((output / "inference.json").read_text(encoding="utf-8"))
    assert preset["selected_on"] == "validation" and preset["neural_primary"] is True
    assert preset["profile"]["method"] == "net+gs+adam"
    assert preset["profile"]["params"] == {"refine_steps": 3, "adam_steps": 5}
    assert calls["benchmark"][-1][1]["split"] == "test"
    assert calls["benchmark"][-1][1]["neural_profiles"] == [preset["profile"]]
    assert result["export"]["neural_profile"] == preset["profile"]


def test_failed_large_pilot_preserves_smaller_candidate(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch, fail_large=True)
    result = a6000.run(minutes=30, device="cpu", precision="fp32", skip_fetch=True,
                       output=tmp_path / "run", weights=tmp_path / "export.pt")
    assert result["pilot_winner"] == "efficient_b32_u4"
    assert result["pilots"]["quality_b48_u8"]["status"] == "failed"
    assert result["pilots"]["capacity_b64_u8"]["status"] == "failed"
    assert calls["train"][-1]["base"] == 32


def test_resume_rejects_changed_artifact_and_configuration(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    options = dict(minutes=30, device="cpu", precision="fp32", skip_fetch=True,
                   output=tmp_path / "run", weights=tmp_path / "export.pt")
    result = a6000.run(**options)
    with pytest.raises(ValueError, match="configuration"):
        a6000.run(**options, resume=True, pool_size=1000)
    Path(result["export"]["weights"]).write_text("another run")
    with pytest.raises(RuntimeError, match="changed"):
        a6000.run(**options, resume=True)


def test_existing_output_is_never_overwritten_without_resume(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "valuable.log").write_text("keep")
    with pytest.raises(FileExistsError, match="not empty"):
        a6000.run(output=out, weights=tmp_path / "weights.pt")


def test_source_counts_require_real_photos_unless_explicit_smoke_override():
    tiny = {"counts": {"train": 10, "val": 4, "test": 1}}
    with pytest.raises(RuntimeError, match="Too few"):
        a6000.check_dataset(tiny, False)
    a6000.check_dataset(tiny, True)
    with pytest.raises(RuntimeError, match="nonempty"):
        a6000.check_dataset({"counts": {"train": 10, "val": 0, "test": 1}}, True)


def test_explicit_resume_removes_mirrored_phase_stop_and_keeps_selected_architecture(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    original_training = a6000.run_training

    def interrupt_main(config, log, stop, deadline):
        if Path(config["output"]).name == "main":
            original_training(config, log, stop, deadline)
            (Path(config["output"]) / "STOP").touch()
            stop.touch()
            raise InterruptedError("synthetic stop")
        original_training(config, log, stop, deadline)

    options = dict(minutes=30, device="cpu", precision="fp32", skip_fetch=True,
                   output=tmp_path / "run", weights=tmp_path / "export.pt")
    monkeypatch.setattr(a6000, "run_training", interrupt_main)
    with pytest.raises(InterruptedError, match="synthetic"):
        a6000.run(**options)
    (options["output"] / "STOP").unlink()

    def continue_main(config, log, stop, deadline):
        assert Path(config["output"]).name == "main"
        assert config["resume"]
        assert not (Path(config["output"]) / "STOP").exists()
        original_training(config, log, stop, deadline)

    monkeypatch.setattr(a6000, "run_training", continue_main)
    result = a6000.run(**options, resume=True)
    assert result["status"] == "completed"
    assert len(calls["train"]) == 5


def test_failed_small_dataset_preparation_can_fetch_on_resume(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    state = {"fetched": False}

    def information():
        return {"version": "fake", "sha256": "source-hash", "sources": [],
                "counts": {"train": 80, "val": 10, "test": 10} if state["fetched"] else
                          {"train": 10, "val": 4, "test": 1}}

    monkeypatch.setattr(a6000, "dataset_info", information)
    monkeypatch.setattr("ml.data.fetch_div2k", lambda: state.update(fetched=True))
    options = dict(minutes=30, device="cpu", precision="fp32",
                   output=tmp_path / "run", weights=tmp_path / "export.pt")
    with pytest.raises(RuntimeError, match="Too few"):
        a6000.run(**options, skip_fetch=True)
    manifest = json.loads((options["output"] / "manifest.json").read_text())
    assert not manifest["data_ready"]
    result = a6000.run(**options, resume=True)
    assert state["fetched"]
    assert result["status"] == "completed"


def test_quality_report_is_honest_when_neural_candidate_trails_baselines():
    report = {"variants": [
        {"method": "gs", "summary": {"pcc": 0.6, "ssim": 0.2}},
        {"method": "gs+adam", "summary": {"pcc": 0.65, "ssim": 0.3}},
        {"method": "net", "summary": {"pcc": 0.55, "ssim": 0.15}},
        {"method": "net+gs", "summary": {"pcc": 0.62, "ssim": 0.25}},
    ]}
    comparison = a6000.quality_comparison(report)
    assert comparison["status"] == "experimental"
    assert comparison["budgets_matched"] is False
    assert comparison["comparisons"]["net"]["gs"]["pcc_delta"] == pytest.approx(-0.05)
    assert comparison["comparisons"]["net+gs"]["gs"]["both_metrics_at_least_baseline"]
    assert not comparison["comparisons"]["net+gs"]["gs+adam"]["both_metrics_at_least_baseline"]


def test_quality_report_limits_positive_claim_to_this_test():
    report = {"variants": [
        {"method": "gs", "summary": {"pcc": 0.6, "ssim": 0.2}},
        {"method": "net", "summary": {"pcc": 0.65, "ssim": 0.3}},
    ]}
    comparison = a6000.quality_comparison(report)
    assert comparison["status"] == "competitive_on_this_test"
    assert "does not establish" in comparison["note"]


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="requires native POSIX bash")
def test_shell_install_dry_run_has_no_install_or_cuda_probe(tmp_path):
    log = tmp_path / "commands.log"
    python = tmp_path / "fake-python"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n')
    python.chmod(0o755)
    env = dict(os.environ, PHOTONIKA_PYTHON=str(python), COMMAND_LOG=str(log))
    subprocess.run(["bash", str(a6000.ROOT / "scripts/train-a6000.sh"), "--dry-run", "--install", "--minutes", "30"],
                   check=True, env=env)
    assert log.read_text().splitlines() == ["-u tools/a6000.py --dry-run --minutes 30"]
