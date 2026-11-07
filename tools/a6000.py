#!/usr/bin/env python3
"""Screen, train and evaluate DOE models on one rented GPU.

Dry run (no torch, CUDA or downloads): python tools/a6000.py --dry-run
Run: python tools/a6000.py --minutes 180
Continue: python tools/a6000.py --minutes 180 --resume --skip-fetch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation_runtime import announce, monitor

CANDIDATES = (
    {"name": "efficient_b32_u4", "base": 32, "unroll": 4, "stage_base": 16},
    {"name": "quality_b48_u8", "base": 48, "unroll": 8, "stage_base": 24},
    {"name": "capacity_b64_u8", "base": 64, "unroll": 8, "stage_base": 32},
)

NEURAL_REFINEMENTS = [
    {"label": "fast-gs25", "method": "net+gs", "params": {"refine_steps": 25}},
    {"label": "adam50", "method": "net+adam", "params": {"refine_steps": 50}},
    {"label": "gs25-adam50", "method": "net+gs+adam", "params": {"refine_steps": 25, "adam_steps": 50}},
    {"label": "gs25-adam150", "method": "net+gs+adam", "params": {"refine_steps": 25, "adam_steps": 150}},
]


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> str:
    """Prevent accidental continuation using different model/training code."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "ml").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_plan(minutes: float, *, precision: str = "bf16", pool_size: int = 100000,
              grad_accum: int = 1, max_steps: int | None = None,
              pilot_count: int = 4, test_count: int = 16, threads: int = 12,
              vram_limit_gib: float = 40.0, pool_in_ram: bool = True) -> dict:
    if minutes < 3 or pool_size < 100 or grad_accum < 1 or pilot_count < 1 or test_count < 1:
        raise ValueError("minutes >= 3, pool_size >= 100, and positive accumulation/evaluation counts required")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if precision not in ("fp32", "bf16", "auto"):
        raise ValueError("precision must be fp32, bf16 or auto")
    if threads < 1 or not 0 < vram_limit_gib <= 45:
        raise ValueError("positive threads and VRAM limit <= 45 GiB required")
    pilot_minutes = min(15.0, minutes * 0.1)
    reserve = min(15.0, max(0.5, minutes * 0.1))
    common = dict(architecture="physics-unrolled", model_version=2, spectral_modes=8,
                  input_mode="imgphase", precision=precision, activation_checkpointing=True,
                  grad_accum=grad_accum, batch="auto", auto_batch_max=128, auto_batch_seconds=60,
                  pool_size=pool_size, val_per_kind=16, eval_every=120.0, lr=2e-4,
                  photo_split="source", deep_supervision=0.2, quantization_aware=True,
                  ema_decay=0.999, teacher_weight=0.1, teacher_every=16,
                  teacher_gs_iters=8, teacher_adam_steps=8, seed=42)
    common["preview"] = False  # Classic test examples are reserved for final evaluation.
    common.update(threads=threads, vram_limit_gib=vram_limit_gib, pool_in_ram=pool_in_ram)
    if max_steps is not None:
        common["max_steps"] = max_steps
    return {
        "schema_version": 1,
        "minutes_per_invocation": minutes,
        "budget_scope": ("wall-time target including data preparation; reserve for selection and final test; "
                         "checkpoint finalization may require a short grace period"),
        "pilot_minutes_each": pilot_minutes,
        "evaluation_reserve_minutes": reserve,
        "nominal_main_training_minutes": max(0.5, minutes - len(CANDIDATES) * pilot_minutes - reserve),
        "candidates": [dict(c) for c in CANDIDATES],
        "hardware_profile": {"reported_vram_gb": 45, "cpu_cores": 12, "ram_gb": 282, "ssd_gb": 30},
        "common_training": common,
        "pilot_validation": {"split": "val", "count": pilot_count, "classic": False,
                             "gs_iters": 50, "adam_steps": 0, "refine_steps": 0,
                             "seeds": [0], "photo_split": "source"},
        "selection_validation": {"split": "val", "count": pilot_count, "classic": False,
                                 "gs_iters": 50, "adam_steps": 0, "refine_steps": 0,
                                 "seeds": [0], "photo_split": "source", "neural_profiles": NEURAL_REFINEMENTS},
        "final_test": {"split": "test", "count": test_count, "classic": True,
                       "gs_iters": 100, "adam_steps": 200, "refine_steps": 0,
                       "seeds": [0, 1, 2], "photo_split": "source"},
        "selection": "joint checkpoint + neural-first refinement selection by quantized validation PCC+SSIM; latency breaks ties",
        "test_policy": "test set is evaluated after all architecture/checkpoint selections are frozen",
    }


def preflight(device: str, precision: str) -> dict:
    import torch

    chosen = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    dev = torch.device(chosen)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA PyTorch build supported by the host driver")
    if dev.type == "cuda":
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 is unsupported on this GPU; pass --precision fp32")
        properties = torch.cuda.get_device_properties(dev)
        gpu = {"name": properties.name, "memory_gib": round(properties.total_memory / 2**30, 2)}
    else:
        if precision == "bf16":
            raise RuntimeError("This GPU training profile requires CUDA for bf16; use --precision fp32 for CPU checks")
        gpu = {"name": "CPU", "memory_gib": None}
    return {"device": str(dev), "gpu": gpu, "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "python": platform.python_version(), "platform": platform.platform()}


def run_training(config: dict, log: Path, stop: Path, deadline: float) -> None:
    """Isolate CUDA allocation between candidates; mirror the coordinator STOP file."""
    request = log.parent / "training_request.json"
    atomic_json(request, config)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--train-config", str(request)]
    started = time.monotonic()
    next_progress = started + 30
    with log.open("a", encoding="utf-8") as stream:
        environment = dict(os.environ, OMP_NUM_THREADS=str(config.get("threads", 12)),
                           MKL_NUM_THREADS=str(config.get("threads", 12)))
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        try:
            while process.poll() is None:
                if stop.exists():
                    (Path(config["output"]) / "STOP").touch()
                # Training itself observes its budget. This cap also covers a
                # stalled data download/evaluation in the child process.
                if time.monotonic() > deadline + 120:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TimeoutError(f"Training exceeded the invocation budget; saved artifacts remain in {log.parent}")
                if time.monotonic() >= next_progress:
                    print(f"[training] {log.parent.name}: {(time.monotonic() - started) / 60:.1f} min; log {log}", flush=True)
                    next_progress += 30
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"Training process exited {process.returncode}; inspect {log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run_benchmark(paths: list[Path], *, device: str, parameters: dict) -> dict:
    from tools.benchmark_models import benchmark

    kwargs = {**parameters, "seeds": tuple(parameters["seeds"])}
    return benchmark(paths, device=device, **kwargs)


def dataset_info() -> dict:
    from ml.data import PHOTO_SPLIT_VERSION, photo_manifest_hash, photo_source_manifest

    sources = photo_source_manifest()
    return {"version": PHOTO_SPLIT_VERSION, "sha256": photo_manifest_hash(sources), "sources": sources,
            "counts": {split: sum(s["split"] == split for s in sources) for split in ("train", "val", "test")}}


def prepare_pool(pool_size: int) -> None:
    from ml.data import load_or_build_pool
    images, categories = load_or_build_pool(pool_size, seed=0, photo_split="train", mmap=True)
    del images, categories


def check_dataset(info: dict, allow_small: bool) -> None:
    counts = info["counts"]
    if any(counts.get(split, 0) < 1 for split in ("train", "val", "test")):
        raise RuntimeError(f"Source-disjoint photos require nonempty train/val/test splits; got {counts}. Add photographs.")
    minimum = {"train": 32, "val": 4, "test": 4}
    if not allow_small and any(counts[split] < size for split, size in minimum.items()):
        raise RuntimeError(f"Too few independent photo sources: {counts}; required {minimum}. "
                           "Run without --skip-fetch to fetch DIV2K, or add photos under data/photos. "
                           "--allow-small-dataset is only for explicit smoke checks.")


def checkpoint_scores(report: dict) -> dict[str, float]:
    scores = {}
    for entry in report["variants"]:
        if entry["method"] != "net":
            continue
        path = str(Path(entry["checkpoint"]["path"]).resolve())
        scores[path] = float(entry["summary"]["pcc"] + entry["summary"]["ssim"])
    return scores


def select_neural_variant(report: dict) -> dict:
    from tools.benchmark_models import NEURAL_METHODS

    variants = [entry for entry in report["variants"]
                if entry["method"] in NEURAL_METHODS and entry.get("checkpoint")]
    if not variants:
        raise ValueError("no neural-first candidates available for deployment")

    def key(entry):
        summary = entry["summary"]
        score = float(summary["pcc"] + summary["ssim"])
        if not float("-inf") < score < float("inf"):
            raise ValueError("non-finite validation score")
        latency = summary.get("synthesis_time_ms_median")
        return score, -float(latency if latency is not None else float("inf"))

    return max(variants, key=key)


def check_storage(directory: Path, pool_size: int) -> dict:
    # Largest model: raw + EMA + two Adam moments, best weights, atomic-save
    # temporaries, four phases, plus DIV2K and a reserve for final artifacts.
    required = pool_size * 128 * 128 + 7 * 2**30
    free = shutil.disk_usage(directory).free
    if free < required:
        raise RuntimeError(f"Not enough free SSD space: {free / 2**30:.1f} GiB; need {required / 2**30:.1f} GiB. "
                           "Free space or reduce --pool-size; existing runs will not be deleted.")
    return {"free_bytes": free, "required_bytes": required, "pool_bytes": pool_size * 128 * 128}


def quality_comparison(report: dict, selected_method: str = "net") -> dict:
    """Report held-out measurements without using test results to choose weights."""
    methods = {}
    for variant in report.get("variants", []):
        summary = variant.get("summary", {})
        if "pcc" in summary and "ssim" in summary:
            methods[variant["method"]] = {"pcc": float(summary["pcc"]), "ssim": float(summary["ssim"]),
                                          "score": float(summary["pcc"] + summary["ssim"]),
                                          "synthesis_time_ms_median": summary.get("synthesis_time_ms_median"),
                                          "params": variant.get("params", {})}
    baseline_names = [name for name in ("gs", "gs+adam") if name in methods]
    if selected_method not in methods or not baseline_names:
        return {"status": "not_available", "methods": methods, "budgets_matched": False,
                "note": "Network and baseline measurements are required; test data never changes the exported weights."}
    comparisons = {}
    for name in ("net", "net+gs", "net+adam", "net+gs+adam"):
        if name not in methods:
            continue
        comparisons[name] = {}
        for reference in baseline_names:
            candidate, baseline = methods[name], methods[reference]
            comparisons[name][reference] = {
                "pcc_delta": candidate["pcc"] - baseline["pcc"],
                "ssim_delta": candidate["ssim"] - baseline["ssim"],
                "both_metrics_at_least_baseline": (candidate["pcc"] >= baseline["pcc"] and
                                                    candidate["ssim"] >= baseline["ssim"]),
            }
    competitive = all(row["both_metrics_at_least_baseline"] for row in comparisons[selected_method].values())
    return {"status": "competitive_on_this_test" if competitive else "experimental",
            "selected_method": selected_method,
            "methods": methods, "comparisons": comparisons, "budgets_matched": False,
            "note": ("These are different inference budgets; compare quality and latency together. "
                     "This test report does not establish state-of-the-art performance and never changes the selected weights.")}


def print_quality(comparison: dict) -> None:
    for name, values in comparison["methods"].items():
        print(f"[quality] {name}: PCC={values['pcc']:.4f} SSIM={values['ssim']:.4f}", flush=True)
    if comparison["status"] == "experimental":
        print("[quality] The selected neural/hybrid pipeline remains experimental: a baseline metric is higher. "
              "Inference budgets differ; compare quality and latency together.", flush=True)
    elif comparison["status"] == "competitive_on_this_test":
        print("[quality] Selected neural/hybrid metrics match or exceed the measured baselines on this test. "
              "Inference budgets differ; this does not establish overall leadership.", flush=True)
    else:
        print("[quality] Baseline comparison is unavailable in this report.", flush=True)


def checked_artifact(record: dict) -> Path:
    path = Path(record["weights"])
    if not path.is_file() or file_hash(path) != record["sha256"]:
        raise RuntimeError(f"Saved checkpoint is missing or changed: {path}")
    return path


def run(*, minutes: float = 180, device: str = "cuda", precision: str = "bf16",
        output: Path = ROOT / "data/runs/a6000", weights: Path = ROOT / "data/weights/doe_a6000.pt",
        resume: bool = False, skip_fetch: bool = False, dry_run: bool = False,
        pool_size: int = 100000, grad_accum: int = 1, max_steps: int | None = None,
        pilot_count: int = 4, test_count: int = 16, threads: int = 12,
        vram_limit_gib: float = 40.0, pool_in_ram: bool = True,
        allow_small_dataset: bool = False, eval_threads: int = 2) -> dict:
    if threads < 1 or eval_threads < 1:
        raise ValueError("threads and eval_threads must be positive")
    plan = make_plan(minutes, precision=precision, pool_size=pool_size, grad_accum=grad_accum,
                     max_steps=max_steps, pilot_count=pilot_count, test_count=test_count, threads=threads,
                     vram_limit_gib=vram_limit_gib, pool_in_ram=pool_in_ram)
    output, weights = Path(output).resolve(), Path(weights).resolve()
    if dry_run:
        return {**plan, "output": str(output), "export_weights": str(weights), "device": device,
                "fetch_div2k": not skip_fetch, "allow_small_dataset": allow_small_dataset}
    started = time.monotonic()
    deadline = started + minutes * 60
    manifest_path, stop = output / "manifest.json", output / "STOP"
    if stop.exists():
        raise RuntimeError(f"Remove {stop} before continuing")
    identity = {"source_sha256": source_hash(), "weights": str(weights), "device": device,
                "common_training": plan["common_training"], "candidates": plan["candidates"],
                "pilot_validation": plan["pilot_validation"], "final_test": plan["final_test"],
                "selection_validation": plan["selection_validation"],
                "allow_small_dataset": allow_small_dataset}
    announce(f"[resume] checking saved experiment: {manifest_path}" if resume else
             f"[prepare] new experiment: {manifest_path}")
    if resume:
        if not manifest_path.is_file():
            raise RuntimeError("No manifest exists; start a new run without --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError("Resume configuration/source changed; use a new --output and --weights")
        for record in manifest["pilots"].values():
            if record.get("status") == "validated":
                checked_artifact(record)
        if manifest.get("export"):
            checked_artifact(manifest["export"])
        if manifest.get("status") == "completed":
            if not Path(manifest["test_report"]).is_file():
                raise RuntimeError("Completed test report is missing; restore the run artifacts")
            if not (output / "inference.json").is_file():
                raise RuntimeError("Validated inference preset is missing; restore inference.json")
            print(f"[done] Existing completed run: {manifest_path}", flush=True)
            return manifest
    else:
        if manifest_path.exists() or (output.is_dir() and any(output.iterdir())):
            raise FileExistsError("Run directory is not empty; use --resume or a new --output")
        if weights.exists():
            raise FileExistsError("Export weights already exist; use --resume or a new --weights")
        manifest = {"schema_version": 1, "created_at": stamp(), "identity": identity,
                    "status": "preparing", "pilots": {}, "data_ready": False,
                    "invocations": [], "plan": plan}
    output.mkdir(parents=True, exist_ok=True)
    storage = check_storage(output, pool_size)
    announce("[prepare] checking CUDA and available storage")
    hardware = preflight(device, precision)
    device = hardware["device"]
    import torch

    torch.set_num_threads(threads)
    manifest["invocations"].append({"started_at": stamp(), "minutes": minutes, "hardware": hardware,
                                    "storage": storage, "eval_threads": eval_threads})

    def save(**updates):
        manifest.update(updates, updated_at=stamp(), invocation_elapsed_seconds=round(time.monotonic() - started, 2))
        atomic_json(manifest_path, manifest)

    def require_time():
        if stop.exists():
            raise InterruptedError(f"Stop requested by {stop}; continue with --resume after removing it")
        if time.monotonic() >= deadline:
            raise TimeoutError("Invocation budget exhausted; continue with --resume")

    def evaluate(paths, parameters, name):
        # Small single-quad FFT/optimization runs benefit from few host threads;
        # child training processes retain their independent 12-thread setting.
        torch.set_num_threads(eval_threads)
        announce(f"[evaluate] {name}; CPU threads={eval_threads}; "
                 f"completed cases cached under {output / 'benchmark_cache' / name}")
        runtime = dict(cache_dir=output / "benchmark_cache" / name,
                       stop_file=stop, deadline=deadline)
        return run_benchmark(paths, device=device, parameters={**parameters, **runtime})

    def training_config(candidate: dict, run_dir: Path, checkpoint: Path, budget: float) -> dict:
        config = dict(plan["common_training"], **{k: candidate[k] for k in ("base", "unroll", "stage_base")},
                      device=device, minutes=budget, output=str(run_dir), weights=str(checkpoint))
        if resume and not stop.exists():
            # Explicit coordinator continuation also resumes the child phase
            # stopped by the previously mirrored global STOP request.
            (run_dir / "STOP").unlink(missing_ok=True)
        if (run_dir / "last.pt").is_file():
            config["resume"] = True
        return config

    try:
        save(status="preparing")
        if not manifest["data_ready"]:
            require_time()
            if not skip_fetch:
                from ml.data import fetch_div2k
                fetch_div2k()
        announce("[prepare] checking photo sources and split hashes (CPU / disk)")
        data = dataset_info()
        announce(f"[prepare] photo sources: {data['counts']}")
        check_dataset(data, allow_small_dataset)
        if manifest.get("data") and manifest["data"]["sha256"] != data["sha256"]:
            raise ValueError("Photo sources changed since the run started; use a new --output and --weights")
        if not manifest["data_ready"]:
            atomic_json(output / "photo_sources.json", data)
            manifest["data"] = {k: data[k] for k in ("version", "sha256", "counts")}
            manifest["data_ready"] = True
            save()
        if not manifest.get("pool_ready"):
            require_time()
            save(status="preparing-pool")
            prepare_pool(pool_size)
            manifest["pool_ready"] = True
            save()
        # Each validated pilot is cached. A failed/OOM candidate does not prevent
        # an otherwise usable smaller candidate from completing the experiment.
        for candidate_index, candidate in enumerate(plan["candidates"]):
            if "pilot_winner" in manifest:
                break  # Main-training continuation keeps its screened architecture.
            name = candidate["name"]
            record = manifest["pilots"].setdefault(name, {"status": "pending", "candidate": candidate})
            if record["status"] == "validated":
                continue
            require_time()
            run_dir, checkpoint = output / name, output / name / "best.pt"
            run_dir.mkdir(parents=True, exist_ok=True)
            remaining = (deadline - time.monotonic()) / 60 - plan["evaluation_reserve_minutes"]
            remaining_phases = len(plan["candidates"]) - candidate_index + 1
            budget = min(plan["pilot_minutes_each"], remaining / remaining_phases)
            if budget < 0.1:
                raise TimeoutError("Not enough time for pilot training; continue with --resume")
            record.update(status="training", started_at=stamp(), allocated_minutes=budget)
            save(status=f"pilot:{name}")
            print(f"[pilot] {name}, budget {budget:.1f} min", flush=True)
            try:
                config = training_config(candidate, run_dir, checkpoint, budget)
                run_training(config, run_dir / "training.log", stop, deadline)
                require_time()
                report = evaluate([checkpoint], plan["pilot_validation"], name)
                report_path = run_dir / "validation_quantized.json"
                atomic_json(report_path, report)
                score = checkpoint_scores(report)[str(checkpoint.resolve())]
                record.update(status="validated", weights=str(checkpoint), sha256=file_hash(checkpoint),
                              score=score, report=str(report_path), completed_at=stamp())
                save()
                print(f"[pilot] {name}: quantized validation PCC+SSIM={score:.5f}", flush=True)
            except (InterruptedError, TimeoutError):
                raise
            except Exception as exc:
                record.update(status="failed", error=str(exc), failed_at=stamp())
                save()
                print(f"[pilot] {name} failed: {exc}", flush=True)
        valid = [r for r in manifest["pilots"].values() if r["status"] == "validated"]
        if not valid:
            raise RuntimeError("Every pilot failed; inspect candidate training.log files")
        winner = max(valid, key=lambda r: r["score"])
        manifest["pilot_winner"] = winner["candidate"]["name"]
        final_dir, final_checkpoint = output / "main", output / "main/best.pt"
        final = manifest.setdefault("main", {"status": "pending"})
        if final["status"] != "trained":
            require_time()
            final_dir.mkdir(parents=True, exist_ok=True)
            budget = (deadline - time.monotonic()) / 60 - plan["evaluation_reserve_minutes"]
            if budget < 0.1:
                raise TimeoutError("Not enough time for main training; continue with --resume")
            config = training_config(winner["candidate"], final_dir, final_checkpoint, budget)
            if not config.get("resume"):
                config["init_weights"] = winner["weights"]
            final.update(status="training", candidate=winner["candidate"], allocated_minutes=budget)
            save(status="main-training")
            print(f"[main] {manifest['pilot_winner']}, budget {budget:.1f} min", flush=True)
            run_training(config, final_dir / "training.log", stop, deadline)
            final.update(status="trained", weights=str(final_checkpoint), sha256=file_hash(final_checkpoint),
                         completed_at=stamp())
            save()
        # Main training is not assumed to improve the pilot: select all trained
        # artifacts on validation, freeze this choice, then evaluate test once.
        if not manifest.get("export"):
            require_time()
            paths = [checked_artifact(r) for r in valid] + [checked_artifact(final)]
            report = evaluate(paths, plan["selection_validation"], "selection")
            atomic_json(output / "selection_validation.json", report)
            variant = select_neural_variant(report)
            selected = Path(variant["checkpoint"]["path"]).resolve()
            if selected not in [path.resolve() for path in paths]:
                raise ValueError("selected checkpoint is outside the validated candidate set")
            profile = {"label": variant.get("label", variant["method"]), "method": variant["method"],
                       "params": variant.get("params", {})}
            if weights.exists():
                raise FileExistsError(f"Export path was created by another process: {weights}")
            weights.parent.mkdir(parents=True, exist_ok=True)
            temporary = weights.with_suffix(weights.suffix + ".tmp")
            shutil.copyfile(selected, temporary)
            os.replace(temporary, weights)
            manifest["export"] = {"weights": str(weights), "sha256": file_hash(weights),
                                  "selected_from": str(selected), "validation_score": variant["summary"]["pcc"] + variant["summary"]["ssim"],
                                  "neural_profile": profile}
            try:
                portable_weights = Path(os.path.relpath(weights, output)).as_posix()
            except ValueError:
                portable_weights = str(weights)
            atomic_json(output / "inference.json", dict(schema_version=1, weights=portable_weights,
                        weights_sha256=manifest["export"]["sha256"], neural_primary=True,
                        selected_on="validation", profile=profile))
            save(status="selected")
        require_time()
        save(status="test-evaluation")
        selected_profile = manifest["export"]["neural_profile"]
        final_parameters = dict(plan["final_test"], neural_profiles=[selected_profile])
        announce("[evaluate] final test; saved weights and inference policy are frozen; no retraining")
        report = evaluate([weights], final_parameters, "test")
        report_path = output / "test_benchmark.json"
        atomic_json(report_path, report)
        comparison = quality_comparison(report, selected_profile["method"])
        save(status="completed", test_report=str(report_path), quality_comparison=comparison, completed_at=stamp())
        print_quality(comparison)
        print(f"[done] Weights: {weights}\n[done] Held-out report: {report_path}", flush=True)
        return manifest
    except BaseException as exc:
        save(status="interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError, TimeoutError)) else "failed",
             error=str(exc))
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=180)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32", "auto"), default="bf16")
    parser.add_argument("--output", type=Path, default=ROOT / "data/runs/a6000")
    parser.add_argument("--weights", type=Path, default=ROOT / "data/weights/doe_a6000.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true", help="use already available local photographs")
    parser.add_argument("--allow-small-dataset", action="store_true",
                        help="explicit smoke-only override for source-count minimums")
    parser.add_argument("--dry-run", action="store_true", help="print plan without torch/CUDA, files or downloads")
    parser.add_argument("--pool-size", type=int, default=100000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps per phase for a smoke check")
    parser.add_argument("--pilot-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--eval-threads", type=int, default=2,
                        help="CPU threads for evaluation only; leaves training/resume settings unchanged")
    parser.add_argument("--vram-limit-gib", type=float, default=40.0)
    parser.add_argument("--pool-in-ram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.train_config:
        from ml.training import train
        train(**json.loads(args.train_config.read_text(encoding="utf-8")))
        return
    try:
        if args.dry_run:
            result = run(**{k: v for k, v in vars(args).items() if k != "train_config"})
        else:
            with monitor():
                result = run(**{k: v for k, v in vars(args).items() if k != "train_config"})
    except (ValueError, RuntimeError, FileExistsError, TimeoutError, InterruptedError) as exc:
        parser.exit(1, f"a6000: {exc}\n")
    if args.dry_run:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
