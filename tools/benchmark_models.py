#!/usr/bin/env python3
"""Compare trained DOE models with GS using the quantized submission pipeline.

    python tools/benchmark_models.py --weights data/weights/physics_unrolled.pt \
        --device auto --count 4 --gs-iters 50 --refine-steps 10

The fixed val/test sets use separate generation seeds, but share photograph
sources with the training pool. Supplementary classic sets include photograph
and text sources reserved by ml.data. Reports separate these populations and
identify the actual inputs by hash. Use --no-classic to omit the classic sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Also support direct execution from any working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import DEFAULT_OPTICS, PHYSICS_VERSION
from ml.data import SPLIT_SEEDS, classic_testsets, fixed_quads
from ml.metrics import DEFAULT_NORM
from tools.evaluation_runtime import announce, monitor

NEURAL_METHODS = ("net", "net+gs", "net+adam", "net+gs+adam")


def validate_neural_profiles(profiles: list[dict]) -> None:
    names = set()
    for profile in profiles:
        if profile.get("method") not in NEURAL_METHODS:
            raise ValueError("deployment profiles must start with neural synthesis")
        label = profile.get("label")
        if not isinstance(label, str) or not label or label in names:
            raise ValueError("neural profile labels must be nonempty and unique")
        names.add(label)
        params = profile.get("params", {})
        if set(params) - {"refine_steps", "adam_steps", "refine_lr"}:
            raise ValueError("unsupported neural refinement parameter")
        for key in ("refine_steps", "adam_steps"):
            value = params.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("neural refinement steps must be nonnegative integers")
        rate = params.get("refine_lr", 0.03)
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("neural refinement learning rate must be positive and finite")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(records: list[dict]) -> dict:
    """Aggregate equally sized quads; report both plane means and worst pairs."""
    pcc = np.asarray([[p["pcc"] for p in row["planes"]] for row in records])
    ssim = np.asarray([[p["ssim"] for p in row["planes"]] for row in records])
    synth = np.asarray([row["synthesis_time_ms"] for row in records])
    total = np.asarray([row["pipeline_time_ms"] for row in records])
    return {
        "runs": len(records),
        "pcc": float(pcc.mean()),
        "ssim": float(ssim.mean()),
        "pcc_by_plane": pcc.mean(0).tolist(),
        "ssim_by_plane": ssim.mean(0).tolist(),
        "min_plane_mean_pcc": float(pcc.mean(0).min()),
        "min_plane_mean_ssim": float(ssim.mean(0).min()),
        "worst_pair_pcc": float(pcc.min()),
        "worst_pair_ssim": float(ssim.min()),
        "synthesis_time_ms_median": float(np.median(synth)),
        "synthesis_time_ms_p95": float(np.percentile(synth, 95)),
        "pipeline_time_ms_median": float(np.median(total)),
    }


def benchmark(weights: list[Path], *, device: str = "auto", split: str = "test",
              count: int = 4, seeds: tuple[int, ...] = (0,), gs_iters: int = 50,
              refine_steps: int = 10, classic: bool = True, adam_steps: int = 0,
              photo_split: str = "legacy", neural_profiles: list[dict] | None = None,
              cache_dir: Path | str | None = None, stop_file: Path | str | None = None,
              deadline: float | None = None) -> dict:
    """Benchmark identical quads for every method; exclude one warmup per method."""
    import torch

    from ml.fresnel import FresnelSingleFFT
    from ml.model import load_model
    from ml.pipeline import run

    if not weights:
        raise ValueError("at least one checkpoint is required")
    if neural_profiles is not None:
        validate_neural_profiles(neural_profiles)
    if photo_split not in ("legacy", "source"):
        raise ValueError("photo_split must be legacy or source")
    if split not in SPLIT_SEEDS or count < 1 or gs_iters < 1 or refine_steps < 0 or adam_steps < 0:
        raise ValueError("invalid split, count or iteration budget")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    paths = [Path(p).resolve(strict=True) for p in weights]
    if len(set(paths)) != len(paths):
        raise ValueError("checkpoint paths must be distinct")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def check_stop():
        if stop_file is not None and Path(stop_file).exists():
            raise InterruptedError(f"Stop requested by {stop_file}; benchmark progress is cached")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Evaluation budget exhausted; continue with --resume to reuse cached results")

    check_stop()
    announce(f"[benchmark] preparing {split} inputs; photo_split={photo_split}")
    if photo_split == "source":
        quads = fixed_quads(split, count, photo_split=split)
        from ml.data import photo_source_manifest, photo_manifest_hash, PHOTO_SPLIT_VERSION
        announce("[benchmark] checking photograph content hashes")
        manifest = photo_source_manifest()
        source_metadata = dict(version=PHOTO_SPLIT_VERSION, hash=photo_manifest_hash(manifest), sources=manifest)
    else:
        quads = fixed_quads(split, count)
        source_metadata = None
    generated_count = sum(len(batch) for batch in quads.values())
    announce(f"[benchmark] preparing classic examples: {classic}")
    classic_sets = classic_testsets() if classic else {}
    # Prefix classic names so generated categories and reserved-source examples
    # remain separately inspectable, even when their subjects overlap.
    quads.update({f"classic/{name}": quad[None] for name, quad in classic_sets.items()})
    cases = [(kind, index, quad) for kind, batch in quads.items()
             for index, quad in enumerate(batch)]
    if not cases:
        raise ValueError("no benchmark inputs were generated")
    prop = FresnelSingleFFT(DEFAULT_OPTICS).to(device)
    report = {
        "schema_version": 1,
        "physics_version": PHYSICS_VERSION,
        "optics": DEFAULT_OPTICS.as_dict(),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device_name": (torch.cuda.get_device_name(torch.device(device))
                        if torch.device(device).type == "cuda" else "CPU"),
        "protocol": {
            "split": split, "split_seed": SPLIT_SEEDS[split],
            "photo_split": photo_split,
            "photo_sources": source_metadata,
            "split_scope": ("content-disjoint photograph sources" if photo_split == "source" else
                            "independent generated crops; photograph sources may overlap training"),
            "quads_per_kind": count, "distinct_quads": len(cases),
            "generated_distinct_quads": generated_count,
            "classic_case_names": list(classic_sets),
            "classic_source_scope": ("photos_classic and text_binary use sources reserved by ml.data; "
                                     "letter/digit subjects and fonts can overlap training"),
            "synthesis_seeds": list(seeds),
            "quantization": "phase -> uint8 DOE -> numpy Fresnel -> uint8 amplitude -> PCC/SSIM",
            "recon_normalisation": DEFAULT_NORM,
            "warmup_runs_per_variant": 1,
            "timing": "synthesis CUDA-synchronized; pipeline includes quantization and CPU reconstruction",
            "input_sha256": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                             for k, v in quads.items()},
        },
        "variants": [],
    }
    report["protocol"]["cpu_threads"] = torch.get_num_threads()
    # Reuse only identical algorithms, inputs, weights, seeds and timing environment.
    # This never substitutes cached runs for another seed or benchmark protocol.
    code = hashlib.sha256(Path(__file__).read_bytes())
    for source in sorted((ROOT / "ml").glob("*.py")):
        code.update(source.name.encode())
        code.update(source.read_bytes())
    cache_context = {k: v for k, v in report.items() if k != "variants"}
    cache_context["code_sha256"] = code.hexdigest()
    tasks = [(kind, index, quad, seed) for kind, index, quad in cases for seed in seeds]
    announce(f"[benchmark] {len(cases)} four-plane inputs x {len(seeds)} seeds; "
             f"{len(tasks)} measured runs per method; CPU threads={torch.get_num_threads()}")

    def measure(label: str, method: str, model=None, checkpoint=None, **params):
        check_stop()
        kwargs = dict(device=device, prop=prop, model=model, **params)
        records = []
        identity = dict(context=cache_context, method=method, params=params,
                        checkpoint=None if checkpoint is None else
                        {k: v for k, v in checkpoint.items() if k != "path"})
        cache_path = None
        if cache_dir is not None:
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            cache_path = Path(cache_dir) / f"{key}.json"
            if cache_path.is_file():
                saved = json.loads(cache_path.read_text(encoding="utf-8"))
                records = saved["records"]
                if saved.get("identity") != identity or len(records) > len(tasks):
                    raise ValueError(f"Invalid benchmark cache: {cache_path}")
                for record, (kind, index, _, seed) in zip(records, tasks):
                    if (record["kind"], record["quad_index"], record["seed"]) != (kind, index, seed):
                        raise ValueError(f"Benchmark cache task mismatch: {cache_path}")
                if records:
                    summarize(records)  # Validate the record structure before reuse.
        announce(f"[benchmark] {label}: {len(records)}/{len(tasks)} cached")
        if len(records) < len(tasks):
            announce(f"[benchmark] {label}: warming up (excluded from measurements)")
            run(cases[0][2], method, norm=DEFAULT_NORM, seed=seeds[0], **kwargs)
        measured_start, initial_count = time.monotonic(), len(records)
        next_progress = measured_start
        for kind, index, quad, seed in tasks[len(records):]:
            check_stop()
            started = time.perf_counter()
            result = run(quad, method, norm=DEFAULT_NORM, seed=seed, **kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000
            records.append({
                    "kind": kind, "quad_index": index, "seed": seed,
                    "population": "classic" if kind.startswith("classic/") else "generated",
                    "planes": result.rows, "synthesis_time_ms": result.time_ms,
                    "pipeline_time_ms": elapsed_ms,
            })
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(dict(identity=identity, records=records),
                                                allow_nan=False), encoding="utf-8")
                os.replace(temporary, cache_path)
            now = time.monotonic()
            if now >= next_progress or len(records) == len(tasks):
                remaining = (now - measured_start) / (len(records) - initial_count) * (len(tasks) - len(records))
                announce(f"[benchmark] {label}: {len(records)}/{len(tasks)}; "
                         f"last={elapsed_ms / 1000:.2f}s; method ETA={remaining / 60:.1f} min")
                next_progress = now + 5
        entry = {
            "label": label, "method": method, "params": params,
            "summary": summarize(records),
            "by_kind": {kind: summarize([r for r in records if r["kind"] == kind])
                        for kind in quads},
            "by_population": {population: summarize([r for r in records if r["population"] == population])
                              for population in dict.fromkeys(r["population"] for r in records)},
            "records": records,
        }
        if checkpoint is not None:
            entry["checkpoint"] = checkpoint
        report["variants"].append(entry)
        announce(f"[benchmark] done {label}: PCC={entry['summary']['pcc']:.4f} "
                 f"SSIM={entry['summary']['ssim']:.4f}")
        return entry

    measure(f"gs-{gs_iters}", "gs", gs_iters=gs_iters)
    if adam_steps:
        measure(f"gs-{gs_iters}+adam-{adam_steps}", "gs+adam",
                gs_iters=gs_iters, adam_steps=adam_steps)
    for path in paths:
        check_stop()
        announce(f"[benchmark] loading weights: {path}")
        net, ckpt = load_model(path, device)
        if ckpt["optics"] != DEFAULT_OPTICS.as_dict():
            # Check normalized values rather than relying on JSON list/tuple forms.
            optics = net.prop.optics
            if optics != DEFAULT_OPTICS:
                raise ValueError(f"checkpoint optics do not match benchmark: {path}")
        metadata = {
            "path": str(path), "sha256": file_sha256(path),
            "model": ckpt["model"], "step": ckpt.get("step"),
        }
        measure(str(path), "net", model=net, checkpoint=metadata)
        if neural_profiles is not None:
            for profile in neural_profiles:
                if profile["method"] != "net":
                    measure(f"{path}:{profile['label']}", profile["method"], model=net,
                            checkpoint=metadata, **profile.get("params", {}))
        elif refine_steps:
            measure(f"{path}+gs-{refine_steps}", "net+gs", model=net,
                    checkpoint=metadata, refine_steps=refine_steps)
        del net
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--photo-split", choices=("legacy", "source"), default="legacy")
    parser.add_argument("--count", type=int, default=4, help="quads per category")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--gs-iters", type=int, default=50)
    parser.add_argument("--adam-steps", type=int, default=100,
                        help="also evaluate GS+Adam with this budget; 0 disables it")
    parser.add_argument("--refine-steps", type=int, default=10,
                        help="also evaluate net+gs with this budget; 0 disables it")
    parser.add_argument("--classic", action=argparse.BooleanOptionalAction, default=True,
                        help="include supplementary classic sets with reserved photograph/text sources")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark.json")
    parser.add_argument("--cache-dir", type=Path, help="save each completed run for interruption recovery")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    with monitor():
        report = benchmark(args.weights, device=args.device, split=args.split, count=args.count,
                           seeds=tuple(args.seeds), gs_iters=args.gs_iters,
                           refine_steps=args.refine_steps, classic=args.classic, adam_steps=args.adam_steps,
                           photo_split=args.photo_split, cache_dir=args.cache_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(f"Benchmark saved: {args.out.resolve()}")
    for variant in report["variants"]:
        summary = variant["summary"]
        print(f"{variant['method']} ({variant['label']}): PCC={summary['pcc']:.4f} "
              f"SSIM={summary['ssim']:.4f} "
              f"synthesis={summary['synthesis_time_ms_median']:.2f} ms")


if __name__ == "__main__":
    main()
