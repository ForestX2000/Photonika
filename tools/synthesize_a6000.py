#!/usr/bin/env python3
"""Apply the neural-first inference policy frozen on validation by a6000.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def synthesize(preset: Path, inputs: list[Path], out: Path, *, weights: Path | None = None,
               device: str = "auto"):
    import torch
    from ml.model import load_model
    from ml.pipeline import load_sources, run, write_outputs
    from tools.benchmark_models import file_sha256, validate_neural_profiles

    preset = Path(preset).resolve()
    config = json.loads(preset.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("neural_primary") is not True:
        raise ValueError("preset must describe a supported neural-first synthesis policy")
    if config.get("selected_on") != "validation":
        raise ValueError("inference policy must be selected on validation")
    profile = config["profile"]
    validate_neural_profiles([profile])
    checkpoint = Path(weights).resolve() if weights is not None else (preset.parent / config["weights"]).resolve()
    if file_sha256(checkpoint) != config["weights_sha256"]:
        raise ValueError("weights do not match the validated inference policy")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(checkpoint, device)
    sources = load_sources(inputs, model.prop.optics)
    result = run(sources, profile["method"], model=model, device=device, optics=model.prop.optics,
                 **profile.get("params", {}))
    result.extra["inference_policy"] = {"selected_on": "validation", "neural_primary": True,
                                         "weights_sha256": config["weights_sha256"]}
    return write_outputs(result, out, optics=model.prop.optics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, default=ROOT / "data/runs/a6000/inference.json")
    parser.add_argument("--weights", type=Path, help="override relocated checkpoint path; hash must still match")
    parser.add_argument("--inputs", type=Path, nargs=4, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/a6000_synthesis")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = synthesize(**vars(args))
    print(f"{result.method}: PCC={result.summary['pcc']:.4f} SSIM={result.summary['ssim']:.4f}; "
          f"synthesis={result.time_ms:.1f} ms; files: {args.out.resolve()}")


if __name__ == "__main__":
    main()
