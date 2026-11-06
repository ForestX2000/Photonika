"""Сравнение прогонов обучения (A/B) по metrics.jsonl.

    python -m ml.compare_runs data/runs/ab_*  [--plot data/runs/compare.png]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def load(run: Path) -> list[dict]:
    path = run / "metrics.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args(argv)
    runs = sorted({Path(p) for pat in args.runs for p in (glob.glob(pat) or [pat]) if Path(p).is_dir()})
    if not runs:
        raise SystemExit("no run directories found")
    print(f"{'run':<22}{'steps':>8}{'min':>7}{'start PCC':>11}{'best PCC':>10}{'SSIM':>7}"
          f"{'letters':>9}{'photos':>8}{'mixed':>7}")
    curves = {}
    for run in runs:
        recs = load(run)
        if not recs:
            print(f"{run.name:<22} (нет metrics.jsonl)")
            continue
        best = max(recs, key=lambda r: r["validation"]["pcc"] + r["validation"]["ssim"])
        v = best["validation"]
        print(f"{run.name:<22}{recs[-1]['step']:>8}{recs[-1]['elapsed_seconds'] / 60:>7.1f}"
              f"{recs[0]['validation']['pcc']:>11.4f}{v['pcc']:>10.4f}{v['ssim']:>7.3f}"
              f"{v['letters']['pcc']:>9.3f}{v['photos']['pcc']:>8.3f}{v['mixed']['pcc']:>7.3f}")
        curves[run.name] = [(r["elapsed_seconds"] / 60, r["validation"]["pcc"]) for r in recs]
    plot = args.plot or runs[0].parent / "compare.png"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, pts in curves.items():
        ax.plot(*zip(*pts), marker=".", label=name)
    ax.set_xlabel("минуты обучения")
    ax.set_ylabel("val PCC (среднее по 4 плоскостям)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot, dpi=120)
    print(f"график: {plot}")


if __name__ == "__main__":
    main()
