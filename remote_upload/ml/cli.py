"""Командная строка.

    python -m ml.cli synth --inputs 1.png 2.png 3.png 4.png --out results/ [--method net+adam]
    python -m ml.cli verify --doe results/doe.png --inputs 1.png 2.png 3.png 4.png
    python -m ml.cli bench --sets data/testsets/photos_classic data/testsets/mephi_letters
    python -m ml.cli info
    python -m ml.cli make-testsets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import DEFAULT_OPTICS, PHYSICS_VERSION
from .metrics import DEFAULT_NORM, NORM_MODES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "data" / "weights" / "doe_multiplane.pt"


def pick_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна; используйте --device cpu")
    return name


def _load_model(weights: Path | None, device: str, required: bool):
    if weights is None or not Path(weights).is_file():
        if required:
            raise SystemExit(f"Нет весов сети: {weights}. Обучите (python -m ml.training) "
                             f"или выберите --method gs+adam.")
        return None
    from .model import load_model

    model, _ = load_model(weights, device)
    return model


def _print_table(report: dict) -> None:
    print(f"{'плоскость':<10}{'z, мм':>8}{'PCC':>9}{'SSIM':>9}{'SSIM_глоб':>11}")
    for p in report["planes"]:
        print(f"{p['index']:<10}{p['distance_mm']:>8.1f}{p['pcc']:>9.4f}{p['ssim']:>9.4f}{p['ssim_global']:>11.4f}")
    m = report["mean"]
    print(f"{'среднее':<10}{'':>8}{m['pcc']:>9.4f}{m['ssim']:>9.4f}{m['ssim_global']:>11.4f}")
    print(f"метод: {report['method']}, время синтеза: {report['synthesis_time_ms']:.1f} мс ({report['device']})")


def cmd_synth(args) -> None:
    from .pipeline import load_sources, run, write_outputs

    device = pick_device(args.device)
    sources = load_sources(args.inputs)
    model = _load_model(args.weights, device, required=args.method.startswith("net"))
    kwargs = dict(device=device, gs_iters=args.gs_iters, adam_steps=args.adam_steps,
                  refine_steps=args.refine_steps, seed=args.seed, model=model)
    if args.warmup and device.startswith("cuda"):
        run(sources, args.method, norm=args.norm, **kwargs)  # прогрев ядер CUDA, cuFFT-планов
    res = run(sources, args.method, norm=args.norm, **kwargs)
    final = write_outputs(res, args.out, norm=args.norm, preview=not args.no_preview)
    report = json.loads((Path(args.out) / "metrics.json").read_text(encoding="utf-8"))
    _print_table(report)
    print(f"файлы: {Path(args.out).resolve()}")
    del final


def cmd_verify(args) -> None:
    """Независимая проверка по готовым PNG (без нейросети и torch)."""
    from .data import load_png
    from .pipeline import load_sources, reconstruct

    sources = load_sources(args.inputs)
    doe = load_png(args.doe)
    recon, _, rows = reconstruct(doe, sources, norm=args.norm)
    for i, r in enumerate(rows, 1):
        print(f"{i}: PCC={r['pcc']:.4f} SSIM={r['ssim']:.4f} SSIM_global={r['ssim_global']:.4f}")
    print(f"mean: PCC={np.mean([r['pcc'] for r in rows]):.4f} SSIM={np.mean([r['ssim'] for r in rows]):.4f}")
    if args.recon:
        from .metrics import pair_metrics

        given = [load_png(Path(p)) for p in args.recon]
        for i, (s, g) in enumerate(zip(sources, given), 1):
            m = pair_metrics(s, g)
            print(f"по присланным recon_{i}: PCC={m['pcc']:.4f} SSIM={m['ssim']:.4f}")


def bench_methods(sets: dict[str, np.ndarray], device: str, model=None, repeats: int = 3,
                  configs: list[tuple[str, dict]] | None = None, norm: str = DEFAULT_NORM) -> list[dict]:
    """Качество и время каждого метода на каждом наборе. Время — медиана repeats запусков
    после прогона-прогрева, с синхронизацией CUDA."""
    from .fresnel import FresnelSingleFFT
    from .pipeline import run

    prop = FresnelSingleFFT(DEFAULT_OPTICS).to(device)
    if configs is None:
        configs = [("gs", {"gs_iters": n}) for n in (10, 25, 50, 100, 200)]
        configs += [("adam", {"adam_steps": 400}), ("gs+adam", {"gs_iters": 200, "adam_steps": 400})]
        if model is not None:
            configs = ([("net", {})] + [("net+gs", {"refine_steps": k}) for k in (5, 10, 25, 50)]
                       + [("net+adam", {"refine_steps": k}) for k in (20, 50)] + configs)
    rows = []
    for name, sources in sets.items():
        for method, kw in configs:
            kwargs = dict(device=device, model=model, prop=prop, norm=norm, **kw)
            run(sources, method, **kwargs)  # прогрев
            times, res = [], None
            for _ in range(max(1, repeats)):
                res = run(sources, method, **kwargs)
                times.append(res.time_ms)
            label = method + "".join(f" {k}={v}" for k, v in kw.items())
            rows.append({"set": name, "method": method, "label": label, "params": kw,
                         "time_ms": float(np.median(times)), "device": device,
                         "pcc": res.summary["pcc"], "ssim": res.summary["ssim"],
                         "ssim_global": res.summary["ssim_global"],
                         "pcc_planes": [r["pcc"] for r in res.rows],
                         "ssim_planes": [r["ssim"] for r in res.rows]})
            print(f"[bench] {name:<16} {label:<32} {rows[-1]['time_ms']:>9.1f} ms  "
                  f"PCC {rows[-1]['pcc']:.3f}  SSIM {rows[-1]['ssim']:.3f}", flush=True)
    return rows


def write_bench(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["| набор | метод | время, мс | PCC | SSIM | PCC по плоскостям |", "|---|---|---:|---:|---:|---|"]
    for r in rows:
        planes = " / ".join(f"{v:.2f}" for v in r["pcc_planes"])
        lines.append(f"| {r['set']} | {r['label']} | {r['time_ms']:.1f} | {r['pcc']:.3f} | {r['ssim']:.3f} | {planes} |")
    (out / "bench.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = sorted({r["set"] for r in rows})
    fig, axes = plt.subplots(1, len(sets), figsize=(6 * len(sets), 4.5), squeeze=False)
    for ax, s in zip(axes[0], sets):
        for family, marker in (("gs", "o-"), ("net+gs", "s-"), ("net+adam", "v-"), ("net", "*"), ("adam", "^"),
                               ("gs+adam", "D")):
            pts = sorted((r["time_ms"], r["pcc"]) for r in rows if r["set"] == s and r["method"] == family)
            if pts:
                ax.plot(*zip(*pts), marker, label=family, ms=9 if family == "net" else 6)
        ax.set_xscale("log")
        ax.set_xlabel("время синтеза, мс (лог.)")
        ax.set_ylabel("PCC (среднее по 4 плоскостям)")
        ax.set_title(s)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "quality_vs_time.png", dpi=120)
    plt.close(fig)


def cmd_bench(args) -> None:
    from .pipeline import load_sources

    device = pick_device(args.device)
    model = _load_model(args.weights, device, required=False)
    sets = {}
    for d in args.sets:
        d = Path(d)
        sets[d.name] = load_sources(sorted(d.glob("*.png"), key=lambda p: p.stem)[:4])
    rows = bench_methods(sets, device, model, args.repeats, norm=args.norm)
    write_bench(rows, Path(args.out))
    print(f"[bench] сохранено в {Path(args.out).resolve()}")


def cmd_info(_args) -> None:
    import torch

    o = DEFAULT_OPTICS
    print(f"физика: {PHYSICS_VERSION}")
    print(f"N = {o.size}, λ = {o.wavelength * 1e9:.0f} нм, Δx = {o.pitch * 1e6:.0f} мкм")
    print(f"критическое расстояние N·Δx²/λ = {o.critical_distance * 1e3:.2f} мм")
    for i, z in enumerate(o.distances, 1):
        print(f"  плоскость {i}: z = {z * 1e3:.1f} мм, шаг пикселя восстановления λz/(NΔx) = "
              f"{o.output_pitch(z) * 1e6:.2f} мкм")
    print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"веса по умолчанию: {DEFAULT_WEIGHTS} ({'есть' if DEFAULT_WEIGHTS.is_file() else 'нет'})")


def build_parser() -> argparse.ArgumentParser:
    from .pipeline import METHOD_HELP, METHODS

    ap = argparse.ArgumentParser(prog="python -m ml.cli", description="Синтез фазового ДОЭ под 4 изображения")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="4 PNG → doe.png + recon_1..4.png + metrics.json/csv")
    s.add_argument("--inputs", nargs=4, type=Path, required=True, help="4 PNG; i-е — на i-м расстоянии")
    s.add_argument("--out", type=Path, default=Path("results"))
    s.add_argument("--method", choices=METHODS, default=None,
                   help="; ".join(f"{k}: {v}" for k, v in METHOD_HELP.items())
                   + ". По умолчанию net+gs при наличии весов, иначе gs+adam")
    s.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    s.add_argument("--device", default="auto")
    s.add_argument("--gs-iters", type=int, default=200)
    s.add_argument("--adam-steps", type=int, default=400)
    s.add_argument("--refine-steps", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--norm", choices=NORM_MODES, default=DEFAULT_NORM, help="перевод модуля в 0–255")
    s.add_argument("--no-preview", action="store_true")
    s.add_argument("--warmup", action="store_true", help="прогрев CUDA перед замером времени")
    s.set_defaults(func=cmd_synth)

    v = sub.add_parser("verify", help="пересчитать восстановление и метрики по doe.png")
    v.add_argument("--doe", type=Path, required=True)
    v.add_argument("--inputs", nargs=4, type=Path, required=True)
    v.add_argument("--recon", nargs=4, type=Path, default=None, help="сравнить и присланные recon_*.png")
    v.add_argument("--norm", choices=NORM_MODES, default=DEFAULT_NORM)
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("bench", help="качество и время: сеть / сеть+K / GS-N / Adam")
    b.add_argument("--sets", nargs="+", type=Path, default=[ROOT / "data/testsets/photos_classic",
                                                           ROOT / "data/testsets/mephi_letters"])
    b.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    b.add_argument("--device", default="auto")
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--norm", choices=NORM_MODES, default=DEFAULT_NORM)
    b.add_argument("--out", type=Path, default=ROOT / "results" / "bench")
    b.set_defaults(func=cmd_bench)

    i = sub.add_parser("info", help="параметры физики и окружения")
    i.set_defaults(func=cmd_info)

    t = sub.add_parser("make-testsets", help="записать тестовые наборы PNG в data/testsets")
    t.add_argument("--out", type=Path, default=None)
    t.set_defaults(func=lambda a: [print(p) for p in __import__("ml.data", fromlist=["x"]).make_testsets(a.out)])
    return ap


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.cmd == "synth" and args.method is None:
        args.method = "net+gs" if Path(args.weights).is_file() else "gs+adam"
    t0 = time.perf_counter()
    args.func(args)
    if args.cmd in ("synth", "bench"):
        print(f"всего: {time.perf_counter() - t0:.1f} с")


if __name__ == "__main__":
    main()
