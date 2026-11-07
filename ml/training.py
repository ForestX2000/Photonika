"""Ограниченное по времени возобновляемое обучение сети «4 изображения → фаза ДОЭ».

    python -m ml.training --minutes 480 --device cuda
    python -m ml.training --minutes 20 --input-mode raw --output data/runs/ab_raw   # A/B-прогон

Инфраструктура из технодемки: AMP, автоподбор батча, возобновление (--resume),
атомарные чекпоинты, metrics.jsonl, status.json, остановка по файлу STOP,
запрет сна Windows на время обучения.

Обучение без разметки: сеть → фаза → дифференцируемый Френель на 4 расстояниях →
multiplane_loss (MSE с оптимальным масштабом + (1−PCC) + (1−SSIM), балансировка плоскостей).
Лучшие веса выбираются по фиксированному val-набору (буквы / фото / смешанные).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_OPTICS, PHYSICS_VERSION
from .data import FIXED_KINDS, QuadSampler, fixed_quads, load_or_build_pool
from .fresnel import FresnelSingleFFT
from .losses import multiplane_loss, plane_metrics
from .model import INPUT_MODES, MultiPlaneUNet

ROOT = Path(__file__).resolve().parent.parent


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def amp_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_loss(net, prop, x, device, loss_kwargs):
    with amp_context(device):
        phase = net(x)
    # комплексный FFT считаем во float32 вне autocast
    return multiplane_loss(prop(phase.float()), x, **loss_kwargs)


@torch.inference_mode()
def evaluate(net, prop, val: dict[str, torch.Tensor], batch: int, device: torch.device, loss_kwargs) -> dict:
    net.eval()
    out = {}
    for kind, quads in val.items():
        losses, pccs, ssims = [], [], []
        for start in range(0, len(quads), batch):
            x = quads[start:start + batch].to(device)
            with amp_context(device):
                phase = net(x)
            mod = prop(phase.float())
            loss, _ = multiplane_loss(mod, x, **loss_kwargs)
            p, s = plane_metrics(mod, x)
            losses.append(loss.cpu())
            pccs.append(p.cpu())
            ssims.append(s.cpu())
        p, s = torch.cat(pccs), torch.cat(ssims)
        out[kind] = {"loss": float(torch.cat(losses).mean()), "pcc": float(p.mean()), "ssim": float(s.mean()),
                     "pcc_planes": [round(float(v), 4) for v in p.mean(0)]}
    out["loss"] = float(np.mean([v["loss"] for v in out.values()]))
    out["pcc"] = float(np.mean([out[k]["pcc"] for k in val]))
    out["ssim"] = float(np.mean([out[k]["ssim"] for k in val]))
    return out


def choose_batch(net, prop, sampler, device, requested, limit, loss_kwargs) -> int:
    if requested != "auto":
        value = int(requested)
        if value < 1:
            raise ValueError("batch must be positive")
        return value
    if device.type != "cuda":
        return 4
    original = {k: v.detach().clone() for k, v in net.state_dict().items()}
    measurements = []
    for candidate in (4, 8, 12, 16, 24, 32, 48):
        if time.monotonic() >= limit:
            break
        opt = torch.optim.AdamW(net.parameters(), lr=0.0)
        scaler = torch.amp.GradScaler("cuda")
        try:
            net.train()
            torch.cuda.reset_peak_memory_stats()
            x = torch.from_numpy(sampler.sample(candidate)).to(device)
            sync(device)
            started = time.monotonic()
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                loss, _ = compute_loss(net, prop, x, device, loss_kwargs)
                scaler.scale(loss.mean()).backward()
                scaler.step(opt)
                scaler.update()
            sync(device)
            rate = candidate * 3 / (time.monotonic() - started)
            peak = torch.cuda.max_memory_reserved() / 2 ** 30
            measurements.append((rate, candidate))
            print(f"[batch] {candidate}: {rate:.1f} quads/s, reserved {peak:.2f} GiB", flush=True)
        except torch.OutOfMemoryError:
            print(f"[batch] {candidate}: VRAM limit", flush=True)
            break
        finally:
            net.zero_grad(set_to_none=True)
            del opt, scaler
            torch.cuda.empty_cache()
    net.load_state_dict(original)
    if not measurements:
        raise RuntimeError("No batch fits in GPU memory")
    best_rate = max(r for r, _ in measurements)
    # берём наименьший батч с пропускной способностью не хуже 90 % лучшей: больше шагов оптимизатора
    chosen = min(c for r, c in measurements if r >= 0.9 * best_rate)
    print(f"[batch] selected {chosen}", flush=True)
    return chosen


@torch.inference_mode()
def final_report(net, output: Path, metadata: dict, device: torch.device, deadline: float) -> dict:
    """Оценка на test-наборах в формате сдачи (8 бит) и сравнение с GS-50; не влияет на выбор весов."""
    from .cli import bench_methods
    from .data import classic_testsets
    from .pipeline import run, save_preview

    net.eval()
    sets = classic_testsets()
    for kind, quads in fixed_quads("test", per_kind=2).items():
        for k, q in enumerate(quads):
            sets[f"random_{kind}_{k}"] = q
    configs = [("net", {}), ("net+gs", {"refine_steps": 10}), ("gs", {"gs_iters": 10}), ("gs", {"gs_iters": 50})]
    if time.monotonic() < deadline:
        with torch.inference_mode(False):
            rows = bench_methods(sets, str(device), net, repeats=1, configs=configs)
    else:
        rows = []
    if "photos_classic" in sets:
        with torch.inference_mode(False):
            res = run(sets["photos_classic"], "net", model=net, device=str(device))
        save_preview(res, output / "preview.png")
    report = {"metadata": metadata, "test": rows,
              "note": "Метрики по 8-битным ДОЭ и восстановлениям, нормировка lsq; test-наборы в обучение не входят."}
    atomic_json(output / "final_report.json", report)
    return report


def train(minutes: float = 120, device: str = "cuda", base: int = 32, input_mode: str = "imgphase",
          unroll: int = 0, stage_base: int = 16,
          batch="auto", lr: float = 1e-3, pool_size: int = 20000, val_per_kind: int = 16,
          output="data/training", weights="data/weights/doe_multiplane.pt", resume: bool = False,
          seed: int = 42, eval_every: float = 120.0, w_mse: float = 1.0, w_pcc: float = 1.0,
          w_ssim: float = 1.0, worst: float = 0.5, ssim_global: float = 0.5, init_weights: str | None = None,
          max_steps: int | None = None) -> Path:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")
    if min(minutes, lr, pool_size, val_per_kind, base) <= 0:
        raise ValueError("minutes, lr, pool_size, val_per_kind, base must be positive")
    output, weights = Path(output), Path(weights)
    output.mkdir(parents=True, exist_ok=True)
    weights.parent.mkdir(parents=True, exist_ok=True)
    if (output / "STOP").exists():
        raise RuntimeError("Remove the STOP file before starting a new run")
    started = time.monotonic()
    deadline = started + minutes * 60
    train_deadline = max(started, deadline - min(120, minutes * 6))  # запас на финальный отчёт
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing a silent CPU fallback")
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85, dev)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    loss_kwargs = dict(w_mse=w_mse, w_pcc=w_pcc, w_ssim=w_ssim, worst=worst, ssim_global=ssim_global)
    model_cfg = {"base": base, "input_mode": input_mode, "unroll": unroll, "stage_base": stage_base}
    metadata = dict(physics_version=PHYSICS_VERSION, optics=DEFAULT_OPTICS.as_dict(), model=model_cfg,
                    loss=loss_kwargs, seed=seed, pool_size=pool_size)
    net = MultiPlaneUNet(base=base, input_mode=input_mode, unroll=unroll, stage_base=stage_base).to(dev)
    prop = FresnelSingleFFT(DEFAULT_OPTICS).to(dev)
    print(f"[model] {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M params, input={input_mode}, unroll={unroll}, "
          f"channels={net.features.channels}", flush=True)

    checkpoint, best, step = None, -float("inf"), 0
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location=dev, weights_only=True)
        for key in ("physics_version", "model", "seed"):
            if checkpoint.get(key) != metadata[key]:
                raise ValueError(f"Resume setting mismatch: {key}")
        net.load_state_dict(checkpoint["net"])
        best, step = float(checkpoint["best_score"]), int(checkpoint["step"])
    elif init_weights:  # дообучение (v2) от готовых весов с новым lr/лоссом
        ckpt = torch.load(init_weights, map_location=dev, weights_only=True)
        if ckpt.get("model") != model_cfg or ckpt.get("physics_version") != PHYSICS_VERSION:
            raise ValueError("init weights have a different model/physics config")
        net.load_state_dict(ckpt["net"])
        print(f"[init] weights from {init_weights}", flush=True)

    state = dict(pid=os.getpid(), status="preparing", device=str(dev),
                 gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
                 started_at=datetime.now(timezone.utc).isoformat(), max_minutes=minutes,
                 deadline_at=(datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat(), **metadata)

    def status(**values):
        state.update(values, elapsed_seconds=round(time.monotonic() - started, 1),
                     updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output / "status.json", state)

    status()
    keep_awake = os.name == "nt" and bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        print(f"[start] {state['gpu']}; {minutes:g} min; base={base}", flush=True)
        images, cats = load_or_build_pool(pool_size, seed=0)
        sampler = QuadSampler(images, cats, seed=seed + step)
        val = {k: torch.from_numpy(v.astype(np.float32) / 255.0)
               for k, v in fixed_quads("val", val_per_kind).items()}
        actual_batch = choose_batch(net, prop, sampler, dev, str(batch),
                                    min(train_deadline, time.monotonic() + 90), loss_kwargs)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        if checkpoint:
            opt.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
        atomic_json(output / "config.json", dict(**metadata, batch=actual_batch, lr=lr, minutes=minutes,
                                                 torch_version=str(torch.__version__),
                                                 cuda_version=torch.version.cuda, amp=dev.type == "cuda",
                                                 resume=resume, init_weights=init_weights))

        def save_last():
            atomic_save(output / "last.pt", dict(**metadata, net=net.state_dict(), optimizer=opt.state_dict(),
                                                 scaler=scaler.state_dict(), best_score=best, step=step,
                                                 torch_rng=torch.get_rng_state()))

        def validate_and_save(train_loss=None, train_diag=None):
            nonlocal best
            metrics = evaluate(net, prop, val, actual_batch, dev, loss_kwargs)
            if not math.isfinite(metrics["loss"]):
                raise RuntimeError("Non-finite validation; last valid checkpoint preserved")
            score = metrics["pcc"] + metrics["ssim"]  # критерий выбора лучших весов
            if score > best:
                best = score
                atomic_save(weights, dict(**metadata, net=net.state_dict(), best_score=best, step=step,
                                          validation=metrics))
            elapsed = time.monotonic() - started
            record = dict(step=step, elapsed_seconds=round(elapsed, 1), train_loss=train_loss, train=train_diag,
                          validation=metrics, best_score=best, lr=opt.param_groups[0]["lr"],
                          quads_seen=step * actual_batch)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, allow_nan=False) + "\n")
            status(status="training", step=step, batch=actual_batch, validation=metrics, best_score=best,
                   train_loss=train_loss,
                   gpu_memory_reserved_gb=round(torch.cuda.memory_reserved(dev) / 2 ** 30, 2) if dev.type == "cuda" else 0)
            save_last()
            per_kind = " ".join(f"{k}:{metrics[k]['pcc']:.3f}/{metrics[k]['ssim']:.3f}" for k in FIXED_KINDS)
            print(f"[val] step={step} PCC={metrics['pcc']:.4f} SSIM={metrics['ssim']:.4f} ({per_kind}) "
                  f"best={best:.4f} {elapsed / 60:.1f}m", flush=True)

        validate_and_save()
        last_eval = time.monotonic()
        losses, diags = [], []
        stop_reason = "time_limit"
        while time.monotonic() < train_deadline:
            if max_steps is not None and step >= max_steps:
                stop_reason = "max_steps"
                break
            if (output / "STOP").exists():
                stop_reason = "stop_requested"
                break
            net.train()
            frac = min(1.0, (time.monotonic() - started) / max(1.0, train_deadline - started))
            warmup = min(1.0, (step + 1) / 200)
            rate = lr * warmup * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
            for group in opt.param_groups:
                group["lr"] = rate
            x = torch.from_numpy(sampler.sample(actual_batch)).to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, diag = compute_loss(net, prop, x, dev, loss_kwargs)
            loss = loss.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; last valid checkpoint preserved")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            losses.append(float(loss.detach()))
            diags.append(diag)
            if time.monotonic() - last_eval >= eval_every:
                mean_diag = {k: round(float(np.mean([d[k] for d in diags])), 5) for k in diags[0]}
                validate_and_save(float(np.mean(losses)), mean_diag)
                losses.clear()
                diags.clear()
                last_eval = time.monotonic()
        mean_diag = {k: round(float(np.mean([d[k] for d in diags])), 5) for k in diags[0]} if diags else None
        validate_and_save(float(np.mean(losses)) if losses else None, mean_diag)
        net.load_state_dict(torch.load(weights, map_location=dev, weights_only=True)["net"])
        status(status="evaluating", stop_reason=stop_reason)
        report = final_report(net, output, metadata, dev, deadline + 600)
        status(status="completed", stop_reason=stop_reason, weights=str(weights),
               test_summary=[{k: r[k] for k in ("set", "label", "pcc", "ssim", "time_ms")} for r in report["test"]])
        print(f"[done] best weights: {weights}", flush=True)
    except BaseException as error:
        status(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return weights


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda", help="cuda | cpu | auto")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--base", type=int, default=32, help="ширина U-Net (32–48 для 6 ГБ)")
    ap.add_argument("--input-mode", choices=INPUT_MODES, default="imgphase")
    ap.add_argument("--unroll", type=int, default=0, help="число развёрнутых стадий GS+CNN после U-Net")
    ap.add_argument("--stage-base", type=int, default=16)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pool-size", type=int, default=20000)
    ap.add_argument("--val-per-kind", type=int, default=16)
    ap.add_argument("--eval-every", type=float, default=120.0, help="секунд между валидациями")
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--w-pcc", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=1.0)
    ap.add_argument("--worst", type=float, default=0.5, help="вес худшей плоскости (0..1)")
    ap.add_argument("--ssim-global", type=float, default=0.5,
                    help="доля глобального ИСС (слайд 3) в SSIM-члене лосса; 0 — только локальный SSIM")
    ap.add_argument("--init-weights", default=None, help="дообучение от готовых весов")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(ROOT / "data" / "training"))
    ap.add_argument("--weights", default=str(ROOT / "data" / "weights" / "doe_multiplane.pt"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    train(minutes=a.minutes, device=a.device, base=a.base, input_mode=a.input_mode, unroll=a.unroll,
          stage_base=a.stage_base, batch=a.batch, lr=a.lr,
          pool_size=a.pool_size, val_per_kind=a.val_per_kind, output=a.output, weights=a.weights,
          resume=a.resume, seed=a.seed, eval_every=a.eval_every, w_mse=a.w_mse, w_pcc=a.w_pcc,
          w_ssim=a.w_ssim, worst=a.worst, ssim_global=a.ssim_global, init_weights=a.init_weights,
          max_steps=a.max_steps)


if __name__ == "__main__":
    main()
