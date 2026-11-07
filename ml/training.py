"""Ограниченное по времени возобновляемое обучение сети «4 изображения → фаза ДОЭ»."""
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

def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def compute_loss(net, prop, x, device, loss_kwargs):
    phase = net(x)  # БЕЗ AMP!
    return multiplane_loss(prop(phase.float()), x, **loss_kwargs)

def evaluate(net, prop, val: dict[str, torch.Tensor], batch: int, device: torch.device, loss_kwargs) -> dict:
    net.eval()
    out = {}
    for kind, quads in val.items():
        losses, pccs, ssims = [], [], []
        for start in range(0, len(quads), batch):
            x = quads[start:start + batch].to(device)
            phase = net(x)  # БЕЗ AMP!
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

def train(minutes: float = 120, device: str = "cuda", base: int = 32, input_mode: str = "imgphase",
          unroll: int = 0, stage_base: int = 16,
          batch="auto", lr: float = 1e-3, pool_size: int = 20000, val_per_kind: int = 16,
          output="data/training", weights="data/weights/doe_multiplane.pt", resume: bool = False,
          seed: int = 42, eval_every: float = 120.0, w_mse: float = 1.0, w_pcc: float = 1.0,
          w_ssim: float = 1.0, worst: float = 0.5, ssim_global: float = 0.5, init_weights: str | None = None,
          max_steps: int | None = None) -> Path:
    output, weights = Path(output), Path(weights)
    output.mkdir(parents=True, exist_ok=True)
    weights.parent.mkdir(parents=True, exist_ok=True)
    if (output / "STOP").exists():
        raise RuntimeError("Remove the STOP file before starting a new run")
    started = time.monotonic()
    deadline = started + minutes * 60
    train_deadline = max(started, deadline - min(120, minutes * 6))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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
    elif init_weights:
        ckpt = torch.load(init_weights, map_location=dev, weights_only=True)
        net.load_state_dict(ckpt["net"], strict=False)
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
        
        # Автоподбор батча
        if batch == "auto":
            actual_batch = 16
            print(f"[batch] using default {actual_batch}", flush=True)
        else:
            actual_batch = int(batch)
            
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        
        atomic_json(output / "config.json", dict(**metadata, batch=actual_batch, lr=lr, minutes=minutes,
                                                 torch_version=str(torch.__version__),
                                                 cuda_version=torch.version.cuda, amp=False,
                                                 resume=resume, init_weights=init_weights))

        def save_last():
            atomic_save(output / "last.pt", dict(**metadata, net=net.state_dict(), optimizer=opt.state_dict(),
                                                 best_score=best, step=step,
                                                 torch_rng=torch.get_rng_state()))

        def validate_and_save(train_loss=None, train_diag=None):
            nonlocal best
            metrics = evaluate(net, prop, val, actual_batch, dev, loss_kwargs)
            if not math.isfinite(metrics["loss"]):
                raise RuntimeError("Non-finite validation; last valid checkpoint preserved")
            score = metrics["pcc"] + metrics["ssim"]
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
                print(f"[warn] NaN loss at step {step}, skipping batch", flush=True)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
                
            loss.backward()
            if step % 100 == 0:
                g = net.unet.head.weight.grad
                print(f"[grad] head: {g.norm().item() if g is not None else 'None'}", flush=True)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)  # clip=10.0
            opt.step()
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
        
        from .pipeline import run, save_preview
        from .data import classic_testsets
        sets = classic_testsets()
        res = run(sets["photos_classic"], "net", model=net, device=str(dev))
        save_preview(res, output / "preview.png")
        
        status(status="completed", stop_reason=stop_reason, weights=str(weights))
        print(f"[done] best weights: {weights}", flush=True)
    except BaseException as error:
        status(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return weights

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--input-mode", choices=INPUT_MODES, default="imgphase")
    ap.add_argument("--unroll", type=int, default=0)
    ap.add_argument("--stage-base", type=int, default=16)
    ap.add_argument("--batch", default="16")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pool-size", type=int, default=20000)
    ap.add_argument("--val-per-kind", type=int, default=16)
    ap.add_argument("--eval-every", type=float, default=120.0)
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--w-pcc", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=1.0)
    ap.add_argument("--worst", type=float, default=0.5)
    ap.add_argument("--ssim-global", type=float, default=0.5)
    ap.add_argument("--init-weights", default=None)
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