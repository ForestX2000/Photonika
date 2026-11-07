"""Ограниченное по времени возобновляемое обучение сети «4 изображения → фаза ДОЭ»."""
from __future__ import annotations
import argparse
import copy
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
from .model import ARCHITECTURES, INPUT_MODES, build_model, load_model

ROOT = Path(__file__).resolve().parent.parent

TRAINING_DEFAULTS = dict(precision="fp32", grad_accum=1, photo_split="legacy", deep_supervision=0.0,
                         quantization_aware=False, ema_decay=0.0, teacher_weight=0.0,
                         teacher_every=16, teacher_gs_iters=8, teacher_adam_steps=8)

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

def precision_context(device, precision: str):
    device = torch.device(device)
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def quantize_phase_ste(phase: torch.Tensor) -> torch.Tensor:
    """Same 256 codes as the submission PNG; identity surrogate gradient."""
    phase = phase.float()
    wrapped = torch.atan2(torch.sin(phase), torch.cos(phase))
    code = torch.round((wrapped + math.pi) * (255.0 / (2 * math.pi))).clamp(0, 255)
    quantized = code * (2 * math.pi / 255.0) - math.pi
    return wrapped + (quantized - wrapped).detach()


def forward_loss(net, prop, x, device, loss_kwargs, *, precision="fp32",
                 quantization_aware=False, deep_supervision=0.0):
    # Autocast is confined to model CNN/linear layers. Model FFTs explicitly
    # remain complex64; propagation and metric losses here remain float32.
    with precision_context(device, precision):
        phases = net(x, return_stages=True) if deep_supervision else (net(x),)
    phase = phases[-1].float()

    def optical_loss(value):
        value = quantize_phase_ste(value) if quantization_aware else value.float()
        return multiplane_loss(prop(value), x, **loss_kwargs)

    loss, diag = optical_loss(phase)
    if deep_supervision and len(phases) > 1:
        auxiliary = torch.stack([optical_loss(value)[0] for value in phases[:-1]])
        weights = torch.arange(1, len(phases), device=loss.device, dtype=loss.dtype)
        auxiliary = (auxiliary * weights[:, None]).sum(0) / weights.sum()
        loss = loss + deep_supervision * auxiliary
        diag["auxiliary_loss"] = float(auxiliary.mean().detach())
    return loss, diag, phase


def compute_loss(net, prop, x, device, loss_kwargs, **options):
    loss, diag, _ = forward_loss(net, prop, x, device, loss_kwargs, **options)
    return loss, diag

@torch.inference_mode()
def evaluate(net, prop, val: dict[str, torch.Tensor], batch: int, device: torch.device, loss_kwargs,
             *, precision="fp32", quantization_aware=False) -> dict:
    if batch < 1 or not val or any(len(q) == 0 for q in val.values()):
        raise ValueError("validation requires nonempty sets and a positive batch")
    net.eval()
    out = {}
    for kind, quads in val.items():
        losses, pccs, ssims = [], [], []
        for start in range(0, len(quads), batch):
            x = quads[start:start + batch].to(device)
            with precision_context(device, precision):
                phase = net(x)
            if quantization_aware:
                phase = quantize_phase_ste(phase)
            mod = prop(phase.float())
            loss, _ = multiplane_loss(mod, x, **loss_kwargs)
            p, s = plane_metrics(mod, x)
            losses.append(loss.cpu())
            pccs.append(p.cpu())
            ssims.append(s.cpu())
        p, s = torch.cat(pccs), torch.cat(ssims)
        out[kind] = {"loss": float(torch.cat(losses).mean()), "pcc": float(p.mean()), "ssim": float(s.mean()),
                     "pcc_planes": [round(float(v), 4) for v in p.mean(0)]}
    count = sum(len(q) for q in val.values())
    for metric in ("loss", "pcc", "ssim"):
        out[metric] = sum(out[k][metric] * len(val[k]) for k in val) / count
    return out

def capture_rng(sampler: QuadSampler, device: torch.device) -> dict:
    numpy_state = np.random.get_state()
    state = {"torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "numpy_rng": (numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)),
                           *numpy_state[2:]),
             "sampler_rng": copy.deepcopy(sampler.rng.bit_generator.state)}
    if device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state: dict, sampler: QuadSampler, device: torch.device) -> None:
    if "torch_rng" in state:
        torch.set_rng_state(state["torch_rng"].cpu())
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if "numpy_rng" in state:
        name, keys, position, has_gauss, cached = state["numpy_rng"]
        np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), position, has_gauss, cached))
    if "sampler_rng" in state:
        sampler.rng.bit_generator.state = state["sampler_rng"]
    if device.type == "cuda" and "cuda_rng" in state:
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)


def choose_batch(net, prop, sampler, device, requested, limit, loss_kwargs, *,
                 max_batch=128, **loss_options) -> int:
    """Select measured throughput, preserving weights, gradients and RNG."""
    if requested != "auto":
        value = int(requested)
        if value < 1:
            raise ValueError("batch must be positive")
        return value
    if device.type != "cuda":
        return 2
    original = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    rng = capture_rng(sampler, device)
    was_training = net.training
    measurements = []
    try:
        candidates = [n for n in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256) if n <= max_batch]
        if max_batch not in candidates:
            candidates.append(max_batch)
            candidates.sort()
        for candidate in candidates:
            if measurements and time.monotonic() >= limit:
                break
            opt, x, loss = None, None, None
            try:
                net.train()
                opt = torch.optim.AdamW(net.parameters(), lr=0.0, weight_decay=0.0)
                x = torch.from_numpy(sampler.sample(candidate)).to(device)
                torch.cuda.reset_peak_memory_stats(device)
                # Warm up this shape, then measure a complete backward + Adam
                # step with optimizer allocations present, at zero learning rate.
                elapsed = None
                for probe in range(2):
                    opt.zero_grad(set_to_none=True)
                    sync(device)
                    probe_started = time.monotonic()
                    loss, _ = compute_loss(net, prop, x, device, loss_kwargs, **loss_options)
                    if not torch.isfinite(loss).all():
                        raise RuntimeError("Non-finite loss during batch selection")
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
                    opt.step()
                    sync(device)
                    elapsed = time.monotonic() - probe_started
                    del loss
                    loss = None
                throughput = candidate / elapsed
                peak = torch.cuda.max_memory_allocated(device)
                total = torch.cuda.get_device_properties(device).total_memory
                if peak > total * 0.75:
                    print(f"[batch] {candidate}: reserving memory headroom for teacher/validation", flush=True)
                    break
                measurements.append((candidate, throughput))
                print(f"[batch] {candidate}: {throughput:.1f} quads/s", flush=True)
            except torch.OutOfMemoryError:
                print(f"[batch] {candidate}: VRAM limit", flush=True)
                break
            finally:
                net.zero_grad(set_to_none=True)
                del opt, x, loss
                torch.cuda.empty_cache()
    finally:
        net.load_state_dict(original)
        net.train(was_training)
        restore_rng(rng, sampler, device)
    if not measurements:
        raise RuntimeError("No batch fits in GPU memory; reduce --base or --unroll")
    fastest = max(rate for _, rate in measurements)
    chosen = min(batch for batch, rate in measurements if rate >= fastest * 0.95)
    print(f"[batch] selected {chosen} (within 5% of peak measured throughput)", flush=True)
    return chosen


def check_checkpoint(checkpoint: dict, metadata: dict, *, resume: bool = False) -> None:
    """Compare normalized metadata returned by load_model before loading training state."""
    keys = ("physics_version", "optics", "model")
    if resume:
        keys += ("seed", "loss", "pool_size")
    for key in keys:
        if checkpoint.get(key) != metadata[key]:
            raise ValueError(f"checkpoint setting mismatch: {key}")
    if resume:
        actual = {**TRAINING_DEFAULTS, **checkpoint.get("training", {})}
        expected = {**TRAINING_DEFAULTS, **metadata.get("training", {})}
        if actual != expected:
            raise ValueError("checkpoint setting mismatch: training")
        if checkpoint.get("photo_manifest_hash") != metadata.get("photo_manifest_hash"):
            raise ValueError("checkpoint setting mismatch: photo sources changed")


def check_best_checkpoint(checkpoint: dict, metadata: dict, best_score: float, last_step: int) -> None:
    """A resumed run must not silently reuse another run's best-weight artifact."""
    check_checkpoint(checkpoint, metadata, resume=True)
    if checkpoint.get("best_score") != best_score:
        raise ValueError("best weights score does not match last.pt; use the original --weights path")
    if not isinstance(checkpoint.get("step"), int) or not 0 <= checkpoint["step"] <= last_step:
        raise ValueError("best weights step does not match the resumed run")


def train(minutes: float = 120, device: str = "cuda", base: int = 32, input_mode: str = "imgphase",
          unroll: int = 4, stage_base: int = 16,
          batch="auto", lr: float = 1e-4, pool_size: int = 20000, val_per_kind: int = 16,
          output="data/training", weights="data/weights/doe_multiplane.pt", resume: bool = False,
          seed: int = 42, eval_every: float = 120.0, w_mse: float = 1.0, w_pcc: float = 1.0,
          w_ssim: float = 1.0, worst: float = 0.5, ssim_global: float = 0.5, init_weights: str | None = None,
          max_steps: int | None = None, architecture: str = "physics-unrolled",
          model_version: int = 1, spectral_modes: int = 8, precision: str = "fp32",
          activation_checkpointing: bool = False, grad_accum: int = 1,
          photo_split: str = "legacy", deep_supervision: float = 0.0,
          quantization_aware: bool = False, ema_decay: float = 0.0,
          teacher_weight: float = 0.0, teacher_every: int = 16,
          teacher_gs_iters: int = 8, teacher_adam_steps: int = 8,
          auto_batch_max: int = 128, auto_batch_seconds: float = 60.0,
          preview: bool = True) -> Path:
    if min(minutes, lr, pool_size, val_per_kind, base, eval_every) <= 0:
        raise ValueError("minutes, lr, pool_size, val_per_kind, base and eval_every must be positive")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    if resume and init_weights:
        raise ValueError("choose either resume or init_weights")
    if precision not in ("fp32", "bf16", "auto") or photo_split not in ("legacy", "source"):
        raise ValueError("invalid precision or photo_split")
    if min(grad_accum, teacher_every, auto_batch_max, auto_batch_seconds) <= 0:
        raise ValueError("grad_accum, teacher_every and batch search limits must be positive")
    if min(deep_supervision, teacher_weight, teacher_gs_iters, teacher_adam_steps) < 0 or not 0 <= ema_decay < 1:
        raise ValueError("invalid supervision, teacher or EMA setting")
    if deep_supervision and (architecture != "physics-unrolled" or model_version != 2):
        raise ValueError("deep supervision requires physics-unrolled model_version=2")
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
    bf16_available = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    if precision == "auto":
        precision = "bf16" if bf16_available else "fp32"
    if precision == "bf16" and not bf16_available:
        raise ValueError("bf16 requires a CUDA GPU with native bfloat16 support; choose fp32")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85, dev)
        # Auto-batch probes several shapes. Exhaustive cuDNN algorithm searches
        # for every BF16 shape can consume an entire short screening budget.
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    loss_kwargs = dict(w_mse=w_mse, w_pcc=w_pcc, w_ssim=w_ssim, worst=worst, ssim_global=ssim_global)
    net = build_model({"architecture": architecture, "base": base, "input_mode": input_mode,
                       "unroll": unroll, "stage_base": stage_base, "model_version": model_version,
                       "spectral_modes": spectral_modes}).to(dev)
    if activation_checkpointing:
        if not hasattr(net, "set_activation_checkpointing"):
            raise ValueError("activation checkpointing requires physics-unrolled model_version=2")
        net.set_activation_checkpointing(True)
    model_cfg = net.config()
    training_options = dict(precision=precision, grad_accum=grad_accum, photo_split=photo_split,
                            deep_supervision=deep_supervision, quantization_aware=quantization_aware,
                            ema_decay=ema_decay, teacher_weight=teacher_weight,
                            teacher_every=teacher_every, teacher_gs_iters=teacher_gs_iters,
                            teacher_adam_steps=teacher_adam_steps)
    metadata = dict(physics_version=PHYSICS_VERSION, optics=DEFAULT_OPTICS.as_dict(), model=model_cfg,
                    loss=loss_kwargs, seed=seed, pool_size=pool_size, training=training_options)
    if photo_split == "source":
        from .data import photo_source_manifest, photo_manifest_hash, PHOTO_SPLIT_VERSION
        photo_manifest = photo_source_manifest()
        metadata.update(photo_manifest_hash=photo_manifest_hash(photo_manifest),
                        photo_split_version=PHOTO_SPLIT_VERSION)
    prop = FresnelSingleFFT(DEFAULT_OPTICS).to(dev)
    print(f"[model] {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M params, input={input_mode}, unroll={unroll}, "
          f"channels={net.features.channels}", flush=True)

    checkpoint, best, step = None, -float("inf"), 0
    schedule_progress = 0.0
    if resume:
        loaded, checkpoint = load_model(output / "last.pt", "cpu")
        del loaded
        check_checkpoint(checkpoint, metadata, resume=True)
        net.load_state_dict(checkpoint["net"])
        best, step = float(checkpoint["best_score"]), int(checkpoint["step"])
        schedule_progress = float(checkpoint.get("schedule_progress", 0.0))
        if not weights.is_file():
            # A relocated run has no earlier best artifact; select from the resumed run.
            best = -float("inf")
        else:
            loaded, best_checkpoint = load_model(weights, "cpu")
            del loaded
            check_best_checkpoint(best_checkpoint, metadata, best, step)
            del best_checkpoint
    elif init_weights:
        loaded, ckpt = load_model(init_weights, "cpu")
        del loaded
        check_checkpoint(ckpt, metadata)
        net.load_state_dict(ckpt["net"])
        print(f"[init] weights from {init_weights}", flush=True)

    ema = copy.deepcopy(net).eval().requires_grad_(False) if ema_decay else None
    if ema is not None and checkpoint:
        if "ema" not in checkpoint:
            raise ValueError("EMA training state missing from last.pt")
        ema.load_state_dict(checkpoint["ema"])

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
        if photo_split == "source":
            images, cats = load_or_build_pool(pool_size, seed=0, photo_split="train", mmap=True)
            raw_val = fixed_quads("val", val_per_kind, photo_split="val")
            atomic_json(output / "photo_sources.json", {"sources": photo_manifest,
                                                        "hash": metadata["photo_manifest_hash"]})
        else:
            images, cats = load_or_build_pool(pool_size, seed=0)
            raw_val = fixed_quads("val", val_per_kind)
        sampler = QuadSampler(images, cats, seed=seed + step)
        val = {k: torch.from_numpy(v.astype(np.float32) / 255.0)
               for k, v in raw_val.items()}
        
        loss_options = dict(precision=precision, quantization_aware=quantization_aware,
                            deep_supervision=deep_supervision)
        actual_batch = choose_batch(net, prop, sampler, dev, str(batch),
                                    min(train_deadline, time.monotonic() + auto_batch_seconds), loss_kwargs,
                                    max_batch=auto_batch_max, **loss_options)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        if checkpoint:
            opt.load_state_dict(checkpoint["optimizer"])
            restore_rng(checkpoint, sampler, dev)
        resume_progress = schedule_progress
        atomic_json(output / "config.json", dict(**metadata, batch=actual_batch, lr=lr, minutes=minutes,
                                                 torch_version=str(torch.__version__),
                                                 cuda_version=torch.version.cuda, amp=precision,
                                                 effective_batch=actual_batch * grad_accum,
                                                 activation_checkpointing=activation_checkpointing,
                                                 preview=preview,
                                                 resume=resume, init_weights=init_weights))

        def save_last():
            extra = {"ema": ema.state_dict()} if ema is not None else {}
            atomic_save(output / "last.pt", dict(**metadata, net=net.state_dict(), optimizer=opt.state_dict(),
                                                 best_score=best, step=step, schedule_progress=schedule_progress,
                                                 **extra, **capture_rng(sampler, dev)))

        def validate_and_save(train_loss=None, train_diag=None):
            nonlocal best
            validation_net = ema if ema is not None else net
            metrics = evaluate(validation_net, prop, val, actual_batch, dev, loss_kwargs,
                               precision=precision, quantization_aware=quantization_aware)
            if not math.isfinite(metrics["loss"]):
                raise RuntimeError("Non-finite validation; last valid checkpoint preserved")
            score = metrics["pcc"] + metrics["ssim"]
            if score > best:
                best = score
                atomic_save(weights, dict(**metadata, net=validation_net.state_dict(), best_score=best, step=step,
                                          weight_source="ema" if ema is not None else "raw",
                                          validation=metrics))
            elapsed = time.monotonic() - started
            record = dict(step=step, elapsed_seconds=round(elapsed, 1), train_loss=train_loss, train=train_diag,
                          validation=metrics, best_score=best, lr=opt.param_groups[0]["lr"],
                          quads_seen=step * actual_batch * grad_accum)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, allow_nan=False) + "\n")
            status(status="training", step=step, batch=actual_batch, validation=metrics, best_score=best,
                   train_loss=train_loss, effective_batch=actual_batch * grad_accum,
                   gpu_memory_reserved_gb=round(torch.cuda.memory_reserved(dev) / 2 ** 30, 2) if dev.type == "cuda" else 0)
            save_last()
            per_kind = " ".join(f"{k}:{metrics[k]['pcc']:.3f}/{metrics[k]['ssim']:.3f}" for k in FIXED_KINDS)
            print(f"[val] step={step} PCC={metrics['pcc']:.4f} SSIM={metrics['ssim']:.4f} ({per_kind}) "
                  f"best={best:.4f} {elapsed / 60:.1f}m", flush=True)

        validate_and_save()
        last_eval = time.monotonic()
        optimizer_started = time.monotonic()
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
            frac = min(1.0, (time.monotonic() - optimizer_started) / max(1.0, train_deadline - optimizer_started))
            schedule_progress = resume_progress + (1.0 - resume_progress) * frac
            warmup = min(1.0, (step + 1) / 200)
            rate = lr * warmup * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * schedule_progress)))
            for group in opt.param_groups:
                group["lr"] = rate
                
            opt.zero_grad(set_to_none=True)
            step_losses, step_diags = [], []
            for microstep in range(grad_accum):
                x = torch.from_numpy(sampler.sample(actual_batch)).to(dev, non_blocking=True)
                loss, diag, phase = forward_loss(net, prop, x, dev, loss_kwargs, **loss_options)
                if teacher_weight and microstep == 0 and step % teacher_every == 0:
                    from .teacher import synthesize_teacher, phase_distillation_loss
                    teacher_parts = []
                    for start in range(0, len(x), 16):
                        teacher_parts.append(synthesize_teacher(x[start:start + 16], prop,
                                                 init_phase=phase[start:start + 16].detach(),
                                                 gs_iters=teacher_gs_iters, adam_steps=teacher_adam_steps,
                                                 seed=seed + step + start, loss_kwargs=loss_kwargs))
                    teacher = torch.cat(teacher_parts)
                    hint = phase_distillation_loss(phase, teacher, target=x)
                    loss = loss + teacher_weight * hint
                    diag["teacher_loss"] = float(hint.mean().detach())
                loss = loss.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; last valid checkpoint preserved")
                (loss / grad_accum).backward()
                step_losses.append(float(loss.detach()))
                step_diags.append(diag)
                del phase, loss, x
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            step += 1
            if ema is not None:
                decay = min(ema_decay, (1 + step) / (10 + step))
                with torch.no_grad():
                    for average, parameter in zip(ema.parameters(), net.parameters()):
                        average.lerp_(parameter.detach(), 1 - decay)
                    for average, value in zip(ema.buffers(), net.buffers()):
                        average.copy_(value)
            losses.append(float(np.mean(step_losses)))
            keys = set().union(*(d.keys() for d in step_diags))
            diags.append({k: float(np.mean([d.get(k, 0.0) for d in step_diags])) for k in keys})
            
            if time.monotonic() - last_eval >= eval_every:
                keys = set().union(*(d.keys() for d in diags))
                mean_diag = {k: round(float(np.mean([d.get(k, 0.0) for d in diags])), 5) for k in keys}
                validate_and_save(float(np.mean(losses)), mean_diag)
                losses.clear()
                diags.clear()
                last_eval = time.monotonic()
                
        keys = set().union(*(d.keys() for d in diags)) if diags else ()
        mean_diag = {k: round(float(np.mean([d.get(k, 0.0) for d in diags])), 5) for k in keys} if diags else None
        validate_and_save(float(np.mean(losses)) if losses else None, mean_diag)
        loaded, best_checkpoint = load_model(weights, "cpu")
        del loaded
        check_best_checkpoint(best_checkpoint, metadata, best, step)
        net.load_state_dict(best_checkpoint["net"])
        del best_checkpoint
        net.eval()
        status(status="evaluating", stop_reason=stop_reason)
        
        from .pipeline import run, save_preview
        from .data import classic_testsets
        sets = classic_testsets() if preview else {}
        if "photos_classic" in sets:
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
    ap.add_argument("--architecture", choices=ARCHITECTURES, default="physics-unrolled")
    ap.add_argument("--model-version", type=int, choices=(1, 2), default=1)
    ap.add_argument("--spectral-modes", type=int, default=8)
    ap.add_argument("--precision", choices=("fp32", "bf16", "auto"), default="fp32")
    ap.add_argument("--activation-checkpointing", action="store_true")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--photo-split", choices=("legacy", "source"), default="legacy")
    ap.add_argument("--deep-supervision", type=float, default=0.0)
    ap.add_argument("--quantization-aware", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.0)
    ap.add_argument("--teacher-weight", type=float, default=0.0)
    ap.add_argument("--teacher-every", type=int, default=16)
    ap.add_argument("--teacher-gs-iters", type=int, default=8)
    ap.add_argument("--teacher-adam-steps", type=int, default=8)
    ap.add_argument("--auto-batch-max", type=int, default=128)
    ap.add_argument("--auto-batch-seconds", type=float, default=60.0)
    ap.add_argument("--no-preview", action="store_true", help="reserve classic examples for the final benchmark")
    ap.add_argument("--input-mode", choices=INPUT_MODES, default="imgphase")
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--stage-base", type=int, default=16)
    ap.add_argument("--batch", default="auto")
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
          max_steps=a.max_steps, architecture=a.architecture, model_version=a.model_version,
          spectral_modes=a.spectral_modes, precision=a.precision, activation_checkpointing=a.activation_checkpointing,
          grad_accum=a.grad_accum, photo_split=a.photo_split, deep_supervision=a.deep_supervision,
          quantization_aware=a.quantization_aware, ema_decay=a.ema_decay, teacher_weight=a.teacher_weight,
          teacher_every=a.teacher_every, teacher_gs_iters=a.teacher_gs_iters, teacher_adam_steps=a.teacher_adam_steps,
          auto_batch_max=a.auto_batch_max, auto_batch_seconds=a.auto_batch_seconds, preview=not a.no_preview)

if __name__ == "__main__":
    main()
