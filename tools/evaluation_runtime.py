"""Visible progress without making CUDA calls from a monitoring thread."""
from __future__ import annotations

from contextlib import contextmanager
import faulthandler
import os
import signal
import threading
import time

_lock = threading.Lock()
_stage = "starting"
_updated = time.monotonic()


def announce(message: str) -> None:
    global _stage, _updated
    with _lock:
        _stage, _updated = message, time.monotonic()
    print(message, flush=True)


@contextmanager
def monitor(interval: float = 30.0):
    """Report liveness even while importing, decoding images or warming up."""
    stop = threading.Event()
    started = time.monotonic()

    def heartbeat():
        previous_cpu = time.process_time()
        previous_time = time.monotonic()
        while not stop.wait(interval):
            now, cpu = time.monotonic(), time.process_time()
            with _lock:
                current, updated = _stage, _updated
            utilization = 100 * (cpu - previous_cpu) / max(now - previous_time, 1e-9)
            print(f"[alive] {(now - started) / 60:.1f} min; CPU={utilization:.0f}% "
                  f"(100%=one core); last progress {now - updated:.0f}s ago; {current}", flush=True)
            previous_cpu, previous_time = cpu, now

    registered = False
    if hasattr(signal, "SIGUSR1") and threading.current_thread() is threading.main_thread():
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True)
            registered = True
        except (OSError, RuntimeError):
            pass
    worker = threading.Thread(target=heartbeat, name="evaluation-heartbeat", daemon=True)
    worker.start()
    announce(f"[start] PID={os.getpid()}; progress heartbeat every {interval:g}s")
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=1)
        if registered:
            faulthandler.unregister(signal.SIGUSR1)
