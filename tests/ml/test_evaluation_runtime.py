"""The progress monitor must remain visible during a blocked main stage."""
import threading

from tools import evaluation_runtime as runtime


def test_heartbeat_reports_stage_while_main_thread_waits(monkeypatch):
    heartbeat = threading.Event()
    messages = []

    def capture(message, **kwargs):
        messages.append(message)
        if message.startswith("[alive]"):
            heartbeat.set()

    monkeypatch.setattr(runtime, "print", capture, raising=False)
    with runtime.monitor(interval=.01):
        runtime.announce("loading photographs")
        assert heartbeat.wait(2)
    assert any("loading photographs" in m and "last progress" in m for m in messages)
