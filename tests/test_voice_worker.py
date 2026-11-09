from queue import Queue
from types import SimpleNamespace
import pytest
from marvel.reasoning.worker import VoiceWorker
from test_speech_alerts import event


class Channel(Queue):
    def cancel_join_thread(self): pass
    def close(self): pass


class Process:
    def __init__(self, **kwargs): self.alive = False
    def start(self): self.alive = True
    def is_alive(self): return self.alive
    def terminate(self): self.alive = False
    def join(self, timeout): pass


def worker():
    clock = [0]
    context = SimpleNamespace(Queue=Channel, Process=Process)
    return VoiceWorker("unused", clock=lambda: clock[0], context=context), clock


def test_startup_timeout_and_cleanup():
    w, clock = worker()
    assert w.submit(event()) == "not_ready"
    clock[0] = 91
    assert w.poll(None)["status"] == "startup_timeout"
    assert w.closed and not w.process.is_alive()
    w.close()


def test_generation_timeout_wins_over_late_completed_result():
    w, clock = worker()
    w.results.put({"status": "ready"})
    assert w.poll(None)["status"] == "ready"
    assert w.submit(event()) == "submitted"
    assert w.submit(event()) == "busy_dropped"
    w.results.put({"status": "generated", "episode_id": "episode"})
    clock[0] = 9
    assert w.poll("episode")["status"] == "generation_timeout"
    assert w.closed


def test_stale_result_and_process_failure_are_visible():
    w, _ = worker()
    w.results.put({"status": "ready"})
    w.poll(None)
    w.submit(event())
    w.results.put({"status": "generated", "episode_id": "episode"})
    assert w.poll("different")["status"] == "stale_discarded"
    w.process.alive = False
    assert w.poll(None)["status"] == "worker_exited"


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_timeouts_fail_before_spawning(duration):
    with pytest.raises(ValueError): VoiceWorker("unused", timeout_s=duration)
