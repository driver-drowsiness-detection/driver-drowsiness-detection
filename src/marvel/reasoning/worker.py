"""Bounded background generation; detection never waits for the language model."""
import multiprocessing as mp
from queue import Empty
import time
import math

from marvel.reasoning.local_model import LocalGenerator, validate_event


def _serve(folder, requests, results):
    try:
        generator = LocalGenerator(folder)
        results.put({"status": "ready"})
        while True:
            event = requests.get()
            if event is None:
                return
            results.put({"status": "generated", **generator.generate(event)})
    except Exception as error:
        results.put({"status": "failed", "error": str(error)[:500]})


class VoiceWorker:
    def __init__(self, folder, timeout_s=8.0, startup_s=90.0,
                 *, clock=time.monotonic, context=None):
        if any(not math.isfinite(v) or v <= 0 for v in (timeout_s, startup_s)):
            raise ValueError("Voice timeouts must be finite and positive")
        self.clock = clock
        self.timeout_s, self.startup_s = timeout_s, startup_s
        self.started = clock()
        self.ready = False
        self.pending = None
        self.closed = False
        ctx = context or mp.get_context("spawn")
        self.requests, self.results = ctx.Queue(1), ctx.Queue(1)
        self.process = ctx.Process(target=_serve, args=(str(folder), self.requests, self.results), daemon=True)
        self.process.start()

    def submit(self, event):
        if event.get("kind") != "PROLONGED_CLOSURE":
            return "ignored"
        validate_event(event)
        if self.closed:
            return "unavailable"
        if not self.ready:
            return "not_ready"
        if self.pending is not None:
            return "busy_dropped"
        self.requests.put_nowait(dict(event))
        self.pending = (event["episode_id"], self.clock())
        return "submitted"

    def poll(self, active_episode):
        if self.closed:
            return None
        # Expired work is rejected even if it completed between polling calls.
        elapsed = self.clock() - (self.pending[1] if self.pending else self.started)
        if (self.pending and elapsed > self.timeout_s) or (not self.ready and elapsed > self.startup_s):
            status = "generation_timeout" if self.pending else "startup_timeout"
            self.close()
            return {"status": status}
        try:
            result = self.results.get_nowait()
        except Empty:
            if not self.process.is_alive():
                self.close()
                return {"status": "worker_exited"}
            return None
        if result["status"] == "ready":
            self.ready = True
        elif result["status"] == "generated":
            expected = self.pending[0] if self.pending else None
            self.pending = None
            if result["episode_id"] != active_episode or result["episode_id"] != expected:
                result["status"] = "stale_discarded"
        elif result["status"] == "failed":
            self.close()
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=1)
        for channel in (self.requests, self.results):
            channel.cancel_join_thread()
            channel.close()
