import multiprocessing as mp
import queue
import time
from .prompt_builder import fallback


def _run(config, event, output):
    from .llm_inference import generate
    try:
        output.put(generate(config, event))
    except Exception as exc:
        output.put({"message": fallback(event), "source": "fallback",
                    "error": type(exc).__name__ + ": " + str(exc)[:200]})


class SpeechWorker:
    """Capacity one. Parent owns deadlines, event freshness and all sound output."""
    def __init__(self, config):
        self.config, self.process, self.output = config, None, None
        self.event = self.started = None

    def submit(self, event):
        if self.process is not None:
            return False
        context = mp.get_context("spawn")
        self.output = context.Queue(maxsize=1)
        self.event, self.started = dict(event), time.monotonic()
        self.process = context.Process(target=_run, args=(self.config, self.event, self.output))
        self.process.start()
        return True

    def poll(self, current_event_id):
        if self.process is None:
            return None
        age = time.monotonic()-self.started
        if current_event_id != self.event["event_id"] or age > self.config["ttl_s"]:
            self.close()
            return {"source": "discarded", "message": None}
        try:
            result = self.output.get_nowait()
        except queue.Empty:
            if age <= self.config["timeout_s"]:
                return None
            result = {"message": fallback(self.event), "source": "timeout_fallback"}
        result["event_id"] = self.event["event_id"]
        result["worker_seconds"] = age
        self.close()
        return result

    def close(self):
        if self.process is not None:
            self.process.join(0.02)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(0.2)
            self.output.cancel_join_thread()
            self.output.close()
        self.process = self.output = None
