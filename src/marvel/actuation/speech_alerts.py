"""Optional speech consumes confirmed events independently of the fixed tone."""
from .voice import VoiceOutput
from marvel.reasoning.local_model import FALLBACK, validate_event


class SpeechAlerts:
    def __init__(self, folder, model=None, *, output=None, worker=None):
        self.output = output if output is not None else VoiceOutput(folder, speak=True)
        self.worker = worker
        if model is not None and self.worker is None:
            from marvel.reasoning.worker import VoiceWorker
            self.worker = VoiceWorker(model)
        self.pending = None

    def _fixed(self, event):
        return self.output.submit(dict(status="generated", text=FALLBACK,
                                       episode_id=event["episode_id"], event_id=event["event_id"],
                                       source="fixed_fallback"))

    def submit(self, event):
        if event.get("kind") != "PROLONGED_CLOSURE":
            return "ignored"
        validate_event(event)
        if self.worker is None:
            return self._fixed(event)
        status = self.worker.submit(event)
        if status == "submitted":
            self.pending = dict(event)
            return status
        if status in ("not_ready", "unavailable"):
            return "fallback:" + self._fixed(event)
        return status

    def poll(self, active_episode):
        records = []
        if self.worker:
            result = self.worker.poll(active_episode)
            if result:
                records.append(result)
                if result["status"] == "generated":
                    records.append(dict(status=self.output.submit(result), episode_id=result["episode_id"]))
                    self.pending = None
                elif result["status"] in ("generation_timeout", "startup_timeout", "worker_exited", "failed"):
                    if self.pending and self.pending["episode_id"] == active_episode:
                        records.append(dict(status="fallback:" + self._fixed(self.pending)))
                    self.pending = None
                elif result["status"] == "stale_discarded":
                    self.pending = None
        completed = self.output.poll(active_episode)
        if completed:
            records.append(completed)
        return records

    def close(self):
        try:
            if self.worker:
                self.worker.close()
        finally:
            self.output.close()
