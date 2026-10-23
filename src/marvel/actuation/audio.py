"""One bounded fixed tone in a worker; capture never waits for playback."""
from concurrent.futures import ThreadPoolExecutor
import sys


def windows_tone():
    import winsound
    winsound.Beep(900, 400)


class FixedAlert:
    def __init__(self, player=None):
        if player is None and sys.platform != "win32":
            raise RuntimeError("Fixed audio currently needs Windows; Jetson audio adapter is pending")
        self.player = player or windows_tone
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="marvel_audio")
        self.future = None
        self.event_id = self.last_submitted = None
        self.closed = False

    def submit(self, event):
        if self.closed:
            raise RuntimeError("Audio worker closed")
        if event["kind"] != "PROLONGED_CLOSURE":
            return "not_a_closure_alert"
        if event["event_id"] == self.last_submitted:
            return "duplicate_ignored"
        # A result must be polled before accepting another event. No queued backlog.
        if self.future is not None:
            return "busy_dropped"
        self.last_submitted = self.event_id = event["event_id"]
        self.future = self.pool.submit(self.player)
        return "sound_requested"

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        try:
            self.future.result()
            result = {"event_id": self.event_id, "status": "playback_call_completed"}
        except Exception as error:
            result = {"event_id": self.event_id, "status": "playback_failed", "error": str(error)}
        self.future = self.event_id = None
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            # The built-in backend is a single 400 ms call. Wait only at shutdown.
            self.pool.shutdown(wait=True, cancel_futures=True)
