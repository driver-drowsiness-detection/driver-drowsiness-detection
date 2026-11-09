"""Render current episode warnings asynchronously; optionally play on Windows."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import uuid

from marvel.actuation.speech import render_speech


class VoiceOutput:
    def __init__(self, folder, speak=False, *, renderer=render_speech, clock=time.monotonic):
        self.folder, self.speak = Path(folder), speak
        self.renderer, self.clock = renderer, clock
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.episode = self.playing = None
        self.submitted = None

    def submit(self, result):
        if result.get("status") != "generated":
            return "ignored"
        if self.future is not None:
            return "busy_dropped"
        self.episode = result["episode_id"]
        self.submitted = self.clock()
        self.future = self.pool.submit(self.renderer, result["text"], self.folder / (uuid.uuid4().hex + ".wav"))
        return "synthesis_submitted"

    def poll(self, active_episode):
        if self.playing and self.playing != active_episode:
            import winsound
            winsound.PlaySound(None, 0)
            self.playing = None
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        try:
            result = future.result()
            stale = self.episode != active_episode or self.clock() - self.submitted > 20
            status = "speech_stale_discarded" if stale else "speech_rendered"
            if self.speak and not stale:
                import winsound
                winsound.PlaySound(result["audio_path"], winsound.SND_FILENAME | winsound.SND_ASYNC)
                self.playing = self.episode
                status = "playback_requested"
            return dict(status=status, episode_id=self.episode, **result)
        except Exception as error:
            return dict(status="speech_failed", episode_id=self.episode, error=str(error)[:500])

    def close(self):
        if self.playing:
            import winsound
            winsound.PlaySound(None, 0)
            self.playing = None
        # The SAPI subprocess has its own 15 second timeout.
        self.pool.shutdown(wait=True, cancel_futures=True)
