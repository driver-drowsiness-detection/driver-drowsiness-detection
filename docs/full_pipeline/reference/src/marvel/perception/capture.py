import multiprocessing as mp
import queue
import time
from marvel.contracts import Frame


def _live_worker(config, output, stop):
    import cv2
    source = config["source"]
    api = cv2.CAP_GSTREAMER if config["kind"] == "gstreamer" else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, api)
    def publish(item):
        try:
            output.put_nowait(item)
        except queue.Full:
            try:
                output.get_nowait()
            except queue.Empty:
                pass
            try:
                output.put_nowait(item)
            except queue.Full:
                pass
    try:
        if not cap.isOpened():
            publish(("error", "camera_open_failed"))
            return
        index = 0
        while not stop.is_set():
            ok, image = cap.read()
            if not ok:
                publish(("error", "camera_read_failed"))
                return
            publish(("frame", Frame(str(index), time.monotonic(), image)))
            index += 1
    finally:
        cap.release()


class LiveCapture:
    def __init__(self, config):
        ctx = mp.get_context("spawn")
        self.queue, self.stop = ctx.Queue(maxsize=1), ctx.Event()
        self.process = ctx.Process(target=_live_worker, args=(config, self.queue, self.stop))
        self.process.start()

    def read(self, timeout=0.1):
        try:
            kind, value = self.queue.get(timeout=timeout)
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError("Capture worker exited")
            return None
        if kind == "error":
            raise RuntimeError(value)
        return value

    def close(self):
        self.stop.set()
        self.process.join(1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1)
        self.queue.close()


class VideoCapture:
    def __init__(self, config):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(config["source"])
        if not self.cap.isOpened():
            raise RuntimeError("video_open_failed")
        self.index, self.last = 0, None
        self.fps = config.get("declared_cfr_fps")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("Declared constant frame rate must be positive")

    def read(self):
        import math
        ok, image = self.cap.read()
        if not ok:
            return None
        t = (self.index / self.fps if self.fps else
             self.cap.get(self.cv2.CAP_PROP_POS_MSEC) / 1000.0)
        if not math.isfinite(t) or t < 0 or (self.last is not None and t <= self.last):
            raise ValueError("Bad media timestamps: supply verified CFR fps or re-encode with audited timing")
        packet = Frame(str(self.index), t, image)
        self.index, self.last = self.index + 1, t
        return packet

    def close(self):
        self.cap.release()
