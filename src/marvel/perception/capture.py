import multiprocessing as mp
import queue
import re
import sys
import time
import uuid

from marvel.contracts import Frame


def _camera_worker(config, output, stop, errors):
    """Own the camera in the child. Errors use a separate, reliable channel."""
    def publish(message):
        try:
            output.put_nowait(message)
        except queue.Full:
            try:
                output.get_nowait()
            except queue.Empty:
                pass
            try:
                output.put_nowait(message)
            except queue.Full:
                pass

    cap = None
    try:
        import cv2

        kind = config["kind"]
        if kind not in ("camera", "gstreamer"):
            raise ValueError("Capture kind must be camera or gstreamer")
        # Preserve perception branch's DirectShow choice on Windows. The
        # automatic MSMF backend can be slow to reopen some laptop cameras.
        backend = cv2.CAP_GSTREAMER if kind == "gstreamer" else (
            cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if kind == "gstreamer" and not re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
            raise RuntimeError("OpenCV has no GStreamer support; inspect the Jetson system build")
        cap = cv2.VideoCapture(config["source"], backend)
        if not cap.isOpened():
            raise RuntimeError("camera_open_failed")

        index = 0
        while not stop.is_set():
            ok, image = cap.read()
            observed = time.monotonic()  # Read completion, not sensor exposure time.
            if not ok or image is None or image.size == 0:
                raise RuntimeError("camera_read_failed")
            if image.ndim != 3 or image.shape[2] != 3 or str(image.dtype) != "uint8":
                raise RuntimeError("unexpected_camera_image_format")
            frame = Frame(config["session_id"] + ":" + str(index), observed, image)
            publish(("frame", frame))
            index += 1
    except Exception as error:
        errors.send(str(error))
    finally:
        if cap is not None:
            cap.release()
        errors.close()
        # A preview may discard queued images at exit. Never wait for a feeder
        # to flush a large image after the parent has stopped consuming.
        output.cancel_join_thread()


class LiveCapture:
    def __init__(self, config, *, _worker=None):
        if config.get("kind") not in ("camera", "gstreamer"):
            raise ValueError("Capture kind must be camera or gstreamer")
        config = dict(config, session_id=uuid.uuid4().hex[:12])
        context = mp.get_context("spawn")
        self.queue = context.Queue(maxsize=1)
        self.stop = context.Event()
        self.errors, sender = context.Pipe(duplex=False)
        self.closed = False
        self.process = context.Process(
            target=_worker or _camera_worker,
            args=(config, self.queue, self.stop, sender),
            daemon=True,
        )
        try:
            self.process.start()
        except BaseException:
            self.queue.close()
            self.errors.close()
            raise
        finally:
            sender.close()

    def _check_error(self):
        if self.errors.poll():
            try:
                message = self.errors.recv()
            except EOFError:
                return
            raise RuntimeError("Capture: " + message)

    def read(self, timeout=0.1):
        if self.closed:
            raise RuntimeError("Capture is closed")
        if timeout < 0:
            raise ValueError("Read timeout must be nonnegative")
        self._check_error()
        try:
            kind, value = self.queue.get(timeout=timeout)
        except queue.Empty:
            self._check_error()
            if not self.process.is_alive():
                raise RuntimeError("Camera worker exited")
            return None
        self._check_error()
        return value

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.0)
        self.queue.close()
        self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
