"""Exercise real spawned processes with controlled camera behavior."""
import queue
import time

import numpy as np
import pytest

from marvel.contracts import Frame
from marvel.perception.capture import LiveCapture, _camera_worker


def stalled(config, output, stop, errors):
    # Simulates a native read ignoring the cooperative stop event.
    time.sleep(30)


def fails(config, output, stop, errors):
    errors.send("simulated_open_failure")
    stop.wait(5)


def produces(config, output, stop, errors):
    index = 0
    while not stop.is_set():
        try:
            output.put_nowait(("frame", Frame(str(index), time.monotonic(),
                                             np.full((8, 8, 3), index % 255, np.uint8))))
        except queue.Full:
            pass
        index += 1
        stop.wait(.01)
    output.cancel_join_thread()


def test_no_frame_read_is_bounded_and_stuck_worker_is_terminated():
    capture = LiveCapture(dict(kind="camera", source=0), _worker=stalled)
    started = time.monotonic()
    try:
        assert capture.read(.05) is None
        assert time.monotonic() - started < 1
    finally:
        capture.close()
    assert not capture.process.is_alive()
    assert time.monotonic() - started < 5
    capture.close()  # Idempotent cleanup.


def test_worker_error_is_delivered_and_cleanup_works():
    with LiveCapture(dict(kind="camera", source=0), _worker=fails) as capture:
        deadline = time.monotonic() + 10
        with pytest.raises(RuntimeError, match="simulated_open_failure"):
            while time.monotonic() < deadline:
                capture.read(.1)
    assert not capture.process.is_alive()


def test_real_spawn_transports_frame_identity_and_can_restart():
    for _ in range(2):
        with LiveCapture(dict(kind="camera", source=0), _worker=produces) as capture:
            deadline = time.monotonic() + 10
            frame = None
            while frame is None and time.monotonic() < deadline:
                frame = capture.read(.1)
            assert isinstance(frame, Frame)
            assert frame.image[0, 0, 0] == int(frame.sample_id) % 255
            assert 0 <= time.monotonic() - frame.t < 3
        assert not capture.process.is_alive()


def test_camera_open_error_releases_handle(monkeypatch):
    import cv2
    from threading import Event

    class Camera:
        released = False
        def isOpened(self):
            return False
        def release(self):
            self.released = True
    class Output(queue.Queue):
        def cancel_join_thread(self):
            pass
    class Errors:
        messages = []
        def send(self, message):
            self.messages.append(message)
        def close(self):
            pass
    camera, errors = Camera(), Errors()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    _camera_worker(dict(kind="camera", source=0, session_id="test"), Output(1), Event(), errors)
    assert errors.messages == ["camera_open_failed"] and camera.released
