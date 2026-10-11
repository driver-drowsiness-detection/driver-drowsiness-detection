 
"""capture.py — Input adapter: Windows webcam / recorded video / verified Jetson CSI.

Job of this module (and nothing more):
  * open a source and VERIFY it really delivers frames
  * hand out FramePackets with a strictly increasing frame_id and a timestamp
  * report failures as packets (READ_FAILED) instead of hiding them
It does NOT mirror, resize, normalise, or fix colour. Those belong in detection/
quality/crops.
"""
from __future__ import annotations

import logging
import math
import re
import sys
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import cv2

from .contracts import FramePacket, FrameStatus

log = logging.getLogger(__name__)


class SourceOpenError(RuntimeError):
    pass


class FrameSource(ABC):
    def __init__(self, source_id: str):
        self.source_id = source_id
        self._next_id = 0

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> FramePacket: ...

    @abstractmethod
    def close(self) -> None: ...

    def _new_id(self) -> int:
        fid = self._next_id
        self._next_id += 1        # ids advance even on failed reads, so gaps are visible
        return fid

    def frames(self) -> Iterator[FramePacket]:
        """Yield packets until END_OF_STREAM (files). Live sources: caller breaks."""
        while True:
            p = self.read()
            yield p
            if p.status is FrameStatus.END_OF_STREAM:
                return

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- live
class _CvLiveSource(FrameSource):
    def __init__(self, source_id: str, warmup_frames: int = 5, reopen_after_failures: int = 30):
        super().__init__(source_id)
        self._cap: Optional[cv2.VideoCapture] = None
        self._warmup = warmup_frames
        self._reopen_after = reopen_after_failures
        self._fail_run = 0
        self.width = self.height = 0

    @abstractmethod
    def _open_capture(self) -> cv2.VideoCapture: ...

    def open(self) -> None:
        cap = self._open_capture()
        if not cap.isOpened():
            cap.release()
            raise SourceOpenError(f"{self.source_id}: could not open capture")
        self._cap = cap
        self._verify()

    def _verify(self) -> None:
        """Prove frames arrive and look like BGR images before the system starts."""
        last = None
        good = 0
        for _ in range(self._warmup):
            ok, img = self._cap.read()
            if ok and img is not None and img.ndim == 3 and img.shape[2] == 3:
                good += 1
                last = img
        if good == 0:
            self.close()
            raise SourceOpenError(f"{self.source_id}: opened but delivered no valid frames")
        if float(last.mean()) < 1.0:
            log.warning("%s: frames are almost black (lens cap / wrong sensor / exposure?)", self.source_id)
        self.height, self.width = last.shape[:2]
        log.info("%s verified: %dx%d, %d/%d warmup frames ok",
                 self.source_id, self.width, self.height, good, self._warmup)

    def read(self) -> FramePacket:
        if self._cap is None:
            raise RuntimeError("source not open")
        ok, img = self._cap.read()
        t = time.monotonic()                     # stamp right after the read returns
        fid = self._new_id()
        if ok and img is not None:
            self._fail_run = 0
            return FramePacket(fid, t, self.source_id, FrameStatus.OK, img)
        self._fail_run += 1
        if self._fail_run % self._reopen_after == 0:
            self._try_reopen()
        time.sleep(0.005)                        # avoid a hot spin while the device is down
        return FramePacket(fid, t, self.source_id, FrameStatus.READ_FAILED, None)

    def _try_reopen(self) -> None:
        log.warning("%s: %d consecutive read failures, reopening", self.source_id, self._fail_run)
        try:
            if self._cap is not None:
                self._cap.release()
            self._cap = self._open_capture()
        except Exception:
            log.exception("%s: reopen failed", self.source_id)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class WebcamSource(_CvLiveSource):
    """Windows webcam via DirectShow (falls back to default backend elsewhere)."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, use_mjpg: bool = True, **kw):
        super().__init__(f"webcam:{index}", **kw)
        self.index, self.req_w, self.req_h, self.req_fps, self.use_mjpg = index, width, height, fps, use_mjpg

    def _open_capture(self) -> cv2.VideoCapture:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if self.use_mjpg:   # many USB cams only reach 30 fps at 720p in MJPG
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
        cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # some backends ignore this
        return cap


def gstreamer_csi_pipeline(sensor_id: int = 0, capture_w: int = 1280, capture_h: int = 720,
                           fps: int = 30, flip_method: int = 0,
                           out_w: Optional[int] = None, out_h: Optional[int] = None) -> str:
    out_w, out_h = out_w or capture_w, out_h or capture_h
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_w}, height={capture_h}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={out_w}, height={out_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


class JetsonCsiSource(_CvLiveSource):
    """CSI camera through GStreamer. Needs OpenCV built WITH GStreamer (JetPack's is)."""

    def __init__(self, sensor_id: int = 0, pipeline: Optional[str] = None, **pipeline_kw):
        super().__init__(f"csi:{sensor_id}")
        self.pipeline = pipeline or gstreamer_csi_pipeline(sensor_id=sensor_id, **pipeline_kw)

    def _open_capture(self) -> cv2.VideoCapture:
        if not re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
            raise SourceOpenError(
                "This OpenCV build has no GStreamer support (pip's opencv-python never does). "
                "Use JetPack's system OpenCV on the Jetson.")
        log.info("CSI pipeline: %s", self.pipeline)
        return cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)


# --------------------------------------------------------------------------- file
class VideoFileSource(FrameSource):
    """Recorded video. Timestamps are media time (frame_index / fps), so a run is
    reproducible and all three methods can be replayed on identical input."""

    def __init__(self, path: str, realtime: bool = False):
        super().__init__(f"file:{path}")
        self.path, self.realtime = path, realtime
        self._cap: Optional[cv2.VideoCapture] = None
        self.fps: Optional[float] = None
        self._wall0: Optional[float] = None
        self._last_t = 0.0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise SourceOpenError(f"cannot open video: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and math.isfinite(fps) and fps > 1e-3 else None
        if self.fps is None:
            log.warning("%s: no usable FPS metadata, falling back to POS_MSEC", self.source_id)
        self._cap = cap

    def read(self) -> FramePacket:
        if self._cap is None:
            raise RuntimeError("source not open")
        ok, img = self._cap.read()
        idx = self._new_id()
        if not ok or img is None:
            return FramePacket(idx, self._last_t, self.source_id, FrameStatus.END_OF_STREAM, None)
        t = idx / self.fps if self.fps else self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self._last_t = t
        if self.realtime:
            if self._wall0 is None:
                self._wall0 = time.monotonic()
            delay = (self._wall0 + t) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return FramePacket(idx, t, self.source_id, FrameStatus.OK, img)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def make_source(kind: str, **kw) -> FrameSource:
    kind = kind.lower()
    if kind == "webcam":
        return WebcamSource(**kw)
    if kind == "video":
        return VideoFileSource(**kw)
    if kind == "csi":
        return JetsonCsiSource(**kw)
    raise ValueError(f"unknown source kind: {kind}")
