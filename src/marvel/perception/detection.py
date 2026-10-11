"""detection.py — Locate the driver, extract named points.

Owns exactly one job: image -> FaceLandmarks (named pixel points), or None if
no driver face is found. It knows nothing about EAR, crops, or validity gates —
that's quality.py and crops.py. This keeps the landmark model swappable: any
other provider just needs to implement detect()/close() with the same point
names (left_eye_p1..p6, right_eye_p1..p6, nose_tip, face_edge_img_left/right).
"""
from __future__ import annotations

import time
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

from .contracts import EyeSide, FaceLandmarks

# MediaPipe Face Mesh indices in EAR order P1..P6:
# P1/P4 = eye corners, P2/P3 = upper lid, P5/P6 = lower lid, P2-P6 and P3-P5 are vertical pairs.
EYE_LANDMARK_IDS = {
    EyeSide.LEFT:  (362, 385, 387, 263, 373, 380),
    EyeSide.RIGHT: (33, 160, 158, 133, 153, 144),
}
FACE_LANDMARK_IDS = {"nose_tip": 1, "face_edge_img_left": 234, "face_edge_img_right": 454}


class LandmarkProvider(Protocol):
    def detect(self, image_bgr: np.ndarray) -> Optional[FaceLandmarks]: ...
    def close(self) -> None: ...


def name_points(pts468: np.ndarray) -> dict:
    named = {}
    for side, ids in EYE_LANDMARK_IDS.items():
        for k, idx in enumerate(ids, start=1):
            named[f"{side.value}_eye_p{k}"] = pts468[idx].astype(np.float32)
    for name, idx in FACE_LANDMARK_IDS.items():
        named[name] = pts468[idx].astype(np.float32)
    return named


class MediaPipeFaceMeshProvider:
    """Default provider. Swap for another landmarker (e.g. a Jetson-friendly one)
    by implementing detect() -> FaceLandmarks with the same point names.

    Newer MediaPipe releases only ship the Tasks API, which needs a model file:
      face_landmarker.task  (download from the MediaPipe Face Landmarker docs page)
    Older releases still have mp.solutions.face_mesh; used only if model_path is None.
    Both give the same 468 mesh indices used in EYE_LANDMARK_IDS.
    """

    def __init__(self, model_path: Optional[str] = None,
                 driver_roi: Optional[Tuple[float, float, float, float]] = None,
                 max_faces: int = 2, det_conf: float = 0.5, track_conf: float = 0.5):
        import mediapipe as mp   # imported lazily so tests run without it
        self._mp = mp
        self._roi = driver_roi   # normalised (x0,y0,x1,y1); the driver's seat area
        self._last_ms = 0
        self._tasks = model_path is not None
        if self._tasks:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=max_faces,
                min_face_detection_confidence=det_conf,
                min_tracking_confidence=track_conf)
            self._lm = vision.FaceLandmarker.create_from_options(opts)
        elif hasattr(mp, "solutions"):
            self._lm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=max_faces, refine_landmarks=False,
                min_detection_confidence=det_conf, min_tracking_confidence=track_conf)
        else:
            raise RuntimeError("This MediaPipe has no mp.solutions; pass model_path=<face_landmarker.task>")

    def _faces_normalised(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self._tasks:
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            self._last_ms = max(self._last_ms + 1, int(time.monotonic() * 1000))  # VIDEO mode needs strictly increasing ms
            res = self._lm.detect_for_video(mp_img, self._last_ms)
            return res.face_landmarks or []
        res = self._lm.process(rgb)
        return [f.landmark for f in (res.multi_face_landmarks or [])]

    def detect(self, image_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        h, w = image_bgr.shape[:2]
        best, best_area = None, -1.0
        for lm in self._faces_normalised(image_bgr):
            pts = np.array([[p.x, p.y] for p in lm[:468]], dtype=np.float32) * np.array([w, h], np.float32)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            if self._roi is not None:
                cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                if not (self._roi[0] <= cx <= self._roi[2] and self._roi[1] <= cy <= self._roi[3]):
                    continue
            area = float((x1 - x0) * (y1 - y0))
            if area > best_area:                 # driver = largest face inside the ROI
                best, best_area = (pts, (float(x0), float(y0), float(x1), float(y1))), area
        if best is None:
            return None
        return FaceLandmarks(points=name_points(best[0]), bbox=best[1])

    def close(self) -> None:
        self._lm.close()
