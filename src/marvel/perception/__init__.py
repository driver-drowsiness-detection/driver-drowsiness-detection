"""MARVEL package: perception. FramePacket -> PerceptionResult.

Runtime behaviour implemented (was: scaffold, pending).

    capture.py    input adapter (webcam / video / CSI)         [not part of this
                  package's job, but the natural caller of it]
    detection.py  locate the driver, extract named landmarks
    quality.py    geometry validation + EAR, PerceptionConfig
    crops.py      eye-crop extraction + crop validation
    types.py      shared dataclasses (FramePacket, PerceptionResult, ...)

The Perception class below is the only thing most callers need:

    from perception import Perception, PerceptionConfig
    from perception.detection import MediaPipeFaceMeshProvider

    p = Perception(MediaPipeFaceMeshProvider(model_path="face_landmarker.task"))
    result = p.process(frame_packet)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .crops import extract_eye_crop, validate_crop
from .detection import LandmarkProvider
from .quality import PerceptionConfig, face_pose_reasons, validate_eye_geometry
from .types import (EyeObservation, EyeSide, FaceLandmarks, FramePacket,
                    FrameStatus, PerceptionResult, Reason)

__all__ = [
    "Perception", "PerceptionConfig",
    "FramePacket", "FrameStatus", "EyeSide", "Reason",
    "FaceLandmarks", "EyeObservation", "PerceptionResult",
]

log = logging.getLogger(__name__)


class Perception:
    """Perception and quality: locate the driver; extract named points and eye
    crops; validate geometry and crops separately; preserve frame identity."""

    def __init__(self, provider: LandmarkProvider, config: Optional[PerceptionConfig] = None):
        self.provider = provider
        self.cfg = config or PerceptionConfig()

    def process(self, packet: FramePacket) -> PerceptionResult:
        t0 = time.perf_counter()
        if packet.image is None:
            return self._unknown(packet, Reason.FRAME_UNAVAILABLE, t0)
        try:
            face = self.provider.detect(packet.image)
        except Exception:
            log.exception("landmark provider failed on frame %d", packet.frame_id)
            return self._unknown(packet, Reason.LANDMARKER_ERROR, t0)
        if face is None:
            return self._unknown(packet, Reason.NO_FACE, t0)

        pose_reasons = face_pose_reasons(face.points, self.cfg)
        left = self._observe_eye(packet.image, EyeSide.LEFT, face.points, pose_reasons)
        right = self._observe_eye(packet.image, EyeSide.RIGHT, face.points, pose_reasons)
        return PerceptionResult(packet.frame_id, packet.t_capture, packet.source_id, True,
                                face.bbox, left, right, (time.perf_counter() - t0) * 1000.0)

    def _observe_eye(self, image: np.ndarray, side: EyeSide, face_pts, pose_reasons) -> EyeObservation:
        cfg = self.cfg
        h, w = image.shape[:2]
        pts = np.stack([face_pts[f"{side.value}_eye_p{i}"] for i in range(1, 7)]).astype(np.float32)

        geom_valid, geom_reasons, ear, degenerate, eye_w = validate_eye_geometry(
            pts, w, h, cfg, pose_reasons)

        crop_base_reasons = pose_reasons if cfg.pose_gates_crop else ()
        crop_img = quad = None
        crop_valid, crop_reasons, brightness, sharpness = False, (Reason.EYE_DEGENERATE,), None, 0.0
        if not degenerate:
            raw, mask, oof, quad = extract_eye_crop(image, pts, cfg)
            crop_valid, crop_reasons, crop_img, brightness, sharpness = validate_crop(
                raw, mask, oof, eye_w, cfg, crop_base_reasons)
        elif crop_base_reasons:
            crop_reasons = tuple(dict.fromkeys((*crop_base_reasons, Reason.EYE_DEGENERATE)))

        return EyeObservation(side, pts, ear, geom_valid, geom_reasons,
                              crop_img, quad, crop_valid, crop_reasons,
                              brightness, sharpness)

    def _unknown(self, packet: FramePacket, reason: Reason, t0: float) -> PerceptionResult:
        def eye(side):
            return EyeObservation(side, None, None, False, (reason,), None, None, False, (reason,))
        return PerceptionResult(packet.frame_id, packet.t_capture, packet.source_id, False, None,
                                eye(EyeSide.LEFT), eye(EyeSide.RIGHT),
                                (time.perf_counter() - t0) * 1000.0)

    def close(self) -> None:
        self.provider.close()