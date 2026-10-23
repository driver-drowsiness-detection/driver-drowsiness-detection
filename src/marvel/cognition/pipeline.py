"""Align current eye evidence, measure EAR/MAR, then update temporal cognition."""
from dataclasses import asdict
import math

from marvel.contracts import UNKNOWN, bilateral
from .ear_classifier import measure, state
from .features import aspect_ratio
from .perclos import ClosureWindow
from .trigger import Trigger
from .pose import HeadPose


class EARCognition:
    def __init__(self, *, ear_threshold, run_id, close_s, recover_s, max_gap_s,
                 unavailable_s, window_s, min_coverage, pose_calibration=None):
        if not math.isfinite(ear_threshold) or ear_threshold <= 0:
            raise ValueError("EAR threshold must be finite and positive")
        self.ear_threshold = ear_threshold
        self.trigger = Trigger(close_s, recover_s, max_gap_s, unavailable_s,
                               mode="ear", run_id=run_id)
        self.window = ClosureWindow(window_s, max_gap_s, min_coverage)
        self.pose = HeadPose(pose_calibration)

    def update(self, sample_id, t, eyes, face_status, metrics=None, image_shape=None):
        estimates, observation, track = [], UNKNOWN, None
        reason = face_status
        # Never combine different samples, times, tracks or duplicate sides.
        if face_status in ("valid", "invalid_eye") and len(eyes) == 2:
            aligned = ({e.side for e in eyes} == {"left", "right"}
                       and all(e.sample_id == sample_id and e.t == t for e in eyes)
                       and eyes[0].track == eyes[1].track
                       and isinstance(eyes[0].track, str) and bool(eyes[0].track.strip()))
            if aligned:
                track = eyes[0].track
                estimates = [measure(e) for e in eyes]
                observation = bilateral(*(state(e, self.ear_threshold) for e in estimates))
                reason = "" if observation != UNKNOWN else "invalid_or_disagreeing_eyes"
            else:
                reason = "unaligned_eye_evidence"
        elif face_status in ("valid", "invalid_eye"):
            reason = "missing_eye_evidence"
        geometry = (metrics or {}).get("geometry", {})
        mar = None
        geometry_aligned = (track is not None and geometry.get("sample_id") == sample_id
                and geometry.get("t") == t and geometry.get("track") == track
                )
        if geometry_aligned and geometry.get("mouth_ok"):
            mar = aspect_ratio(geometry.get("mouth_points"))
        pose_points = geometry.get("pose_points") if geometry_aligned and geometry.get("pose_ok") else None
        pose = self.pose.estimate(pose_points, image_shape)
        decision = self.trigger.update(t, observation, track, sample_id=sample_id)
        rolling = self.window.update(t, observation, track)
        return {"sample_id": sample_id, "t": t, "face_status": face_status,
                "reason": reason, "eye_estimates": [asdict(e) for e in estimates],
                "mar": mar, "mar_is_yawn_label": False, "pose": pose,
                "decision": asdict(decision), "rolling": rolling}
