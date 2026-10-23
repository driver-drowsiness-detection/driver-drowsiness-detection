"""Calibrated six-point PnP measurement; never a fatigue decision."""
import math
import numpy as np

from marvel.perception.detection import POSE


class HeadPose:
    def __init__(self, calibration=None):
        self.calibration = calibration
        if calibration is None:
            return
        try:
            self.size = calibration["image_size"]
            self.objects = np.asarray(calibration["object_points"], dtype=np.float64)
            self.matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
            self.distortion = np.asarray(calibration["distortion"], dtype=np.float64)
            self.limit = float(calibration["max_reprojection_px"])
            if (len(self.size) != 2 or any(type(x) is not int or x <= 0 for x in self.size)
                    or calibration["landmark_indices"] != list(POSE)
                    or not isinstance(calibration["template_axes"], str)
                    or not calibration["template_axes"].strip()
                    or self.objects.shape != (6, 3) or self.matrix.shape != (3, 3)
                    or self.distortion.ndim != 1 or self.distortion.size not in (4, 5, 8, 12, 14)
                    or not all(np.isfinite(x).all() for x in (self.objects, self.matrix, self.distortion))
                    or np.linalg.matrix_rank(self.objects - self.objects.mean(axis=0)) != 3
                    or self.matrix[0, 0] <= 0 or self.matrix[1, 1] <= 0
                    or not np.allclose(self.matrix[2], [0, 0, 1])
                    or self.matrix[0, 1] != 0 or self.matrix[1, 0] != 0
                    or not math.isfinite(self.limit) or self.limit <= 0):
                raise ValueError("Invalid camera or face-template parameters")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid pose calibration: " + str(error)) from error

    def estimate(self, points, image_shape):
        if self.calibration is None:
            return {"valid": False, "reason": "camera_calibration_missing"}
        if image_shape is None or len(image_shape) < 2 or list(image_shape[:2][::-1]) != self.size:
            return {"valid": False, "reason": "intrinsic_resolution_mismatch"}
        try:
            observed = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError):
            return {"valid": False, "reason": "pose_points"}
        if (observed.shape != (6, 2) or not np.isfinite(observed).all()
                or np.linalg.matrix_rank(observed - observed.mean(axis=0)) < 2
                or (observed < 0).any() or (observed >= self.size).any()):
            return {"valid": False, "reason": "pose_points"}
        import cv2
        try:
            ok, rvec, tvec = cv2.solvePnP(self.objects, observed, self.matrix, self.distortion,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                return {"valid": False, "reason": "pnp_failed"}
            rotation, _ = cv2.Rodrigues(rvec)
            camera_points = self.objects @ rotation.T + tvec.reshape(1, 3)
            if (camera_points[:, 2] <= 0).any():
                return {"valid": False, "reason": "behind_camera"}
            projected, _ = cv2.projectPoints(self.objects, rvec, tvec, self.matrix, self.distortion)
            error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - observed)**2, axis=1))))
            if not math.isfinite(error) or error > self.limit:
                return {"valid": False, "reason": "reprojection"}
            angles = cv2.RQDecomp3x3(rotation)[0]
            if not all(math.isfinite(a) for a in angles):
                return {"valid": False, "reason": "nonfinite_angles"}
            return {"valid": True, "reason": "", "pitch_deg": float(angles[0]),
                    "yaw_deg": float(angles[1]), "roll_deg": float(angles[2]),
                    "reprojection_rmse_px": error, "template_axes": self.calibration["template_axes"],
                    "is_fatigue_label": False}
        except cv2.error:
            return {"valid": False, "reason": "pnp_failed"}
