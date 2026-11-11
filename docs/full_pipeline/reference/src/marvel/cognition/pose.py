def estimate_pose(points, image_shape, calibration):
    """Optional PnP measurement. Requires real camera calibration + declared 3D template."""
    import cv2
    import numpy as np
    if calibration is None:
        return {"valid": False, "reason": "camera_calibration_undecided"}
    h, w = image_shape[:2]
    if calibration["image_size"] != [w, h]:
        return {"valid": False, "reason": "intrinsic_resolution_mismatch"}
    observed = np.asarray(points, dtype=np.float64)
    objects = np.asarray(calibration["object_points"], dtype=np.float64)
    matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion"], dtype=np.float64)
    if observed.shape != (6, 2) or objects.shape != (6, 3) or not np.isfinite(observed).all():
        return {"valid": False, "reason": "pose_points"}
    ok, rvec, tvec = cv2.solvePnP(objects, observed, matrix, distortion,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        return {"valid": False, "reason": "pnp_failed"}
    projected, _ = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)
    error = float(np.linalg.norm(projected.reshape(-1, 2)-observed, axis=1).mean())
    if error > calibration["max_reprojection_px"]:
        return {"valid": False, "reason": "reprojection", "error_px": error}
    rotation, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rotation)[0]
    return {"valid": True, "pitch_deg": float(angles[0]), "yaw_deg": float(angles[1]),
            "roll_deg": float(angles[2]), "error_px": error}
