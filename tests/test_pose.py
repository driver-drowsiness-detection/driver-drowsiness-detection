from copy import deepcopy
import cv2
import numpy as np
import pytest

from marvel.cognition.pose import HeadPose
from marvel.perception.detection import POSE


def calibration():
    # Synthetic test geometry and intrinsics, never camera calibration for a demo.
    return dict(image_size=[640, 480], landmark_indices=list(POSE),
                template_axes="Synthetic x right, y down, z forward",
                object_points=[[0, 0, 0], [0, 50, -10], [-30, -25, -15],
                               [30, -25, -15], [-20, 20, -10], [20, 20, -10]],
                camera_matrix=[[600, 0, 320], [0, 600, 240], [0, 0, 1]],
                distortion=[0, 0, 0, 0, 0], max_reprojection_px=2)


def projected(c):
    rotation = np.array([.1, -.15, .05])
    points, _ = cv2.projectPoints(np.array(c["object_points"], float), rotation,
                                  np.array([0., 0., 600.]), np.array(c["camera_matrix"], float),
                                  np.array(c["distortion"], float))
    return points.reshape(-1, 2), rotation


def test_known_projection_recovers_rotation():
    c = calibration()
    points, rotation = projected(c)
    result = HeadPose(c).estimate(points, (480, 640, 3))
    expected = cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0]
    assert result["valid"]
    assert [result[k] for k in ("pitch_deg", "yaw_deg", "roll_deg")] == pytest.approx(expected, abs=1e-5)
    assert result["reprojection_rmse_px"] < 1e-6
    assert not result["is_fatigue_label"]


def test_missing_calibration_resolution_mismatch_and_bad_points():
    assert HeadPose().estimate(None, None)["reason"] == "camera_calibration_missing"
    pose = HeadPose(calibration())
    assert pose.estimate(None, (240, 320, 3))["reason"] == "intrinsic_resolution_mismatch"
    for points in (None, [], [[1, 1]] * 6, [[float("nan"), 0]] * 6):
        assert pose.estimate(points, (480, 640, 3))["reason"] == "pose_points"


@pytest.mark.parametrize("field,value", [
    ("image_size", [0, 480]), ("landmark_indices", [1, 2, 3, 4, 5, 6]),
    ("camera_matrix", [[0, 0, 320], [0, 600, 240], [0, 0, 1]]),
    ("object_points", [[0, 0, 0]] * 6), ("distortion", [float("nan")] * 5),
    ("template_axes", ""), ("max_reprojection_px", float("nan")),
])
def test_invalid_calibration_fails_before_processing(field, value):
    c = deepcopy(calibration())
    c[field] = value
    with pytest.raises(ValueError):
        HeadPose(c)


def test_inconsistent_correspondences_rejected_by_residual():
    c = calibration()
    points, _ = projected(c)
    points[0] += [30, 30]
    assert not HeadPose(c).estimate(points, (480, 640, 3))["valid"]
