"""
Unit tests for capture.py - Person 1
Run with: pytest tests/test_capture.py

Uses unittest.mock to fake a camera, so this runs on any machine with
no physical webcam needed - useful once this also needs to pass on a
CI pipeline that has no camera attached.
"""

import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "capture"))
from capture import CameraCapture


def test_read_frame_resizes_to_requested_size():
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # a fake 720p frame

    with patch("capture.cv2.VideoCapture") as mock_video_capture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_video_capture.return_value = mock_cap

        cam = CameraCapture(source=0, width=640, height=480)
        frame = cam.read_frame()

        assert frame is not None
        assert frame.shape[1] == 640, "frame width should match requested width"
        assert frame.shape[0] == 480, "frame height should match requested height"


def test_read_frame_returns_none_on_camera_failure():
    with patch("capture.cv2.VideoCapture") as mock_video_capture:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        cam = CameraCapture(source=0)
        frame = cam.read_frame()

        assert frame is None


if __name__ == "__main__":
    test_read_frame_resizes_to_requested_size()
    test_read_frame_returns_none_on_camera_failure()
    print("All tests passed.")
