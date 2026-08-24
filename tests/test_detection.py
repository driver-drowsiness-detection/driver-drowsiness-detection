"""
Unit test for detection.py - Person 2
Run with: pytest tests/test_detection.py

Requires mediapipe to be installed (pip install mediapipe) - same as
detection.py itself. Doesn't need a live camera; uses a synthetic
blank frame instead, so it runs anywhere.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "detection"))
from detection import FaceLandmarkDetector


def test_detect_returns_none_on_blank_frame():
    detector = FaceLandmarkDetector()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # solid black, no face
    result = detector.detect(blank_frame)
    assert result is None, "A blank frame should never produce a detected face"


if __name__ == "__main__":
    test_detect_returns_none_on_blank_frame()
    print("All tests passed.")
