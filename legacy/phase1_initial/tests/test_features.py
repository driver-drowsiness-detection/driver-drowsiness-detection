"""
Unit tests for features.py - Person 3
Run with: pytest tests/test_features.py

These need NO camera, NO MediaPipe, NO other team member's code to be
finished. That's the whole point of the pure-function design in
features.py - you can start this on day 1.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "features"))
from features import eye_aspect_ratio


def test_open_eye_gives_high_ear():
    # A wide-open eye: corners 40px apart, lids 12px apart top and bottom
    points = [(200, 96), (210, 90), (230, 90), (240, 96), (230, 102), (210, 102)]
    ear = eye_aspect_ratio(points)
    assert ear > 0.25, f"Expected open-eye EAR > 0.25, got {ear:.3f}"


def test_closed_eye_gives_low_ear():
    # A closed eye: corners still 40px apart, lids only 2px apart
    points = [(200, 96), (210, 95), (230, 95), (240, 96), (230, 97), (210, 97)]
    ear = eye_aspect_ratio(points)
    assert ear < 0.10, f"Expected closed-eye EAR < 0.10, got {ear:.3f}"


if __name__ == "__main__":
    test_open_eye_gives_high_ear()
    test_closed_eye_gives_low_ear()
    print("All tests passed.")
