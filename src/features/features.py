"""
Module: features
Owner: Person 3

PURPOSE
    Converts raw face geometry (landmarks) into the physiological signal
    we actually care about: how open are the eyes (EAR). MAR (mouth,
    for yawning) is added in Week 2.

TERMINOLOGY (for your notes)
    - EAR (Eye Aspect Ratio): (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
      Vertical eyelid distance over horizontal eye width. Falls toward 0
      as the eye closes. Source: Soukupova & Cech, 2016.
    - MAR (Mouth Aspect Ratio): same idea applied to mouth landmarks,
      used to detect yawning. (Week 2.)
    - Euclidean distance: straight-line distance between two points -
      the building block of both formulas above.

WHY eye_aspect_ratio() TAKES PLAIN POINTS, NOT MEDIAPIPE LANDMARKS
    This function has ZERO dependency on MediaPipe. It just takes 6
    (x, y) points and does math. That means you can unit-test it RIGHT
    NOW with hardcoded numbers, without waiting for Person 2's detector
    or a live camera. This is "separation of concerns" - it's exactly
    why 4 people can build this in parallel without blocking each other.

INTERFACE CONTRACT
    eye_aspect_ratio(points: list[(x, y)]) -> float
        points = [p1, p2, p3, p4, p5, p6] in that exact order.
    compute_features(landmarks, frame_w, frame_h) -> dict
        landmarks = MediaPipe's raw output from detection.py's .detect()
        Returns {"ear": float, "mar": None}   # mar filled in Week 2
"""

import math

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(points):
    """points = [p1, p2, p3, p4, p5, p6] as (x, y) tuples, pixels or normalized."""
    p1, p2, p3, p4, p5, p6 = points
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    return vertical / (2.0 * horizontal)


def _landmarks_to_points(landmarks, indices, frame_w, frame_h):
    return [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in indices]


def compute_features(landmarks, frame_w, frame_h):
    right_pts = _landmarks_to_points(landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
    left_pts = _landmarks_to_points(landmarks, LEFT_EYE_IDX, frame_w, frame_h)
    right_ear = eye_aspect_ratio(right_pts)
    left_ear = eye_aspect_ratio(left_pts)
    avg_ear = (right_ear + left_ear) / 2.0
    return {"ear": avg_ear, "mar": None}  # TODO Week 2 (Person 3): add MAR
