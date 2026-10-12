"""Freshness boundary shared by preview and later live consumers."""
import math
import time

from .quality import frame_problem


def process_frame(frame, detector, max_age, clock=time.monotonic):
    """Return current evidence only: (eyes, status, metrics, read_age)."""
    problem = frame_problem(frame)
    age = clock() - frame.t
    if problem or not math.isfinite(age) or age < 0 or age > max_age:
        if detector is not None:
            detector.invalidate()
        return [], problem or "stale_frame", {}, age
    eyes, status, metrics = ([], "camera_only", {}) if detector is None else detector.process(frame)
    age = clock() - frame.t
    if not math.isfinite(age) or age < 0 or age > max_age:
        if detector is not None:
            detector.invalidate()
        return [], "stale_frame", {}, age
    return eyes, status, metrics, age
