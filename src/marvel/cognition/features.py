"""Pure geometry; no camera, OpenCV or model dependency."""
import math


def aspect_ratio(points, min_width=2.0):
    """Six pixel points: corners 0/3, vertical pairs 1/5 and 2/4.

    Return None for missing/unusable evidence. A closed eye with separated
    corners legitimately has ratio 0. Uniform scale and rotation preserve EAR.
    """
    if not math.isfinite(min_width) or min_width <= 0:
        raise ValueError("min_width must be finite and positive")
    try:
        if len(points) != 6 or any(len(p) != 2 for p in points):
            return None
        if not all(math.isfinite(v) for p in points for v in p):
            return None
        width = math.dist(points[0], points[3])
        if width < min_width or not math.isfinite(width):
            return None
        value = (math.dist(points[1], points[5]) / width
                 + math.dist(points[2], points[4]) / width) / 2.0
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def sigmoid(value):
    """Stable finite-logit conversion; raw EAR itself is not a probability."""
    if not math.isfinite(value):
        raise ValueError("Logit must be finite")
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)
