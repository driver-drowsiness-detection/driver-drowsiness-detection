import math


def aspect_ratio(points, min_width=2.0):
    """Six ordered points: corners 0/3, vertical pairs 1/5 and 2/4."""
    if len(points) != 6 or any(len(p) != 2 for p in points):
        return None
    if not all(math.isfinite(v) for p in points for v in p):
        return None
    width = math.dist(points[0], points[3])
    if width < min_width:
        return None
    return (math.dist(points[1], points[5]) +
            math.dist(points[2], points[4])) / (2.0 * width)


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
