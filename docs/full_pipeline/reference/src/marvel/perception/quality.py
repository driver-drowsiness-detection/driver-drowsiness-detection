import math


def points_ok(points, width, height):
    return len(points) == 6 and all(
        math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height
        for x, y in points)


def image_quality(crop, config):
    import cv2
    if crop is None or crop.size == 0:
        return False, ("crop_bounds",)
    if min(crop.shape[:2]) < config["min_crop_pixels"]:
        return False, ("crop_small",)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if not config["brightness_min"] <= mean <= config["brightness_max"]:
        reasons.append("exposure")
    if blur < config["laplacian_min"]:
        reasons.append("blur")
    return not reasons, tuple(reasons)
