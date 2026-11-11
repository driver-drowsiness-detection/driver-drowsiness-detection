import math


def eye_crop(image, points, width_factor=1.6, height_factor=0.8):
    """Height follows eye WIDTH, so a closed eyelid does not create a zero-height crop."""
    if len(points) != 6:
        return None
    w = math.dist(points[0], points[3])
    if not math.isfinite(w) or w < 2:
        return None
    cx = (points[0][0] + points[3][0]) / 2
    cy = (points[0][1] + points[3][1]) / 2
    x1, x2 = math.floor(cx - width_factor*w/2), math.ceil(cx + width_factor*w/2)
    y1, y2 = math.floor(cy - height_factor*w/2), math.ceil(cy + height_factor*w/2)
    h, iw = image.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > iw or y2 > h or x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()
