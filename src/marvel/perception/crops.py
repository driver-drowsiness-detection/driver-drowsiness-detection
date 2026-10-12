import math


def eye_box(points):
    """Rectangle from corner width, so a closed eyelid retains crop area."""
    try:
        if len(points) != 6 or not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
            return None
        eye_width = math.hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])
        if eye_width < 2:
            return None
        cx = (points[0][0] + points[3][0]) / 2
        cy = (points[0][1] + points[3][1]) / 2
        return (math.floor(cx - 0.8 * eye_width), math.floor(cy - 0.4 * eye_width),
                math.ceil(cx + 0.8 * eye_width), math.ceil(cy + 0.4 * eye_width))
    except (TypeError, ValueError):
        return None


def eye_crop(image, points):
    box = eye_box(points)
    return None if box is None else crop_box(image, box)[0]


def crop_box(image, box):
    if image is None or image.size == 0:
        return None, "empty_frame"
    if image.ndim != 3 or image.shape[2] != 3:
        return None, "unexpected_image_shape"
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None, "invalid_box"
    left, top, right, bottom = box
    height, width = image.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None, "box_outside_or_empty"
    x0, y0 = math.floor(left), math.floor(top)
    x1, y1 = math.ceil(right), math.ceil(bottom)
    crop = image[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None, "empty_crop"
    return crop, None
