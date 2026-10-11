import math


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