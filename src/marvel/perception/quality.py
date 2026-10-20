"""Independent geometry and crop validity with reasons.
Missing observations must remain unknown."""
from dataclasses import dataclass
import math


def frame_problem(frame, previous=None):
    """Adapt perception01's packet checks to the canonical Frame contract."""
    image = frame.image
    if image is None or image.size == 0:
        return "empty_frame"
    if image.ndim != 3 or image.shape[2] != 3 or str(image.dtype) != "uint8":
        return "unexpected_image_format"
    if not frame.sample_id or not math.isfinite(frame.t):
        return "invalid_frame_identity_or_time"
    if previous is not None:
        if frame.sample_id == previous.sample_id:
            return "repeated_frame"
        if frame.t <= previous.t:
            return "non_increasing_time"
    return None


def points_ok(points, width, height):
    try:
        return len(points) == 6 and all(
            math.isfinite(x) and math.isfinite(y)
            and 0 <= x < width and 0 <= y < height for x, y in points
        )
    except (TypeError, ValueError):
        return False


def validate_quality(config):
    required = ("min_crop_pixels", "brightness_min", "brightness_max", "laplacian_min")
    if not isinstance(config, dict) or any(k not in config for k in required):
        raise ValueError("Missing quality settings")
    if not all(isinstance(config[k], (int, float)) and not isinstance(config[k], bool)
               and math.isfinite(config[k]) for k in required):
        raise ValueError("Quality settings must be finite numbers")
    if config["min_crop_pixels"] < 1 or int(config["min_crop_pixels"]) != config["min_crop_pixels"]:
        raise ValueError("min_crop_pixels must be a positive integer")
    if not 0 <= config["brightness_min"] < config["brightness_max"] <= 255:
        raise ValueError("Brightness limits must satisfy 0 <= min < max <= 255")
    if config["laplacian_min"] < 0:
        raise ValueError("laplacian_min must be nonnegative")


def image_quality(crop, config):
    if crop is None or crop.size == 0:
        return False, ("crop_missing_or_outside",), {}
    if crop.ndim != 3 or crop.shape[2] != 3 or str(crop.dtype) != "uint8":
        return False, ("unexpected_image_format",), {}
    if min(crop.shape[:2]) < config["min_crop_pixels"]:
        return False, ("crop_too_small",), {}
    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if not config["brightness_min"] <= brightness <= config["brightness_max"]:
        reasons.append("exposure")
    if sharpness < config["laplacian_min"]:
        reasons.append("blur")
    return not reasons, tuple(reasons), {"brightness": brightness, "sharpness": sharpness}


@dataclass(frozen=True)
class Validity:
    valid: bool
    reasons: tuple = ()


def crop_validity(crop, crop_reason, min_side_px):
    if crop is None:
        return Validity(False, (crop_reason or "no_crop",))
    height, width = crop.shape[:2]
    if min(height, width) < min_side_px:
        return Validity(False, ("crop_too_small",))
    return Validity(True)

@dataclass(frozen=True)
class EyeObservation:
    geometry: Validity
    crop: Validity


def combine_eye_validity(face_found, geometry, crop):
    if not face_found:
        bad = Validity(False, ("no_face",))
        return EyeObservation(bad, bad)
    return EyeObservation(geometry, crop)
