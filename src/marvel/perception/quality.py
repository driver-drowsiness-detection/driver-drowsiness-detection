"""Independent geometry and crop validity with reasons.
Missing observations must remain unknown."""
from dataclasses import dataclass


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