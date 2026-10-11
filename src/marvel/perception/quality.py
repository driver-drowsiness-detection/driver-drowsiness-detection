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