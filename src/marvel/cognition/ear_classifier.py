"""Independent EAR baseline. Owner: Vishruth; reviewer: Arushi."""
import math

from marvel.contracts import Estimate, OPEN, CLOSED, UNKNOWN
from .features import aspect_ratio, sigmoid


def measure(eye):
    """Retain the perception identity; never measure rejected geometry."""
    value = aspect_ratio(eye.points) if eye.geometry_ok else None
    reason = "" if value is not None else (
        ",".join(eye.reasons) or "geometry_invalid")
    return Estimate(eye.sample_id, eye.t, eye.track, eye.side, "ear",
                    value, value is not None, reason)


def state(estimate, threshold):
    """Explicit raw-EAR threshold. Equality is OPEN; no default fitted cutoff."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("EAR threshold must be finite and positive")
    if estimate.method != "ear":
        raise ValueError("Expected raw EAR, not a CNN or calibrated score")
    if not estimate.valid:
        return UNKNOWN
    return CLOSED if estimate.value < threshold else OPEN


def probability(estimate, calibration):
    """Apply externally fitted coefficients; this function does not fit them."""
    slope, intercept = calibration["slope"], calibration["intercept"]
    if not all(math.isfinite(x) for x in (slope, intercept)) or slope >= 0:
        raise ValueError("EAR mapping needs a finite negative slope and finite intercept")
    if estimate.method != "ear":
        raise ValueError("Expected raw EAR")
    value = sigmoid(slope * estimate.value + intercept) if estimate.valid else None
    return Estimate(estimate.sample_id, estimate.t, estimate.track, estimate.side,
                    "ear_probability", value, value is not None, estimate.reason)
