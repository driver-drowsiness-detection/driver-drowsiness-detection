"""Experimental weighted score fusion; settings must be supplied explicitly."""
import math

from marvel.contracts import Estimate, OPEN, CLOSED, UNKNOWN


def _unit(value, name):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")


def validate_score(estimate, method):
    if estimate.method != method:
        raise ValueError(f"Expected {method} estimate")
    if estimate.valid:
        _unit(estimate.value, "Score")


def combine(ear, cnn, *, alpha):
    """alpha weights CNN. Both branches remain required even at endpoints."""
    _unit(alpha, "Alpha")
    validate_score(ear, "ear_probability")
    validate_score(cnn, "cnn")
    identity = (ear.sample_id, ear.t, ear.track, ear.side)
    if identity != (cnn.sample_id, cnn.t, cnn.track, cnn.side):
        raise ValueError("Cannot fuse different samples, times, tracks or eyes")
    if any(not isinstance(x, str) or not x.strip() for x in (ear.sample_id, ear.track)):
        raise ValueError("Fusion requires explicit sample and track identity")
    reason = "" if ear.valid and cnn.valid else "missing_branch"
    value = alpha * cnn.value + (1 - alpha) * ear.value if not reason else None
    return Estimate(*identity, "fusion", value, value is not None, reason)


def score_state(estimate, threshold):
    """Equality is CLOSED for a closure score; raw EAR uses its own rule."""
    _unit(threshold, "Threshold")
    if estimate.method not in ("cnn", "ear_probability", "fusion"):
        raise ValueError("Expected a closure score, not raw EAR")
    validate_score(estimate, estimate.method)
    if not estimate.valid:
        return UNKNOWN
    return CLOSED if estimate.value >= threshold else OPEN
