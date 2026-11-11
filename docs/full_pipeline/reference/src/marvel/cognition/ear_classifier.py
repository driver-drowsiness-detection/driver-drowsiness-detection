from marvel.contracts import Estimate, OPEN, CLOSED, UNKNOWN
from .features import aspect_ratio, sigmoid


def measure(eye):
    value = aspect_ratio(eye.points) if eye.geometry_ok else None
    return Estimate(eye.sample_id, eye.t, eye.track, eye.side, "ear",
                    value, value is not None, "" if value is not None else "geometry")


def state(estimate, threshold):
    if not estimate.valid:
        return UNKNOWN
    return CLOSED if estimate.value < threshold else OPEN


def probability(estimate, calibration):
    slope, intercept = calibration["slope"], calibration["intercept"]
    if slope >= 0:
        raise ValueError("EAR mapping must decrease as eye opening increases")
    value = sigmoid(slope * estimate.value + intercept) if estimate.valid else None
    return Estimate(estimate.sample_id, estimate.t, estimate.track, estimate.side,
                    "ear_probability", value, estimate.valid, estimate.reason)
