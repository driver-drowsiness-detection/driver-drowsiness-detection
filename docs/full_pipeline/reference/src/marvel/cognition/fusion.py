from marvel.contracts import Estimate, OPEN, CLOSED, UNKNOWN


def combine(ear, cnn, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if (ear.sample_id, ear.t, ear.track, ear.side) != (
            cnn.sample_id, cnn.t, cnn.track, cnn.side):
        raise ValueError("Cannot fuse different observations")
    if ear.method != "ear_probability" or cnn.method != "cnn":
        raise ValueError("Fuse calibrated scores, not raw EAR")
    valid = ear.valid and cnn.valid
    for x in (ear, cnn):
        if x.valid and not 0 <= x.value <= 1:
            raise ValueError("Closure score outside [0, 1]")
    value = alpha * cnn.value + (1 - alpha) * ear.value if valid else None
    return Estimate(ear.sample_id, ear.t, ear.track, ear.side, "fusion", value,
                    valid, "" if valid else "missing_branch")


def state(estimate, threshold):
    if not estimate.valid:
        return UNKNOWN
    return CLOSED if estimate.value >= threshold else OPEN
