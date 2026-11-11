class FatigueState:
    """Optional experimental tiers. Configuration supplies unvalidated policy values."""
    def __init__(self, policy):
        self.policy = policy
        if policy is not None:
            if not 0 <= policy["low"] < policy["mild"] < policy["high"] <= 1:
                raise ValueError("Expected low < mild < high")
            if min(policy["hold_s"], policy["recover_s"], policy["max_gap_s"]) <= 0:
                raise ValueError("Tier durations must be positive")
        self.state = "UNKNOWN"
        self.candidate = self.since = self.track = self.last = None

    def update(self, t, fraction, alarm, valid, track):
        if self.policy is None:
            return "UNDECIDED"
        p = self.policy
        if (self.last is not None and t <= self.last):
            raise ValueError("Time must increase")
        if track != self.track or (self.last is not None and t-self.last > p["max_gap_s"]):
            self.state, self.candidate, self.since = "UNKNOWN", None, None
        self.track, self.last = track, t
        if not valid:
            self.candidate = self.since = None
            return "UNKNOWN"
        if alarm:
            self.state, self.candidate, self.since = "PROLONGED_CLOSURE", None, None
            return self.state
        if fraction is None:
            self.candidate = self.since = None
            return "INSUFFICIENT_WINDOW"
        if not 0 <= fraction <= 1:
            raise ValueError("Fraction outside [0, 1]")
        if fraction >= p["high"]:
            desired = "HIGH_CLOSURE_BURDEN"
        elif fraction >= p["mild"]:
            desired = "ELEVATED_CLOSURE_BURDEN"
        elif fraction <= p["low"]:
            desired = "LOW_CLOSURE_BURDEN"
        else:
            desired = self.state
        if desired == self.state:
            self.candidate = self.since = None
        else:
            if desired != self.candidate:
                self.candidate, self.since = desired, t
            required = p["recover_s"] if desired == "LOW_CLOSURE_BURDEN" else p["hold_s"]
            if t - self.since >= required:
                self.state, self.candidate, self.since = desired, None, None
        return self.state
