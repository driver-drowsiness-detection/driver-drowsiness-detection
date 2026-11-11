from collections import deque
import math
from marvel.contracts import OPEN, CLOSED, UNKNOWN


class ClosureWindow:
    """Observed-time closure fraction proxy. Does not establish physical P80."""
    def __init__(self, window_s, max_gap_s, min_coverage):
        if window_s <= 0 or max_gap_s <= 0 or not 0 < min_coverage <= 1:
            raise ValueError("Invalid rolling window configuration")
        self.window_s, self.max_gap_s, self.min_coverage = window_s, max_gap_s, min_coverage
        self.segments = deque()
        self.previous = None
        self.track = None
        self.started = None

    def update(self, t, state, track):
        if not math.isfinite(t) or state not in (OPEN, CLOSED, UNKNOWN):
            raise ValueError("Invalid rolling observation")
        if self.previous and t <= self.previous[0]:
            raise ValueError("Time must increase")
        if track != self.track:
            self.segments.clear()
            self.previous = None
            self.started = t
        if self.previous:
            pt, ps = self.previous
            # Count only intervals whose TWO endpoints agree and are valid.
            if t - pt <= self.max_gap_s and ps == state and state != UNKNOWN:
                self.segments.append((pt, t, state))
        self.previous, self.track = (t, state), track
        cutoff = t - self.window_s
        while self.segments and self.segments[0][1] <= cutoff:
            self.segments.popleft()
        valid = closed = 0.0
        for start, end, s in self.segments:
            dt = end - max(start, cutoff)
            valid += dt
            closed += dt if s == CLOSED else 0
        coverage = valid / self.window_s
        ready = t - self.started >= self.window_s and coverage >= self.min_coverage
        return {"closure_fraction": closed / valid if ready and valid else None,
                "coverage": coverage, "valid_seconds": valid, "ready": ready}
