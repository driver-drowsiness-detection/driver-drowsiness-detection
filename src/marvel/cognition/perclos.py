"""Time-weighted binary closure proxy, not validated physical P80 PERCLOS."""
from collections import deque
import math

from marvel.contracts import OPEN, CLOSED, UNKNOWN


class ClosureWindow:
    def __init__(self, window_s, max_gap_s, min_coverage):
        if (not all(math.isfinite(x) for x in (window_s, max_gap_s, min_coverage))
                or window_s <= 0 or max_gap_s <= 0 or not 0 < min_coverage <= 1):
            raise ValueError("Invalid rolling window configuration")
        self.window_s, self.max_gap_s, self.min_coverage = window_s, max_gap_s, min_coverage
        self.segments = deque()
        self.previous = self.track = self.started = None

    def update(self, t, state, track):
        if not math.isfinite(t) or state not in (OPEN, CLOSED, UNKNOWN):
            raise ValueError("Invalid rolling observation")
        if self.previous is not None and t <= self.previous[0]:
            raise ValueError("Time must strictly increase")
        if track is not None and (not isinstance(track, str) or not track.strip()):
            raise ValueError("Invalid track")
        if state != UNKNOWN and track is None:
            raise ValueError("Valid observations require a track")
        if track is not None and track != self.track:
            self.segments.clear()
            self.previous = None
            self.started = t
            self.track = track
        if self.started is None:
            self.started = t
        if self.previous is not None:
            pt, ps = self.previous
            # A transition interval has unknown transition time, so exclude it.
            if t - pt <= self.max_gap_s and ps == state and state != UNKNOWN:
                self.segments.append((pt, t, state))
        self.previous = (t, state)
        cutoff = t - self.window_s
        while self.segments and self.segments[0][1] <= cutoff:
            self.segments.popleft()
        valid = closed = 0.0
        for start, end, observed in self.segments:
            duration = end - max(start, cutoff)
            valid += duration
            if observed == CLOSED:
                closed += duration
        coverage = min(1.0, valid / self.window_s)
        ready = t - self.started >= self.window_s and coverage >= self.min_coverage
        return {"metric": "binary_closure_fraction_proxy", "window_s": self.window_s,
                "closure_fraction": closed / valid if ready and valid else None,
                "coverage": coverage, "valid_seconds": valid, "closed_seconds": closed,
                "ready": ready, "track": self.track}
