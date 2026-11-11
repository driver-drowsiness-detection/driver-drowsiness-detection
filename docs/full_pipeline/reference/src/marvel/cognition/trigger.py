import math
from marvel.contracts import OPEN, CLOSED, UNKNOWN


class Trigger:
    """One instance per method. Unknown breaks evidence but never clears an alarm."""
    def __init__(self, close_s, recover_s, max_gap_s, unavailable_s):
        if min(close_s, recover_s, max_gap_s, unavailable_s) <= 0:
            raise ValueError("All durations must be positive")
        self.close_s, self.recover_s = close_s, recover_s
        self.max_gap_s, self.unavailable_s = max_gap_s, unavailable_s
        self.last = self.track = None
        self.closed_since = self.open_since = self.unknown_since = None
        self.active = False
        self.unavailable_sent = False
        self.counter = 0

    def update(self, t, observation, track):
        if not math.isfinite(t) or observation not in (OPEN, CLOSED, UNKNOWN):
            raise ValueError("Invalid observation")
        if self.last is not None and t <= self.last:
            raise ValueError("Observation time must strictly increase")
        events = []
        if self.track is not None and track != self.track:
            if self.active:
                events.append("TRACK_ENDED_UNRESOLVED")
            self.active = False
            self.closed_since = self.open_since = self.unknown_since = None
            self.unavailable_sent = False
        if self.last is not None and t - self.last > self.max_gap_s:
            self.closed_since = self.open_since = None
        self.last, self.track = t, track
        if observation == UNKNOWN:
            self.closed_since = self.open_since = None
            if self.unknown_since is None:
                self.unknown_since = t
            if t - self.unknown_since >= self.unavailable_s and not self.unavailable_sent:
                events.append("MONITORING_UNAVAILABLE")
                self.unavailable_sent = True
        else:
            self.unknown_since = None
            self.unavailable_sent = False
            if observation == CLOSED:
                self.open_since = None
                if self.closed_since is None:
                    self.closed_since = t
                if t - self.closed_since >= self.close_s and not self.active:
                    self.active = True
                    events.append("PROLONGED_CLOSURE")
            else:
                self.closed_since = None
                if self.open_since is None:
                    self.open_since = t
                if self.active and t - self.open_since >= self.recover_s:
                    self.active = False
                    events.append("RECOVERED")
        state = (UNKNOWN if observation == UNKNOWN else
                 "ALARM" if self.active else
                 "CLOSURE_PENDING" if observation == CLOSED else OPEN)
        closure_s = 0.0 if self.closed_since is None else t - self.closed_since
        records = []
        for kind in events:
            self.counter += 1
            records.append({"sequence": self.counter, "kind": kind, "t": t,
                            "track": track, "closure_s": closure_s})
        return {"state": state, "alarm_active": self.active,
                "closure_s": closure_s, "events": records}
