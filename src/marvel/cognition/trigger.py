"""Elapsed-time closure controller. Instantiate independently for each mode."""

from dataclasses import dataclass
import math

from marvel.contracts import OPEN, CLOSED, UNKNOWN


@dataclass(frozen=True)
class Event:
    event_id: str
    episode_id: str | None
    kind: str
    sample_id: str
    t: float
    track: str | None
    mode: str
    closure_s: float


@dataclass(frozen=True)
class Decision:
    observation: str
    state: str
    alarm_active: bool
    episode_id: str | None
    track: str | None
    closure_s: float
    recovery_s: float
    unknown_s: float
    events: tuple[Event, ...]


class Trigger:
    """No sleeping, frame counters or learned thresholds inside the controller.

    Missing detections use UNKNOWN with track=None, preserving the last target
    and any outstanding alarm. A different explicit track ends that target's
    episode as unresolved. The caller must deliver UNKNOWN heartbeat updates
    during capture stalls; this synchronous controller cannot detect silence.
    """

    def __init__(self, close_s: float, recover_s: float, max_gap_s: float,
                 unavailable_s: float, *, mode: str, run_id: str):
        durations = (close_s, recover_s, max_gap_s, unavailable_s)
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError("Durations must be finite and positive")
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("A mode name is required")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("A run identifier is required")
        self.close_s, self.recover_s = close_s, recover_s
        self.max_gap_s, self.unavailable_s = max_gap_s, unavailable_s
        self.mode, self.run_id = mode, run_id
        self.last = self.track = None
        self.closed_since = self.open_since = self.unknown_since = None
        self.episode_id = None
        self.unavailable_sent = False
        self.counter = 0

    def _event(self, kind, sample_id, t, closure_s, *, begins_episode=False):
        self.counter += 1
        event_id = f"{self.run_id}:{self.mode}:{self.counter}"
        if begins_episode:
            self.episode_id = event_id
        return Event(event_id, self.episode_id, kind, sample_id, t,
                     self.track, self.mode, closure_s)

    def update(self, t: float, observation: str, track: str | None,
               *, sample_id: str) -> Decision:
        # Validate before mutating state: rejected input must not consume time.
        if not math.isfinite(t) or observation not in (OPEN, CLOSED, UNKNOWN):
            raise ValueError("Invalid observation or timestamp")
        if self.last is not None and t <= self.last:
            raise ValueError("Observation time must strictly increase")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("A sample identifier is required")
        if track is not None and (not isinstance(track, str) or not track.strip()):
            raise ValueError("Track must be a nonempty string or None")
        if observation != UNKNOWN and track is None:
            raise ValueError("Valid eye states require a track")

        events = []
        if track is not None and self.track is not None and track != self.track:
            if self.episode_id is not None:
                events.append(self._event("TRACK_ENDED_UNRESOLVED", sample_id, t, 0.0))
            self.episode_id = None
            self.closed_since = self.open_since = None
            # Unavailability is system-wide; target churn must not postpone it.
        if track is not None:
            self.track = track
        if self.last is not None and t - self.last > self.max_gap_s:
            self.closed_since = self.open_since = None
        self.last = t

        if observation == UNKNOWN:
            self.closed_since = self.open_since = None
            if self.unknown_since is None:
                self.unknown_since = t
            if t - self.unknown_since >= self.unavailable_s and not self.unavailable_sent:
                events.append(self._event("MONITORING_UNAVAILABLE", sample_id, t, 0.0))
                self.unavailable_sent = True
        else:
            self.unknown_since = None
            self.unavailable_sent = False
            if observation == CLOSED:
                self.open_since = None
                if self.closed_since is None:
                    self.closed_since = t
                duration = t - self.closed_since
                if duration >= self.close_s and self.episode_id is None:
                    events.append(self._event("PROLONGED_CLOSURE", sample_id, t,
                                              duration, begins_episode=True))
            else:
                self.closed_since = None
                if self.open_since is None:
                    self.open_since = t
                if self.episode_id is not None and t - self.open_since >= self.recover_s:
                    events.append(self._event("RECOVERED", sample_id, t, 0.0))
                    self.episode_id = None

        active = self.episode_id is not None
        state = (UNKNOWN if observation == UNKNOWN else
                 "ALARM" if active else
                 "CLOSURE_PENDING" if observation == CLOSED else OPEN)
        return Decision(
            observation, state, active, self.episode_id, self.track,
            0.0 if self.closed_since is None else t - self.closed_since,
            0.0 if self.open_since is None else t - self.open_since,
            0.0 if self.unknown_since is None else t - self.unknown_since,
            tuple(events),
        )
