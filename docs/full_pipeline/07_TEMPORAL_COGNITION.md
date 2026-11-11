# 07. Time, rolling closure and operational states

## Requirement

Distinguish a short blink from sustained sampled closure, avoid repeated events on every frame, preserve unresolved alarms through missing observations, and summarize recent closure with known observation coverage.

## Concepts and choices

| Concept | Meaning |
| --- | --- |
| Debounce | Require evidence to persist before a transition |
| Recovery | Require valid reopening before clearing an active event |
| Hysteresis | Use different conditions for escalation and de-escalation |
| Maximum gap | Largest interval allowed to support continuity between samples |
| Coverage | Fraction of the requested window supported by usable observations |
| Watchdog | Detect missing progress even if the camera stops providing frames |
| Outstanding event | Earlier alarm that has not been resolved by valid evidence |

Frame-count rules are easy but change meaning with FPS. Elapsed-time rules handle variable processing rate. Learned sequence models can capture richer patterns but need separately labelled sequences. This reference uses explicit elapsed time and preserves learned sequence models as an optional later experiment.

## Closure state machine

| Current input | Behavior |
| --- | --- |
| First closed observation | Start closure candidate |
| Further closed samples with small gaps | Increase sampled closure duration |
| Duration reaches configured threshold | Emit one `PROLONGED_CLOSURE` event |
| Continued closure | Keep event active; no event every frame |
| Open before threshold | Cancel closure candidate |
| Open after alarm | Start valid recovery interval |
| Sustained valid recovery | Emit `RECOVERED` and clear active event |
| Unknown | Break closure/recovery candidates, preserve active event |
| Long observation gap | Break candidate continuity |
| Track change | End old history; log unresolved old event if necessary |
| Persistent unknown | Emit one `MONITORING_UNAVAILABLE` notification per unknown episode |

`UNKNOWN` and `alarm_active` are separate fields. You can have unknown current visibility while an earlier alarm remains unresolved. Treating unknown as open would invent recovery. Treating unknown as closed would invent evidence of closure.

The reference breaks candidate continuity on **any** explicit unknown sample. `max_gap_s` also prevents bridging long intervals with no sample. This policy is conservative and may reduce event recall when observations frequently fail. Measure that effect; do not hide it.

## Walk through the trigger variables

1. `last` checks strictly increasing observation time.
2. `track` isolates one continuity epoch.
3. `closed_since` is the first closed sample of a current continuous candidate.
4. `open_since` starts recovery, never recovery from unknown.
5. `unknown_since` starts the monitoring-unavailable timer.
6. `active` preserves unresolved event state independently of the current observation.
7. `unavailable_sent` bounds notifications during one unknown episode.
8. `counter` produces unique local event sequences; the runtime prefixes run and method IDs.

Time and memory per trigger update are O(1). Each method receives a distinct `Trigger` instance. Sharing one instance would mix CNN and EAR histories and invalidate the experiment.

## PERCLOS versus the supplied rolling proxy

The original guide's frame-count approximation is insufficient when FPS varies. The reference accumulates observed time, not frame counts.

For each adjacent sample pair, an interval contributes only if:

1. The gap is below the maximum.
2. Both observations are valid.
3. Both endpoints agree on open or closed.
4. The track is unchanged.

At each timestamp, clip retained intervals to the latest window. Then:

\[
coverage=\frac{observed\ valid\ seconds}{requested\ window\ seconds}
\]

\[
closure\ fraction=\frac{observed\ closed\ seconds}{observed\ valid\ seconds}.
\]

The value is withheld until a complete time window has elapsed and coverage reaches the required minimum. An unknown interval is excluded from the denominator and lowers coverage. It does not become zero closure.

Example: a 60-second window with 40 seconds of usable observations has coverage 2/3. If 20 usable seconds are closed, the fraction is 0.5, but the default minimum coverage of 0.8 prevents presenting it as a ready summary.

This is an explicit interval-estimation policy, not continuous ground truth between frames. Rejecting state-transition intervals also lowers coverage. Other interpolation policies are possible, but they must be declared and compared consistently.

`ClosureWindow` keeps recent intervals in a deque. Append/pop operations are O(1) amortized; the reference recomputes the clipped sum in O(K) per update, where K is intervals within the window. At high rates, maintain running totals after correctness tests if profiling shows a need.

**Do not call the output validated P80 PERCLOS merely because a CNN labels eyes closed.** P80 requires a declared physical closure criterion and annotation/measurement protocol. EAR divided by a personal open reference is not a direct physical eyelid percentage. To report PERCLOS accuracy, define and validate the closure criterion, treatment of blinks and coverage against an independently labelled reference.

## Richer state machine and proposal terminology

The optional `FatigueState` can maintain low, elevated and high closure-burden tiers using separate low/mild/high cutoffs and hold/recovery durations. It keeps the immediate prolonged-closure path independent of a long rolling window.

Default `fatigue_policy: null` means undecided. The code is supplied, but no arbitrary cutoffs are presented as clinical fatigue boundaries.

When you have development-sequence labels, configure `low`, `mild`, `high`, `hold_s`, `recover_s` and `max_gap_s`. The ordering must satisfy `low < mild < high`. Validate transitions and delays. The reference names are observable proxy states. Mapping these states to the proposal's “Alert, Mild Fatigue, Drowsy, Microsleep” labels needs independent evidence and honest wording. Valid eyes-open evidence does not prove a person is alert.

Do not let a rolling statistic suppress an immediate prolonged-closure alarm. The rolling window answers a different question: recent burden, rather than the current uninterrupted event.

## Acceptance scenarios

Run brief blink, sustained closure, wink, one-eye failure, unknown during closure, unknown during recovery, long capture gap, repeated timestamp, track change, repeated episode and low window coverage. Include multiple frame rates with the same underlying observation timeline.

Compare event times within sampling resolution. At 10 Hz, a threshold crossing cannot generally be localized more precisely than the observation spacing without assumptions.

Used elapsed time because FPS changes. Used independent state and alarm-history fields because unknown observations cannot erase an earlier event.

Interview explanation: “We distinguish per-frame eye state from event state, use timestamped evidence with maximum-gap rules, and evaluate both alert delay and unavailable monitoring time.”

Retention exercise: trace `CLOSED` at 0, 0.25 and 0.5 seconds, `UNKNOWN` at 0.75, then `CLOSED` at 1.0. What is `closed_since` at 1.0, and why?

<!-- CODE -->

## Complete source files

### File: `src/marvel/cognition/trigger.py`

[Open the companion file](reference/src/marvel/cognition/trigger.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `Trigger`: lines 5 to 67. One instance per method. Unknown breaks evidence but never clears an alarm.

```python
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
```

### File: `src/marvel/cognition/perclos.py`

[Open the companion file](reference/src/marvel/cognition/perclos.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `ClosureWindow`: lines 6 to 43. Observed-time closure fraction proxy. Does not establish physical P80.

```python
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
```

### File: `src/marvel/cognition/state_machine.py`

[Open the companion file](reference/src/marvel/cognition/state_machine.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `FatigueState`: lines 1 to 49. Optional experimental tiers. Configuration supplies unvalidated policy values.

```python
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
```

