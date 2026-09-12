"""
Module: trigger
Owner: Person 4

PURPOSE
    The decision-making layer. Takes a continuous EAR value every frame
    and decides: has this been low for long enough to count as real
    drowsiness, not just a blink?

TERMINOLOGY (for your notes)
    - Debounce: requiring a condition to hold for N consecutive readings
      before acting on it - same idea used in physical buttons/switches,
      borrowed here to ignore momentary noise.
    - Alert fatigue: a named failure mode in safety engineering where too
      many false alarms cause people to start ignoring real ones. This is
      exactly what a single-frame threshold (no debounce) would cause -
      every normal blink would fire an alert.
    - Hysteresis (preview, Phase 2 material): a smarter version of
      debounce with different "enter alert" and "exit alert" thresholds
      so the alert doesn't flicker right at the boundary. Not needed
      yet - this week's counter is the intentional Phase 1 "naive"
      baseline the proposal describes.

INTERFACE CONTRACT
    DrowsinessTrigger(ear_threshold=0.21, consec_frames=15)
        .update(ear: float) -> bool   # True = fire alert this frame
        .reset()                      # clears the internal counter

NOTE ON THE BUZZER
    This module does NOT call any hardware directly. It only returns
    True/False. Wiring that boolean to the real buzzer is a shared team
    task once hardware arrives (src/buzzer/ is intentionally left
    unowned for now) - keeps this module's logic testable with zero
    hardware dependency, same reasoning as features.py.
"""


class DrowsinessTrigger:
    def __init__(self, ear_threshold=0.21, consec_frames=15):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self._counter = 0

    def update(self, ear):
        if ear < self.ear_threshold:
            self._counter += 1
        else:
            self._counter = 0
        return self._counter >= self.consec_frames

    def reset(self):
        self._counter = 0
