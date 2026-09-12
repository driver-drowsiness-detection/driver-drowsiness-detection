"""
Unit tests for trigger.py - Person 4
Run with: pytest tests/test_trigger.py

Feeds a sequence of EAR values simulating a normal blink (should NOT
alert) vs. sustained closed eyes (SHOULD alert). No camera, no
MediaPipe, no hardware needed - pure logic testing.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "trigger"))
from trigger import DrowsinessTrigger


def test_single_blink_does_not_alert():
    trigger = DrowsinessTrigger(ear_threshold=0.21, consec_frames=15)
    # Simulate 3 low-EAR frames (a quick blink) - well under the 15-frame threshold
    ear_sequence = [0.30] * 5 + [0.10] * 3 + [0.30] * 5
    alerts = [trigger.update(ear) for ear in ear_sequence]
    assert not any(alerts), "A quick blink should never trigger an alert"


def test_sustained_closed_eyes_alerts():
    trigger = DrowsinessTrigger(ear_threshold=0.21, consec_frames=15)
    ear_sequence = [0.30] * 5 + [0.10] * 20  # eyes closed for 20 straight frames
    alerts = [trigger.update(ear) for ear in ear_sequence]
    assert any(alerts), "20 consecutive low-EAR frames should trigger an alert"


if __name__ == "__main__":
    test_single_blink_does_not_alert()
    test_sustained_closed_eyes_alerts()
    print("All tests passed.")
