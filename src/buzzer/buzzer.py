"""
Module: buzzer
Owner: UNASSIGNED - team decision, same as hardware wiring in general.

PURPOSE
    Turns a True/False alert decision (from trigger.py) into a real
    physical buzzer sound on the Jetson's GPIO pin. Kept as its own
    tiny file so trigger.py never needs to know anything about GPIO.

STATUS: Week 1 stub only
    Nobody is individually assigned this yet. Whoever's around when the
    buzzer + GPIO header arrive wires this up together, the same way
    final integration is a team session, not a pre-assigned role.

INTERFACE CONTRACT
    trigger_alert()
        Week 1: just prints (placeholder so trigger.py has something to
                 call today without waiting on hardware).
        Week 3: replace the body with the real GPIO.output() call once
                 the buzzer is wired.
"""


def trigger_alert():
    """Week 1 stub. Replace with a real GPIO buzzer call once hardware arrives."""
    print("DROWSINESS ALERT!")
