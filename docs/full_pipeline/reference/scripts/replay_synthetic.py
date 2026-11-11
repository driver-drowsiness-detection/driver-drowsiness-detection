"""No camera, CNN, audio or LLM dependency. Exercises the actual trigger and logger."""
import sys
from marvel.cognition.trigger import Trigger
from marvel.observability.logger import JsonLogger


def main():
    output = sys.argv[1]
    tracker = Trigger(1.5, 0.5, 0.25, 2.0)
    log = JsonLogger(output)
    try:
        for index in range(81):
            t = index/10
            state = "CLOSED" if 1 <= t < 3 else "UNKNOWN" if 4 <= t < 7 else "OPEN"
            value = tracker.update(t, state, "synthetic_track")
            record = {"type": "observation", "synthetic": True, "sample_id": str(index),
                      "t": t, "track": "synthetic_track", "modes": {"ear": dict(value, observation=state)}}
            log.write(record)
            for event in value["events"]:
                print(event)
    finally:
        log.close()


if __name__ == "__main__":
    main()
