"""Exercise actual temporal logic with synthetic observations, never accuracy data."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marvel.cognition.trigger import Trigger
from marvel.observability.logger import JsonLogger


def replay(output, run_id):
    # Demonstration values only; these are not fitted detection settings.
    config = dict(close_s=1.5, recover_s=0.5, max_gap_s=0.25, unavailable_s=2.0)
    trigger = Trigger(**config, mode="synthetic", run_id=run_id)
    events = []
    with JsonLogger(output) as log:
        log.write(dict(schema=1, type="run", synthetic=True, run_id=run_id,
                       input_source="generated_eye_states", config=config))
        for index in range(81):
            t = index / 10
            state = "CLOSED" if 1 <= t < 3 else "UNKNOWN" if 4 <= t < 7 else "OPEN"
            track = None if state == "UNKNOWN" else "synthetic_track"
            sample_id = f"synthetic:{index}"
            decision = trigger.update(t, state, track, sample_id=sample_id)
            log.write(dict(schema=1, type="observation", synthetic=True,
                           run_id=run_id, sample_id=sample_id, t=t,
                           source_track=track, decision=asdict(decision)))
            events.extend(asdict(event) for event in decision.events)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New JSONL file; existing logs are refused")
    args = parser.parse_args()
    for event in replay(args.output, str(uuid.uuid4())):
        print(json.dumps(event, sort_keys=True))


if __name__ == "__main__":
    main()
