"""Summarize measured runtime JSONL; no inferred accuracy or sensor latency."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def describe(values):
    if not values:
        return None
    values = sorted(values)
    return dict(n=len(values), median=statistics.median(values),
                p95=values[max(0, math.ceil(.95 * len(values)) - 1)], maximum=max(values))


def summarize(path):
    counts, events, valid = Counter(), Counter(), Counter()
    latency, age, frames, end, header = [], [], 0, None, None
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["type"] == "run": header = row
            elif row["type"] == "run_end": end = row
            elif row["type"] == "observation":
                counts[row["face_status"]] += 1
                if "cognition_ms" in row:
                    frames += 1
                    latency.append(row["cognition_ms"])
                if row.get("decision_age_s") is not None: age.append(row["decision_age_s"])
                for mode, result in row.get("modes", {}).items():
                    valid[mode] += int(result["decision"]["observation"] != "UNKNOWN")
                    for event in result["decision"]["events"]:
                        events[mode + ":" + event["kind"]] += 1
    if header is None or end is None:
        raise ValueError("A completed runtime log with run and run_end is required")
    elapsed = end.get("loop_seconds")
    return dict(schema=1, source=str(path), synthetic=header["synthetic"],
                source_kind=header["source_kind"], frames=frames, loop_seconds=elapsed,
                processed_frames_per_second=frames / elapsed if elapsed and elapsed > 0 else None,
                cognition_ms=describe(latency), estimated_read_completion_to_decision_s=describe(age),
                observations_by_status=dict(counts), valid_observations_by_mode=dict(valid),
                events=dict(events), active_episodes=end.get("active_episodes"),
                accuracy_evaluated=False, memory_measured=False, temperature_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.log)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
