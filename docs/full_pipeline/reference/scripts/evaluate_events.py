"""python scripts/evaluate_events.py observations.jsonl annotations.json report.json"""
import json
import math
import sys
from pathlib import Path


def evaluate(records, annotations, mode):
    intervals = annotations["intervals"]
    # Exhaustive, nonoverlapping independently labelled recording intervals.
    previous_end = None
    for v in intervals:
        a, b, label = v["start"], v["end"], v["label"]
        if not math.isfinite(a) or not math.isfinite(b) or b <= a or label not in ("negative", "closure_event", "unscorable"):
            raise ValueError("Invalid annotated interval")
        if previous_end is not None and abs(a-previous_end) > 1e-6:
            raise ValueError("Annotations must partition the recording without gaps")
        previous_end = b
    truth = [x for x in intervals if x["label"] == "closure_event"]
    observed = [r for r in records if r.get("type") == "observation"]
    if not observed or any(mode not in r["modes"] for r in observed):
        raise ValueError("Missing observations/mode")
    if observed[0]["t"] > intervals[0]["start"] or observed[-1]["t"] < intervals[-1]["end"]:
        raise ValueError("Logs do not cover the full annotation interval")
    events = [e for r in observed for e in r["modes"][mode]["events"] if e["kind"] == "PROLONGED_CLOSURE"]
    matched, delays, false, duplicates, ignored = set(), [], 0, 0, 0
    for event in events:
        t = event["t"]
        candidates = [(i, v) for i, v in enumerate(truth) if v["start"] <= t < v["end"]]
        if candidates:
            i, v = candidates[0]
            if i in matched:
                duplicates += 1
            else:
                matched.add(i)
                delays.append(t-v["start"])
        elif any(v["start"] <= t < v["end"] and v["label"] == "negative" for v in intervals):
            false += 1
        else:
            ignored += 1
    negative_s = sum(v["end"]-v["start"] for v in intervals if v["label"] == "negative")
    valid_s = total_s = 0
    for a, b in zip(observed, observed[1:]):
        if b["t"] <= a["t"]:
            raise ValueError("Log time must increase")
        dt = b["t"]-a["t"]
        total_s += dt
        if (dt <= annotations["max_gap_s"] and a["track"] == b["track"] and
                a["modes"][mode]["observation"] != "UNKNOWN" and b["modes"][mode]["observation"] != "UNKNOWN"):
            valid_s += dt
    return {"truth_events": len(truth), "matched_events": len(matched),
            "event_recall": len(matched)/len(truth) if truth else None,
            "false_alerts": false, "negative_seconds": negative_s,
            "false_alerts_per_negative_hour": false*3600/negative_s if negative_s else None,
            "duplicate_alerts_in_true_events": duplicates, "unscorable_alerts": ignored,
            "delay_from_annotated_closure_onset_s": delays,
            "observation_coverage": valid_s/total_s if total_s else None}


def main():
    log, truth, target = sys.argv[1:]
    records = [json.loads(s) for s in Path(log).read_text().splitlines() if s]
    if any(r.get("synthetic") for r in records):
        raise ValueError("Synthetic logs are not real-data accuracy evidence")
    annotations = json.loads(Path(truth).read_text())
    modes = next(r["modes"] for r in records if r.get("type") == "observation")
    output = {m: evaluate(records, annotations, m) for m in modes}
    with open(target, "x", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
