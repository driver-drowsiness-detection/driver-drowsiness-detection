# 10. Full execution procedure and testing

## The rule for every checkpoint

State the expected behavior, run the smallest relevant command, inspect the actual output, save evidence, and only then connect the next component. All paths below are relative to the reference root. Create the `data`, `models` and `reports` folders locally before commands that write there.

Use a new versioned filename for each experiment. Most scripts deliberately refuse to overwrite reports/models. Repeated execution should not erase earlier evidence.

## G0. Environment and pure behavior

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
New-Item -ItemType Directory -Force data,models,reports
python scripts\inspect_environment.py > environment.json
python -m unittest discover -s tests -v
python scripts\replay_synthetic.py reports\synthetic_run01.jsonl
```

Expected synthetic events: prolonged closure at observation time 2.5 seconds, recovery at 3.5, and monitoring unavailable at 6.0. They come from explicit artificial states. No camera, CNN or LLM accuracy claim follows from this run.

The test suite uses `unittest`, so core tests do not require pytest. Tests requiring missing image/model libraries are clearly skipped. A skipped test has not passed.

## G1. Capture and perception

```powershell
python scripts\preview_camera.py configs\runtime.yaml
python -m marvel.main configs\runtime.yaml
```

Before the second command, acquire the selected `.task` asset, validate its location, install compatible image dependencies and keep `mode: ear`, `sound: false`, `llm.enabled: false`.

Evidence: frame shapes and timestamps, a saved screenshot showing correct eyelid points, quality failure records and clean shutdown. Test multiple faces, missing face, glasses, a head turn and a closed eyelid crop.

## G2. EAR and time

Use controlled desk recordings, not driving experiments. Test normal blinking, a brief voluntary eye closure, sustained closure, one-eye closure, face absent, and camera disconnected. Verify the state and events from logs rather than relying only on an overlay.

To test the sound, enable `sound: true` after confirming the output device and use a controlled event. Record that someone heard it. For offline replay, sound stays disabled.

## G3. Data preparation

```powershell
python scripts\extract_frames.py data\raw\session01.mp4 data\frames\session01 A:p1:s1:v1 0.2
```

Annotate and assign globally namespaced groups. Merge reviewed labels into the manifest. Confirm the split keeps each person, session and video together. Then:

```powershell
python scripts\prepare_dataset.py data\annotations.csv models\face_landmarker.task data\crops_v1 data\prepared_v1.csv
```

Evidence: annotation policy, independent label source, subject lists, per-split counts, class balance, missing crop rates, example crop montage and data licence/permission record. The runtime never sees these labels.

## G4. CNN

```powershell
python scripts\train_cnn.py data\prepared_v1.csv models\eye_v1.pt 10 cpu
python -m unittest discover -s tests -p test_optional.py -v
```

Evidence: finite losses, validation history, checkpoint reload, tensor-shape test and preprocessing parity. Do not move forward if the model predicts one class for every sample without understanding why.

## G5. Calibration and fusion

```powershell
python scripts\score_dataset.py data\prepared_v1.csv models\eye_v1.pt calibration reports\calibration_scores.csv
python scripts\score_dataset.py data\prepared_v1.csv models\eye_v1.pt validation reports\validation_scores.csv
python scripts\calibrate_scores.py reports\calibration_scores.csv models\calibration_v1.json
python scripts\tune_fusion.py reports\validation_scores.csv models\calibration_v1.json models\eye_v1.pt models\bundle_v1.json
```

Evidence: artifact hashes match, subject partitions are disjoint, EAR mapping decreases, temperature is positive and alpha/threshold selections are recorded. Review paired coverage as well as F1.

Configure `mode: compare` and the bundle path, replay the same development videos, and tune temporal policy using development annotations. Freeze that runtime configuration before test evaluation.

## G6. Frozen image evaluation

```powershell
python scripts\score_dataset.py data\prepared_v1.csv models\eye_v1.pt test reports\test_scores.csv
python scripts\evaluate.py reports\test_scores.csv models\bundle_v1.json reports\image_metrics_v1.json
```

The report provides per-method usable-sample metrics, common-paired-subset metrics, coverage and per-subject results. It also includes Brier score, log loss and reliability bins. Brier score alone is not pure calibration error; inspect the reliability bins and sample counts.

If one method has high F1 on only a small accepted subset, that is not automatically better overall monitoring. Report abstentions and difficult conditions. Confidence intervals should be estimated by resampling participants or recordings, not by treating adjacent frames as independent observations. That statistical extension is not implemented in this reference evaluator.

## G7. Event evaluation

Recordings require independently labelled intervals. Create one JSON annotation file per complete replay with this structure:

```json
{
  "max_gap_s": 0.25,
  "intervals": [
    {"start": 0.0, "end": 5.0, "label": "negative"},
    {"start": 5.0, "end": 8.0, "label": "closure_event"},
    {"start": 8.0, "end": 12.0, "label": "negative"},
    {"start": 12.0, "end": 14.0, "label": "unscorable"}
  ]
}
```

These times illustrate the schema. Replace them with actual annotations. Intervals must be contiguous, nonoverlapping and covered by the log. Use half-open intervals `[start, end)`. `closure_event` means an independently annotated target event under your declared duration criterion, not every blink. End the annotation at a timestamp actually covered by replay records.

```powershell
python scripts\evaluate_events.py reports\runs\RUN_ID\observations.jsonl data\session01_events.json reports\events_v1.json
python scripts\profile_runtime.py reports\runs\RUN_ID\observations.jsonl
```

Replace `RUN_ID` with the actual directory printed/listed after the run. The evaluator refuses synthetic logs as real-data evidence.

Event matching policy: the first alert inside a ground-truth event interval matches that event. Additional alerts in that same interval are duplicates. Alerts during negative intervals count as false alerts. Unscorable intervals are reported separately. Delay is measured from annotated closure onset. This is a simple explicit protocol; if you adopt onset tolerance or episode merging, version that protocol and apply it equally to all modes.

The evaluator reports false alerts per **negative annotated hour**, not per arbitrary count of frames. Short recordings make this rate unstable, so include the exposure duration.

## G8. Rolling statistics and richer states

Validate warm-up, partial coverage, unknown gaps, edge clipping, subject change and unequal frame rates. Compare the reported proxy against independently annotated time intervals. Leave severity policy null until its thresholds and labels are defensible. See Guide 07 for the physical PERCLOS boundary.

## G9. LLM and TTS

```powershell
python scripts\check_llm.py configs\runtime.yaml
```

First select `backend: mock`. Then test the deliberately slow backend. Only after choosing a real model/server, configure `local_chat` and evaluate that endpoint. Run the controlled live pipeline with language generation enabled, then with TTS, and measure CV slowdown.

Evidence must distinguish `mock`, `local_chat`, `fallback`, `timeout_fallback`, `discarded`, speech requested and speech completed. A fallback sentence is valid failure handling but not a successful LLM-generation measurement.

## G10. Jetson acceptance

Follow Guide 11. Repeat portable tests, verify camera and landmarker separately, run the same saved crops and clips through the target runtime, then test live with audio and LLM load. Save exact environment and compatibility failures.

## Test matrix

| Layer | Positive test | Failure test | Evidence |
| --- | --- | --- | --- |
| Capture | Increasing fresh frames | Unplug or stalled worker | Status and timestamps |
| Landmarks | Correct eyelid overlay | Multiple/missing face | Unknown reason |
| Crop | Valid closed-eye image | Crop out of bounds | Validity and saved sample |
| Geometry | Hand calculation/invariants | Degenerate/NaN points | Unit result |
| CNN | Shape, gradients, reload | Wrong metadata/class map | Rejection or parity result |
| Data | Subject-separated manifest | Group leakage | Audit result |
| Fusion | Same observation, expected weighted score | Mismatched eye/time/track | Rejection |
| Trigger | One event then recovery | Unknown/gap during event | Event sequence |
| Rolling window | Known time fraction | Low coverage | Withheld value |
| Runtime | Three independent histories | Stale inference | Recorded stop/error |
| LLM | Valid current-event sentence | Slow/malformed/stale result | Fallback/discard |
| TTS | Heard approved output | Unavailable engine or interruption | Execution status and listening test |
| Export | Same saved inputs | Unsupported operator | Parity/parse evidence |

## Troubleshooting

| Symptom | First inspection | Likely next action |
| --- | --- | --- |
| `No module named marvel` | Current folder and `PYTHONPATH` | Point to reference `src` |
| `No module named cv2/torch` | Exact interpreter in environment report | Install compatible library in that venv |
| `.task` file missing | Asset path, file hash | Acquire actual selected model asset |
| All outputs unknown | Logged quality reasons and actual crop images | Fix visibility/mapping before thresholds |
| CNN always one class | Label distribution, class mapping, preprocessing | Audit data and inspect loss curves |
| Fusion no better | Correlated errors and validation coverage | Keep result honestly; improve observations/data |
| Timing changes with playback speed | Observation timestamp source | Use media time, not execution time |
| EAR fit fails | Paired calibration variation/classes | Obtain usable independently labelled examples |
| LLM output repeatedly falls back | Logged failure, response format/model template | Fix the adapter/server contract |
| Nano install fails | CPU/Python/JetPack/OpenCV details | Choose a compatible target route rather than replacing the system blindly |

## Definition of completion

All intended stages must have reproducible evidence. A teammate should be able to start from setup notes, reproduce the chosen mode on a fixed recording, obtain the expected event sequence, run the model/checks on the target, and demonstrate fixed sound plus a real LLM/TTS path with tested failure behavior. Until then, name the specific pending gate.

Used separate image and event evaluation because classifying an eye correctly does not automatically produce correct alerts over time.

Interview explanation: “We evaluate classifier quality, abstention, event recall, false alerts, delay and target resource cost separately. Every report is tied to a frozen configuration.”

<!-- CODE -->

## Complete source files

### File: `scripts/replay_synthetic.py`

[Open the companion file](reference/scripts/replay_synthetic.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 7 to 22.

```python
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
```

### File: `scripts/evaluate.py`

[Open the companion file](reference/scripts/evaluate.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `summarize`: lines 9 to 38.
* `main`: lines 41 to 57.

```python
"""python scripts/evaluate.py test_scores.csv models/bundle.json reports/image_metrics.json"""
import json
import sys
from pathlib import Path
from marvel.training.calibration import scores, binary_metrics, logloss
from marvel.cognition.features import sigmoid


def summarize(rows, bundle, mode, common=False):
    labelled = [r for r in rows if r["observable"] == "1"]
    used, y, probabilities, predictions = [], [], [], []
    for r in labelled:
        ev, cv = r["ear"] != "", r["logit"] != ""
        eligible = ev and cv if common or mode == "fusion" else ev if mode == "ear" else cv
        if not eligible:
            continue
        pe = sigmoid(bundle["ear_calibration"]["slope"]*float(r["ear"])+bundle["ear_calibration"]["intercept"]) if ev else None
        pc = sigmoid(float(r["logit"])/bundle["temperature"]) if cv else None
        p = pe if mode == "ear" else pc if mode == "cnn" else bundle["alpha"]*pc+(1-bundle["alpha"])*pe
        pred = (int(float(r["ear"]) < bundle["ear_threshold"]) if mode == "ear" else
                int(p >= bundle[mode+"_threshold"]))
        used.append(r)
        y.append(int(r["label"]))
        probabilities.append(p)
        predictions.append(pred)
    result = {"eligible": len(used), "labelled": len(labelled),
              "coverage": len(used)/len(labelled) if labelled else None,
              "metrics": binary_metrics(y, predictions) if used else None}
    if used:
        result["logloss"] = logloss(y, probabilities)
        result["brier"] = sum((a-b)**2 for a, b in zip(y, probabilities))/len(y)
        result["reliability_bins"] = []
        for i in range(10):
            values = [(a, p) for a, p in zip(y, probabilities) if i/10 <= p < (i+1)/10 or (i == 9 and p == 1)]
            result["reliability_bins"].append({"bin": i, "count": len(values),
                "mean_score": sum(p for _, p in values)/len(values) if values else None,
                "closed_fraction": sum(a for a, _ in values)/len(values) if values else None})
    return result


def main():
    source, bundle_path, target = sys.argv[1:]
    rows = scores(source, "test")
    bundle = json.loads(Path(bundle_path).read_text())
    if rows[0]["checkpoint_sha256"] != bundle["checkpoint_sha256"] or rows[0]["manifest_sha256"] != bundle["manifest_sha256"]:
        raise ValueError("Test artifacts differ from frozen bundle")
    if {r["subject_id"] for r in rows} & set(bundle["calibration_subjects"]+bundle["validation_subjects"]):
        raise ValueError("Test group leakage")
    output = {"all_observable": {}, "paired_observable": {}, "per_subject": {}}
    for mode in ("ear", "cnn", "fusion"):
        output["all_observable"][mode] = summarize(rows, bundle, mode)
        output["paired_observable"][mode] = summarize(rows, bundle, mode, True)
    for subject in sorted({r["subject_id"] for r in rows}):
        subset = [r for r in rows if r["subject_id"] == subject]
        output["per_subject"][subject] = {m: summarize(subset, bundle, m) for m in ("ear", "cnn", "fusion")}
    with open(target, "x", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
```

### File: `scripts/evaluate_events.py`

[Open the companion file](reference/scripts/evaluate_events.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `evaluate`: lines 8 to 57.
* `main`: lines 60 to 69.

```python
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
```

### File: `scripts/profile_runtime.py`

[Open the companion file](reference/scripts/profile_runtime.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 8 to 23.

```python
"""python scripts/profile_runtime.py observations.jsonl"""
import json
import statistics
import sys
from pathlib import Path


def main():
    records = [json.loads(s) for s in Path(sys.argv[1]).read_text().splitlines() if s]
    samples = [r for r in records if r.get("type") == "observation" and "processing_s" in r]
    values = sorted(r["processing_s"] for r in samples)
    if not values:
        raise ValueError("No measured runtime samples")
    def percentile(p):
        return values[min(len(values)-1, int(p*(len(values)-1)))]
    # This includes waiting for capture; label it honestly, not model-only latency.
    report = {"records": len(samples), "loop_median_s": statistics.median(values),
              "loop_p95_s": percentile(0.95), "loop_p99_s": percentile(0.99),
              "loop_max_s": max(values), "loop_throughput_hz": len(values)/sum(values)}
    age = [r["read_to_decision_s"] for r in samples if r.get("read_to_decision_s") is not None]
    if age:
        report["read_to_decision_max_s"] = max(age)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

### File: `tests/test_core.py`

[Open the companion file](reference/tests/test_core.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `GeometryTests`: lines 24 to 46.
* `FusionTests`: lines 49 to 88.
* `TriggerTests`: lines 91 to 157.
* `RollingTests`: lines 160 to 194.
* `DataAndLogTests`: lines 197 to 228.
* `SpeechTests`: lines 231 to 268.
* `NumericTests`: lines 272 to 288.

```python
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from marvel.contracts import Estimate, bilateral
from marvel.cognition.features import aspect_ratio, sigmoid
from marvel.cognition.ear_classifier import probability
from marvel.cognition.fusion import combine
from marvel.cognition.trigger import Trigger
from marvel.cognition.perclos import ClosureWindow
from marvel.cognition.state_machine import FatigueState
from marvel.reasoning.prompt_builder import validate, fallback, build
from marvel.reasoning.worker import SpeechWorker
from marvel.observability.logger import JsonLogger
from marvel.training.dataset import read_manifest, REQUIRED

POINTS = ((0., 0.), (1., 1.), (3., 1.), (4., 0.), (3., -1.), (1., -1.))


class GeometryTests(unittest.TestCase):
    def test_hand_computation(self):
        self.assertEqual(aspect_ratio(POINTS), 0.5)

    def test_translation_and_scale(self):
        changed = tuple((x*3+100, y*3+20) for x, y in POINTS)
        self.assertAlmostEqual(aspect_ratio(changed), aspect_ratio(POINTS))

    def test_flat_closed_is_valid(self):
        self.assertEqual(aspect_ratio(tuple((x, 0) for x, _ in POINTS)), 0)

    def test_degenerate(self):
        self.assertIsNone(aspect_ratio(((0, 0),)*6))

    def test_nonfinite(self):
        self.assertIsNone(aspect_ratio(((float("nan"), 0),)+POINTS[1:]))

    def test_wrong_point_count(self):
        self.assertIsNone(aspect_ratio(POINTS[:5]))

    def test_stable_sigmoid(self):
        self.assertEqual(sigmoid(-1000), 0)
        self.assertEqual(sigmoid(1000), 1)


class FusionTests(unittest.TestCase):
    def estimate(self, method, value, sample="1", side="left", track="a", t=1.0):
        return Estimate(sample, t, track, side, method, value, value is not None)

    def test_weighted_mean(self):
        a = self.estimate("ear_probability", 0.7)
        b = self.estimate("cnn", 0.9)
        self.assertAlmostEqual(combine(a, b, 0.5).value, 0.8)

    def test_missing_branch_not_zero(self):
        a = self.estimate("ear_probability", None)
        b = self.estimate("cnn", 0.9)
        self.assertFalse(combine(a, b, 0).valid)

    def test_mismatched_identity(self):
        a = self.estimate("ear_probability", 0.7)
        for field in ({"sample": "2"}, {"side": "right"}, {"track": "b"}, {"t": 2.0}):
            with self.subTest(field=field), self.assertRaises(ValueError):
                combine(a, self.estimate("cnn", 0.9, **field), 0.5)

    def test_raw_ear_rejected(self):
        with self.assertRaises(ValueError):
            combine(self.estimate("ear", 0.2), self.estimate("cnn", 0.9), 0.5)

    def test_bad_alpha(self):
        with self.assertRaises(ValueError):
            combine(self.estimate("ear_probability", 0.7), self.estimate("cnn", 0.9), 2)

    def test_mapping_direction(self):
        cal = {"slope": -10, "intercept": 2}
        self.assertGreater(probability(self.estimate("ear", 0.1), cal).value,
                           probability(self.estimate("ear", 0.3), cal).value)

    def test_positive_mapping_rejected(self):
        with self.assertRaises(ValueError):
            probability(self.estimate("ear", 0.1), {"slope": 2, "intercept": 0})

    def test_bilateral(self):
        self.assertEqual(bilateral("OPEN", "CLOSED"), "UNKNOWN")
        self.assertEqual(bilateral("CLOSED", "CLOSED"), "CLOSED")


class TriggerTests(unittest.TestCase):
    def setUp(self):
        self.tr = Trigger(1.5, 0.5, 0.3, 1.0)

    def drive(self, times, state, track="a"):
        return [self.tr.update(t, state, track) for t in times]

    def alarm(self):
        return self.drive([i/4 for i in range(7)], "CLOSED")

    def test_normal_blink(self):
        self.drive([0, .25], "CLOSED")
        self.assertFalse(self.tr.update(.5, "OPEN", "a")["alarm_active"])

    def test_one_event_per_continuous_closure(self):
        results = self.drive([i/4 for i in range(20)], "CLOSED")
        self.assertEqual(sum(len(x["events"]) for x in results), 1)

    def test_threshold_boundary(self):
        result = self.alarm()[-1]
        self.assertEqual(result["events"][0]["t"], 1.5)

    def test_unknown_does_not_recover(self):
        self.alarm()
        r = self.tr.update(1.75, "UNKNOWN", "a")
        self.assertTrue(r["alarm_active"])
        self.assertEqual(r["state"], "UNKNOWN")

    def test_recovery_requires_duration(self):
        self.alarm()
        self.assertTrue(self.tr.update(1.75, "OPEN", "a")["alarm_active"])
        self.tr.update(2, "OPEN", "a")
        self.assertEqual(self.tr.update(2.25, "OPEN", "a")["events"][0]["kind"], "RECOVERED")

    def test_gap_breaks_closure(self):
        self.drive([0, .25, .5], "CLOSED")
        self.assertFalse(self.tr.update(3, "CLOSED", "a")["alarm_active"])

    def test_unknown_breaks_closure(self):
        self.drive([0, .25, .5], "CLOSED")
        self.tr.update(.75, "UNKNOWN", "a")
        self.assertEqual(self.tr.update(1, "CLOSED", "a")["closure_s"], 0)

    def test_driver_change_not_recovery(self):
        self.alarm()
        r = self.tr.update(1.75, "OPEN", "b")
        self.assertEqual(r["events"][0]["kind"], "TRACK_ENDED_UNRESOLVED")

    def test_nonincreasing_time(self):
        self.tr.update(1, "OPEN", "a")
        with self.assertRaises(ValueError):
            self.tr.update(1, "OPEN", "a")

    def test_unknown_notification_bounded(self):
        r = self.drive([i/4 for i in range(20)], "UNKNOWN")
        self.assertEqual(sum(len(x["events"]) for x in r), 1)

    def test_independent_trackers(self):
        other = Trigger(1.5, .5, .3, 1)
        self.alarm()
        self.assertFalse(other.update(2, "OPEN", "a")["alarm_active"])

    def test_rearm(self):
        self.alarm()
        self.drive([1.75, 2, 2.25], "OPEN")
        r = self.drive([2.5+i/4 for i in range(7)], "CLOSED")
        self.assertEqual(r[-1]["events"][0]["kind"], "PROLONGED_CLOSURE")


class RollingTests(unittest.TestCase):
    def test_all_closed_full_window(self):
        w = ClosureWindow(2, .3, .8)
        for i in range(9):
            r = w.update(i/4, "CLOSED", "a")
        self.assertEqual(r["closure_fraction"], 1)

    def test_unknown_reduces_coverage(self):
        w = ClosureWindow(2, .3, .8)
        for i in range(9):
            r = w.update(i/4, "UNKNOWN" if i < 5 else "CLOSED", "a")
        self.assertIsNone(r["closure_fraction"])

    def test_sparse_samples_are_not_continuity(self):
        w = ClosureWindow(2, .3, .8)
        w.update(0, "CLOSED", "a")
        self.assertEqual(w.update(2, "CLOSED", "a")["coverage"], 0)

    def test_rollout_and_track_reset(self):
        w = ClosureWindow(2, .3, .8)
        for i in range(20):
            r = w.update(i/4, "OPEN", "a")
        self.assertEqual(r["closure_fraction"], 0)
        self.assertFalse(w.update(5, "CLOSED", "b")["ready"])

    def test_fatigue_policy_undecided(self):
        self.assertEqual(FatigueState(None).update(0, None, False, True, "a"), "UNDECIDED")

    def test_tier_hysteresis(self):
        f = FatigueState({"low": .1, "mild": .2, "high": .4, "hold_s": .5,
                          "recover_s": .5, "max_gap_s": .3})
        for i in range(3):
            r = f.update(i/4, .5, False, True, "a")
        self.assertEqual(r, "HIGH_CLOSURE_BURDEN")
        self.assertEqual(f.update(.75, None, False, False, "a"), "UNKNOWN")


class DataAndLogTests(unittest.TestCase):
    def manifest(self, directory, second_split="train"):
        path = Path(directory)/"data.csv"
        rows = []
        for i, split in enumerate(("train", second_split)):
            rows.append(dict(sample_id=str(i), subject_id="a", session_id="a:s1", video_id="a:v1",
                             source="test", split=split, frame_path=f"frame{i}.png", timestamp_s=str(i),
                             eye_side="left", label=str(i), observable="1", label_source="human"))
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(REQUIRED))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_group_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                read_manifest(self.manifest(d, "test"))

    def test_manifest_valid(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(read_manifest(self.manifest(d))), 2)

    def test_json_missing_not_nan(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"a.jsonl"
            log = JsonLogger(p)
            log.write({"value": None})
            with self.assertRaises(ValueError):
                log.write({"value": float("nan")})
            log.close()
            self.assertIsNone(json.loads(p.read_text())["value"])


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.event = {"event_id": "r:1", "kind": "PROLONGED_CLOSURE", "method": "fusion", "closure_s": 2}

    def test_valid_output(self):
        text = json.dumps({"event_id": "r:1", "message": fallback(self.event)})
        self.assertEqual(validate(text, self.event), fallback(self.event))

    def test_wrong_event(self):
        with self.assertRaises(ValueError):
            validate(json.dumps({"event_id": "r:2", "message": fallback(self.event)}), self.event)

    def test_unsupported_claim(self):
        with self.assertRaises(ValueError):
            validate(json.dumps({"event_id": "r:1", "message": "You are safe to keep driving."}), self.event)

    def test_no_raw_video(self):
        text = json.dumps(build(dict(self.event, raw_image="private_pixels")))
        self.assertNotIn("private_pixels", text)

    def test_queue_bounded_and_stale_cancelled(self):
        w = SpeechWorker({"backend": "test_slow", "timeout_s": .5, "ttl_s": 1})
        try:
            self.assertTrue(w.submit(self.event))
            self.assertFalse(w.submit(self.event))
            self.assertEqual(w.poll("other")["source"], "discarded")
        finally:
            w.close()

    def test_hard_timeout(self):
        w = SpeechWorker({"backend": "test_slow", "timeout_s": .25, "ttl_s": 3})
        try:
            w.submit(self.event)
            time.sleep(.3)
            self.assertEqual(w.poll("r:1")["source"], "timeout_fallback")
            self.assertIsNone(w.process)
        finally:
            w.close()


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
class NumericTests(unittest.TestCase):
    def test_fit_ear_separates_fixture(self):
        from marvel.training.calibration import fit_ear
        rows = [{"ear": .1, "label": "1"}, {"ear": .11, "label": "1"},
                {"ear": .3, "label": "0"}, {"ear": .31, "label": "0"}]
        cal = fit_ear(rows)
        self.assertLess(cal["slope"], 0)
        self.assertGreater(sigmoid(cal["slope"]*.1+cal["intercept"]), .9)

    def test_closed_crop_does_not_collapse(self):
        import numpy as np
        from marvel.perception.crops import eye_crop
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        points = tuple((x*5+20, 50) for x, _ in POINTS)
        crop = eye_crop(image, points)
        self.assertEqual(crop.shape, (16, 32, 3))
        self.assertFalse(np.shares_memory(image, crop))


if __name__ == "__main__":
    unittest.main()
```

### File: `tests/test_integration.py`

[Open the companion file](reference/tests/test_integration.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `FakeCNN`: lines 17 to 19.
* `PipelineTests`: lines 22 to 44.
* `OfflineCLITests`: lines 48 to 79.
* `EventEvaluationTests`: lines 82 to 101.

```python
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from marvel.contracts import Eye, Estimate
from marvel.pipeline import Pipeline
from marvel.cognition.cnn_inference import sha256

ROOT = Path(__file__).resolve().parents[1]


class FakeCNN:
    def estimate(self, eye):
        return Estimate(eye.sample_id, eye.t, eye.track, eye.side, "cnn", .1, True)


class PipelineTests(unittest.TestCase):
    def test_three_modes_do_not_share_closure_history(self):
        config = {"mode": "compare", "ear_threshold": .2, "fatigue_policy": None,
                  "temporal": {"close_s": 1, "recover_s": .5, "max_gap_s": .3, "unavailable_s": 1},
                  "window": {"window_s": 2, "max_gap_s": .3, "min_coverage": .8}}
        bundle = {"ear_threshold": .2, "cnn_threshold": .5, "fusion_threshold": .5,
                  "ear_calibration": {"slope": -20, "intercept": 5}, "alpha": .5}
        pipe = Pipeline(config, bundle, FakeCNN())
        for index in range(5):
            t = index/4
            points = tuple((float(x), 0.) for x in (0, 1, 3, 4, 3, 1))
            eyes = [Eye(str(index), t, "a", side, points, None, True, True) for side in ("left", "right")]
            result = pipe.update(str(index), t, eyes)
        self.assertTrue(result["modes"]["ear"]["alarm_active"])
        self.assertFalse(result["modes"]["cnn"]["alarm_active"])
        self.assertTrue(result["modes"]["fusion"]["alarm_active"])

    def test_empty_observation_preserves_unknown(self):
        config = {"mode": "ear", "ear_threshold": .2, "fatigue_policy": None,
                  "temporal": {"close_s": 1, "recover_s": .5, "max_gap_s": .3, "unavailable_s": 1},
                  "window": {"window_s": 2, "max_gap_s": .3, "min_coverage": .8}}
        r = Pipeline(config).update("0", 0, [])
        self.assertEqual(r["modes"]["ear"]["observation"], "UNKNOWN")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
class OfflineCLITests(unittest.TestCase):
    def test_calibrate_tune_evaluate_contracts(self):
        # Artificial predictions; validates file/CLI integration, not a trained model.
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            checkpoint = folder/"identity_only.pt"
            checkpoint.write_bytes(b"synthetic file identity, not torch weights")
            for split in ("calibration", "validation", "test"):
                rows = []
                for i in range(8):
                    label = i % 2
                    rows.append({"sample_id": split+str(i), "subject_id": split+":p"+str(i),
                        "session_id": split+":s"+str(i), "video_id": split+":v"+str(i), "source": "synthetic",
                        "split": split, "frame_path": str(folder/(split+str(i)+".png")),
                        "timestamp_s": i, "eye_side": "left", "label": label, "observable": "1",
                        "label_source": "human", "crop_path": "fixture.png", "ear": .1 if label else .3,
                        "logit": 3 if label else -3, "checkpoint_sha256": sha256(checkpoint),
                        "manifest_sha256": "synthetic_manifest_identity"})
                with (folder/(split+".csv")).open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            commands = [
                ["calibrate_scores.py", str(folder/"calibration.csv"), str(folder/"cal.json")],
                ["tune_fusion.py", str(folder/"validation.csv"), str(folder/"cal.json"), str(checkpoint), str(folder/"bundle.json")],
                ["evaluate.py", str(folder/"test.csv"), str(folder/"bundle.json"), str(folder/"metrics.json")]]
            for args in commands:
                result = subprocess.run([sys.executable, str(ROOT/"scripts"/args[0]), *args[1:]],
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            metrics = json.loads((folder/"metrics.json").read_text())
            self.assertEqual(metrics["all_observable"]["fusion"]["metrics"]["f1"], 1)


class EventEvaluationTests(unittest.TestCase):
    def test_one_to_one_event_matching(self):
        spec = importlib.util.spec_from_file_location("event_evaluator", ROOT/"scripts"/"evaluate_events.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        annotations = {"max_gap_s": .3, "intervals": [
            {"start": 0, "end": 1, "label": "negative"},
            {"start": 1, "end": 3, "label": "closure_event"},
            {"start": 3, "end": 4, "label": "negative"}]}
        records = []
        for i in range(17):
            t = i/4
            events = [{"kind": "PROLONGED_CLOSURE", "t": t}] if t in (0.5, 2, 2.5) else []
            records.append({"type": "observation", "t": t, "track": "a",
                            "modes": {"ear": {"observation": "OPEN", "events": events}}})
        result = module.evaluate(records, annotations, "ear")
        self.assertEqual(result["matched_events"], 1)
        self.assertEqual(result["duplicate_alerts_in_true_events"], 1)
        self.assertEqual(result["false_alerts"], 1)
        self.assertEqual(result["delay_from_annotated_closure_onset_s"], [1])


if __name__ == "__main__":
    unittest.main()
```

### File: `tests/test_optional.py`

[Open the companion file](reference/tests/test_optional.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `CNNTests`: lines 8 to 27.
* `ImageTests`: lines 31 to 42.

```python
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"), "PyTorch/Pillow unavailable")
class CNNTests(unittest.TestCase):
    def test_forward_backward_checkpoint(self):
        import torch
        from marvel.cognition.cnn_architecture import EyeCNN
        from marvel.cognition.cnn_inference import CNN
        from marvel.training.transforms import PREPROCESS
        model = EyeCNN()
        x = torch.rand((4, 1, 32, 64))
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4,))
        loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0., 1., 0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"weights.pt"
            torch.save({"state_dict": model.state_dict(), "class_map": {"open": 0, "closed": 1},
                        "preprocess": PREPROCESS}, p)
            restored = CNN(p).model
            model.eval()
            torch.testing.assert_close(restored(x), model(x))


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("PIL"), "OpenCV/Pillow unavailable")
class ImageTests(unittest.TestCase):
    def test_saved_crop_preprocess_matches_live(self):
        import cv2
        import numpy as np
        from PIL import Image
        from marvel.training.transforms import array_from_image, array_from_bgr
        crop = np.random.default_rng(7).integers(0, 256, (25, 50, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"crop.png"
            self.assertTrue(cv2.imwrite(str(path), crop))
            with Image.open(path) as image:
                np.testing.assert_array_equal(array_from_image(image), array_from_bgr(crop))


if __name__ == "__main__":
    unittest.main()
```

