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
