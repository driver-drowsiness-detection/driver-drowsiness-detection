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
