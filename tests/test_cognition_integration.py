from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from marvel.contracts import Eye
from marvel.cognition.pipeline import EARCognition
from marvel.cognition.perclos import ClosureWindow


def engine():
    return EARCognition(ear_threshold=.2, run_id="test", close_s=1, recover_s=.5,
                        max_gap_s=.5, unavailable_s=1, window_s=2, min_coverage=.5)


def eyes(t, closed=False):
    gap = 0 if closed else 2
    points = ((0, 0), (2, gap), (6, gap), (8, 0), (6, -gap), (2, -gap))
    return [Eye(str(t), t, "target", side, points, None, True, False)
            for side in ("left", "right")]


def test_geometry_to_bilateral_state_to_episode():
    c = engine()
    assert c.update("0", 0, eyes(0), "valid")["decision"]["state"] == "OPEN"
    records = [c.update(str(t), t, eyes(t, True), "valid") for t in (.5, 1, 1.5)]
    assert records[-1]["decision"]["events"][0]["kind"] == "PROLONGED_CLOSURE"
    assert records[-1]["eye_estimates"][0]["value"] == 0
    unknown = c.update("2", 2, [], "no_face")
    assert unknown["decision"]["alarm_active"]
    assert unknown["decision"]["state"] == "UNKNOWN"


@pytest.mark.parametrize("changes", [{"sample_id": "wrong"}, {"t": 9},
                                     {"track": "other"}, {"side": "left"}])
def test_mismatched_evidence_cannot_enter_timers(changes):
    pair = eyes(0)
    pair[1] = replace(pair[1], **changes)
    result = engine().update("0", 0, pair, "valid")
    assert result["reason"] == "unaligned_eye_evidence"
    assert result["decision"]["state"] == "UNKNOWN"
    assert not result["eye_estimates"]


def test_disagreement_invalid_geometry_and_stale_face_remain_unknown():
    for pair, status in [(eyes(0)[:1], "valid"),
                         ([eyes(0)[0], eyes(0, True)[1]], "valid"),
                         ([eyes(0)[0], replace(eyes(0)[1], geometry_ok=False)], "invalid_eye"),
                         (eyes(0), "stale_frame")]:
        result = engine().update("0", 0, pair, status)
        assert result["decision"]["state"] == "UNKNOWN"


def test_mar_requires_matching_geometry_and_is_only_telemetry():
    geometry = dict(sample_id="0", t=0, track="target", mouth_ok=True,
                    mouth_points=eyes(0)[0].points)
    result = engine().update("0", 0, eyes(0), "valid", {"geometry": geometry})
    assert result["mar"] == .5 and not result["mar_is_yawn_label"]
    geometry["sample_id"] = "old"
    assert engine().update("0", 0, eyes(0), "valid", {"geometry": geometry})["mar"] is None


def test_window_time_weighting_transition_exclusion_and_warmup():
    w = ClosureWindow(2, .5, .5)
    observations = [(0, "OPEN"), (.5, "OPEN"), (1, "CLOSED"), (1.5, "CLOSED"), (2, "CLOSED")]
    results = [w.update(t, s, "target") for t, s in observations]
    assert not results[-2]["ready"] and results[-2]["closure_fraction"] is None
    assert results[-1]["valid_seconds"] == 1.5
    assert results[-1]["coverage"] == .75
    assert results[-1]["closure_fraction"] == pytest.approx(2/3)


def test_window_unknown_at_start_no_target_and_track_reset():
    w = ClosureWindow(1, .5, .5)
    assert not w.update(0, "UNKNOWN", None)["ready"]
    w.update(.5, "CLOSED", "a")
    w.update(1, "CLOSED", "a")
    assert w.update(1.5, "CLOSED", "a")["closure_fraction"] == 1
    result = w.update(2, "UNKNOWN", None)
    assert result["track"] == "a" and result["coverage"] == .5
    assert w.update(2.5, "CLOSED", "b")["valid_seconds"] == 0


def test_window_clips_boundary_and_expires_long_gaps():
    w = ClosureWindow(1.25, .5, .5)
    for t in (0, .5, 1, 1.5):
        result = w.update(t, "OPEN", "a")
    assert result["valid_seconds"] == 1.25 and result["closure_fraction"] == 0
    result = w.update(4, "CLOSED", "a")
    assert result["coverage"] == 0 and result["closure_fraction"] is None


@pytest.mark.parametrize("settings", [(math.nan, .5, .8), (1, math.inf, .8),
                                      (1, .5, math.nan), (0, .5, .8), (1, .5, 0)])
def test_bad_window_config(settings):
    with pytest.raises(ValueError):
        ClosureWindow(*settings)


def test_actual_detector_connects_to_ear_and_mar():
    import numpy as np
    from types import SimpleNamespace
    from marvel.contracts import Frame
    from marvel.perception.detection import Detector, EYES, MOUTH
    landmarks = [SimpleNamespace(x=.5, y=.5) for _ in range(478)]
    for indices in [EYES["left"], EYES["right"], MOUTH]:
        coords = ((.3, .5), (.35, .45), (.45, .45), (.5, .5), (.45, .55), (.35, .55))
        for i, (x, y) in zip(indices, coords):
            landmarks[i] = SimpleNamespace(x=x, y=y)
    backend = SimpleNamespace(detect_bgr=lambda image: [landmarks], close=lambda: None)
    quality = dict(min_crop_pixels=8, brightness_min=10, brightness_max=245, laplacian_min=0)
    d = Detector(None, [0, 0, 1, 1], quality, backend=backend)
    frame = Frame("0", 0, np.full((400, 400, 3), 100, np.uint8))
    pair, status, metrics = d.process(frame)
    result = engine().update("0", 0, pair, status, metrics)
    assert result["decision"]["state"] == "OPEN"
    assert result["mar"] == pytest.approx(.5)
    assert len(metrics["geometry"]["pose_points"]) == 6


@pytest.mark.skipif(not os.environ.get("MARVEL_TEST_MODEL"), reason="Real model opt-in")
def test_real_model_image_and_video_cli(tmp_path):
    import cv2
    import numpy as np
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/runtime.yaml").read_text())
    config["landmarker"] = os.environ["MARVEL_TEST_MODEL"]
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(yaml.safe_dump(config))
    image_path, video_path = tmp_path / "blank.png", tmp_path / "blank.avi"
    blank = np.zeros((128, 128, 3), np.uint8)
    assert cv2.imwrite(str(image_path), blank)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 128))
    assert writer.isOpened()
    for _ in range(22):
        writer.write(blank)
    writer.release()
    for kind, path in [("image", image_path), ("video", video_path)]:
        output = tmp_path / f"{kind}.jsonl"
        run = subprocess.run([sys.executable, str(root / "scripts/run_cognition.py"),
                              "--" + kind, str(path), "--config", str(config_path),
                              "--output", str(output), "--ear-threshold", ".2", "--synthetic-source"],
                             cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr
        records = [json.loads(line) for line in output.read_text().splitlines()]
        observations = [r for r in records if r["type"] == "observation"]
        assert all(r["decision"]["state"] == "UNKNOWN" for r in observations)
        events = [e for r in observations for e in r["decision"]["events"]]
        assert [e["kind"] for e in events] == ([] if kind == "image" else ["MONITORING_UNAVAILABLE"])
        assert records[-1]["type"] == "run_end"


def test_camera_heartbeat_and_late_frame_drop_without_camera(tmp_path, monkeypatch):
    import numpy as np
    import yaml
    from types import SimpleNamespace
    from marvel import runtime
    from marvel.contracts import Frame
    from marvel.perception import capture
    root = Path(__file__).resolve().parents[1]
    model = tmp_path / "fake.task"
    model.write_bytes(b"synthetic test asset")
    settings = yaml.safe_load((root / "configs/runtime.yaml").read_text())
    settings["landmarker"] = str(model)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(settings))
    image = np.zeros((10, 10, 3), np.uint8)
    frames = iter([Frame("first", .1, image), None,
                   Frame("late", .5, image), Frame("fresh", 1.7, image)])
    closed = []
    monkeypatch.setattr(capture, "LiveCapture", lambda _: SimpleNamespace(
        read=lambda: next(frames), close=lambda: closed.append("capture")))
    clock = iter([0, .1, 1.5, 1.6, 1.7])
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(runtime, "Detector", lambda *args: SimpleNamespace(
        invalidate=lambda: None, close=lambda: closed.append("detector")))
    monkeypatch.setattr(runtime, "process_frame", lambda *args: ([], "no_face", {}, 0))
    output = tmp_path / "camera.jsonl"
    runtime.main(["--camera", "--config", str(config), "--output", str(output),
                  "--ear-threshold", ".2", "--max-frames", "2", "--unavailable-seconds", "1",
                  "--synthetic-source"])
    records = [json.loads(line) for line in output.read_text().splitlines()]
    observations = [r for r in records if r["type"] == "observation"]
    assert [r["t"] for r in observations] == [.1, 1.5, 1.7]
    assert observations[1]["decision"]["events"][0]["kind"] == "MONITORING_UNAVAILABLE"
    assert [r["sample_id"] for r in records if r["type"] == "discarded_frame"] == ["late"]
    assert closed == ["capture", "detector"]
