from dataclasses import replace
from types import SimpleNamespace
import pytest
from marvel.contracts import Estimate
from marvel.cognition.camera_pipeline import CameraCognition
from test_cognition_integration import eyes


POLICY = dict(ear_threshold=.2, close_s=1, recover_s=.5, max_gap_s=.6,
              unavailable_s=1, window_s=1, min_coverage=.5)

def model(value=.9):
    return SimpleNamespace(estimate=lambda eye: Estimate(eye.sample_id, eye.t, eye.track,
        eye.side, "cnn", value if eye.crop_ok else None, eye.crop_ok))

def pair(t):
    return [replace(e, crop_ok=True) for e in eyes(t, True)]

def test_camera_estimates_connect_to_three_modes_and_selected_alert():
    c = CameraCognition(policy=POLICY, run_id="test", model=model(), cnn_threshold=.5,
        comparison=dict(ear_calibration={"slope": -20, "intercept": 4}, alpha=.5, fusion_threshold=.5),
        alert_mode="fusion")
    for t in (0, .5, 1): result = c.update(str(t), t, pair(t), "valid")
    assert set(result["modes"]) == {"ear", "cnn", "fusion"}
    assert result["decision"]["events"][0]["mode"] == "fusion"
    assert len(set(c.active_episodes().values())) == 3

def test_missing_crop_keeps_ear_usable_and_bad_face_skips_inference():
    c = CameraCognition(policy=POLICY, run_id="test", model=model(), cnn_threshold=.5)
    result = c.update("0", 0, eyes(0, True), "invalid_eye")
    assert result["modes"]["ear"]["decision"]["observation"] == "CLOSED"
    assert result["modes"]["cnn"]["decision"]["observation"] == "UNKNOWN"
    c.model.estimate = lambda eye: pytest.fail("Must not infer on stale frame")
    assert c.update("1", 1, pair(1), "stale_frame")["decision"]["observation"] == "UNKNOWN"

def test_slow_cnn_cannot_advance_closure_timer(monkeypatch):
    from marvel.cognition import camera_pipeline
    clock = iter([0, 2])
    monkeypatch.setattr(camera_pipeline, "perf_counter", lambda: next(clock))
    c = CameraCognition(policy=POLICY, run_id="test", model=model(), cnn_threshold=.5)
    result = c.update("0", 0, pair(0), "valid", evidence_age_s=.1, max_age_s=1)
    assert result["face_status"] == "stale_after_cognition"
    assert all(m["decision"]["observation"] == "UNKNOWN" for m in result["modes"].values())

def test_wrong_identity_fails_before_history_changes():
    m = model()
    m.estimate = lambda eye: Estimate("old", eye.t, eye.track, eye.side, "cnn", .9, True)
    c = CameraCognition(policy=POLICY, run_id="test", model=m, cnn_threshold=.5)
    with pytest.raises(ValueError, match="identity"): c.update("0", 0, pair(0), "valid")
    assert c.trigger.last is None

def test_no_silent_fusion_or_cnn_without_artifact():
    for mode in ("cnn", "fusion"):
        with pytest.raises(ValueError): CameraCognition(policy=POLICY, run_id="test", alert_mode=mode)

def test_cli_missing_model_fails_before_detector_or_camera(tmp_path, monkeypatch):
    from marvel import runtime
    monkeypatch.setattr(runtime, "Detector", lambda *args: pytest.fail("Must fail before camera setup"))
    with pytest.raises(FileNotFoundError):
        runtime.main(["--camera", "--ear-threshold", ".2", "--cnn-threshold", ".5",
                      "--cnn-checkpoint", str(tmp_path / "missing.pt"),
                      "--output", str(tmp_path / "run.jsonl")])
    assert not (tmp_path / "run.jsonl").exists()
