import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from marvel.contracts import Estimate, Frame

ROOT = Path(__file__).resolve().parents[1]


def test_camera_image_cli_runs_outside_repository_without_mediapipe_model(tmp_path):
    path = tmp_path / "blank.png"
    assert cv2.imwrite(str(path), np.zeros((48, 64, 3), np.uint8))
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/preview_perception.py"), "camera",
        "--image", str(path), "--no-display",
    ], cwd=tmp_path, capture_output=True, text=True, timeout=30, check=True)
    record = json.loads(result.stdout)
    assert record["status"] == "camera_only" and record["shape"] == [48, 64, 3]
    assert "1 received frame" in result.stderr


def test_q_while_waiting_closes_camera_and_windows(monkeypatch):
    spec = importlib.util.spec_from_file_location("preview_cli_test", ROOT / "scripts/preview_perception.py")
    preview = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preview)

    class Capture:
        closed = False
        def read(self):
            return None
        def close(self):
            self.closed = True
    capture = Capture()
    destroyed = []
    monkeypatch.setattr(preview, "LiveCapture", lambda config: capture)
    for name in ("namedWindow", "resizeWindow", "imshow"):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: destroyed.append(True))
    preview.main(["camera"])
    assert capture.closed and destroyed == [True]


def test_preview_cnn_summary_preserves_missing_evidence():
    spec = importlib.util.spec_from_file_location("preview_cnn_test", ROOT / "scripts/preview_perception.py")
    preview = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preview)
    frame = Frame("f1", 1.0, np.zeros((32, 64, 3), np.uint8))
    estimates = [Estimate("f1", 1.0, "track_1", "left", "cnn", 0.8, True),
                 Estimate("f1", 1.0, "track_1", "right", "cnn", None, False,
                          "crop_quality")]
    record = preview.summary(frame, [], "invalid_eye", {}, 0.01, 5.0, estimates)
    assert record["cnn_status"] == "experimental_raw_eye_scores_no_alert"
    assert record["cnn"][0]["value"] == 0.8
    assert record["cnn"][1]["value"] is None
    assert record["cnn"][1]["reason"] == "crop_quality"


def test_standalone_jetson_probe_parses_as_python36():
    # Syntax compatibility only; actual ARM64/OpenCV execution remains a board test.
    ast.parse((ROOT / "scripts/jetson_camera_probe.py").read_text(), feature_version=(3, 6))
