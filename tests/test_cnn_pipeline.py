"""Synthetic fixtures test software plumbing, never model accuracy."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

from marvel.contracts import Eye
from marvel.training.dataset import read_manifest
from marvel.training.preparation import prepare_rows
from marvel.training.audit import audit_manifest

ROOT = Path(__file__).resolve().parents[1]


def row(sample, split="train", **extra):
    result = dict(sample_id=sample, subject_id=split+":person",
                  session_id=split+":session", video_id=split+":video", source="synthetic_test",
                  split=split, frame_path="frame.png", timestamp_s="0", eye_side="left",
                  label="0", observable="1", label_source="human")
    result.update(extra)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_portable_manifest_and_inconsistent_frame(tmp_path, monkeypatch):
    path = tmp_path / "annotations.csv"
    write_csv(path, [row("f")])
    monkeypatch.chdir(tmp_path.parent)
    assert read_manifest(path)[0]["frame_path"] == str(tmp_path / "frame.png")
    write_csv(path, [row("f"), row("f", eye_side="right", timestamp_s="1")])
    with pytest.raises(ValueError, match="frame identity"):
        read_manifest(path)


def test_unobservable_label_is_not_a_binary_training_target(tmp_path):
    path = tmp_path / "annotations.csv"
    write_csv(path, [row("f", observable="0", label="1")])
    with pytest.raises(ValueError, match="blank"):
        read_manifest(path)


def test_exact_duplicate_bytes_across_subjects_and_splits(tmp_path):
    Image.new("RGB", (20, 10)).save(tmp_path / "a.png")
    Image.new("RGB", (20, 10)).save(tmp_path / "b.png")
    path = tmp_path / "annotations.csv"
    write_csv(path, [row("a", frame_path="a.png"),
                     row("b", "test", frame_path="b.png")])
    with pytest.raises(ValueError, match="Identical image"):
        audit_manifest(path)


def test_offline_preparation_three_returns_and_timestamp_reset(tmp_path):
    image = np.full((80, 100, 3), 100, dtype=np.uint8)
    frame_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(frame_path), image)
    rows = [row("v1:f", timestamp_s="10", frame_path=str(frame_path)),
            row("v1:f", timestamp_s="10", eye_side="right", frame_path=str(frame_path)),
            row("v2:f", timestamp_s="0", frame_path=str(frame_path))]

    class Detector:
        def __init__(self):
            self.times = []

        def invalidate(self):
            pass

        def process(self, frame):
            self.times.append(frame.t)
            if len(self.times) == 2:
                return [], "no_face", {}
            points = ((20, 20), (21, 19), (23, 19), (24, 20), (23, 21), (21, 21))
            return [Eye(frame.sample_id, frame.t, "track", "left", points,
                        image[:10, :20], True, True),
                    Eye(frame.sample_id, frame.t, "track", "right", points,
                        None, False, False, ("exposure",))], "invalid_eye", {}

    detector = Detector()
    target = tmp_path / "prepared.csv"
    result = prepare_rows(rows, detector, tmp_path / "crops", target)
    assert detector.times == [0, 2]  # original video time can go backwards
    assert result[0]["sample_id"] == "v1:f" and result[0]["timestamp_s"] == "10"
    assert json.loads(result[0]["points_json"])[0] == [20, 20]
    assert (target.parent / result[0]["crop_path"]).exists()
    assert result[1]["crop_path"] == "" and result[1]["invalid_reason"] == "exposure"
    assert result[2]["invalid_reason"] == "no_face" and result[2]["geometry_ok"] == "0"
    with pytest.raises(FileExistsError):
        prepare_rows(rows, detector, tmp_path / "crops", target)


def test_cnn_preprocessing_color_parity_and_fail_closed():
    from marvel.training.transforms import array_from_image, array_from_bgr
    from marvel.cognition.cnn_inference import CNN
    rgb = np.random.default_rng(3).integers(0, 256, (12, 20, 3), dtype=np.uint8)
    assert np.array_equal(array_from_image(Image.fromarray(rgb)), array_from_bgr(rgb[:, :, ::-1]))
    model = CNN.__new__(CNN)
    model.temperature = 1
    model.logit = lambda crop: float("nan")
    current = Eye("a", 0, "t", "left", (), rgb, True, True)
    assert model.estimate(current).reason == "nonfinite_logit"
    assert model.estimate(Eye("b", 0, "t", "right", (), None, True, True)).reason == "crop_format"
    with pytest.raises(ValueError, match="uint8"):
        array_from_bgr(rgb.astype(float))


def test_real_cpu_train_reload_evaluate(tmp_path):
    torch = pytest.importorskip("torch")
    from marvel.cognition.cnn_inference import CNN
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for label in (0, 1):
            name = f"{split}-{label}.png"
            # No real humans or claimed annotations: isolated software fixtures.
            Image.new("RGB", (64, 32), (30+label*170, 40+split_index, 50)).save(tmp_path / name)
            rows.append(row(name, split, frame_path=name, crop_path=name, label=str(label)))
    # One observable but rejected row is included in coverage, not classification.
    rows.append(row("missing", "test", frame_path="test-0.png", crop_path="", label="1"))
    manifest = tmp_path / "prepared.csv"
    write_csv(manifest, rows)
    checkpoint, report = tmp_path / "model.pt", tmp_path / "report.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for command in [
        ["train_cnn.py", str(manifest), str(checkpoint), "1", "cpu"],
        ["evaluate.py", str(manifest), str(checkpoint), "test", str(report), "cpu"],
    ]:
        run = subprocess.run([sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]],
                             cwd=tmp_path, capture_output=True, text=True, env=env, timeout=120)
        assert run.returncode == 0, run.stderr
    result = json.loads(report.read_text())
    assert result["n"] == 2 and result["coverage_observable"] == pytest.approx(2/3)
    assert result["threshold_status"] == "unvalidated_baseline"
    model = CNN(checkpoint)
    estimate = model.estimate(Eye("f", 0, "t", "left", (), np.zeros((32, 64, 3), np.uint8), True, True))
    assert estimate.valid and 0 <= estimate.value <= 1
    saved = torch.load(checkpoint, weights_only=True)
    saved["class_map"] = {"open": 1, "closed": 0}
    bad = tmp_path / "bad.pt"
    torch.save(saved, bad)
    with pytest.raises(ValueError, match="class mapping"):
        CNN(bad)
    for temperature in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Temperature"):
            CNN(checkpoint, temperature=temperature)
