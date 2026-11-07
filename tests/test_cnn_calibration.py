"""Calibration uses only reserved subjects and preserves threshold tradeoffs."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from marvel.training.calibration import (
    calibration_quality, choose_thresholds, fit_temperature, metrics, probabilities,
)


ROOT = Path(__file__).resolve().parents[1]


def test_temperature_reduces_calibration_nll_without_changing_ranking():
    logits = np.array([4, 3, 2, 1, -1, -2, -3, -4], dtype=float)
    labels = np.array([1, 1, 0, 1, 0, 1, 0, 0])
    temperature, fit = fit_temperature(logits, labels)
    assert temperature > 1
    assert fit["calibrated_nll"] < fit["uncalibrated_nll"]
    assert np.array_equal(np.argsort(probabilities(logits, 1)),
                          np.argsort(probabilities(logits, temperature)))
    assert calibration_quality(labels, probabilities(logits, temperature))["brier"] >= 0


def test_threshold_candidates_keep_open_false_positives_within_calibration_limit():
    labels = [1, 1, 1, 0, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.85, 0.6, 0.4, 0.2, 0.1]
    choices = choose_thresholds(labels, scores)
    selected = metrics(labels, scores, choices["calibration_open_fpr_at_most_5pct"])
    assert selected["fp"] == 0
    assert selected["tp"] == 1
    high_recall = metrics(labels, scores, choices["calibration_closed_recall_at_least_95pct"])
    assert high_recall["recall_closed"] == 1
    assert metrics(labels, scores, choices["calibration_f1"])["f1_closed"] >= selected["f1_closed"]
    assert metrics([0, 1], [0.9, 0.1], 0.5)["fp"] == 1


def test_calibration_script_scores_no_test_images(tmp_path):
    torch = pytest.importorskip("torch")
    from marvel.cognition.cnn_architecture import EyeCNN
    from marvel.training.transforms import PREPROCESS

    rows = []
    for split in ("calibration", "validation", "test"):
        for label in (0, 1):
            name = f"{split}-{label}.png"
            if split != "test":
                Image.new("RGB", (64, 32), (30 + 170 * label,) * 3).save(tmp_path / name)
            rows.append(dict(sample_id=name, subject_id=f"{split}:person",
                             session_id=f"{split}:session", video_id=f"{split}:video",
                             source="synthetic_test", split=split, frame_path=name,
                             crop_path=name, timestamp_s="0", eye_side="left",
                             label=str(label), observable="1", label_source="human"))
    manifest = tmp_path / "eyes.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": EyeCNN().state_dict(),
                "class_map": {"open": 0, "closed": 1}, "preprocess": PREPROCESS,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}, checkpoint)
    report = tmp_path / "report.json"
    command = [sys.executable, str(ROOT / "scripts" / "calibrate_cnn.py"),
               str(manifest), str(checkpoint), str(report), "cpu"]
    run = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path,
                         env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"), timeout=90)
    assert run.returncode == 0, run.stderr
    result = json.loads(report.read_text())
    assert result["test_scored"] is False
    assert result["operating_points"]["calibrated_0_5"]["validation"]["n"] == 2
    assert len(result["calibration_subjects"]) == len(result["validation_subjects"]) == 1
    assert Path(str(report) + ".false_positives.csv").exists()
