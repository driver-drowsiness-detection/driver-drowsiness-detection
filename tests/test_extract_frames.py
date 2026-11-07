"""Video extraction creates review work, never ready-made CNN labels."""

import csv
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from marvel.training.dataset import read_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_extract_frames_requires_human_review(tmp_path):
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG video writer unavailable in this OpenCV environment")
    for index in range(5):
        writer.write(np.full((48, 64, 3), index * 35, dtype=np.uint8))
    writer.release()
    frames, queue = tmp_path / "frames", tmp_path / "review.csv"
    command = [sys.executable, str(ROOT / "scripts" / "extract_frames.py"),
               str(video), str(frames), str(queue), "--subject-id", "local:p1",
               "--session-id", "local:p1:s1", "--video-id", "local:p1:s1:v1",
               "--source", "test_video", "--step", "2"]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    with queue.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["eye_side"] for row in rows} == {"left", "right"}
    assert all(row["label"] == row["observable"] == row["split"] == "" for row in rows)
    assert all(row["annotation_status"] == "needs_human_review" for row in rows)
    assert rows[-1]["timestamp_s"] == "0.400000"
    assert all((queue.parent / row["frame_path"]).is_file() for row in rows)
    with pytest.raises(ValueError):
        read_manifest(queue)  # blank labels and split cannot be used to train
    for row in rows:
        row.update(split="train", label="0" if row["eye_side"] == "left" else "1",
                   observable="1", label_source="human")
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="human review"):
        read_manifest(queue)
    for row in rows:
        row.update(annotation_status="reviewed", orientation_status="verified")
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert len(read_manifest(queue)) == 6
    second = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert second.returncode != 0 and "FileExistsError" in second.stderr
