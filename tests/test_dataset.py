import csv

import pytest

from marvel.training.dataset import binary_rows, read_manifest


FIELDS = [
    "sample_id", "subject_id", "session_id", "video_id", "source", "split",
    "frame_path", "timestamp_s", "eye_side", "label", "observable", "label_source",
]


def _row(sample_id="set:p1:s1:0001", side="left", **changes):
    row = {
        "sample_id": sample_id, "subject_id": "set:p1", "session_id": "set:p1:s1",
        "video_id": "set:p1:s1:v1", "source": "consented_collection", "split": "train",
        "frame_path": "data/frames/0001.png", "timestamp_s": "0.0", "eye_side": side,
        "label": "0", "observable": "1", "label_source": "human",
    }
    row.update(changes)
    return row


def _write_manifest(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_read_manifest_rejects_subject_leakage(tmp_path):
    path = _write_manifest(tmp_path, [
        _row(),
        _row("set:p1:s1:0002", split="test", frame_path="data/frames/0002.png"),
    ])
    with pytest.raises(ValueError, match="Group leakage"):
        read_manifest(path)


def test_read_manifest_rejects_ear_generated_labels(tmp_path):
    path = _write_manifest(tmp_path, [_row(label_source="ear_threshold")])
    with pytest.raises(ValueError, match="independent"):
        read_manifest(path)


def test_binary_rows_requires_two_labels_after_crop_filtering():
    rows = [_row(crop_path="left.png"), _row("set:p2:s2:0002", subject_id="set:p2",
                 session_id="set:p2:s2", video_id="set:p2:s2:v1", crop_path="right.png", label="1")]
    assert len(binary_rows(rows, "train")) == 2
