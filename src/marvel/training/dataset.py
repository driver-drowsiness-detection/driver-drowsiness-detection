"""Manifest validation and PyTorch dataset access for independently labelled eyes."""

import csv
import math
from pathlib import Path


SPLITS = {"train", "calibration", "validation", "test"}
REQUIRED = {
    "sample_id", "subject_id", "session_id", "video_id", "source", "split",
    "frame_path", "timestamp_s", "eye_side", "label", "observable", "label_source",
}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV and reject data leakage or non-independent labels early."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED - fieldnames
        if missing:
            raise ValueError("Manifest missing: " + str(missing))
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")

    seen: set[tuple[str, str]] = set()
    ownership: dict[tuple[str, str], str] = {}
    frame_splits: dict[str, str] = {}
    for row in rows:
        sample_eye = (row["sample_id"], row["eye_side"])
        if sample_eye in seen or row["eye_side"] not in ("left", "right"):
            raise ValueError("Duplicate sample/eye or wrong eye side")
        seen.add(sample_eye)
        if row["split"] not in SPLITS or row["observable"] not in ("0", "1"):
            raise ValueError("Invalid split or observable flag")
        try:
            timestamp = float(row["timestamp_s"])
        except ValueError as exc:
            raise ValueError("Invalid timestamp") from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid timestamp")
        if row["observable"] == "1" and row["label"] not in ("0", "1"):
            raise ValueError("Observable samples need independent 0/1 eye labels")
        if row["label_source"] not in ("human", "dataset_eye_annotation"):
            raise ValueError("EAR-generated labels are not independent ground truth")
        for field in ("source", "subject_id", "session_id", "video_id"):
            if not row[field]:
                raise ValueError("Missing grouping identity: " + field)
        for field in ("subject_id", "session_id", "video_id"):
            key = (field, row[field])
            if key in ownership and ownership[key] != row["split"]:
                raise ValueError("Group leakage: " + str(key))
            ownership[key] = row["split"]
        frame = str(Path(row["frame_path"]).resolve())
        if frame in frame_splits and frame_splits[frame] != row["split"]:
            raise ValueError("Same frame in different partitions")
        frame_splits[frame] = row["split"]
    return rows


def binary_rows(rows: list[dict[str, str]], split: str, paired: bool = False) -> list[dict[str, str]]:
    """Keep usable labelled crops and require both classes after filtering."""
    selected = [
        row for row in rows
        if row["split"] == split and row["observable"] == "1" and row.get("crop_path")
        and (not paired or row.get("ear", "") != "")
    ]
    if {row["label"] for row in selected} != {"0", "1"}:
        raise ValueError("Partition needs both labels after quality filtering: " + split)
    return selected


class EyeDataset:
    """Lazy image loader; tensors are produced only when a batch is requested."""

    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image
        from .transforms import array_from_image

        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            x = array_from_image(image)
        return torch.from_numpy(x), torch.tensor(float(row["label"]), dtype=torch.float32)
