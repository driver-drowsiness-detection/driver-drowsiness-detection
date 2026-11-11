import csv
from pathlib import Path

SPLITS = {"train", "calibration", "validation", "test"}
REQUIRED = {"sample_id", "subject_id", "session_id", "video_id", "source", "split",
            "frame_path", "timestamp_s", "eye_side", "label", "observable", "label_source"}


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError("Manifest missing: " + str(REQUIRED-set(reader.fieldnames or [])))
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    seen, ownership, frame_splits = set(), {}, {}
    for row in rows:
        key = (row["sample_id"], row["eye_side"])
        if key in seen or row["eye_side"] not in ("left", "right"):
            raise ValueError("Duplicate sample/eye or wrong eye side")
        seen.add(key)
        if row["split"] not in SPLITS or row["observable"] not in ("0", "1"):
            raise ValueError("Invalid split or observable flag")
        import math
        t = float(row["timestamp_s"])
        if not math.isfinite(t) or t < 0:
            raise ValueError("Invalid timestamp")
        if row["observable"] == "1" and row["label"] not in ("0", "1"):
            raise ValueError("Observable samples need independent 0/1 eye labels")
        if row["label_source"] not in ("human", "dataset_eye_annotation"):
            raise ValueError("EAR-generated labels are not independent ground truth")
        for field in ("source", "subject_id", "session_id", "video_id"):
            if not row[field]:
                raise ValueError("Missing grouping identity: " + field)
        # IDs must be globally namespaced, e.g. datasetA:person12:session2.
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


def binary_rows(rows, split, paired=False):
    selected = [r for r in rows if r["split"] == split and r["observable"] == "1"
                and r.get("crop_path") and (not paired or r.get("ear", "") != "")]
    if {r["label"] for r in selected} != {"0", "1"}:
        raise ValueError("Partition needs both labels after quality filtering: " + split)
    return selected


class EyeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        from .transforms import array_from_image
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            x = array_from_image(image)
        return torch.from_numpy(x), torch.tensor(float(row["label"]), dtype=torch.float32)
