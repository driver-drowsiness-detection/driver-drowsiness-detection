"""Offline perception adapter. No EAR classifier or training framework import."""
import csv
import hashlib
import json
import os
from pathlib import Path

from marvel.contracts import Frame


def prepare_rows(rows, detector, crop_dir, target):
    """Retain invalid rows and geometry for independent EAR work.

    Detector timestamps represent offline iteration order only; original
    recording times are preserved in timestamp_s.
    """
    import cv2

    crop_dir, target = Path(crop_dir).resolve(), Path(target).resolve()
    if not rows:
        raise ValueError("Empty preparation input")
    if target.exists():
        raise FileExistsError(target)
    crop_dir.mkdir(parents=True, exist_ok=False)
    output, cache = [], {}
    for index, row in enumerate(rows):
        key = (row["sample_id"], row["frame_path"], row["timestamp_s"])
        if key not in cache:
            image = cv2.imread(row["frame_path"])
            if image is None:
                raise ValueError("Unreadable frame: " + row["frame_path"])
            detector.invalidate()
            eyes, status, _ = detector.process(Frame(f"offline:{index}", float(index), image))
            cache = {key: ({eye.side: eye for eye in eyes}, status)}
        eyes, status = cache[key]
        eye = eyes.get(row["eye_side"])
        result = dict(row, crop_path="", points_json="", geometry_ok="0",
                      invalid_reason=status if eye is None else ",".join(eye.reasons))
        result["frame_path"] = os.path.relpath(row["frame_path"], target.parent)
        if eye is not None:
            result["geometry_ok"] = str(int(eye.geometry_ok))
            if eye.geometry_ok:
                result["points_json"] = json.dumps(eye.points, allow_nan=False)
            if eye.crop_ok:
                name = hashlib.sha256((row["sample_id"] + ":" + eye.side).encode()).hexdigest()
                crop_path = crop_dir / (name + ".png")
                if not cv2.imwrite(str(crop_path), eye.crop):
                    raise RuntimeError("Crop write failed: " + str(crop_path))
                result["crop_path"] = os.path.relpath(crop_path, target.parent)
        output.append(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return output
