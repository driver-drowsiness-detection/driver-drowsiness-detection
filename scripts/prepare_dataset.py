"""Generate versioned eye crops from independently labelled full-frame images.

Usage: python scripts/prepare_dataset.py annotations.csv detector.task crops prepared.csv
"""

import csv
import hashlib
import sys
from pathlib import Path

from marvel.contracts import Frame
from marvel.training.dataset import read_manifest


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 4:
        raise SystemExit("Usage: prepare_dataset.py ANNOTATIONS.csv DETECTOR.task CROP_DIR PREPARED.csv")
    source, task, folder, target = argv
    target_path = Path(target)
    if target_path.exists():
        raise FileExistsError(target_path)

    import cv2
    from marvel.cognition.ear_classifier import measure
    from marvel.perception.detection import Detector

    rows = read_manifest(source)
    crop_dir = Path(folder)
    crop_dir.mkdir(parents=True, exist_ok=False)
    quality = {
        "min_crop_pixels": 8,
        "brightness_min": 10,
        "brightness_max": 245,
        "laplacian_min": 0.0,
    }
    detector = Detector(task, [0, 0, 1, 1], quality)
    output: list[dict[str, str]] = []
    cache: dict[tuple[str, str, str], tuple[dict, str]] = {}
    try:
        for row in rows:
            key = (row["sample_id"], row["frame_path"], row["timestamp_s"])
            if key not in cache:
                image = cv2.imread(row["frame_path"])
                if image is None:
                    raise ValueError("Unreadable frame: " + row["frame_path"])
                detector.invalidate()
                eyes, _, _, reason = detector.process(
                    Frame(row["sample_id"], float(row["timestamp_s"]), image)
                )
                cache = {key: ({eye.side: eye for eye in eyes}, reason)}
            eyes, reason = cache[key]
            eye = eyes.get(row["eye_side"])
            result = dict(row, crop_path="", ear="", invalid_reason=reason)
            if eye is not None:
                ear = measure(eye)
                result["ear"] = str(ear.value) if ear.valid else ""
                result["invalid_reason"] = ",".join(eye.reasons)
                if eye.crop_ok:
                    name = hashlib.sha256(
                        (row["sample_id"] + ":" + eye.side).encode("utf-8")
                    ).hexdigest() + ".png"
                    crop_path = crop_dir / name
                    if not cv2.imwrite(str(crop_path), eye.crop):
                        raise RuntimeError("Crop write failed")
                    result["crop_path"] = str(crop_path)
            output.append(result)
    finally:
        detector.close()

    if not output:
        raise ValueError("No rows written")
    with target_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print({"rows": len(output), "usable_crops": sum(bool(row["crop_path"]) for row in output)})


if __name__ == "__main__":
    main()
