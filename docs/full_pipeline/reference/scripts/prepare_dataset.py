"""python scripts/prepare_dataset.py annotations.csv detector.task crops prepared.csv"""
import csv
import hashlib
import sys
from pathlib import Path
from marvel.contracts import Frame
from marvel.training.dataset import read_manifest


def main():
    import cv2
    from marvel.perception.detection import Detector
    from marvel.cognition.ear_classifier import measure
    source, task, folder, target = sys.argv[1:]
    rows = read_manifest(source)
    Path(folder).mkdir(parents=True, exist_ok=True)
    quality = {"min_crop_pixels": 8, "brightness_min": 10, "brightness_max": 245,
               "laplacian_min": 0.0}
    detector = Detector(task, [0, 0, 1, 1], quality)
    output, cache = [], {}
    try:
        for row in rows:
            key = (row["sample_id"], row["frame_path"], row["timestamp_s"])
            if key not in cache:
                image = cv2.imread(row["frame_path"])
                if image is None:
                    raise ValueError("Unreadable frame: " + row["frame_path"])
                detector.invalidate()
                eyes, _, _, reason = detector.process(Frame(row["sample_id"], float(row["timestamp_s"]), image))
                cache = {key: ({e.side: e for e in eyes}, reason)}
            eyes, reason = cache[key]
            eye = eyes.get(row["eye_side"])
            result = dict(row, crop_path="", ear="", invalid_reason=reason)
            if eye:
                ear = measure(eye)
                result["ear"] = ear.value if ear.valid else ""
                result["invalid_reason"] = ",".join(eye.reasons)
                if eye.crop_ok:
                    name = hashlib.sha256((row["sample_id"]+":"+eye.side).encode()).hexdigest()+".png"
                    path = Path(folder)/name
                    if not cv2.imwrite(str(path), eye.crop):
                        raise RuntimeError("Crop write failed")
                    result["crop_path"] = str(path)
            output.append(result)
    finally:
        detector.close()
    with open(target, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print({"rows": len(output), "usable_crops": sum(bool(r["crop_path"]) for r in output)})


if __name__ == "__main__":
    main()
