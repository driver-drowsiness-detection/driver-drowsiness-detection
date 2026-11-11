"""python scripts/score_dataset.py prepared.csv models/eye.pt calibration scores.csv"""
import csv
import sys
from marvel.training.dataset import read_manifest


def main():
    import cv2
    from marvel.cognition.cnn_inference import CNN, sha256
    manifest, checkpoint, split, target = sys.argv[1:]
    if split not in ("calibration", "validation", "test"):
        raise ValueError("Never calibrate on training predictions")
    model = CNN(checkpoint)
    rows = [dict(r) for r in read_manifest(manifest) if r["split"] == split]
    if not rows:
        raise ValueError("Empty split")
    for row in rows:
        row["logit"] = ""
        row["checkpoint_sha256"] = sha256(checkpoint)
        row["manifest_sha256"] = sha256(manifest)
        if row["crop_path"]:
            crop = cv2.imread(row["crop_path"])
            if crop is None:
                raise ValueError("Missing prepared crop")
            row["logit"] = model.logit(crop)
    with open(target, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
