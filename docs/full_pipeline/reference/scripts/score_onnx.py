"""python scripts/score_onnx.py prepared.csv models/eye.onnx target_validation_scores.csv"""
import csv
import json
import sys
from pathlib import Path


def main():
    import cv2
    from marvel.training.dataset import read_manifest
    from marvel.cognition.cnn_inference import OpenCVCNN, sha256
    source, onnx_path, target = sys.argv[1:]
    metadata = json.loads(Path(onnx_path+".json").read_text())
    model = OpenCVCNN(onnx_path, metadata["checkpoint_sha256"])
    rows = [dict(r) for r in read_manifest(source) if r["split"] == "validation"]
    if not rows:
        raise ValueError("No validation samples")
    for r in rows:
        r["logit"] = ""
        r["checkpoint_sha256"] = metadata["checkpoint_sha256"]
        r["manifest_sha256"] = sha256(source)
        if r["crop_path"]:
            crop = cv2.imread(r["crop_path"])
            if crop is None:
                raise ValueError("Missing crop; preserve relative dataset paths across machines")
            r["logit"] = model.logit(crop)
    with open(target, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
