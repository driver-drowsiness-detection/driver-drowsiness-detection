"""Compare PyTorch and OpenCV ONNX logits on stratified validation eye crops.

Usage: python scripts/check_onnx.py MANIFEST.csv CHECKPOINT.pt MODEL.onnx REPORT.json
Test subjects are never scored by this script.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marvel.cognition.cnn_inference import CNN, OpenCVCNN, sha256
from marvel.training.dataset import binary_rows, read_manifest


def select_validation_rows(rows, maximum=100):
    """Spread a small deterministic parity sample across subject and eye label."""
    groups = defaultdict(list)
    for row in binary_rows(rows, "validation"):
        groups[(row["subject_id"], row["label"])].append(row)
    rng = random.Random(7)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < maximum and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < maximum:
                selected.append(groups[key].pop())
    return selected


def main():
    if len(sys.argv) != 5:
        raise SystemExit("Usage: check_onnx.py MANIFEST.csv CHECKPOINT.pt MODEL.onnx REPORT.json")
    manifest, checkpoint, onnx_path, report = map(Path, sys.argv[1:])
    if report.exists():
        raise FileExistsError(report)

    import cv2
    import torch
    metadata = json.loads(Path(str(onnx_path) + ".json").read_text(encoding="utf-8"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("manifest_sha256") != sha256(manifest):
        raise ValueError("Checkpoint and manifest do not match")
    reference = CNN(checkpoint, device="cpu")
    exported = OpenCVCNN(onnx_path, sha256(checkpoint))
    selected = select_validation_rows(read_manifest(manifest))
    if not selected:
        raise ValueError("No validation crops")
    errors, decision_disagreements, tolerance_failures = [], 0, 0
    counts = defaultdict(int)
    for row in selected:
        crop = cv2.imread(row["crop_path"])
        if crop is None:
            raise ValueError("Unreadable validation crop: " + row["crop_path"])
        expected, actual = reference.logit(crop), exported.logit(crop)
        if not math.isfinite(expected) or not math.isfinite(actual):
            raise ValueError("Nonfinite parity output")
        errors.append(abs(expected - actual))
        decision_disagreements += (expected >= 0) != (actual >= 0)
        tolerance_failures += not math.isclose(expected, actual, rel_tol=1e-4, abs_tol=1e-4)
        counts[(row["subject_id"], row["label"])] += 1
    result = {
        "status": "laptop_parity_only_not_jetson_validation",
        "manifest_sha256": sha256(manifest),
        "checkpoint_sha256": sha256(checkpoint),
        "onnx_sha256": sha256(onnx_path),
        "sample_count": len(selected), "selection_seed": 7,
        "split": "validation", "test_scored": False,
        "maximum_absolute_logit_error": max(errors),
        "median_absolute_logit_error": statistics.median(errors),
        "decision_disagreements_at_logit_zero": decision_disagreements,
        "logits_outside_tolerance": tolerance_failures,
        "tolerance": {"absolute": 1e-4, "relative": 1e-4},
        "subject_class_counts": {f"{subject}:{label}": count
                                 for (subject, label), count in sorted(counts.items())},
    }
    result["passed"] = decision_disagreements == 0 and tolerance_failures == 0
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("PyTorch/OpenCV ONNX logit parity failed; see report")


if __name__ == "__main__":
    main()
