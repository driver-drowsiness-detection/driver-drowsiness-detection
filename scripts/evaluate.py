"""Evaluate a CNN checkpoint on a frozen labelled partition.

Usage: python scripts/evaluate.py prepared.csv models/eye.pt test reports/cnn_test.json cpu
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marvel.training.dataset import binary_rows, read_manifest


def _metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    true_positive = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    true_negative = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    false_positive = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    false_negative = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "n": len(labels), "tp": true_positive, "tn": true_negative,
        "fp": false_positive, "fn": false_negative,
        "accuracy": (true_positive + true_negative) / len(labels),
        "precision_closed": precision,
        "recall_closed": recall,
        "f1_closed": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 5:
        raise SystemExit("Usage: evaluate.py PREPARED.csv CHECKPOINT.pt SPLIT REPORT.json DEVICE")
    manifest, checkpoint, split, report, device = argv
    if Path(report).exists():
        raise FileExistsError(report)
    if split not in {"validation", "test"}:
        raise ValueError("Evaluate only a frozen validation or test partition")

    import cv2
    from marvel.cognition.cnn_inference import CNN, sha256, _sigmoid

    partition = [r for r in read_manifest(manifest) if r["split"] == split]
    rows = binary_rows(partition, split)
    model = CNN(checkpoint, device=device)
    labels, predictions, scores = [], [], []
    for row in rows:
        crop = cv2.imread(row["crop_path"])
        if crop is None:
            raise ValueError("Unreadable prepared crop: " + row["crop_path"])
        logit = model.logit(crop)
        score = _sigmoid(logit)
        labels.append(int(row["label"]))
        predictions.append(int(score >= 0.5))
        scores.append(score)

    result = _metrics(labels, predictions)
    result.update({
        "split": split,
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest": str(Path(manifest)),
        "manifest_sha256": sha256(manifest),
        "threshold": 0.5,
        "threshold_status": "unvalidated_baseline",
        "total_rows": len(partition),
        "observable_rows": sum(r["observable"] == "1" for r in partition),
        "coverage_observable": len(rows) / sum(r["observable"] == "1" for r in partition),
        "mean_closed_score": sum(scores) / len(scores),
    })
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
