"""Fit CNN temperature on calibration subjects; inspect validation errors.

Usage: python scripts/calibrate_cnn.py MANIFEST.csv CHECKPOINT.pt REPORT.json cpu
No test images are scored. Output paths must be new.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from marvel.cognition.cnn_inference import CNN, sha256
from marvel.training.calibration import (
    calibration_quality, choose_thresholds, fit_temperature, metrics, probabilities,
)
from marvel.training.dataset import EyeDataset, binary_rows, read_manifest


MRL_NAME = re.compile(r"s\d{4}_\d{5}_([01])_([01])_([01])_([012])_([01])_(0[123])\.png")
MRL_FIELDS = ("gender", "glasses", "source_eye_state", "reflections", "image_quality", "sensor")


def metadata(row):
    """Decode only fields documented in the inspected MRL annotation.txt."""
    if row["source"] != "MRL_Eye_2018_01":
        return {}
    match = MRL_NAME.fullmatch(Path(row["crop_path"]).name)
    if match is None or row["label"] != ("1" if match[3] == "0" else "0"):
        raise ValueError("MRL filename metadata or eye-state label mismatch")
    return dict(zip(MRL_FIELDS, match.groups()))


def score_rows(rows, model, device, split):
    import torch
    from torch.utils.data import DataLoader

    selected = binary_rows(rows, split)
    loader = DataLoader(EyeDataset(selected), batch_size=128, shuffle=False, num_workers=0)
    logits = []
    with torch.inference_mode():
        for batch_number, (images, _) in enumerate(loader, 1):
            output = model.model(images.to(device)).detach().cpu().numpy()
            if not np.all(np.isfinite(output)):
                raise ValueError("Nonfinite CNN output")
            logits.extend(float(value) for value in output)
            if batch_number % 40 == 0:
                print(f"{split}: scored {len(logits)}/{len(selected)} eyes", flush=True)
    if len(logits) != len(selected):
        raise RuntimeError("Scoring count does not match manifest")
    return selected, np.asarray(logits, dtype=np.float64), np.asarray(
        [int(row["label"]) for row in selected], dtype=np.int8)


def breakdown(rows, labels, scores, threshold):
    categories = {"subject": {}, "glasses": {}, "reflections": {},
                  "image_quality": {}, "sensor": {}}
    for index, row in enumerate(rows):
        values = metadata(row)
        for category in categories:
            value = row["subject_id"] if category == "subject" else values.get(category)
            if value is not None:
                categories[category].setdefault(value, []).append(index)
    return {category: {value: metrics(labels[indices], scores[indices], threshold)
                       for value, indices in sorted(groups.items())}
            for category, groups in categories.items() if groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("device")
    args = parser.parse_args()
    review_csv = Path(str(args.report) + ".false_positives.csv")
    if args.report.exists() or review_csv.exists():
        raise FileExistsError("Use a new report path; an output already exists")

    import torch
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest_hash = sha256(args.manifest)
    if saved.get("manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint was trained from a different manifest")
    model = CNN(args.checkpoint, device=args.device)
    all_rows = read_manifest(args.manifest)
    calibration, cal_logits, cal_labels = score_rows(all_rows, model, args.device, "calibration")
    validation, val_logits, val_labels = score_rows(all_rows, model, args.device, "validation")
    cal_subjects = {row["subject_id"] for row in calibration}
    val_subjects = {row["subject_id"] for row in validation}
    if cal_subjects & val_subjects:
        raise ValueError("Calibration and validation subjects overlap")

    temperature, fit = fit_temperature(cal_logits, cal_labels)
    cal_raw, val_raw = probabilities(cal_logits, 1.0), probabilities(val_logits, 1.0)
    cal_scores, val_scores = probabilities(cal_logits, temperature), probabilities(val_logits, temperature)
    candidates = {"raw_0_5": 0.5, "calibrated_0_5": 0.5}
    candidates.update(choose_thresholds(cal_labels, cal_scores))
    result = {
        "status": "development_analysis_not_frozen",
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": manifest_hash,
        "temperature": temperature,
        "temperature_fit": fit,
        "threshold_selection": "calibration subjects only; F1 maximum and <=5% open-eye FPR candidate",
        "validation_role": "checkpoint epoch was previously selected on validation; this is not a final unbiased test",
        "test_scored": False,
        "calibration_subjects": sorted(cal_subjects),
        "validation_subjects": sorted(val_subjects),
        "score_quality": {
            "calibration_raw": calibration_quality(cal_labels, cal_raw),
            "calibration_temperature": calibration_quality(cal_labels, cal_scores),
            "validation_raw": calibration_quality(val_labels, val_raw),
            "validation_temperature": calibration_quality(val_labels, val_scores),
        },
        "operating_points": {},
    }
    for name, threshold in candidates.items():
        cal_used, val_used = (cal_raw, val_raw) if name == "raw_0_5" else (cal_scores, val_scores)
        result["operating_points"][name] = {
            "threshold": threshold,
            "calibration": metrics(cal_labels, cal_used, threshold),
            "validation": metrics(val_labels, val_used, threshold),
            "validation_breakdown": breakdown(validation, val_labels, val_used, threshold),
        }

    # Keep paths and sample IDs in a local, ignored review queue, never in the shared report.
    fp_indices = np.flatnonzero((val_labels == 0) & (val_scores >= candidates["calibration_open_fpr_at_most_5pct"]))
    fp_indices = sorted(fp_indices, key=lambda index: (-val_scores[index], validation[index]["subject_id"]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with review_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "subject_id", "crop_path", "calibrated_closed_score",
                                                     "gender", "glasses", "source_eye_state", "reflections",
                                                     "image_quality", "sensor"))
        writer.writeheader()
        for index in fp_indices:
            row = validation[index]
            writer.writerow(dict(sample_id=row["sample_id"], subject_id=row["subject_id"],
                                 crop_path=row["crop_path"], calibrated_closed_score=val_scores[index],
                                 **metadata(row)))
    result["local_false_positive_review_csv"] = str(review_csv)
    result["local_false_positive_review_count"] = len(fp_indices)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"temperature": temperature,
                      "score_quality": result["score_quality"],
                      "operating_points": {name: {"threshold": item["threshold"],
                                                  "calibration": item["calibration"],
                                                  "validation": item["validation"]}
                                           for name, item in result["operating_points"].items()},
                      "false_positive_review_count": len(fp_indices)}, indent=2))


if __name__ == "__main__":
    main()
