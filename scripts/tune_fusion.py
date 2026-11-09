"""Compare weighted fusion on reviewed validation eyes; output a candidate report."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marvel.cognition.cnn_inference import CNN, sha256
from marvel.training.ear_calibration import prepared_development_rows
from marvel.training.fusion_analysis import analyze_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument("ear_report", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--minimum-recall", type=float, required=True)
    parser.add_argument("--ear-threshold", type=float, required=True)
    parser.add_argument("--cnn-threshold", type=float, required=True)
    parser.add_argument("--cnn-temperature", type=float, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    rows, provenance = prepared_development_rows(args.prepared_manifest)
    fitted = json.loads(args.ear_report.read_text(encoding="utf-8"))
    if fitted.get("schema") != 1 or fitted.get("status") != "experimental_not_frozen":
        raise ValueError("Expected an experimental EAR calibration report")
    for key in ("manifest_sha256", "preparation_metadata_sha256"):
        if fitted.get(key) != provenance[key]:
            raise ValueError("EAR calibration provenance mismatch")
    import torch
    import cv2
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not saved.get("training_subjects") or "validation_subjects" not in saved:
        raise ValueError("Checkpoint needs training and validation subject provenance")
    if set(saved["training_subjects"]) & {r["subject_id"] for r in rows}:
        raise ValueError("CNN training subject leakage into development evaluation")
    if set(saved["validation_subjects"]) & set(fitted["calibration_subjects"]):
        raise ValueError("Calibration subjects were used for CNN checkpoint selection")
    model = CNN(args.checkpoint, temperature=args.cnn_temperature, device=args.device)
    validation = [r for r in rows if r["split"] == "validation"]
    scores = []
    from marvel.cognition.cnn_inference import _sigmoid
    for row in validation:
        if row["observable"] != "1" or not row.get("crop_path"):
            scores.append(None)
            continue
        crop = cv2.imread(row["crop_path"])
        if crop is None: raise ValueError("Unreadable prepared crop")
        scores.append(_sigmoid(model.logit(crop) / args.cnn_temperature))
    report = analyze_validation(validation, scores, fitted["ear_calibration"],
        alphas=args.alphas, thresholds=args.thresholds, minimum_recall=args.minimum_recall,
        ear_threshold=args.ear_threshold, cnn_threshold=args.cnn_threshold)
    report.update(schema=1, **provenance, checkpoint_sha256=sha256(args.checkpoint),
        ear_report_sha256=sha256(args.ear_report), cnn_temperature=args.cnn_temperature,
        validation_subjects=sorted({r["subject_id"] for r in validation}),
        ear_mapping_at_direction_boundary=fitted["fit"]["at_direction_boundary"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(candidate=report["candidate"], coverage=report["coverage"],
                         runtime_settings_adopted=False)))

if __name__ == "__main__":
    main()
