"""python scripts/tune_fusion.py validation_scores.csv calibration.json eye.pt bundle.json"""
import json
import os
import sys
from pathlib import Path
from marvel.training.calibration import scores, paired, binary_metrics
from marvel.cognition.features import sigmoid
from marvel.cognition.cnn_inference import sha256


def main():
    source, calibration, checkpoint, target = sys.argv[1:]
    rows = scores(source, "validation")
    used = paired(rows)
    cal = json.loads(Path(calibration).read_text())
    if set(cal["subjects"]) & {r["subject_id"] for r in rows}:
        raise ValueError("Calibration and validation subjects overlap")
    if cal["checkpoint_sha256"] != sha256(checkpoint) or rows[0]["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Checkpoint identity changed")
    if cal["manifest_sha256"] != rows[0]["manifest_sha256"]:
        raise ValueError("Manifest identity changed")
    y = [int(r["label"]) for r in used]
    ear = [float(r["ear"]) for r in used]
    pe = [sigmoid(cal["ear"]["slope"]*x+cal["ear"]["intercept"]) for x in ear]
    pc = [sigmoid(float(r["logit"])/cal["temperature"]) for r in used]
    thresholds = [i/20 for i in range(1, 20)]
    def quality(pred):
        m = binary_metrics(y, pred)
        return (m["f1"], -m["fp"])
    unique = sorted(set(ear))
    ear_choices = [unique[0]-1e-6]+[(a+b)/2 for a, b in zip(unique, unique[1:])]+[unique[-1]+1e-6]
    ear_t = max(ear_choices, key=lambda t: quality([int(x < t) for x in ear]))
    cnn_t = max(thresholds, key=lambda t: quality([int(x >= t) for x in pc]))
    choices = [(a/10, t) for a in range(11) for t in thresholds]
    alpha, fused_t = max(choices, key=lambda at: quality(
        [int(at[0]*c+(1-at[0])*e >= at[1]) for c, e in zip(pc, pe)]))
    bundle = {"schema": 1, "status": "frozen", "checkpoint": os.path.relpath(checkpoint, Path(target).parent),
              "checkpoint_sha256": sha256(checkpoint), "manifest_sha256": cal["manifest_sha256"],
              "ear_calibration": cal["ear"], "temperature": cal["temperature"],
              "ear_threshold": ear_t, "cnn_threshold": cnn_t,
              "alpha": alpha, "fusion_threshold": fused_t,
              "selection": "paired validation eye F1, fewer false positives on ties",
              "calibration_subjects": cal["subjects"],
              "validation_subjects": sorted({r["subject_id"] for r in rows}),
              "temporal_policy": "configure and freeze separately using development videos"}
    with open(target, "x", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print({"paired_validation_rows": len(used), "alpha": alpha})


if __name__ == "__main__":
    main()
