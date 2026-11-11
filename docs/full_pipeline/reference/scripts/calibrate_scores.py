"""python scripts/calibrate_scores.py calibration_scores.csv models/calibration.json"""
import json
import sys
from pathlib import Path
from marvel.training.calibration import scores, paired, fit_ear, logloss
from marvel.cognition.features import sigmoid


def main():
    source, target = sys.argv[1:]
    rows = scores(source, "calibration")
    used = paired(rows)
    y = [int(r["label"]) for r in used]
    temperatures = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    temperature = min(temperatures, key=lambda t: logloss(y, [sigmoid(float(r["logit"])/t) for r in used]))
    result = {"ear": fit_ear(used), "temperature": temperature,
              "checkpoint_sha256": rows[0]["checkpoint_sha256"],
              "manifest_sha256": rows[0]["manifest_sha256"],
              "paired_rows": len(used), "total_rows": len(rows),
              "subjects": sorted({r["subject_id"] for r in rows})}
    with open(target, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
