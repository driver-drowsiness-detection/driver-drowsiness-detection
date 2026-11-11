"""python scripts/evaluate.py test_scores.csv models/bundle.json reports/image_metrics.json"""
import json
import sys
from pathlib import Path
from marvel.training.calibration import scores, binary_metrics, logloss
from marvel.cognition.features import sigmoid


def summarize(rows, bundle, mode, common=False):
    labelled = [r for r in rows if r["observable"] == "1"]
    used, y, probabilities, predictions = [], [], [], []
    for r in labelled:
        ev, cv = r["ear"] != "", r["logit"] != ""
        eligible = ev and cv if common or mode == "fusion" else ev if mode == "ear" else cv
        if not eligible:
            continue
        pe = sigmoid(bundle["ear_calibration"]["slope"]*float(r["ear"])+bundle["ear_calibration"]["intercept"]) if ev else None
        pc = sigmoid(float(r["logit"])/bundle["temperature"]) if cv else None
        p = pe if mode == "ear" else pc if mode == "cnn" else bundle["alpha"]*pc+(1-bundle["alpha"])*pe
        pred = (int(float(r["ear"]) < bundle["ear_threshold"]) if mode == "ear" else
                int(p >= bundle[mode+"_threshold"]))
        used.append(r)
        y.append(int(r["label"]))
        probabilities.append(p)
        predictions.append(pred)
    result = {"eligible": len(used), "labelled": len(labelled),
              "coverage": len(used)/len(labelled) if labelled else None,
              "metrics": binary_metrics(y, predictions) if used else None}
    if used:
        result["logloss"] = logloss(y, probabilities)
        result["brier"] = sum((a-b)**2 for a, b in zip(y, probabilities))/len(y)
        result["reliability_bins"] = []
        for i in range(10):
            values = [(a, p) for a, p in zip(y, probabilities) if i/10 <= p < (i+1)/10 or (i == 9 and p == 1)]
            result["reliability_bins"].append({"bin": i, "count": len(values),
                "mean_score": sum(p for _, p in values)/len(values) if values else None,
                "closed_fraction": sum(a for a, _ in values)/len(values) if values else None})
    return result


def main():
    source, bundle_path, target = sys.argv[1:]
    rows = scores(source, "test")
    bundle = json.loads(Path(bundle_path).read_text())
    if rows[0]["checkpoint_sha256"] != bundle["checkpoint_sha256"] or rows[0]["manifest_sha256"] != bundle["manifest_sha256"]:
        raise ValueError("Test artifacts differ from frozen bundle")
    if {r["subject_id"] for r in rows} & set(bundle["calibration_subjects"]+bundle["validation_subjects"]):
        raise ValueError("Test group leakage")
    output = {"all_observable": {}, "paired_observable": {}, "per_subject": {}}
    for mode in ("ear", "cnn", "fusion"):
        output["all_observable"][mode] = summarize(rows, bundle, mode)
        output["paired_observable"][mode] = summarize(rows, bundle, mode, True)
    for subject in sorted({r["subject_id"] for r in rows}):
        subset = [r for r in rows if r["subject_id"] == subject]
        output["per_subject"][subject] = {m: summarize(subset, bundle, m) for m in ("ear", "cnn", "fusion")}
    with open(target, "x", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
