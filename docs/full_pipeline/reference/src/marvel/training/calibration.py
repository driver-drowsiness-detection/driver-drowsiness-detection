import csv
import math
from marvel.cognition.features import sigmoid


def scores(path, expected_split):
    from .dataset import read_manifest
    rows = read_manifest(path)
    if any(r["split"] != expected_split for r in rows):
        raise ValueError("Wrong partition for this operation")
    for field in ("checkpoint_sha256", "manifest_sha256"):
        if len({r[field] for r in rows}) != 1:
            raise ValueError("Mixed artifact identities")
    return rows


def paired(rows):
    result = [r for r in rows if r["observable"] == "1" and r["ear"] != "" and r["logit"] != ""]
    if {r["label"] for r in result} != {"0", "1"}:
        raise ValueError("Paired partition must contain both eye classes")
    if any(not math.isfinite(float(r[f])) for r in result for f in ("ear", "logit")):
        raise ValueError("Nonfinite training evidence")
    return result


def logloss(y, p):
    return -sum(a*math.log(max(1e-9, b))+(1-a)*math.log(max(1e-9, 1-b))
                for a, b in zip(y, p))/len(y)


def fit_ear(rows):
    # Projected gradient descent constrains the EAR slope to be negative.
    import numpy as np
    x = np.array([float(r["ear"]) for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    mean, scale = float(x.mean()), float(x.std())
    if scale < 1e-6:
        raise ValueError("EAR has no usable variation")
    z = (x-mean)/scale
    a, b = -1.0, 0.0
    for _ in range(5000):
        p = 1/(1+np.exp(-np.clip(a*z+b, -40, 40)))
        error = p-y
        a = min(-1e-6, a-0.05*float((error*z).mean()))
        b -= 0.05*float(error.mean())
    return {"slope": a/scale, "intercept": b-a*mean/scale}


def binary_metrics(y, pred):
    tp = sum(a == 1 and b == 1 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, pred))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
