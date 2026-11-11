# 06. Calibration, combination and model bundles

## Requirement

Combine evidence about the **same eye in the same frame** without averaging incompatible quantities or fitting against final-test labels.

Raw EAR might be 0.18 while CNN output is 0.85. These numbers have different meanings. Averaging them is not meaningful fusion.

## Prerequisites

A sigmoid maps a real number to the interval 0 to 1. Calibration fits the relationship between a model's score and observed class frequency. A well calibrated 0.8 score means that about 80 percent of comparable examples with such a score are positive. It does not mean “this one prediction is certainly correct.”

Calibration is not driver personalization. It also does not guarantee good performance under a new camera or lighting distribution.

## Partition discipline

| Partition | Allowed use |
| --- | --- |
| Train | CNN weight fitting |
| Calibration | EAR score mapping and CNN temperature |
| Validation | Checkpoint/architecture selection, EAR cutoff, alpha, score cutoffs and temporal policy |
| Test | One frozen evaluation |

Reusing validation for several choices is development, not an untouched performance estimate. Keep that distinction in the report. If development is extensive, use grouped nested validation or collect more held-out people.

## Step 1: produce logits on held-out data

```powershell
python scripts\score_dataset.py data\prepared_v1.csv models\eye_v1.pt calibration reports\calibration_scores.csv
python scripts\score_dataset.py data\prepared_v1.csv models\eye_v1.pt validation reports\validation_scores.csv
```

Every score row preserves the manifest fields, raw EAR, raw CNN logit and artifact hashes. Missing branch evidence remains blank. Scoring does not change the checkpoint.

## Step 2: fit EAR and CNN mappings

The reference fits:

\[
p_E=\sigma(a\,EAR+b),\qquad a<0
\]

It standardizes EAR for optimization, runs projected gradient descent, constrains the slope to remain negative, then converts coefficients back to raw EAR units. Standardization is fitted only on the calibration partition. The constraint expresses the intended geometry: more open eyes should not increase predicted closure.

Projection is the line `a = min(-1e-6, ...)`. After taking a gradient step, it clamps any nonnegative slope back into the allowed region. This is O(iterations times calibration samples), performed offline.

CNN temperature uses:

\[
p_C=\sigma(z/T),\qquad T>0.
\]

The reference searches a small declared temperature grid using calibration log loss. It is intentionally inspectable. A boundary winner is a reason to investigate fit and data, not to claim perfect calibration.

```powershell
python scripts\calibrate_scores.py reports\calibration_scores.csv models\calibration_v1.json
```

Only paired, observable calibration rows are used. The output reports selected row count and total count. This conditions the learned mapping on the usable paired subset; report that limitation and verify calibration by condition.

## Step 3: compare fusion approaches

| Approach | Formula/rule | Main advantage | Main drawback |
| --- | --- | --- | --- |
| Agreement | Both methods say closed | Simple | Can miss closure when one branch fails |
| Weighted calibrated scores | `alpha*p_C + (1-alpha)*p_E` | Interpretable and cheap | Correlated errors can persist |
| Learned stacking | Fit a classifier on held-out branch outputs | Can learn interactions | More data and careful out-of-fold fitting |

The reference implements weighted scores and keeps strict validity: both branches must be valid, even when alpha is 0 or 1. A single-branch fallback would be a separate declared policy with its own coverage comparison.

## Step 4: select thresholds and alpha

```powershell
python scripts\tune_fusion.py reports\validation_scores.csv models\calibration_v1.json models\eye_v1.pt models\bundle_v1.json
```

The script:

1. Rejects wrong partitions, overlapping calibration/validation subjects and mixed hashes.
2. Converts EAR and logits using frozen calibration.
3. Searches raw EAR threshold candidates between observed EAR values.
4. Searches CNN score cutoffs.
5. Searches alpha from 0 through 1 and fused score cutoffs.
6. Maximizes paired validation eye F1, preferring fewer false positives on ties.
7. Writes the selected configuration and provenance as a new frozen bundle.

This F1 objective is a transparent reference objective. It is not automatically the final project operating requirement. If you choose a minimum recall plus maximum false-alert constraint, implement that selection rule on development data and document the change before test evaluation.

For `p_E=0.7`, `p_C=0.9` and `alpha=0.5`, the result is 0.8. These are example values. A weighted result is a score; check calibration again before calling it a reliable probability.

## Fusion contract checks

`combine` requires equal sample ID, timestamp, continuity track and eye side. It checks that EAR has already been converted to a closure score, that alpha is in range, and that valid scores are finite and in range. One invalid branch returns unknown evidence. It does not silently copy the valid branch.

Combine each eye first. Then use the same bilateral policy for all three modes. Combining left EAR with right CNN would answer a different question and is rejected.

## Frozen bundle versus complete release

`bundle_v1.json` freezes eye-model weights, score mappings, alpha and eye thresholds. It does **not** freeze the whole system by itself. A complete release must also include:

1. Runtime YAML with temporal and quality settings.
2. Landmarker asset hash and crop policy.
3. Dataset and label versions.
4. Environment versions.
5. Optional ONNX metadata and target benchmark results.
6. Optional pose policy and LLM/TTS configuration.

The main program saves the bundle contents and configuration identity in the run log. Change any decision setting and create a new release/run; do not quietly overwrite the evidence of a previous test.

## Failure interpretation

If alpha selects 0 or 1, the experiment found no benefit from weighting both on the chosen objective. That is a legitimate result. If calibration has almost no paired examples, improve observability/data instead of forcing a fit. If CNN and EAR fail together on bad landmarks, changing alpha cannot restore absent information.

Used a decreasing logistic mapping because raw EAR has an inverse relationship to eye closure. Used held-out temperature because sigmoid scores can be poorly calibrated.

Interview explanation: “We aligned evidence by frame and eye, calibrated geometric and neural scores on held-out subjects, and selected fusion parameters only on validation data. Final-test labels were excluded from fitting.”

Retention exercise: describe the difference between increasing the closure threshold and increasing alpha. Which changes the operating point, and which changes the relative contribution of the CNN?

<!-- CODE -->

## Complete source files

### File: `src/marvel/cognition/fusion.py`

[Open the companion file](reference/src/marvel/cognition/fusion.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `combine`: lines 4 to 18.
* `state`: lines 21 to 24.

```python
from marvel.contracts import Estimate, OPEN, CLOSED, UNKNOWN


def combine(ear, cnn, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if (ear.sample_id, ear.t, ear.track, ear.side) != (
            cnn.sample_id, cnn.t, cnn.track, cnn.side):
        raise ValueError("Cannot fuse different observations")
    if ear.method != "ear_probability" or cnn.method != "cnn":
        raise ValueError("Fuse calibrated scores, not raw EAR")
    valid = ear.valid and cnn.valid
    for x in (ear, cnn):
        if x.valid and not 0 <= x.value <= 1:
            raise ValueError("Closure score outside [0, 1]")
    value = alpha * cnn.value + (1 - alpha) * ear.value if valid else None
    return Estimate(ear.sample_id, ear.t, ear.track, ear.side, "fusion", value,
                    valid, "" if valid else "missing_branch")


def state(estimate, threshold):
    if not estimate.valid:
        return UNKNOWN
    return CLOSED if estimate.value >= threshold else OPEN
```

### File: `src/marvel/training/calibration.py`

[Open the companion file](reference/src/marvel/training/calibration.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `scores`: lines 6 to 14.
* `paired`: lines 17 to 23.
* `logloss`: lines 26 to 28.
* `fit_ear`: lines 31 to 46.
* `binary_metrics`: lines 49 to 57.

```python
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
```

### File: `scripts/calibrate_scores.py`

[Open the companion file](reference/scripts/calibrate_scores.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 9 to 22.

```python
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
```

### File: `scripts/tune_fusion.py`

[Open the companion file](reference/scripts/tune_fusion.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 11 to 48.

```python
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
```

