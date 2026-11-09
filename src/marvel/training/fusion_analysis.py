"""Validation-only fusion analysis; never freezes or adopts runtime settings."""
import math
import numpy as np
from .calibration import metrics, probabilities, calibration_quality


def _summary(labels, scores, threshold):
    if not len(labels): return None
    result = metrics(labels, scores, threshold)
    if not result["closed"]: result["recall_closed"] = None
    return result


def analyze_validation(rows, cnn_scores, mapping, *, alphas, thresholds,
                       ear_threshold, cnn_threshold, minimum_recall):
    if len(rows) != len(cnn_scores) or any(r["split"] != "validation" for r in rows):
        raise ValueError("Matching validation rows and scores required")
    if not alphas or not thresholds or not {0, 1}.issubset(set(alphas)):
        raise ValueError("Alpha grid must include endpoints zero and one")
    for value in [*alphas, *thresholds, cnn_threshold, minimum_recall]:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Score settings must be finite and in [0, 1]")
    if minimum_recall <= 0 or not math.isfinite(ear_threshold) or ear_threshold <= 0:
        raise ValueError("Explicit positive recall target and EAR cutoff required")
    if not all(math.isfinite(mapping[k]) for k in ("slope", "intercept")) or mapping["slope"] >= 0:
        raise ValueError("A finite decreasing EAR mapping is required")
    for score in cnn_scores:
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid CNN score")
    labelled = [i for i, r in enumerate(rows) if r["observable"] == "1"]
    ear_ok = [i for i in labelled if rows[i]["ear_value"] is not None]
    cnn_ok = [i for i in labelled if cnn_scores[i] is not None]
    common = sorted(set(ear_ok) & set(cnn_ok))
    if {rows[i]["label"] for i in common} != {"0", "1"}:
        raise ValueError("Common observable validation subset needs both classes")
    labels = lambda indices: np.array([int(rows[i]["label"]) for i in indices])
    raw = lambda indices: np.array([rows[i]["ear_value"] for i in indices])
    cnn = lambda indices: np.array([cnn_scores[i] for i in indices])
    y = labels(common)
    ear_scores = probabilities(mapping["slope"] * raw(common) + mapping["intercept"], 1)
    cnn_values = cnn(common)
    candidates = []
    for alpha in sorted(set(alphas)):
        scores = alpha * cnn_values + (1-alpha) * ear_scores
        for threshold in sorted(set(thresholds)):
            candidates.append(dict(alpha=alpha, threshold=threshold, **metrics(y, scores, threshold)))
    feasible = [c for c in candidates if c["recall_closed"] >= minimum_recall]
    best = min(feasible, key=lambda c: (c["false_positive_rate_open"], -c["recall_closed"],
               -c["f1_closed"], c["alpha"], c["threshold"])) if feasible else None
    baselines = {
        "ear": dict(threshold=ear_threshold,
            common=_summary(y, (raw(common) < ear_threshold).astype(float), .5),
            own_observable=_summary(labels(ear_ok), (raw(ear_ok) < ear_threshold).astype(float), .5)),
        "cnn": dict(threshold=cnn_threshold,
            common=_summary(y, cnn_values, cnn_threshold),
            own_observable=_summary(labels(cnn_ok), cnn(cnn_ok), cnn_threshold)),
    }
    subjects = {}
    for subject in sorted({rows[i]["subject_id"] for i in common}):
        positions = [j for j, i in enumerate(common) if rows[i]["subject_id"] == subject]
        subjects[subject] = dict(
            ear=_summary(y[positions], (raw(common)[positions] < ear_threshold).astype(float), .5),
            cnn=_summary(y[positions], cnn_values[positions], cnn_threshold),
            fusion=_summary(y[positions], (best["alpha"]*cnn_values + (1-best["alpha"])*ear_scores)[positions], best["threshold"]) if best else None)
    return dict(status="validation_analysis_not_frozen", objective="minimize_open_fpr_subject_to_closed_recall",
        minimum_closed_recall=minimum_recall, candidate=best, feasible_candidates=len(feasible),
        grid=candidates, baselines=baselines, common_by_subject=subjects,
        coverage=dict(total_rows=len(rows), independently_labelled=len(labelled), ear=len(ear_ok),
                      cnn=len(cnn_ok), fusion=len(common), common=len(common)),
        common_score_quality=dict(mapped_ear=calibration_quality(y, ear_scores),
                                  cnn=calibration_quality(y, cnn_values)),
        test_evaluated=False, event_accuracy_evaluated=False, runtime_settings_adopted=False)
