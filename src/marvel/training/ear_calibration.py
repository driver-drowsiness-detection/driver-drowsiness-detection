"""Fit a decreasing EAR mapping on independently labelled calibration eyes."""
import json
import math
from pathlib import Path

import numpy as np

from marvel.cognition.features import aspect_ratio
from marvel.cognition.cnn_inference import sha256
from .dataset import read_manifest
from .calibration import probabilities, calibration_quality


def prepared_development_rows(manifest):
    rows = read_manifest(manifest)
    if any(r["split"] not in ("calibration", "validation") for r in rows):
        raise ValueError("Use a development manifest containing only calibration and validation; no train/test rows")
    if any(r["eye_side"] not in ("left", "right") or not r["timestamp_s"] for r in rows):
        raise ValueError("Full-frame anatomical eye identities and timestamps are required")
    if any(r.get("annotation_status") != "reviewed" or r.get("orientation_status") != "verified" for r in rows):
        raise ValueError("Reviewed annotations and verified anatomical orientation are required")
    metadata_path = Path(str(manifest) + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != 1 or metadata.get("geometry") != "six_original_pixel_points":
        raise ValueError("Expected prepared full-frame pixel geometry metadata")
    if metadata.get("rows") != len(rows):
        raise ValueError("Preparation metadata row count mismatch")
    # Check source files now: cross-partition copies must not bypass subject IDs.
    from .audit import audit_manifest
    audit = audit_manifest(manifest)
    for row in rows:
        if row.get("geometry_ok") not in ("0", "1"):
            raise ValueError("Explicit prepared geometry validity is required")
        row["ear_value"] = (aspect_ratio(json.loads(row["points_json"]))
                            if row["geometry_ok"] == "1" else None)
        if row["geometry_ok"] == "1" and row["ear_value"] is None:
            raise ValueError("Invalid geometry marked usable")
    return rows, dict(manifest_sha256=sha256(manifest),
                      preparation_metadata_sha256=sha256(metadata_path),
                      preparation=metadata, audit=audit)


def fit_decreasing(ears, labels, *, regularization=1e-3, max_iterations=200):
    """Projected Newton logistic fit with line search and standardized EAR.

    Constrain slope < 0. L2 regularization keeps separable fixtures finite.
    The sign constraint is an assumption, not evidence that EAR is predictive.
    """
    x, y = np.asarray(ears, dtype=float), np.asarray(labels, dtype=float)
    if x.ndim != 1 or x.size < 2 or x.shape != y.shape or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("Finite nonnegative EAR measurements with matching labels required")
    if set(y.tolist()) != {0., 1.}:
        raise ValueError("Both independent eye classes are required")
    if not math.isfinite(regularization) or regularization <= 0 or max_iterations <= 0:
        raise ValueError("Positive regularization and iteration limit required")
    mean, scale = float(x.mean()), float(x.std())
    if scale < 1e-8:
        raise ValueError("EAR has no usable variation")
    design = np.column_stack(((x - mean) / scale, np.ones_like(x)))
    beta = np.array([-1., math.log(y.mean() / (1 - y.mean()))])
    def objective(b):
        z = design @ b
        return float(np.mean(np.logaddexp(0, z) - y*z) + regularization * np.dot(b, b) / 2)
    converged = False
    for iteration in range(max_iterations):
        p = probabilities(design @ beta, 1)
        gradient = design.T @ (p - y) / len(y) + regularization * beta
        hessian = (design.T * (p * (1 - p))) @ design / len(y) + regularization * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        target = beta - step
        if target[0] > -1e-6:
            target[0] = -1e-6
            target[1] = beta[1] - (gradient[1] + hessian[1, 0] * (target[0] - beta[0])) / hessian[1, 1]
        old = objective(beta)
        rate = 1.
        for _ in range(60):
            candidate = beta + rate * (target - beta)
            if objective(candidate) <= old:
                break
            rate *= .5
        else:
            raise RuntimeError("EAR calibration line search failed")
        if np.max(np.abs(candidate - beta)) < 1e-8:
            converged = True
            beta = candidate
            break
        beta = candidate
    if not converged:
        raise RuntimeError("EAR calibration did not converge")
    slope = float(beta[0] / scale)
    mapping = dict(slope=slope, intercept=float(beta[1] - slope * mean))
    scores = probabilities(slope*x + mapping["intercept"], 1)
    return mapping, dict(iterations=iteration + 1, regularization=regularization,
                         standardized_ear_mean=mean, standardized_ear_scale=scale,
                         at_direction_boundary=bool(beta[0] >= -1.00001e-6),
                         calibration_quality=calibration_quality(y, scores))


def calibrate_manifest(manifest):
    rows, provenance = prepared_development_rows(manifest)
    selected = [r for r in rows if r["split"] == "calibration" and r["observable"] == "1"
                and r["ear_value"] is not None]
    mapping, diagnostics = fit_decreasing([r["ear_value"] for r in selected], [int(r["label"]) for r in selected])
    return dict(schema=1, status="experimental_not_frozen", **provenance,
                ear_calibration=mapping, fit=diagnostics,
                calibration_subjects=sorted({r["subject_id"] for r in rows if r["split"] == "calibration"}),
                calibration_labelled=sum(r["split"] == "calibration" and r["observable"] == "1" for r in rows),
                calibration_usable=len(selected), test_evaluated=False,
                validation_labels_used_for_fit=False)
