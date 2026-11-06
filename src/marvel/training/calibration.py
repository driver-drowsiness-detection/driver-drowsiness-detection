"""Calibration and operating-point analysis for independently labelled eyes."""

import math

import numpy as np


def probabilities(logits, temperature: float):
    """Stable sigmoid(logit / temperature) without clipping the model output."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    values = np.asarray(logits, dtype=np.float64) / temperature
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite logits")
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


def log_loss(logits, labels, temperature: float) -> float:
    values = np.asarray(logits, dtype=np.float64) / temperature
    targets = np.asarray(labels, dtype=np.float64)
    if values.size == 0 or values.shape != targets.shape or not np.all(np.isfinite(values)):
        raise ValueError("Finite logits and matching nonempty labels required")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Binary labels required")
    return float(np.mean(np.logaddexp(0, values) - targets * values))


def fit_temperature(logits, labels) -> tuple[float, dict]:
    """Minimize calibration-set binary NLL over log temperature in [0.05, 20]."""
    log_low, log_high = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = log_low, log_high
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = log_loss(logits, labels, math.exp(c)), log_loss(logits, labels, math.exp(d))
    for _ in range(90):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = log_loss(logits, labels, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = log_loss(logits, labels, math.exp(d))
    candidates = [(log_loss(logits, labels, math.exp(x)), math.exp(x))
                  for x in (log_low, (a + b) / 2, log_high)]
    loss, temperature = min(candidates)
    return temperature, {
        "uncalibrated_nll": log_loss(logits, labels, 1.0),
        "calibrated_nll": loss,
        "search_bounds": [0.05, 20.0],
        "at_search_boundary": temperature in (0.05, 20.0),
    }


def metrics(labels, scores, threshold: float) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or y.shape != p.shape or not np.all(np.isfinite(p)):
        raise ValueError("Finite scores and matching nonempty labels required")
    predicted = p >= threshold
    tp = int(np.sum((y == 1) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    open_count = tn + fp
    return {
        "n": len(y), "open": open_count, "closed": tp + fn,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(y),
        "precision_closed": precision, "recall_closed": recall,
        "f1_closed": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate_open": fp / open_count if open_count else None,
    }


def calibration_quality(labels, scores) -> dict:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or y.shape != p.shape:
        raise ValueError("Matching nonempty labels and scores required")
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    nll = -np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    brier = np.mean((p - y) ** 2)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if np.any(selected):
            ece += float(np.mean(selected) * abs(np.mean(p[selected]) - np.mean(y[selected])))
    return {"nll": float(nll), "brier": float(brier), "ece_10_bins": ece}


def choose_thresholds(labels, scores, maximum_open_fpr: float = 0.05) -> dict:
    """Select candidates on calibration eyes only; group tied scores together."""
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or y.shape != p.shape or not np.all(np.isfinite(p)):
        raise ValueError("Finite scores and matching nonempty labels required")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Threshold selection needs both eye classes")
    if not 0 <= maximum_open_fpr <= 1:
        raise ValueError("Maximum open-eye false-positive rate must be in [0, 1]")
    order = np.argsort(-p, kind="stable")
    sorted_y, sorted_p = y[order], p[order]
    open_count = int(np.sum(y == 0))
    total_closed = int(np.sum(y == 1))
    tp = fp = 0
    best_f1 = (0.0, float(np.nextafter(sorted_p[0], math.inf)))
    best_fpr = (0, float(np.nextafter(sorted_p[0], math.inf)))
    index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and sorted_p[end] == sorted_p[index]:
            end += 1
        group = sorted_y[index:end]
        tp += int(np.sum(group == 1))
        fp += int(np.sum(group == 0))
        threshold = float(sorted_p[index])
        f1 = 2 * tp / (tp + fp + total_closed)
        if f1 > best_f1[0]:
            best_f1 = (f1, threshold)
        if fp / open_count <= maximum_open_fpr and tp > best_fpr[0]:
            best_fpr = (tp, threshold)
        index = end
    return {"calibration_f1": best_f1[1],
            "calibration_open_fpr_at_most_5pct": best_fpr[1]}
