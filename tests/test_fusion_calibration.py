import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from marvel.training.ear_calibration import fit_decreasing, calibrate_manifest, prepared_development_rows
from marvel.training.fusion_analysis import analyze_validation


def fixture_manifest(folder):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, split in enumerate(("calibration", "validation")):
        for index, value in enumerate((.05, .1, .15, .18, .25, .3, .35, .4)):
            image = folder / f"{split}_{index}.png"
            Image.new("RGB", (64, 32), (40 + group*60 + index, 80, 100)).save(image)
            points = ((0, 0), (.5, value), (1.5, value), (2, 0), (1.5, -value), (.5, -value))
            rows.append(dict(sample_id=f"{split}:{index}", subject_id=f"{split}:person",
                session_id=f"{split}:session", video_id=f"{split}:video", source="synthetic_fixture",
                split=split, frame_path=image.name, timestamp_s=str(index), eye_side="left",
                label=str(int(value < .2)), observable="1", label_source="human",
                annotation_status="reviewed", orientation_status="verified", crop_path=image.name,
                geometry_ok="1", points_json=json.dumps(points), invalid_reason=""))
    path = folder / "prepared.csv"
    write_rows(path, rows)
    Path(str(path) + ".metadata.json").write_text(json.dumps(dict(schema=1, rows=len(rows),
        geometry="six_original_pixel_points", synthetic=True)), encoding="utf-8")
    return path, rows


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_monotonic_fit_finite_on_separable_data_and_boundary_diagnostic():
    x = [.05, .1, .3, .4]
    mapping, diagnostic = fit_decreasing(x, [1, 1, 0, 0])
    assert mapping["slope"] < 0 and np.isfinite(mapping["intercept"])
    assert diagnostic["calibration_quality"]["nll"] < .1
    reverse, diagnostic = fit_decreasing(x, [0, 0, 1, 1])
    assert reverse["slope"] < 0 and diagnostic["at_direction_boundary"]

def test_boundary_fit_still_fits_imbalanced_intercept():
    mapping, diagnostic = fit_decreasing([.1,.2,.3,.4,.5], [0,1,1,1,1])
    from marvel.training.calibration import probabilities
    score = probabilities(np.array([mapping["intercept"] + mapping["slope"]*.3]), 1)[0]
    assert diagnostic["at_direction_boundary"] and score == pytest.approx(.8, abs=.002)


@pytest.mark.parametrize("x,y", [([.2,.2],[0,1]), ([.1,.2],[1,1]),
    ([float('nan'),.2],[0,1]), ([-.1,.2],[0,1]), ([.1,.2],[0,2])])
def test_bad_calibration_inputs(x,y):
    with pytest.raises(ValueError): fit_decreasing(x,y)


def test_validation_labels_cannot_change_fitted_mapping(tmp_path):
    path, rows = fixture_manifest(tmp_path)
    first = calibrate_manifest(path)
    for row in rows:
        if row["split"] == "validation": row["label"] = str(1 - int(row["label"]))
    write_rows(path, rows)
    second = calibrate_manifest(path)
    assert first["ear_calibration"] == second["ear_calibration"]
    assert first["manifest_sha256"] != second["manifest_sha256"]


@pytest.mark.parametrize("problem", ["test", "leakage", "generated_label", "unreviewed", "duplicate_image"])
def test_manifest_rejects_leakage_test_and_unreviewed_labels(tmp_path, problem):
    path, rows = fixture_manifest(tmp_path)
    if problem == "test":
        for row in rows:
            if row["split"] == "validation": row["split"] = "test"
    elif problem == "leakage": rows[-1]["subject_id"] = rows[0]["subject_id"]
    elif problem == "generated_label": rows[0]["label_source"] = "ear"
    elif problem == "unreviewed": rows[0]["annotation_status"] = "needs_human_review"
    else:
        (tmp_path / rows[-1]["frame_path"]).write_bytes((tmp_path / rows[0]["frame_path"]).read_bytes())
    write_rows(path, rows)
    with pytest.raises(ValueError): prepared_development_rows(path)


def analysis_rows():
    return [dict(split="validation", subject_id="person", observable="1", label=str(y), ear_value=x)
            for x,y in ((.1,1),(.1,1),(.4,0),(.4,0))]


def analyze(rows, scores, **changes):
    settings = dict(alphas=[0,.5,1], thresholds=[.3,.5,.7], ear_threshold=.2,
                    cnn_threshold=.5, minimum_recall=.9)
    settings.update(changes)
    return analyze_validation(rows, scores, dict(slope=-20, intercept=4), **settings)


def test_missing_evidence_coverage_and_endpoints_not_forced_fusion_win():
    result = analyze(analysis_rows(), [.1,None,.9,.9])
    assert result["coverage"] == dict(total_rows=4, independently_labelled=4, ear=4, cnn=3, fusion=3, common=3)
    assert result["candidate"]["alpha"] == 0
    assert result["baselines"]["ear"]["own_observable"]["n"] == 4
    assert not result["runtime_settings_adopted"] and not result["test_evaluated"]


def test_infeasible_target_has_no_candidate():
    result = analyze(analysis_rows(), [.1,.1,.1,.1], thresholds=[1])
    assert result["candidate"] is None and result["feasible_candidates"] == 0


def test_no_common_classes_no_tuning():
    with pytest.raises(ValueError): analyze(analysis_rows(), [None,None,.1,.1])
    with pytest.raises(ValueError): analyze(analysis_rows(), [.1]*4, alphas=[.5])


def test_real_cli_with_controlled_checkpoint_and_provenance_guard(tmp_path):
    import torch
    from marvel.cognition.cnn_architecture import EyeCNN
    from marvel.training.transforms import PREPROCESS
    manifest, _ = fixture_manifest(tmp_path)
    model = EyeCNN()
    for parameter in model.parameters(): parameter.data.zero_()
    checkpoint = tmp_path / "synthetic_only.pt"
    saved = dict(state_dict=model.state_dict(), class_map={"open":0,"closed":1}, preprocess=PREPROCESS,
                 training_subjects=["train:person"], validation_subjects=["validation:person"], synthetic=True)
    torch.save(saved, checkpoint)
    root = Path(__file__).resolve().parents[1]
    fit, report = tmp_path / "fit.json", tmp_path / "analysis.json"
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def run(script, args):
        return subprocess.run([sys.executable, str(root / "scripts" / script), *map(str,args)],
                              capture_output=True, text=True, timeout=60, env=environment)
    result = run("calibrate_scores.py", [manifest, fit])
    assert result.returncode == 0, result.stderr
    first_fit_bytes = fit.read_bytes()
    assert run("calibrate_scores.py", [manifest, fit]).returncode != 0
    assert fit.read_bytes() == first_fit_bytes
    args = [manifest, fit, checkpoint, report, "--alphas", 0,.5,1, "--thresholds", .3,.5,.7,
            "--minimum-recall", .9, "--ear-threshold", .2, "--cnn-threshold", .5]
    result = run("tune_fusion.py", args)
    assert result.returncode == 0, result.stderr
    output = json.loads(report.read_text())
    assert output["candidate"] is not None and not output["event_accuracy_evaluated"]
    # A trained subject cannot quietly enter calibration or validation.
    saved["training_subjects"] = ["calibration:person"]
    torch.save(saved, checkpoint)
    args[3] = tmp_path / "must_not_exist.json"
    result = run("tune_fusion.py", args)
    assert result.returncode != 0 and "leakage" in result.stderr
    assert not args[3].exists()
