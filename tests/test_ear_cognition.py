import math
from dataclasses import replace

import pytest

from marvel.contracts import Eye, Estimate, UNKNOWN, OPEN, CLOSED, bilateral
from marvel.cognition.features import aspect_ratio, sigmoid
from marvel.cognition.ear_classifier import measure, state, probability

POINTS = ((0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1))


def eye(**changes):
    return replace(Eye("f1", 1.0, "track_1", "left", POINTS, None, True, False), **changes)


def test_hand_calculated_geometry_and_closed_eye():
    assert aspect_ratio(POINTS) == pytest.approx(0.5)
    assert aspect_ratio(tuple((x, 0) for x, y in POINTS)) == 0


def test_uniform_scale_translation_and_rotation():
    transformed = tuple((-3*y+10, 3*x+20) for x, y in POINTS)
    assert aspect_ratio(transformed) == pytest.approx(0.5)


@pytest.mark.parametrize("points", [None, (), [(0, 0)]*6,
    ((0, 0),)*5 + ((float("nan"), 0),), [(1,)]*6, [("a", 0)]*6])
def test_malformed_geometry_is_missing(points):
    assert aspect_ratio(points) is None


def test_estimate_identity_quality_and_threshold_boundary():
    estimate = measure(eye())
    assert (estimate.sample_id, estimate.t, estimate.track, estimate.side) == ("f1", 1, "track_1", "left")
    assert state(estimate, .5) == OPEN
    assert state(estimate, .6) == CLOSED
    invalid = measure(eye(geometry_ok=False, reasons=("exposure",)))
    assert invalid.value is None and invalid.reason == "exposure"
    assert state(invalid, .2) == UNKNOWN
    assert bilateral(CLOSED, UNKNOWN) == UNKNOWN


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_bad_thresholds_rejected(value):
    with pytest.raises(ValueError):
        state(measure(eye()), value)


def test_calibrated_mapping_monotonic_and_missing():
    calibration = {"slope": -10, "intercept": 2}
    opened = probability(measure(eye()), calibration)
    closed = probability(measure(eye(points=tuple((x, 0) for x, y in POINTS))), calibration)
    assert opened.value < closed.value
    assert probability(measure(eye(geometry_ok=False)), calibration).value is None
    assert sigmoid(-1000) == 0 and sigmoid(1000) == 1
    with pytest.raises(ValueError):
        probability(measure(eye()), {"slope": 1, "intercept": 2})
    with pytest.raises(ValueError):
        state(Estimate("f", 1, "t", "left", "cnn", .5, True), .2)


def test_saved_image_cli_no_face(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import cv2
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    if not os.environ.get("MARVEL_TEST_MODEL"):
        pytest.skip("Set MARVEL_TEST_MODEL to opt into real MediaPipe execution")
    image = tmp_path / "blank.png"
    assert cv2.imwrite(str(image), np.zeros((128, 128, 3), np.uint8))
    config = tmp_path / "runtime.yaml"
    import yaml
    settings = yaml.safe_load((root / "configs/runtime.yaml").read_text())
    settings["landmarker"] = str(Path(os.environ["MARVEL_TEST_MODEL"]).resolve())
    config.write_text(yaml.safe_dump(settings))
    result = subprocess.run(
        [sys.executable, str(root / "scripts/inspect_ear.py"), str(image),
         "--config", str(config)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["face_status"] == "no_face"
    assert output["bilateral"] == UNKNOWN and output["eyes"] == []
