"""Opt-in real model test: set MARVEL_TEST_MODEL to the downloaded .task file."""
import os
from pathlib import Path

import numpy as np
import pytest

from marvel.contracts import Frame
from marvel.perception.detection import Detector


@pytest.mark.skipif(not os.environ.get("MARVEL_TEST_MODEL"), reason="Explicit local model required")
def test_real_mediapipe_model_loads_detects_blank_image_and_closes():
    quality = dict(min_crop_pixels=8, brightness_min=10, brightness_max=245, laplacian_min=0)
    detector = Detector(Path(os.environ["MARVEL_TEST_MODEL"]), [0, 0, 1, 1], quality)
    try:
        result = detector.process(Frame("blank_smoke", 1.0, np.zeros((480, 640, 3), np.uint8)))
        assert result == ([], "no_face", {})
    finally:
        detector.close()
