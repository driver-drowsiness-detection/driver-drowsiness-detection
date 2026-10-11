import numpy as np
import pytest

from marvel.contracts import Eye


def test_preprocessing_has_expected_shape_and_range():
    pytest.importorskip("PIL")
    from marvel.training.transforms import array_from_bgr

    crop = np.full((10, 20, 3), 255, dtype=np.uint8)
    output = array_from_bgr(crop)
    assert output.shape == (1, 32, 64)
    assert output.dtype == np.float32
    assert float(output.min()) == 1.0
    assert float(output.max()) == 1.0


def test_invalid_crop_never_becomes_a_numeric_cnn_estimate(monkeypatch):
    torch = pytest.importorskip("torch")
    from marvel.cognition.cnn_architecture import EyeCNN
    from marvel.cognition.cnn_inference import CNN
    from marvel.training.transforms import PREPROCESS

    checkpoint = {"state_dict": EyeCNN().state_dict(), "class_map": {"open": 0, "closed": 1},
                  "preprocess": PREPROCESS}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    model = CNN("unused.pt")
    eye = Eye("f1", 1.0, "track1", "left", (), None, True, False, ("dark",))
    estimate = model.estimate(eye)
    assert not estimate.valid
    assert estimate.value is None
    assert estimate.reason == "crop_quality"
