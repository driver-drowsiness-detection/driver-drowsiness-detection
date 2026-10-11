"""Deterministic preprocessing shared by training and live inference."""

import numpy as np
from PIL import Image


PREPROCESS = {
    "width": 64,
    "height": 32,
    "channels": 1,
    "scale": "0_to_1",
    "resize": "PIL_BILINEAR",
    "crop": "width_1.6_height_0.8",
}


def array_from_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL eye crop into a float32 tensor-shaped array, ``(1, 32, 64)``."""
    image = image.convert("L").resize((PREPROCESS["width"], PREPROCESS["height"]),
                                       Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)[None, :, :] / 255.0


def array_from_bgr(crop: np.ndarray) -> np.ndarray:
    """Apply exactly the training path to an OpenCV BGR crop."""
    if not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("Expected a non-empty BGR image with three channels")
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        raise ValueError("Cannot preprocess an empty eye crop")
    return array_from_image(Image.fromarray(crop[:, :, ::-1].copy()))
