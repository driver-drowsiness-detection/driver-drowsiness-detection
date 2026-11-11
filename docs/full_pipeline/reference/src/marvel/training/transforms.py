import numpy as np
from PIL import Image

PREPROCESS = {"width": 64, "height": 32, "channels": 1,
              "scale": "0_to_1", "resize": "PIL_BILINEAR", "crop": "width_1.6_height_0.8"}


def array_from_image(image):
    image = image.convert("L").resize((64, 32), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)[None, :, :] / 255.0


def array_from_bgr(crop):
    return array_from_image(Image.fromarray(crop[:, :, ::-1].copy()))
