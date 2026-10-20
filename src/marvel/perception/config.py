"""Validate preview settings before opening a camera or loading a model."""
import math
from pathlib import Path

from .quality import validate_quality


def load_config(path, root):
    import yaml

    root = Path(root)
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    capture = config.get("capture", {})
    if capture.get("kind") not in ("camera", "gstreamer"):
        raise ValueError("capture.kind must be camera or gstreamer")
    source = capture.get("source")
    if capture["kind"] == "camera":
        if not isinstance(source, int) or isinstance(source, bool) or source < 0:
            raise ValueError("Camera source must be a nonnegative integer index")
    elif not isinstance(source, str) or "appsink" not in source:
        raise ValueError("GStreamer source must be a pipeline ending in appsink")
    for key in ("max_frame_age_s", "capture_timeout_s"):
        value = config.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(key + " must be finite and positive")
    roi = config.get("driver_roi", [])
    if (not isinstance(roi, list) or len(roi) != 4
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in roi)
            or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)):
        raise ValueError("driver_roi must be normalized [left, top, right, bottom]")
    validate_quality(config.get("quality"))
    model = config.get("landmarker")
    if not isinstance(model, str) or not model:
        raise ValueError("landmarker must name a model file")
    model = Path(model)
    config["landmarker"] = str(model if model.is_absolute() else root / model)
    return config
