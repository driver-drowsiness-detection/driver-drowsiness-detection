"""Checkpoint-verified CNN inference for current valid eye crops."""

import hashlib
import json
import math
from pathlib import Path

from marvel.contracts import Estimate


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sigmoid(value: float) -> float:
    """Numerically stable conversion from a logit to a closure probability."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class CNN:
    def __init__(self, checkpoint: str | Path, temperature: float = 1.0, device: str = "cpu"):
        import torch
        from .cnn_architecture import EyeCNN
        from marvel.training.transforms import PREPROCESS

        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("class_map") != {"open": 0, "closed": 1}:
            raise ValueError("Checkpoint class mapping does not match the runtime contract")
        if saved.get("preprocess") != PREPROCESS:
            raise ValueError("Checkpoint preprocessing does not match the runtime contract")
        self.torch, self.device, self.temperature = torch, device, temperature
        self.model = EyeCNN().to(device)
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()

    def logit(self, crop) -> float:
        from marvel.training.transforms import array_from_bgr

        x = self.torch.from_numpy(array_from_bgr(crop)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return float(self.model(x).item())

    def estimate(self, eye) -> Estimate:
        value = _sigmoid(self.logit(eye.crop) / self.temperature) if eye.crop_ok else None
        return Estimate(
            eye.sample_id, eye.t, eye.track, eye.side, "cnn", value,
            value is not None, "" if value is not None else "crop_quality",
        )


def load_bundle(path: str | Path) -> tuple[dict, Path]:
    """Load only a frozen bundle that names the exact checkpoint it calibrated."""
    bundle_path = Path(path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("status") != "frozen" or bundle.get("schema") != 1:
        raise ValueError("Bundle is not frozen")
    checkpoint = bundle_path.parent / bundle["checkpoint"]
    if sha256(checkpoint) != bundle.get("checkpoint_sha256"):
        raise ValueError("Checkpoint hash mismatch")
    return bundle, checkpoint


class OpenCVCNN(CNN):
    """ONNX/OpenCV-DNN variant for a deployment-compatible inference path."""

    def __init__(self, onnx_path: str | Path, checkpoint_hash: str, temperature: float = 1.0):
        import cv2
        from marvel.training.transforms import PREPROCESS

        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        onnx_path = Path(onnx_path)
        metadata = json.loads(Path(str(onnx_path) + ".json").read_text(encoding="utf-8"))
        if (metadata.get("checkpoint_sha256") != checkpoint_hash
                or metadata.get("onnx_sha256") != sha256(onnx_path)
                or metadata.get("preprocess") != PREPROCESS):
            raise ValueError("ONNX artifact identity mismatch")
        self.temperature = temperature
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))

    def logit(self, crop) -> float:
        from marvel.training.transforms import array_from_bgr

        self.net.setInput(array_from_bgr(crop)[None, ...])
        return float(self.net.forward().reshape(-1)[0])
