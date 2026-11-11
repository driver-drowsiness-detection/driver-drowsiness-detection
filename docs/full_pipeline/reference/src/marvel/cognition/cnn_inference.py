import json
import hashlib
from pathlib import Path
from marvel.contracts import Estimate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CNN:
    def __init__(self, checkpoint, temperature=1.0, device="cpu"):
        import torch
        from .cnn_architecture import EyeCNN
        from marvel.training.transforms import PREPROCESS
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.torch, self.device, self.temperature = torch, device, temperature
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["class_map"] != {"open": 0, "closed": 1} or saved["preprocess"] != PREPROCESS:
            raise ValueError("Checkpoint preprocessing or labels do not match")
        self.model = EyeCNN().to(device)
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()

    def logit(self, crop):
        from marvel.training.transforms import array_from_bgr
        x = self.torch.from_numpy(array_from_bgr(crop)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return float(self.model(x).item())

    def estimate(self, eye):
        from .features import sigmoid
        value = sigmoid(self.logit(eye.crop)/self.temperature) if eye.crop_ok else None
        return Estimate(eye.sample_id, eye.t, eye.track, eye.side, "cnn", value,
                        value is not None, "" if value is not None else "crop_quality")


def load_bundle(path):
    p = Path(path)
    bundle = json.loads(p.read_text(encoding="utf-8"))
    if bundle["status"] != "frozen" or bundle["schema"] != 1:
        raise ValueError("Bundle is not frozen")
    checkpoint = p.parent / bundle["checkpoint"]
    if sha256(checkpoint) != bundle["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    return bundle, checkpoint


class OpenCVCNN(CNN):
    def __init__(self, onnx_path, checkpoint_hash, temperature=1.0):
        import cv2
        from marvel.training.transforms import PREPROCESS
        metadata = json.loads(Path(str(onnx_path)+".json").read_text())
        if (metadata["checkpoint_sha256"] != checkpoint_hash or
                metadata["onnx_sha256"] != sha256(onnx_path) or metadata["preprocess"] != PREPROCESS):
            raise ValueError("ONNX artifact identity mismatch")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))

    def logit(self, crop):
        from marvel.training.transforms import array_from_bgr
        self.net.setInput(array_from_bgr(crop)[None, ...])
        return float(self.net.forward().reshape(-1)[0])
