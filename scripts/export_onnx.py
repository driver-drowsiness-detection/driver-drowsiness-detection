"""Export one EyeCNN checkpoint to ONNX opset 11 with identity metadata.

Usage: python scripts/export_onnx.py CHECKPOINT.pt OUTPUT.onnx
This is an export/parity candidate, not Jetson compatibility certification.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: export_onnx.py CHECKPOINT.pt OUTPUT.onnx")
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    metadata_path = Path(str(target) + ".json")
    if target.exists() or metadata_path.exists():
        raise FileExistsError("Use a new ONNX output path")

    import onnx
    import torch
    from marvel.cognition.cnn_inference import CNN, sha256
    from marvel.training.transforms import PREPROCESS

    model = CNN(source, device="cpu").model
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Legacy opset 11 is an explicit OpenCV/older-target candidate.
        # Actual Jetson parser support still requires board validation.
        torch.onnx.export(model, torch.zeros((1, 1, 32, 64)), str(target),
                          input_names=["eye"], output_names=["logit"],
                          opset_version=11, dynamo=False)
        graph = onnx.load(str(target))
        onnx.checker.check_model(graph)
        opset = next(item.version for item in graph.opset_import if item.domain == "")
        if opset != 11:
            raise ValueError(f"Unexpected ONNX opset {opset}")
        metadata = {
            "schema": 1, "status": "development_export_unvalidated_on_jetson",
            "checkpoint_sha256": sha256(source), "onnx_sha256": sha256(target),
            "preprocess": PREPROCESS, "opset": opset,
            "torch_version": torch.__version__, "onnx_version": onnx.__version__,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except BaseException:
        target.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
