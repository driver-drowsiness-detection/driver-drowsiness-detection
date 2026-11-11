"""python scripts/export_onnx.py models/eye.pt models/eye.onnx"""
import json
import sys
from pathlib import Path


def main():
    import torch
    import onnx
    from marvel.cognition.cnn_inference import CNN, sha256
    from marvel.training.transforms import PREPROCESS
    source, target = sys.argv[1:]
    if Path(target).exists():
        raise FileExistsError(target)
    model = CNN(source).model
    x = torch.zeros((1, 1, 32, 64))
    # Legacy exporter and opset are explicit compatibility candidates, not Jetson certification.
    torch.onnx.export(model, x, target, input_names=["eye"], output_names=["logit"],
                      opset_version=11, dynamo=False)
    onnx.checker.check_model(onnx.load(target))
    Path(target+".json").write_text(json.dumps({"checkpoint_sha256": sha256(source),
        "onnx_sha256": sha256(target), "preprocess": PREPROCESS, "opset": 11}, indent=2))


if __name__ == "__main__":
    main()
