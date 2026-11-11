"""python scripts/check_onnx.py prepared.csv models/eye.pt models/eye.onnx"""
import sys


def main():
    import cv2
    import numpy as np
    from marvel.training.dataset import read_manifest, binary_rows
    from marvel.training.transforms import array_from_bgr
    from marvel.cognition.cnn_inference import CNN
    manifest, checkpoint, onnx = sys.argv[1:]
    reference = CNN(checkpoint)
    net = cv2.dnn.readNetFromONNX(onnx)
    errors = []
    for row in binary_rows(read_manifest(manifest), "validation")[:100]:
        crop = cv2.imread(row["crop_path"])
        if crop is None:
            raise ValueError("Missing crop")
        net.setInput(array_from_bgr(crop)[None, ...])
        result = float(net.forward().reshape(-1)[0])
        expected = reference.logit(crop)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
        errors.append(abs(result-expected))
    print({"examples": len(errors), "max_logit_error": max(errors)})


if __name__ == "__main__":
    main()
