import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"), "PyTorch/Pillow unavailable")
class CNNTests(unittest.TestCase):
    def test_forward_backward_checkpoint(self):
        import torch
        from marvel.cognition.cnn_architecture import EyeCNN
        from marvel.cognition.cnn_inference import CNN
        from marvel.training.transforms import PREPROCESS
        model = EyeCNN()
        x = torch.rand((4, 1, 32, 64))
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4,))
        loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0., 1., 0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"weights.pt"
            torch.save({"state_dict": model.state_dict(), "class_map": {"open": 0, "closed": 1},
                        "preprocess": PREPROCESS}, p)
            restored = CNN(p).model
            model.eval()
            torch.testing.assert_close(restored(x), model(x))


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("PIL"), "OpenCV/Pillow unavailable")
class ImageTests(unittest.TestCase):
    def test_saved_crop_preprocess_matches_live(self):
        import cv2
        import numpy as np
        from PIL import Image
        from marvel.training.transforms import array_from_image, array_from_bgr
        crop = np.random.default_rng(7).integers(0, 256, (25, 50, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"crop.png"
            self.assertTrue(cv2.imwrite(str(path), crop))
            with Image.open(path) as image:
                np.testing.assert_array_equal(array_from_image(image), array_from_bgr(crop))


if __name__ == "__main__":
    unittest.main()
