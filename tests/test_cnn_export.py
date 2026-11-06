"""Export and runtime parity are checked on real code, not mock logits."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from marvel.contracts import Eye


ROOT = Path(__file__).resolve().parents[1]


def test_export_and_opencv_parity_without_scoring_test(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from marvel.cognition.cnn_architecture import EyeCNN
    from marvel.cognition.cnn_inference import OpenCVCNN, sha256
    from marvel.training.transforms import PREPROCESS

    rows = []
    for split in ("validation", "test"):
        for label in (0, 1):
            name = f"{split}-{label}.png"
            if split == "validation":
                Image.new("RGB", (64, 32), (60 + 130 * label,) * 3).save(tmp_path / name)
            rows.append(dict(sample_id=name, subject_id=f"{split}:subject",
                             session_id=f"{split}:session", video_id=f"{split}:video",
                             source="synthetic_test", split=split, frame_path=name,
                             crop_path=name, timestamp_s="0", eye_side="left",
                             label=str(label), observable="1", label_source="human"))
    manifest = tmp_path / "eyes.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = tmp_path / "eye.pt"
    torch.save({"state_dict": EyeCNN().state_dict(),
                "class_map": {"open": 0, "closed": 1},
                "preprocess": PREPROCESS,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}, checkpoint)
    exported, report = tmp_path / "eye.onnx", tmp_path / "parity.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for script, arguments in (("export_onnx.py", [checkpoint, exported]),
                              ("check_onnx.py", [manifest, checkpoint, exported, report])):
        run = subprocess.run([sys.executable, str(ROOT / "scripts" / script),
                              *map(str, arguments)], cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr + run.stdout
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["passed"] and result["sample_count"] == 2
    assert result["test_scored"] is False
    assert result["decision_disagreements_at_logit_zero"] == 0
    assert result["maximum_absolute_logit_error"] < 1e-4

    crop = np.full((32, 64, 3), 60, dtype=np.uint8)
    model = OpenCVCNN(exported, sha256(checkpoint))
    eye = Eye("sample", 1.0, "track", "left", (), crop, True, True)
    assert model.estimate(eye).valid
    assert not model.estimate(Eye("missing", 2.0, "track", "right",
                                  (), None, True, False)).valid
    with pytest.raises(ValueError, match="identity mismatch"):
        OpenCVCNN(exported, "0" * 64)
