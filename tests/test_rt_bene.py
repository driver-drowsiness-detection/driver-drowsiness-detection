"""Input-contract tests without downloading a checkpoint or dataset."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from marvel.training.rt_bene import RTBene, transform_bgr, transform_training_rgb, aligned_crops, reviewed_pairs


def test_transform_keeps_bgr_channel_order():
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    result = transform_bgr(image)
    assert result.shape == (3, 36, 60)
    assert torch.allclose(result[:, 0, 0], torch.tensor([(1-.485)/.229, -.456/.224, -.406/.225]))
    with pytest.raises(ValueError):
        transform_bgr(np.zeros((4, 4), dtype=np.uint8))
    rgb = transform_training_rgb(image)
    assert torch.allclose(rgb[:, 0, 0], torch.tensor([-.485/.229, -.456/.224, (1-.406)/.225]))


def eye(cx, cy):
    return np.array([(cx-10, cy), (cx-5, cy-3), (cx+5, cy-3),
                     (cx+10, cy), (cx+5, cy+3), (cx-5, cy+3)], dtype=float)


def test_alignment_preserves_anatomical_order_and_removes_roll():
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    left, right = eye(160, 90), eye(80, 70)
    cv2.circle(image, (160, 90), 8, (255, 0, 0), -1)
    cv2.circle(image, (80, 70), 8, (0, 0, 255), -1)
    lc, rc = aligned_crops(image, left, right)
    assert lc.shape == rc.shape == (36, 60, 3)
    assert lc[:, :, 0].sum() > lc[:, :, 2].sum()
    assert rc[:, :, 2].sum() > rc[:, :, 0].sum()
    with pytest.raises(ValueError, match="unmirrored"):
        aligned_crops(image, right, left)


def test_pair_filter_does_not_invent_uncertain_or_mixed_labels():
    rows = []
    for sample, labels in enumerate((("0", "0"), ("0", "1"), ("1", ""))):
        for side, label in zip(("left", "right"), labels):
            rows.append(dict(sample_id=str(sample), frame_path=str(sample), timestamp_s=str(sample),
                             subject_id="person", split="calibration", eye_side=side,
                             label=label, observable="1" if label else "0", geometry_ok="1"))
    pairs, excluded = reviewed_pairs(rows)
    assert len(pairs) == 1
    assert excluded == {"mixed_eye_labels": 1, "uncertain_label": 1}
    rows[1]["timestamp_s"] = "9"
    with pytest.raises(ValueError, match="identity"):
        reviewed_pairs(rows)


def test_reject_checkpoint_before_deserialization(tmp_path):
    checkpoint = tmp_path/"bad.model"
    checkpoint.write_bytes(b"not an official model")
    with pytest.raises(ValueError, match="official"):
        RTBene(checkpoint)


def test_unknown_predictions_are_not_counted_as_correct_open():
    path = Path(__file__).resolve().parents[1]/"scripts/evaluate_rt_bene.py"
    spec = importlib.util.spec_from_file_location("evaluate_rt_bene", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.state_metrics([0, 0, 1, 1], ["OPEN", "UNKNOWN", "CLOSED", "UNKNOWN"])
    assert result["coverage"] == .5
    assert result["closed_recall_all"] == .5
    assert result["correct_fraction_all"] == .5
    assert result["accuracy"] == 1  # Only on the explicitly covered subset.


def test_public_partition_duplicate_filter_removes_both_copies(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from train_rt_bene_pilot import reject_partition_duplicates
    clean, counts = reject_partition_duplicates({
        "train": [{"hash": "copied"}, {"hash": "unique_train"}],
        "calibration": [{"hash": "unique_cal"}],
        "validation": [{"hash": "copied"}, {"hash": "unique_val"}]})
    assert counts == {"train": 1, "calibration": 0, "validation": 1}
    assert clean["train"] == [{"hash": "unique_train"}]
    assert clean["validation"] == [{"hash": "unique_val"}]
