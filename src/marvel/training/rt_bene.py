"""Offline RT-BENE adapter. Deliberately not connected to live alerts.

Architecture and crop geometry adapted from Tobias-Fischer/rt_gene at
fa46a87e5f6fdc6f13be5d59572037c29ff9fc8b, by Cortacero, Fischer and Demiris.
SPDX-License-Identifier: CC-BY-NC-SA-4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
Changes: MediaPipe six-point input, explicit validation, offline batch scoring.
"""
from collections import defaultdict, Counter
import hashlib

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .rt_bene_model import BlinkEstimationModelResnet18

SOURCE_COMMIT = "fa46a87e5f6fdc6f13be5d59572037c29ff9fc8b"
MODEL_MD5 = "7c228fe7b95ce5960c4c5cae8f2d3a09"
MODEL_URL = "https://imperialcollegelondon.box.com/shared/static/p8jmekxhw4k8xtbz6vph3924g6ywnbre.model"


def transform_bgr(image):
    """Preserve BGR, exactly as the upstream OpenCV inference path does."""
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError("Nonempty uint8 BGR eye image required")
    image = cv2.resize(image, (60, 36), interpolation=cv2.INTER_CUBIC)
    return transforms.functional.normalize(transforms.functional.to_tensor(image),
                                           [.485, .456, .406], [.229, .224, .225])


def transform_training_rgb(image):
    """Diagnostic for the archived training path: PIL RGB, 224 then 60x36.

    The upstream HDF5 builder resized to 224 square before the validation loader.
    Keep real distinct eye inputs; do not replicate its apparent left/right typo.
    """
    transform_bgr(image)
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((224, 224), Image.Resampling.BICUBIC)
    rgb = rgb.resize((60, 36), Image.Resampling.BICUBIC)
    return transforms.functional.normalize(transforms.functional.to_tensor(rgb),
                                           [.485, .456, .406], [.229, .224, .225])


class RTBene:
    def __init__(self, checkpoint, device="cpu", preprocessing="upstream_bgr"):
        with open(checkpoint, "rb") as handle:
            digest = hashlib.file_digest(handle, "md5").hexdigest()
        if digest != MODEL_MD5:
            raise ValueError("Checkpoint does not match the official ResNet18 model 1")
        self.device = torch.device(device)
        self.transform = {"upstream_bgr": transform_bgr, "training_rgb": transform_training_rgb}[preprocessing]
        self.model = BlinkEstimationModelResnet18()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval()

    def predict(self, pairs, batch_size=32):
        if batch_size < 1:
            raise ValueError("Positive batch size required")
        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                left = torch.stack([self.transform(p[0]) for p in batch]).to(self.device)
                right = torch.stack([self.transform(p[1]) for p in batch]).to(self.device)
                values = torch.sigmoid(self.model(left, right)).flatten().cpu().numpy()
                if not np.all(np.isfinite(values)):
                    raise ValueError("Nonfinite RT-BENE output")
                scores.extend(values.astype(float).tolist())
        return scores


def reviewed_pairs(rows):
    """Same-frame anatomical pairs; never infer labels for uncertain/mixed eyes."""
    groups = defaultdict(dict)
    for row in rows:
        group = groups[row["sample_id"]]
        if row["eye_side"] in group:
            raise ValueError("Duplicate eye")
        group[row["eye_side"]] = row
    selected, excluded = [], Counter()
    for group in groups.values():
        if set(group) != {"left", "right"}:
            excluded["missing_eye"] += 1
            continue
        left, right = group["left"], group["right"]
        if any(left[k] != right[k] for k in ("frame_path", "timestamp_s", "subject_id", "split")):
            raise ValueError("Eye pair identity mismatch")
        if any(r["observable"] != "1" for r in (left, right)):
            excluded["uncertain_label"] += 1
        elif left["label"] != right["label"]:
            excluded["mixed_eye_labels"] += 1
        elif any(r.get("geometry_ok") != "1" for r in (left, right)):
            excluded["invalid_geometry"] += 1
        else:
            selected.append((left, right))
    return selected, dict(excluded)


def aligned_crops(frame, left_points, right_points):
    """Upstream roll alignment and 2 x eye-width crop with MediaPipe corners.

    Uses the full frame as alignment canvas instead of an upstream face box.
    This preserves the crop geometry but changes intermediate resampling scale.
    Left and right are anatomical, matching upstream landmarks 42/45 and 36/39.
    """
    transform_bgr(frame)  # validate before geometry or OpenCV slicing
    left, right = [np.asarray(p, dtype=float) for p in (left_points, right_points)]
    if any(p.shape != (6, 2) or not np.isfinite(p).all() for p in (left, right)):
        raise ValueError("Two finite six-point eyes required")
    corners = [p[[0, 3]][np.argsort(p[[0, 3], 0])] for p in (left, right)]
    lc, rc = [p.mean(axis=0) for p in corners]
    if lc[0] <= rc[0] or np.linalg.norm(lc - rc) < 1:
        raise ValueError("Expected unmirrored anatomical left eye on image right")
    h, w = frame.shape[:2]
    delta = rc - lc
    angle = np.degrees(np.arctan2(delta[1], delta[0])) - 180
    center = tuple((lc + rc) // 2)
    matrix = cv2.getRotationMatrix2D(center, float(angle), .3 * w / np.linalg.norm(delta))
    matrix[0, 2] += w * .5 - center[0]
    matrix[1, 2] += h * .35 - center[1]
    aligned = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_NEAREST)
    output = []
    for p in corners:
        p = np.column_stack((p, np.ones(2))) @ matrix.T
        width = p[1, 0] - p[0, 0]
        cy = p[:, 1].mean()
        box = np.array([p[0, 0] - width/2, cy - width*.6,
                        p[1, 0] + width/2, cy + width*.6], dtype=int)
        x1, y1, x2, y2 = box
        if width <= 0 or x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
            raise ValueError("Aligned eye crop outside frame")
        output.append(cv2.resize(aligned[y1:y2, x1:x2], (60, 36), interpolation=cv2.INTER_CUBIC))
    return tuple(output)
