"""Bounded public-data pilot with separate subjects and ImageNet initialization.

Does not load the RT-BENE all-subjects checkpoint. Camera labels never enter fit.
Uses the RT-BENE architecture under CC BY-NC-SA 4.0; see rt_bene_model.py.
"""
import csv
import hashlib
import json
from pathlib import Path
import random
import tarfile
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights

from marvel.training.rt_bene_model import BlinkEstimationModelResnet18
from marvel.training.ear_calibration import prepared_development_rows
from marvel.training.rt_bene import aligned_crops, reviewed_pairs
from marvel.training.calibration import metrics, choose_thresholds
from marvel.contracts import OPEN, CLOSED, UNKNOWN, bilateral
from download_rt_bene_pilot import FILES
from evaluate_rt_bene import digest, state_metrics

SPLITS = {"train": [3, 7, 12], "calibration": [15], "validation": [16]}
SEED = 20261110


def read_subject(root, subject):
    prefix = f"s{subject:03d}"
    records = []
    excluded = 0
    for name in (prefix + "_blink_labels.csv", prefix + "_noglasses_eyes.tar"):
        if digest(root/name, "md5") != FILES[name]:
            raise ValueError("Checksum mismatch: " + name)
    with (root/(prefix + "_blink_labels.csv")).open(newline="") as handle:
        labels = list(csv.reader(handle))
    with tarfile.open(root/(prefix + "_noglasses_eyes.tar")) as archive:
        members = {m.name.removeprefix("./"): m for m in archive.getmembers() if m.isfile()}
        for name, label in labels:
            if label == "0.5":
                excluded += 1
                continue
            if label not in ("0.0", "1.0") or not name.startswith("left_"):
                raise ValueError("Unexpected public annotation")
            eyes = []
            for side in ("left", "right"):
                path = f"{prefix}_noglasses/natural/{side}/{name.replace('left_', side + '_', 1)}"
                raw = archive.extractfile(members[path]).read()
                bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Unreadable eye: " + path)
                eyes.append(cv2.cvtColor(cv2.resize(bgr, (60, 36), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
            images = np.stack(eyes)
            records.append(dict(subject=subject, name=name, label=int(float(label)), images=images,
                                hash=hashlib.sha256(images.tobytes()).hexdigest()))
    return records, excluded


def reject_partition_duplicates(partitions):
    owners = {}
    for split, rows in partitions.items():
        for row in rows:
            owners.setdefault(row["hash"], set()).add(split)
    duplicates = {key for key, values in owners.items() if len(values) > 1}
    clean = {split: [r for r in rows if r["hash"] not in duplicates] for split, rows in partitions.items()}
    return clean, {split: len(partitions[split])-len(rows) for split, rows in clean.items()}


def loader(rows, shuffle=False):
    images = torch.from_numpy(np.stack([r["images"] for r in rows])).permute(0, 1, 4, 2, 3)
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
    return DataLoader(TensorDataset(images, labels), batch_size=64, shuffle=shuffle, num_workers=0)


def forward(model, images, device):
    images = images.to(device).float()/255
    mean = images.new_tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
    std = images.new_tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
    images = (images-mean)/std
    return model(images[:, 0], images[:, 1]).flatten()


def evaluate(model, batches, device):
    model.eval()
    scores, labels, losses = [], [], []
    with torch.inference_mode():
        for images, y in batches:
            z = forward(model, images, device)
            losses.extend(nn.functional.binary_cross_entropy_with_logits(z, y.to(device), reduction="none").cpu().tolist())
            scores.extend(torch.sigmoid(z).cpu().tolist())
            labels.extend(y.int().tolist())
    return labels, scores, float(np.mean(losses))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("data/raw/rt_bene")
    output = Path("reports/local/rt_bene_public_pilot_2026_11_10")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path("models/checkpoints/rt_bene/public_pilot_imagenet_2026_11_10.pt")
    if checkpoint.exists():
        raise FileExistsError("Preserve the existing experiment; choose a new run name")
    partitions, exclusions = {}, {}
    for split, subjects in SPLITS.items():
        partitions[split] = []
        for subject in subjects:
            rows, excluded = read_subject(root, subject)
            partitions[split].extend(rows)
            exclusions[str(subject)] = excluded
    partitions, duplicates = reject_partition_duplicates(partitions)
    counts = {split: dict(n=len(rows), closed=sum(r["label"] for r in rows), subjects=SPLITS[split])
              for split, rows in partitions.items()}
    if any(not 0 < c["closed"] < c["n"] for c in counts.values()):
        raise ValueError("Every split must contain both classes")
    print(json.dumps({"counts": counts, "duplicates": duplicates}), flush=True)
    batches = {split: loader(rows, split == "train") for split, rows in partitions.items()}
    model = BlinkEstimationModelResnet18()
    # Initialization is ImageNet only, never the RT-BENE all-subjects weights.
    pretrained = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(pretrained.children())[:-1])
    for features in (model.left_features, model.right_features):
        features.load_state_dict(backbone.state_dict(), strict=True)
        for parameter in features.parameters():
            parameter.requires_grad = False
        for parameter in features[7].parameters():
            parameter.requires_grad = True
    del pretrained, backbone
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    positive_weight = (counts["train"]["n"]-counts["train"]["closed"])/counts["train"]["closed"]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    best_loss, best_epoch, history = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(1, 6):
        model.train()
        for features in (model.left_features, model.right_features):
            features.eval()  # Freeze early BatchNorm statistics too.
            features[7].train()
        losses = []
        for images, y in batches["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward(model, images, device), y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        labels, scores, calibration_loss = evaluate(model, batches["calibration"], device)
        row = dict(epoch=epoch, train_weighted_loss=float(np.mean(losses)), calibration_nll=calibration_loss,
                   calibration_fixed=metrics(labels, scores, .5))
        history.append(row)
        if calibration_loss < best_loss:
            best_loss, best_epoch = calibration_loss, epoch
            torch.save({"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        "initialization": "torchvision ResNet18 IMAGENET1K_V1", "splits": SPLITS,
                        "epoch": epoch, "seed": SEED, "preprocessing": "RGB cubic 60x36 normalized ImageNet"}, checkpoint)
        print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    cal_y, cal_p, _ = evaluate(model, batches["calibration"], device)
    threshold = choose_thresholds(cal_y, cal_p)["calibration_open_fpr_at_most_5pct"]
    # Validation is evaluated only after epoch and threshold selection finish.
    val_y, val_p, _ = evaluate(model, batches["validation"], device)
    camera_manifest = Path("data/processed/camera_user_001_session_02_prepared.csv")
    rows, provenance = prepared_development_rows(camera_manifest)
    pairs, excluded = reviewed_pairs(rows)
    camera_rows, camera_ear = [], []
    for left, right in pairs:
        crops = aligned_crops(cv2.imread(left["frame_path"]), json.loads(left["points_json"]), json.loads(right["points_json"]))
        images = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops])
        camera_rows.append(dict(images=images, label=int(left["label"]), name=left["sample_id"]))
        camera_ear.append(bilateral(*[CLOSED if r["ear_value"] < .2 else OPEN for r in (left, right)]))
    cy, cp, _ = evaluate(model, loader(camera_rows), device)
    cn = [CLOSED if p >= threshold else OPEN for p in cp]
    fusion = [CLOSED if CLOSED in (e, c) else OPEN if e == c == OPEN else UNKNOWN for e,c in zip(camera_ear,cn)]
    for name, rows, labels, scores in (("calibration",partitions["calibration"],cal_y,cal_p),
                                      ("validation",partitions["validation"],val_y,val_p), ("camera",camera_rows,cy,cp)):
        with (output/(name+".csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "label", "score"])
            writer.writeheader()
            writer.writerows(dict(name=r["name"], label=y, score=p) for r,y,p in zip(rows, labels, scores))
    report = dict(schema=1, status="bounded_public_pilot_not_deployment_ready", counts=counts,
                  splits=SPLITS, uncertain_excluded=exclusions, cross_split_duplicates_removed=duplicates,
                  seed=SEED, epochs=5, selected_epoch=best_epoch, history=history,
                  initialization="ImageNet only; RT-BENE allsubjects checkpoint never loaded",
                  checkpoint_sha256=digest(checkpoint), model_path=str(checkpoint),
                  dataset_hashes={name:digest(root/name) for name in FILES},
                  threshold=threshold, threshold_selection="public s015 max recall at <=5% open FPR",
                  calibration=metrics(cal_y,cal_p,threshold), validation=metrics(val_y,val_p,threshold),
                  validation_fixed_0_5=metrics(val_y,val_p,.5),
                  camera=dict(status="previously inspected one-person development diagnostic",
                              manifest_sha256=provenance["manifest_sha256"], excluded=excluded,
                              fixed_0_5=metrics(cy,cp,.5), selected_threshold=metrics(cy,cp,threshold),
                              ear=state_metrics(cy,camera_ear), fusion_or=state_metrics(cy,fusion)),
                  runtime_changed=False, device=device, seconds=time.perf_counter()-started)
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"selected_epoch":best_epoch,"threshold":threshold,"validation":report["validation"],"camera":report["camera"]}), flush=True)


if __name__ == "__main__":
    main()
