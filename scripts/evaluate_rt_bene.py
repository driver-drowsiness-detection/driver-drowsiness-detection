"""Reproducible offline pretrained blink diagnostic, never a deployment gate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np
import torch

from marvel.contracts import OPEN, CLOSED, UNKNOWN, bilateral
from marvel.training.ear_calibration import prepared_development_rows
from marvel.training.rt_bene import RTBene, SOURCE_COMMIT, MODEL_URL, aligned_crops, reviewed_pairs
from marvel.training.calibration import metrics


def digest(path, algorithm="sha256"):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def state_metrics(labels, states):
    y, s = np.asarray(labels), np.asarray(states)
    covered = s != UNKNOWN
    result = metrics(y[covered], (s[covered] == CLOSED).astype(float), .5) if covered.any() else {}
    result.update(total=len(y), coverage=float(covered.mean()),
                  unknown=int((~covered).sum()),
                  unknown_closed=int(((~covered) & (y == 1)).sum()),
                  unknown_open=int(((~covered) & (y == 0)).sum()),
                  closed_recall_all=float(((s == CLOSED) & (y == 1)).sum() / (y == 1).sum()),
                  correct_fraction_all=float((((s == CLOSED) & (y == 1)) | ((s == OPEN) & (y == 0))).mean()))
    return result


def camera(model, manifest, baseline_csv):
    rows, provenance = prepared_development_rows(manifest)
    paired, excluded = reviewed_pairs(rows)
    if not paired:
        raise ValueError("No reviewed pairs")
    with open(baseline_csv, newline="", encoding="utf-8-sig") as handle:
        baseline = {(r["sample_id"], r["eye_side"]): r for r in csv.DictReader(handle)}
    original_crops, aligned, selected = [], [], []
    for left, right in paired:
        frame = cv2.imread(left["frame_path"])
        try:
            crops = aligned_crops(frame, json.loads(left["points_json"]), json.loads(right["points_json"]))
        except ValueError:
            excluded["invalid_aligned_crop"] = excluded.get("invalid_aligned_crop", 0) + 1
            continue
        original_crops.append(tuple(cv2.imread(r["crop_path"]) for r in (left, right)))
        aligned.append(crops)
        selected.append((left, right))
    # Both preparations are declared before inspecting any pretrained scores.
    scores_original = model.predict(original_crops)
    scores_aligned = model.predict(aligned)
    records = []
    for (left, right), po, pa in zip(selected, scores_original, scores_aligned):
        old = [baseline[(r["sample_id"], r["eye_side"])] for r in (left, right)]
        for row, stored in zip((left, right), old):
            if row["label"] != stored["label"] or abs(float(row["timestamp_s"]) - float(stored["timestamp_s"])) > 1e-8:
                raise ValueError("Baseline identity/label mismatch")
        ear = bilateral(*[CLOSED if r["ear_value"] < .2 else OPEN for r in (left, right)])
        cnn = CLOSED if pa >= .5 else OPEN
        record = dict(sample_id=left["sample_id"], timestamp_s=left["timestamp_s"], label=int(left["label"]),
                      rt_bene_existing_crop=po, rt_bene_aligned=pa, ear=ear,
                      rt_bene=cnn,
                      cnn_original=bilateral(*[CLOSED if float(r["cnn_original"]) >= .5 else OPEN for r in old]),
                      cnn_10epoch=bilateral(*[CLOSED if float(r["cnn_10epoch"]) >= .5 else OPEN for r in old]))
        # Prespecified logical sensitivity/specificity diagnostics, not fitted fusion.
        record["fusion_or"] = CLOSED if CLOSED in (ear, cnn) else OPEN if ear == cnn == OPEN else UNKNOWN
        record["fusion_consensus"] = bilateral(ear, cnn)
        records.append(record)
    labels = [r["label"] for r in records]
    methods = {name: state_metrics(labels, [r[name] for r in records]) for name in
               ("ear", "cnn_original", "cnn_10epoch", "rt_bene", "fusion_or", "fusion_consensus")}
    methods["rt_bene_existing_crop"] = state_metrics(labels, [CLOSED if r["rt_bene_existing_crop"] >= .5 else OPEN for r in records])
    return dict(status="one_participant_calibration_diagnostic_not_independent_validation",
                subjects=sorted({r["subject_id"] for r in rows}),
                input_frames=len(rows)//2, scored_pairs=len(records), excluded=excluded,
                closed=sum(labels), open=len(labels)-sum(labels), methods=methods,
                manifest_sha256=provenance["manifest_sha256"], baseline_sha256=digest(baseline_csv),
                threshold=.5, ear_threshold=.2,
                preprocessing="BGR, cubic 60x36, upstream channel normalization; anatomical left/right",
                crop_variants=["existing MARVEL crops", "upstream geometry on full-frame canvas using MediaPipe corners"],
                threshold_fitted=False, weights_fitted=False, runtime_changed=False), records


def public_sanity(model, root):
    files = {"rt_bene_subjects.csv": "a07810daaf6d2053a9ed1f782e5f360c",
             "s012_blink_labels.csv": "7722524624f3dcf53116e7081a7de5c7",
             "s012_noglasses_eyes.tar": "e925025262025b3c6145c42b08255ba1"}
    for name, expected in files.items():
        if digest(root/name, "md5") != expected:
            raise ValueError("Public dataset checksum mismatch: " + name)
    with (root/"s012_blink_labels.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    pairs, labels, names = [], [], []
    excluded = 0
    # Read archive members in memory: no archive extraction or path traversal.
    with tarfile.open(root/"s012_noglasses_eyes.tar") as archive:
        members = {m.name.removeprefix("./"): m for m in archive.getmembers() if m.isfile()}
        for name, label in rows:
            if label == "0.5":
                excluded += 1
                continue
            if label not in ("0.0", "1.0") or not name.startswith("left_"):
                raise ValueError("Unexpected public label/name")
            pair = []
            for side in ("left", "right"):
                path = f"s012_noglasses/natural/{side}/{name.replace('left_', side + '_', 1)}"
                raw = archive.extractfile(members[path]).read()
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Unreadable public image: " + path)
                pair.append(image)
            pairs.append(pair)
            labels.append(int(float(label)))
            names.append(name)
    scores = model.predict(pairs)
    return dict(status="source_training_subject_compatibility_only_not_independent_validation",
                subject="s012", source_split="training", excluded_annotator_disagreement=excluded,
                dataset="https://zenodo.org/records/3685316", threshold=.5,
                hashes={name: digest(root/name) for name in files},
                metrics=metrics(labels, scores, .5)), [dict(name=n, label=y, score=s) for n, y, s in zip(names, labels, scores)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preprocessing", choices=("upstream_bgr", "training_rgb"), default="upstream_bgr")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = RTBene(args.checkpoint, args.device, args.preprocessing)
    local, local_scores = camera(model, args.manifest, args.baseline_scores)
    local["preprocessing"] = args.preprocessing
    print(json.dumps({"camera": local}, indent=2), flush=True)
    public, public_scores = public_sanity(model, args.public_data)
    report = dict(schema=1, source_commit=SOURCE_COMMIT, checkpoint_url=MODEL_URL,
                  checkpoint_sha256=digest(args.checkpoint), torch=torch.__version__,
                  device=args.device, preprocessing=args.preprocessing, camera=local, public=public,
                  elapsed_seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    for suffix, records in (("camera", local_scores), ("public", public_scores)):
        with args.output.with_suffix("."+suffix+".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({"public": public, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
