"""Inventory the original CEW face release and optionally probe perception.

This produces a review queue, never a training manifest or automatic eye labels.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def source_identity(name, face_class):
    """Return documented filename identity hints; never guess closed subjects."""
    if face_class == "open":
        match = re.fullmatch(r"(.+)_(\d{4})\.jpg", name)
        if not match:
            raise ValueError(f"Unexpected open face filename: {name}")
        return match[1], name
    match = re.fullmatch(r"(closed_eye_\d+\.[^.]+)_face_\d+\.jpg", name)
    if not match:
        raise ValueError(f"Unexpected closed face filename: {name}")
    return "", match[1]


def coordinate_index(path):
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        name, *values = line.split()
        if name in result or len(values) != 4:
            raise ValueError(f"Invalid coordinate row: {line}")
        coordinates = tuple(int(value) for value in values)
        if not all(0 <= value < 100 for value in coordinates):
            raise ValueError(f"Out of bounds source coordinates: {name}")
        result[name] = coordinates
    return result


def inspect_release(root):
    from PIL import Image

    rows, hashes, dimensions = [], defaultdict(list), Counter()
    for folder, face_class in (("ClosedFace", "closed"), ("OpenFace", "open")):
        coordinates = coordinate_index(root / f"EyeCoordinatesInfo_{folder}.txt")
        paths = sorted((root / folder).glob("*.jpg"))
        if not paths or {p.name for p in paths} != set(coordinates):
            raise ValueError(f"Images and coordinate records disagree: {folder}")
        for path in paths:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[str(image.size)] += 1
                # Pixel hashes catch equivalent images with differing file metadata.
                rgb = image.convert("RGB")
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            identity, source_group = source_identity(path.name, face_class)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[pixel_hash].append(f"{folder}/{path.name}")
            rows.append(dict(
                image_path=str(path.resolve()), source_face_class=face_class,
                source_identity_hint=identity, source_image_group=source_group,
                sha256=digest, decoded_pixel_sha256=pixel_hash,
                source_eye_centers=json.dumps(coordinates[path.name]),
            ))
    duplicates = [values for values in hashes.values() if len(values) > 1]
    closed_groups = Counter(r["source_image_group"] for r in rows if r["source_face_class"] == "closed")
    summary = dict(
        images=len(rows), classes=dict(Counter(r["source_face_class"] for r in rows)),
        dimensions=dict(dimensions), decoded_pixel_duplicate_groups=duplicates,
        open_filename_identities=len({r["source_identity_hint"] for r in rows if r["source_face_class"] == "open"}),
        closed_source_image_groups=len(closed_groups),
        closed_source_groups_with_multiple_faces=sum(n > 1 for n in closed_groups.values()),
        status="inspection_only_not_ready_for_training",
        blockers=["Closed-face person identities and cross-class identity overlap unresolved",
                  "Per-eye visibility and labels require review",
                  "No subject-disjoint train/calibration/validation/test assignment"],
    )
    return rows, summary


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probe(rows, task, config_path, output):
    import cv2
    from marvel.contracts import Frame
    from marvel.perception.config import load_config
    from marvel.perception.detection import Detector

    root = Path(__file__).resolve().parents[1]
    config = load_config(config_path, root)
    detector = Detector(task, config["driver_roi"], config["quality"])
    review, statuses, reasons, usable = [], Counter(), Counter(), Counter()
    crop_dir = output / "crops"
    crop_dir.mkdir()
    try:
        for index, row in enumerate(rows):
            image = cv2.imread(row["image_path"])
            if image is None:
                raise ValueError(f"OpenCV cannot read {row['image_path']}")
            detector.invalidate()
            # Monotonic adapter sequence, NOT a source video timestamp.
            eyes, status, _ = detector.process(Frame(f"cew_probe:{index}", float(index), image))
            statuses[row["source_face_class"] + ":" + status] += 1
            by_side = {eye.side: eye for eye in eyes}
            for side in ("left", "right"):
                eye = by_side.get(side)
                crop_path = ""
                if eye is not None and eye.crop_ok:
                    target = crop_dir / f"{index:04d}_{side}.png"
                    if not cv2.imwrite(str(target), eye.crop):
                        raise IOError(target)
                    crop_path = str(target.relative_to(output))
                    usable[row["source_face_class"]] += 1
                rejection = ";".join(eye.reasons) if eye is not None else status
                if rejection:
                    reasons[rejection] += 1
                review.append(dict(
                    **row, eye_side=side, perception_status=status,
                    geometry_ok=int(eye is not None and eye.geometry_ok),
                    crop_ok=int(bool(crop_path)), crop_path=crop_path,
                    rejection_reason=rejection,
                    points_json=json.dumps(eye.points if eye is not None else []),
                    reviewed_subject_id="", reviewed_label="", reviewed_observable="",
                    reviewer="", review_notes="", split="",
                ))
            if (index + 1) % 250 == 0:
                print(f"Inspected {index + 1}/{len(rows)} images", flush=True)
    finally:
        detector.close()
    write_csv(output / "eye_review_queue.csv", review)
    return dict(face_statuses=dict(statuses), usable_crops_by_source_class=dict(usable),
                eye_rejection_reasons=dict(reasons), review_rows=len(review),
                config=config, landmark_model_sha256=hashlib.sha256(Path(task).read_bytes()).hexdigest(),
                sequence_time="synthetic_index_only_not_source_time")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.yaml"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, summary = inspect_release(args.source)
    args.output.mkdir(parents=True)
    write_csv(args.output / "image_inventory.csv", rows)
    if args.task:
        summary["perception_probe"] = probe(rows, args.task, args.config, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
