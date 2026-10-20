"""Read-only checks before a dataset is used for fitting."""
import hashlib
from collections import Counter
from pathlib import Path

from .dataset import SPLITS, read_manifest


def audit_manifest(path):
    from PIL import Image

    rows = read_manifest(path)
    ownership, checked = {}, {}
    for row in rows:
        for field in ("frame_path", "crop_path"):
            if not row.get(field):
                continue
            image_path = row[field]
            if image_path not in checked:
                with Image.open(image_path) as image:
                    image.verify()
                with Path(image_path).open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                checked[image_path] = digest
            digest = checked[image_path]
            if digest in ownership and ownership[digest] != row["split"]:
                raise ValueError("Identical image bytes in different partitions: " + image_path)
            ownership[digest] = row["split"]
    summary = {}
    for split in sorted(SPLITS):
        subset = [r for r in rows if r["split"] == split]
        usable = [r for r in subset if r["observable"] == "1" and r.get("crop_path")]
        summary[split] = dict(
            rows=len(subset), subjects=len({r["subject_id"] for r in subset}),
            observable=sum(r["observable"] == "1" for r in subset),
            usable_crops=len(usable), usable_labels=dict(Counter(r["label"] for r in usable)),
            both_classes_after_filtering={r["label"] for r in usable} == {"0", "1"})
    return dict(rows=len(rows), checked_images=len(checked), splits=summary,
                path_base="manifest_directory",
                duplicate_check="exact_file_bytes_only")
