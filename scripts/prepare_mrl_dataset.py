"""Import original MRL eye-only archive for supplemental CNN training."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from marvel.training.dataset import read_manifest

PATTERN = re.compile(r"mrlEyes_2018_01/(s\d{4})/s\d{4}_(\d{5})_[01]_[01]_([01])_[012]_[01]_0[123]\.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    archive, manifest = args.archive.resolve(), args.manifest.resolve()
    if manifest.exists() or manifest.with_suffix(".provenance.json").exists():
        raise FileExistsError("Use a new versioned manifest path")
    extraction_root = archive.parent / "mrlEyes_2018_01"
    if extraction_root.exists():
        raise FileExistsError("Extraction folder already exists; inspect it before retrying")
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        images = []
        for member in members:
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] != "mrlEyes_2018_01"
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe archive member: " + member.filename)
            if member.is_dir():
                continue
            if member.filename.endswith(".png"):
                match = PATTERN.fullmatch(member.filename)
                if not match or match[1] != path.parts[1]:
                    raise ValueError("Unexpected image label or subject: " + member.filename)
                images.append((member.filename, match[1], match[3]))
            elif member.filename not in ("mrlEyes_2018_01/annotation.txt", "mrlEyes_2018_01/stats_2018_01.ods"):
                raise ValueError("Unexpected member: " + member.filename)
        if len(images) != 84898:
            raise ValueError(f"Expected original release's 84898 eye images; got {len(images)}")
        annotation = z.read("mrlEyes_2018_01/annotation.txt").decode("utf-8")
        if "0 - close" not in annotation or "1 - open" not in annotation:
            raise ValueError("Source eye-label mapping differs from inspected release")
        damaged = z.testzip()
        if damaged:
            raise ValueError("Archive CRC failed: " + damaged)
        z.extractall(archive.parent)

    subjects = sorted({s for _, s, _ in images})
    if len(subjects) != 37:
        raise ValueError("Expected 37 source subjects")
    random.Random(7).shuffle(subjects)
    assignments = {s: split for split, names in (
        ("train", subjects[:25]), ("calibration", subjects[25:29]),
        ("validation", subjects[29:33]), ("test", subjects[33:])) for s in names}
    counts = Counter()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=(
            "sample_id", "subject_id", "session_id", "video_id", "source", "split",
            "frame_path", "timestamp_s", "eye_side", "label", "observable", "label_source",
            "crop_path", "source_eye_state", "metadata_note"))
        writer.writeheader()
        for name, subject, source_state in images:
            split = assignments[subject]
            counts[(split, source_state)] += 1
            local = archive.parent / name
            relative = os.path.relpath(local, manifest.parent)
            writer.writerow(dict(sample_id="mrl:" + name, subject_id="mrl:" + subject,
                session_id="mrl:" + subject + ":all_unknown_sessions",
                video_id="mrl:" + subject + ":all_unknown_videos", source="MRL_Eye_2018_01",
                split=split, frame_path=relative, timestamp_s="", eye_side="unspecified",
                label="1" if source_state == "0" else "0", observable="1",
                label_source="dataset_eye_annotation", crop_path=relative,
                source_eye_state=source_state,
                metadata_note="Eye-only image; anatomical side, timestamp, session and video unavailable"))
    read_manifest(manifest)
    provenance = dict(source_url="https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip",
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_label_map={"0": "closed", "1": "open"}, project_label_map={"0": "open", "1": "closed"},
        split_seed=7, split_subjects={split: sorted(s for s in subjects if assignments[s] == split)
                                for split in ("train", "calibration", "validation", "test")},
        image_counts={split: {state: counts[(split, state)] for state in ("0", "1")}
                      for split in ("train", "calibration", "validation", "test")},
        purpose="supplemental eye-only CNN; cannot evaluate EAR or temporal drowsiness")
    manifest.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
