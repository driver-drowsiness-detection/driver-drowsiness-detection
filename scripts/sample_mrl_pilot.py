"""Make a small, reproducible subject-disjoint MRL pilot manifest."""
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marvel.training.dataset import read_manifest


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: sample_mrl_pilot.py FULL.csv PILOT.csv")
    source, target = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if target.exists() or target.with_suffix(".provenance.json").exists():
        raise FileExistsError(target)
    rows = read_manifest(source)
    if any(r["source"] != "MRL_Eye_2018_01" for r in rows):
        raise ValueError("Expected inspected MRL release")
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            continue  # untouched test data is not used in the pilot
        groups[(row["subject_id"], row["split"], row["label"])].append(row)
    rng = random.Random(7)
    selected = []
    for key in sorted(groups):
        candidates = groups[key]
        selected.extend(rng.sample(candidates, min(50, len(candidates))))
    selected.sort(key=lambda r: (r["split"], r["subject_id"], r["sample_id"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        for row in selected:
            result = dict(row)
            for field in ("frame_path", "crop_path"):
                if result[field]:
                    result[field] = os.path.relpath(result[field], target.parent)
            writer.writerow(result)
    read_manifest(target)
    provenance = dict(purpose="pilot_only_no_test_set", seed=7, maximum_per_subject_and_class=50,
                      source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      selected_rows=len(selected), subjects=len({r["subject_id"] for r in selected}),
                      note="MRL eye-only images; no temporal/paired EAR evaluation")
    target.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
