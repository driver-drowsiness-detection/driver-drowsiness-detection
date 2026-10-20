"""Generate versioned eye crops from independently labelled full-frame images.

Usage: python scripts/prepare_dataset.py annotations.csv detector.task crops prepared.csv
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marvel.training.dataset import read_manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("task")
    parser.add_argument("folder")
    parser.add_argument("target")
    parser.add_argument("--config", default=str(ROOT / "configs/runtime.yaml"))
    args = parser.parse_args(argv)
    source, task, folder, target = args.source, args.task, args.folder, args.target
    target_path = Path(target)
    metadata = Path(target + ".metadata.json")
    if target_path.exists() or metadata.exists():
        raise FileExistsError(target_path)

    from marvel.training.preparation import prepare_rows
    from marvel.perception.config import load_config
    from marvel.perception.detection import Detector

    rows = read_manifest(source)
    config = load_config(args.config, ROOT)
    detector = Detector(task, config["driver_roi"], config["quality"])
    try:
        output = prepare_rows(rows, detector, folder, target)
    finally:
        detector.close()

    with metadata.open("x", encoding="utf-8") as handle:
        json.dump(dict(schema=1, path_base="manifest_directory",
                       annotation_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                       model_sha256=hashlib.sha256(Path(task).read_bytes()).hexdigest(),
                       driver_roi=config["driver_roi"], quality=config["quality"],
                       geometry="six_original_pixel_points", crop="width_1.6_height_0.8",
                       rows=len(output)), handle, indent=2, allow_nan=False)
    print({"rows": len(output), "usable_crops": sum(bool(row["crop_path"]) for row in output)})


if __name__ == "__main__":
    main()
