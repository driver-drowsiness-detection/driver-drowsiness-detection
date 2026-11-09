"""Fit the EAR mapping using only reviewed calibration eyes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marvel.training.ear_calibration import calibrate_manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    report = calibrate_manifest(args.prepared_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report["ear_calibration"]))

if __name__ == "__main__":
    main()
