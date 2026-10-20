"""Validate labels/groups/files and report counts without training."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    args = parser.parse_args()
    from marvel.training.audit import audit_manifest
    print(json.dumps(audit_manifest(args.manifest), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
