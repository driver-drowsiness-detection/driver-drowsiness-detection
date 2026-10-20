"""Download the official version-1 Face Landmarker bundle without overwriting files."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/face_landmarker.task")
    parser.add_argument("--sha256", default=MODEL_SHA256, help="Required model SHA256")
    args = parser.parse_args()
    path = args.output
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves another teammate's existing asset.
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        if len(content) < 1024:
            raise ValueError("Unexpectedly small model download")
        digest = hashlib.sha256(content).hexdigest()
        if args.sha256 and args.sha256.lower() != digest:
            raise ValueError("Downloaded model SHA256 mismatch; no file was written")
        with path.open("xb") as handle:
            handle.write(content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.sha256 and args.sha256.lower() != digest:
        raise ValueError("Existing model SHA256 mismatch; file preserved")
    print("Model:", path)
    print("SHA256:", digest)


if __name__ == "__main__":
    main()
