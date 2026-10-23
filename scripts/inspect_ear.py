"""Inspect EAR on one saved full-face image; no camera or trained CNN needed."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--config", default=str(ROOT / "configs/runtime.yaml"))
    parser.add_argument("--threshold", type=float,
                        help="Explicit development cutoff; omit to report raw EAR only")
    args = parser.parse_args()
    import cv2
    from marvel.contracts import Frame, UNKNOWN, bilateral
    from marvel.perception.config import load_config
    from marvel.perception.detection import Detector
    from marvel.cognition.ear_classifier import measure, state

    config = load_config(args.config, ROOT)
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError("Unreadable image: " + args.image)
    detector = Detector(config["landmarker"], config["driver_roi"], config["quality"])
    try:
        eyes, status, _ = detector.process(Frame("saved_image:0", 0.0, image))
        output, states = [], {}
        for eye in eyes:
            estimate = measure(eye)
            eye_state = state(estimate, args.threshold) if args.threshold is not None else UNKNOWN
            states[eye.side] = eye_state
            output.append(dict(sample_id=eye.sample_id, t=eye.t, track=eye.track,
                               side=eye.side, ear=estimate.value, valid=estimate.valid,
                               reason=estimate.reason, state=eye_state))
        print(json.dumps(dict(face_status=status, eyes=output, threshold=args.threshold,
                              bilateral=bilateral(states.get("left", UNKNOWN),
                                                  states.get("right", UNKNOWN))),
                         indent=2, allow_nan=False))
    finally:
        detector.close()


if __name__ == "__main__":
    main()
