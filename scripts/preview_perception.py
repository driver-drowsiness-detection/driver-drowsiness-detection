"""Camera preview or complete eye perception, using the same settings and contract."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marvel.contracts import Frame
from marvel.perception.capture import LiveCapture
from marvel.perception.config import load_config
from marvel.perception.pipeline import process_frame


def summary(frame, eyes, status, metrics, age, fps):
    return {
        "sample_id": frame.sample_id, "t": frame.t, "shape": list(frame.image.shape),
        "status": status, "read_age_s": age, "processed_fps": fps,
        "eyes": [{"sample_id": e.sample_id, "t": e.t, "track": e.track, "side": e.side,
                  "geometry_ok": e.geometry_ok, "crop_ok": e.crop_ok,
                  "reasons": list(e.reasons), "points": e.points,
                  "crop_shape": list(e.crop.shape) if e.crop is not None else None}
                 for e in eyes], "metrics": metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("camera", "perception"))
    parser.add_argument("config", nargs="?", default="configs/runtime.yaml")
    parser.add_argument("--image", help="Saved-image landmark check; no camera is opened")
    parser.add_argument("--no-display", action="store_true", help="Print observations without GUI windows")
    parser.add_argument("--max-frames", type=int, help="Stop after this many received frames")
    parser.add_argument("--max-seconds", type=float, help="Stop after this many elapsed seconds")
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.max_seconds is not None:
        import math
        if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
            parser.error("--max-seconds must be finite and positive")
    config = load_config(args.config, ROOT)
    max_age, timeout = config["max_frame_age_s"], config["capture_timeout_s"]

    import cv2
    import numpy as np
    from marvel.observability.overlay import draw_preview, show_eyes

    capture = detector = None
    display = not args.no_display
    try:
        if args.mode == "perception":
            from marvel.perception.detection import Detector
            detector = Detector(config["landmarker"], config["driver_roi"], config["quality"])
            if config["quality"]["laplacian_min"] == 0:
                print("NOTE: blur rejection disabled; sharpness is measured only.", file=sys.stderr)
        if args.image:
            image = cv2.imread(args.image)
            if image is None:
                raise ValueError("Unreadable image: " + args.image)
        else:
            capture = LiveCapture(config["capture"])

        if display:
            cv2.namedWindow("MARVEL preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("MARVEL preview", 960, 720)
        started = last_received = time.monotonic()
        last_done, last_print, count = None, -float("inf"), 0
        while True:
            if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds:
                if count == 0:
                    raise RuntimeError("Run ended without receiving a camera frame")
                break
            frame = Frame("saved_image", time.monotonic(), image) if args.image else capture.read()
            now = time.monotonic()
            if frame is None:
                if detector is not None and now - last_received > max_age:
                    detector.invalidate()
                if display:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for a new frame; q=quit", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("MARVEL preview", blank)
                    if detector is not None:
                        show_eyes([])
                if now - last_received > timeout:
                    raise RuntimeError("No camera frame within capture_timeout_s")
            else:
                last_received = now
                eyes, status, metrics, age = process_frame(frame, detector, max_age)
                done = time.monotonic()
                fps = 0.0 if last_done is None else 1.0 / max(done - last_done, 1e-6)
                last_done = done
                if display:
                    cv2.imshow("MARVEL preview", draw_preview(frame, eyes, status, fps, age, metrics))
                    if detector is not None:
                        show_eyes(eyes)
                if done - last_print >= 1.0 or args.image:
                    print(json.dumps(summary(frame, eyes, status, metrics, age, fps)), flush=True)
                    last_print = done
                count += 1
                if args.image:
                    if display:
                        while cv2.waitKey(50) & 0xFF not in (ord("q"), 27):
                            if cv2.getWindowProperty("MARVEL preview", cv2.WND_PROP_VISIBLE) < 1:
                                break
                    break
                if args.max_frames is not None and count >= args.max_frames:
                    break
            if display:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("MARVEL preview", cv2.WND_PROP_VISIBLE) < 1:
                    break
        print("Completed preview: {} received frame(s).".format(count), file=sys.stderr)
    finally:
        try:
            if capture is not None:
                capture.close()
        finally:
            try:
                if detector is not None:
                    detector.close()
            finally:
                if display:
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
