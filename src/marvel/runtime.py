"""Runnable EAR cognition for an image, timestamped video, or explicit live camera."""
import argparse
import json
import math
from pathlib import Path
import time
import uuid

from marvel.contracts import Frame
from marvel.cognition.pipeline import EARCognition
from marvel.cognition.cnn_inference import sha256
from marvel.observability.logger import JsonLogger
from marvel.perception.config import load_config
from marvel.perception.detection import Detector
from marvel.perception.pipeline import process_frame

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", action="store_true")
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-calibration", type=Path,
                        help="Measured camera parameters and declared 3D face template JSON")
    parser.add_argument("--ear-threshold", type=float, required=True,
                        help="Explicit experimental cutoff; no fitted default")
    for flag, default in (("close-seconds", 1.5), ("recover-seconds", .5),
                          ("max-gap-seconds", .25), ("unavailable-seconds", 2.0),
                          ("window-seconds", 60.0), ("min-coverage", .8)):
        parser.add_argument("--" + flag, type=float, default=default)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--audio", action="store_true", help="Opt into a bounded Windows fixed tone")
    parser.add_argument("--synthetic-source", action="store_true",
                        help="Mark generated/test media; never use it as accuracy evidence")
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("max-frames must be positive")
    settings = load_config(args.config, ROOT)
    run_id = str(uuid.uuid4())
    policy = dict(ear_threshold=args.ear_threshold, close_s=args.close_seconds,
                  recover_s=args.recover_seconds, max_gap_s=args.max_gap_seconds,
                  unavailable_s=args.unavailable_seconds, window_s=args.window_seconds,
                  min_coverage=args.min_coverage)
    pose_calibration = (json.loads(args.pose_calibration.read_text(encoding="utf-8"))
                        if args.pose_calibration else None)
    cognition = EARCognition(**policy, run_id=run_id, pose_calibration=pose_calibration)
    if not Path(settings["landmarker"]).is_file():
        raise FileNotFoundError(settings["landmarker"])
    import cv2
    capture = detector = audio = None
    count, heartbeat_index = 0, 0
    last_received = time.monotonic()
    source_kind = "camera" if args.camera else "video" if args.video else "image"
    try:
        with JsonLogger(args.output) as log:
            log.write(dict(schema=1, type="run", run_id=run_id,
                           synthetic=args.synthetic_source, source_kind=source_kind,
                           source=str(args.video or args.image or settings["capture"]["source"]),
                           clock="monotonic_read_completion" if args.camera else "media_seconds",
                           experimental_policy=policy, perception_config=settings,
                           pose_calibration=pose_calibration,
                           landmark_sha256=sha256(settings["landmarker"])))
            detector = Detector(settings["landmarker"], settings["driver_roi"], settings["quality"])
            if args.audio:
                from marvel.actuation.audio import FixedAlert
                audio = FixedAlert()
            if args.camera:
                from marvel.perception.capture import LiveCapture
                capture = LiveCapture(settings["capture"])
            elif args.video:
                capture = cv2.VideoCapture(str(args.video))
                if not capture.isOpened():
                    raise ValueError("Cannot open video")
            else:
                image = cv2.imread(str(args.image))
                if image is None:
                    raise ValueError("Cannot read image")
            while True:
                if audio is not None:
                    completed = audio.poll()
                    if completed is not None:
                        log.write(dict(schema=1, type="actuation", run_id=run_id, **completed))
                if args.camera:
                    frame = capture.read()
                    now = time.monotonic()
                    if frame is None:
                        if now - last_received > settings["max_frame_age_s"]:
                            detector.invalidate()
                            heartbeat_index += 1
                            record = cognition.update(f"heartbeat:{heartbeat_index}", now, [], "capture_stall")
                            log.write(dict(schema=1, type="observation", run_id=run_id,
                                           synthetic=args.synthetic_source, **record))
                        if now - last_received > settings["capture_timeout_s"]:
                            raise RuntimeError("Camera stalled beyond capture_timeout_s")
                        if args.display and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                        continue
                    last_received = now
                    # A queued pre-heartbeat frame is historical evidence: discard it.
                    if cognition.trigger.last is not None and frame.t <= cognition.trigger.last:
                        log.write(dict(schema=1, type="discarded_frame", run_id=run_id,
                                       sample_id=frame.sample_id, t=frame.t, reason="before_last_update"))
                        continue
                    eyes, status, metrics, age = process_frame(frame, detector, settings["max_frame_age_s"])
                else:
                    if args.video:
                        ok, image = capture.read()
                        if not ok:
                            break
                        t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    else:
                        t = 0.0
                    if not math.isfinite(t) or (cognition.trigger.last is not None and t <= cognition.trigger.last):
                        raise ValueError("Video timestamps must be finite and strictly increasing")
                    frame = Frame(f"{run_id}:{count}", t, image)
                    eyes, status, metrics = detector.process(frame)
                    age = None  # Media time is not comparable to a live wall clock.
                record = cognition.update(frame.sample_id, frame.t, eyes, status, metrics, frame.image.shape)
                log.write(dict(schema=1, type="observation", run_id=run_id,
                               synthetic=args.synthetic_source, read_age_s=age, **record))
                for event in record["decision"]["events"]:
                    print(f"{event['t']:.3f} {event['kind']} {event['episode_id']}", flush=True)
                    if audio is not None:
                        log.write(dict(schema=1, type="actuation", run_id=run_id,
                                       event_id=event["event_id"], status=audio.submit(event)))
                count += 1
                if args.display:
                    view = image.copy() if not args.camera else frame.image.copy()
                    label = f"EAR {record['decision']['state']} | coverage {record['rolling']['coverage']:.0%}"
                    cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                    cv2.imshow("MARVEL cognition", view)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                if args.image or (args.max_frames is not None and count >= args.max_frames):
                    break
            if count == 0:
                raise RuntimeError("No frames processed")
            if audio is not None:
                audio.close()
                completed = audio.poll()
                if completed is not None:
                    log.write(dict(schema=1, type="actuation", run_id=run_id, **completed))
            log.write(dict(schema=1, type="run_end", run_id=run_id, frames=count,
                           active_episode=cognition.trigger.episode_id,
                           disposition="unresolved_at_end" if cognition.trigger.episode_id else "no_active_alarm"))
    finally:
        try:
            if capture is not None:
                capture.close() if args.camera else capture.release()
        finally:
            try:
                if detector is not None:
                    detector.close()
            finally:
                try:
                    if audio is not None:
                        audio.close()
                finally:
                    if args.display:
                        cv2.destroyAllWindows()
    print(f"Processed {count} frame(s). Log: {args.output}")
