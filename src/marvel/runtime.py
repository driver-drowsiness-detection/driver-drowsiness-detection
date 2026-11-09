"""Laptop cognition with EAR, optional CNN comparison, fixed audio and speech."""
import argparse
import json
import math
from pathlib import Path
import time
import uuid
from time import perf_counter

from marvel.contracts import Frame
from marvel.cognition.camera_pipeline import CameraCognition
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
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--cnn-checkpoint", type=Path)
    models.add_argument("--cnn-onnx", type=Path)
    parser.add_argument("--cnn-threshold", type=float)
    parser.add_argument("--cnn-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--comparison-config", type=Path)
    parser.add_argument("--alert-mode", choices=("ear", "cnn", "fusion"), default="ear")
    parser.add_argument("--voice", choices=("fixed", "llm"))
    parser.add_argument("--voice-model", type=Path, default=Path("models/llm/SmolLM2-360M-Instruct"))
    parser.add_argument("--max-seconds", type=float, help="Bound total capture-loop duration")
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
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("max-seconds must be finite and positive")
    if args.voice == "llm" and not (args.voice_model / "model-manifest.json").is_file():
        parser.error("Local voice model manifest is missing")
    settings = load_config(args.config, ROOT)
    run_id = str(uuid.uuid4())
    policy = dict(ear_threshold=args.ear_threshold, close_s=args.close_seconds,
                  recover_s=args.recover_seconds, max_gap_s=args.max_gap_seconds,
                  unavailable_s=args.unavailable_seconds, window_s=args.window_seconds,
                  min_coverage=args.min_coverage)
    pose_calibration = (json.loads(args.pose_calibration.read_text(encoding="utf-8"))
                        if args.pose_calibration else None)
    model, model_identity = None, None
    if args.cnn_checkpoint or args.cnn_onnx:
        if args.cnn_threshold is None:
            parser.error("CNN inference requires an explicit experimental cnn-threshold")
        from marvel.cognition.cnn_inference import CNN, OpenCVCNN
        model_path = args.cnn_checkpoint or args.cnn_onnx
        model_identity = dict(path=str(model_path), sha256=sha256(model_path),
                              temperature=args.cnn_temperature, threshold=args.cnn_threshold,
                              backend="pytorch" if args.cnn_checkpoint else "opencv_dnn")
        if args.cnn_checkpoint:
            model = CNN(model_path, temperature=args.cnn_temperature, device=args.device)
        else:
            metadata = json.loads(Path(str(model_path) + ".json").read_text(encoding="utf-8"))
            model_identity["metadata_sha256"] = sha256(str(model_path) + ".json")
            model_identity["checkpoint_sha256"] = metadata["checkpoint_sha256"]
            model = OpenCVCNN(model_path, metadata["checkpoint_sha256"], temperature=args.cnn_temperature)
    elif args.cnn_threshold is not None or args.comparison_config or args.alert_mode != "ear":
        parser.error("CNN or fusion settings require a CNN checkpoint or ONNX artifact")
    comparison_config = (json.loads(args.comparison_config.read_text(encoding="utf-8"))
                         if args.comparison_config else None)
    cognition = CameraCognition(policy=policy, run_id=run_id, model=model,
        cnn_threshold=args.cnn_threshold, comparison=comparison_config,
        alert_mode=args.alert_mode, pose_calibration=pose_calibration)
    if not Path(settings["landmarker"]).is_file():
        raise FileNotFoundError(settings["landmarker"])
    import cv2
    capture = detector = audio = speech = None
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
                           cnn=model_identity, comparison_config=comparison_config,
                           alert_mode=args.alert_mode, voice_mode=args.voice,
                           voice_manifest_sha256=sha256(args.voice_model / "model-manifest.json") if args.voice == "llm" else None,
                           pose_calibration=pose_calibration,
                           landmark_sha256=sha256(settings["landmarker"])))
            detector = Detector(settings["landmarker"], settings["driver_roi"], settings["quality"])
            if args.audio:
                from marvel.actuation.audio import FixedAlert
                audio = FixedAlert()
            if args.voice:
                from marvel.actuation.speech_alerts import SpeechAlerts
                speech = SpeechAlerts(args.output.parent / (args.output.stem + "_speech"),
                                      args.voice_model if args.voice == "llm" else None)
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
            loop_started = perf_counter()
            while True:
                if args.max_seconds is not None and perf_counter() - loop_started >= args.max_seconds:
                    break
                if speech is not None:
                    for item in speech.poll(cognition.trigger.episode_id):
                        log.write(dict(schema=1, type="voice", run_id=run_id, **item))
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
                decision_started = perf_counter()
                record = cognition.update(frame.sample_id, frame.t, eyes, status, metrics, frame.image.shape,
                    evidence_age_s=age, max_age_s=settings["max_frame_age_s"] if args.camera else None)
                cognition_ms = (perf_counter() - decision_started) * 1000
                decision_age_s = age + cognition_ms / 1000 if args.camera else None
                log.write(dict(schema=1, type="observation", run_id=run_id,
                               synthetic=args.synthetic_source, read_age_s=age,
                               cognition_ms=cognition_ms, decision_age_s=decision_age_s, **record))
                for event in record["decision"]["events"]:
                    print(f"{event['t']:.3f} {event['kind']} {event['episode_id']}", flush=True)
                    if audio is not None:
                        log.write(dict(schema=1, type="actuation", run_id=run_id,
                                       event_id=event["event_id"], status=audio.submit(event)))
                    if speech is not None:
                        log.write(dict(schema=1, type="voice", run_id=run_id,
                                       event_id=event["event_id"], status=speech.submit(event)))
                count += 1
                if args.display:
                    view = image.copy() if not args.camera else frame.image.copy()
                    label = f"{args.alert_mode.upper()} {record['decision']['state']} | coverage {record['rolling']['coverage']:.0%}"
                    cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                    for index, (mode, mode_record) in enumerate(record["modes"].items()):
                        values = ', '.join(f"{e['side']}={e['value']:.3f}" if e['valid'] else f"{e['side']}=UNKNOWN"
                                           for e in mode_record["estimates"])
                        text = f"{mode}: {mode_record['decision']['state']} {values}"
                        cv2.putText(view, text, (10, 60 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                    cv2.putText(view, f"{status} | experimental settings | q to quit", (10, view.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
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
            if speech is not None:
                speech.close()
                for item in speech.poll(None):
                    log.write(dict(schema=1, type="voice", run_id=run_id, **item))
            log.write(dict(schema=1, type="run_end", run_id=run_id, frames=count,
                           loop_seconds=perf_counter() - loop_started,
                           active_episodes=cognition.active_episodes(),
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
                    try:
                        if speech is not None:
                            speech.close()
                    finally:
                        if args.display:
                            cv2.destroyAllWindows()
    print(f"Processed {count} frame(s). Log: {args.output}")
