import json
import sys
import time
import uuid
from pathlib import Path
from marvel.pipeline import Pipeline
from marvel.observability.logger import JsonLogger


def run(config_path):
    import cv2
    import yaml
    from marvel.perception.capture import LiveCapture, VideoCapture
    from marvel.perception.detection import Detector
    from marvel.cognition.cnn_inference import CNN, OpenCVCNN, load_bundle, sha256
    from marvel.cognition.features import aspect_ratio
    from marvel.cognition.pose import estimate_pose
    from marvel.observability.overlay import show
    from marvel.reasoning.worker import SpeechWorker
    from marvel.actuation.audio import Audio
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config["mode"] not in ("ear", "cnn", "fusion", "compare"):
        raise ValueError("Unknown mode")
    live = config["capture"]["kind"] != "video"
    if config["capture"]["kind"] not in ("camera", "gstreamer", "video"):
        raise ValueError("Unknown capture kind")
    if config["llm"]["enabled"] and (not live or config["llm"]["backend"] == "undecided"):
        raise ValueError("Speech requires live mode and an explicitly selected backend")
    if config["llm"]["ttl_s"] <= config["llm"]["timeout_s"]:
        raise ValueError("Speech TTL must exceed the hard timeout")
    run_id = uuid.uuid4().hex
    folder = Path(config["runs_dir"])/run_id
    folder.mkdir(parents=True)
    (folder/"runtime.yaml").write_text(Path(config_path).read_text(), encoding="utf-8")
    detector = capture = audio = worker = log = None
    try:
        log = JsonLogger(folder/"observations.jsonl")
        bundle = cnn = None
        if config["bundle"]:
            bundle, checkpoint = load_bundle(config["bundle"])
            if config["mode"] != "ear":
                if config["inference_backend"] == "torch":
                    cnn = CNN(checkpoint, bundle["temperature"], config["device"])
                elif config["inference_backend"] == "opencv_onnx":
                    cnn = OpenCVCNN(config["onnx_path"], bundle["checkpoint_sha256"], bundle["temperature"])
                else:
                    raise ValueError("Unsupported inference backend")
        pipeline = Pipeline(config, bundle, cnn)
        selected = config["audible_mode"] if config["mode"] == "compare" else config["mode"]
        if selected not in pipeline.modes:
            raise ValueError("Audible mode must be one of the running modes")
        detector = Detector(config["landmarker"], config["driver_roi"], config["quality"])
        capture = LiveCapture(config["capture"]) if live else VideoCapture(config["capture"])
        audio = Audio(folder, config["sound"] and live, config["tts"])
        worker = SpeechWorker(config["llm"])
        camera = json.loads(Path(config["pose_calibration"]).read_text()) if config["pose_calibration"] else None
        log.write({"type": "run", "run_id": run_id, "config": config,
                   "bundle": bundle, "config_sha256": sha256(config_path),
                   "landmarker_sha256": sha256(config["landmarker"]),
                   "synthetic": False, "capture_clock": "monotonic_read_completion" if live else "media"})
        last_t, current_event, pending_speech, current_kind = None, None, None, None
        start = time.monotonic()
        while time.monotonic()-start < config["max_run_s"]:
            started = time.monotonic()
            frame = capture.read()
            if frame is None and not live:
                break
            eyes, mouth, pose_points, reason = [], (), (), "capture_wait"
            if frame is None:
                t, sample_id = time.monotonic(), "watchdog:"+uuid.uuid4().hex
                detector.invalidate()
            else:
                t, sample_id = frame.t, frame.sample_id
                if last_t is not None and t <= last_t:
                    continue
                age = time.monotonic()-t if live else 0
                if age > config["max_frame_age_s"]:
                    reason = "stale_before_inference"
                    detector.invalidate()
                else:
                    eyes, mouth, pose_points, reason = detector.process(frame)
            record = pipeline.update(sample_id, t, eyes)
            # CNN can be expensive. Discard stale predictions before recording/actuation.
            # Pipeline timing is sampled observation time; abort run if processing exceeds budget.
            if live and frame is not None and time.monotonic()-t > config["max_frame_age_s"]:
                log.write({"type": "error", "reason": "inference_exceeded_age_budget", "sample_id": sample_id})
                audio.alert()
                raise RuntimeError("Inference too slow for configured age budget; profile and reconfigure")
            record["reason"] = reason
            record["mar"] = aspect_ratio(mouth) if mouth else None
            record["pose"] = estimate_pose(pose_points, frame.image.shape, camera) if pose_points else {"valid": False}
            record["processing_s"] = time.monotonic()-started
            record["read_to_decision_s"] = time.monotonic()-t if live else None
            for mode, result in record["modes"].items():
                for event in result["events"]:
                    event.update(event_id=f"{run_id}:{mode}:{event['sequence']}", method=mode)
                    if mode == selected:
                        if event["kind"] in ("RECOVERED", "TRACK_ENDED_UNRESOLVED"):
                            current_event, pending_speech, current_kind = None, None, None
                            worker.close()
                            audio.stop_speech()
                        elif event["kind"] in ("PROLONGED_CLOSURE", "MONITORING_UNAVAILABLE"):
                            event["sound_status"] = audio.alert()
                            current_event, current_kind = event["event_id"], event["kind"]
                            worker.close()
                            pending_speech = None
                            if config["llm"]["enabled"]:
                                event["speech_submitted"] = worker.submit(event)
            if current_kind == "MONITORING_UNAVAILABLE" and record["modes"][selected]["observation"] != "UNKNOWN":
                current_event, pending_speech, current_kind = None, None, None
                worker.close()
                audio.stop_speech()
            response = worker.poll(current_event)
            if response:
                log.write(dict(response, type="llm_result"))
                if response["message"]:
                    pending_speech = (response["message"], time.monotonic(), current_event)
            if pending_speech and time.monotonic() >= audio.tone_until:
                message, received, event_id = pending_speech
                if event_id == current_event and time.monotonic()-received < 1:
                    log.write({"type": "tts", "status": audio.speak(message), "event_id": event_id})
                pending_speech = None
            status = audio.poll()
            if status:
                log.write({"type": "audio", "status": status})
            log.write(record)
            last_t = t
            if config["display"] and frame is not None:
                if not show(frame.image, eyes, record["modes"], selected):
                    break
    except KeyboardInterrupt:
        if log:
            log.write({"type": "shutdown", "reason": "keyboard_interrupt"})
    except Exception as exc:
        if log:
            log.write({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        if audio:
            audio.alert()
        raise
    finally:
        for resource in (worker, audio, capture, detector, log):
            if resource:
                resource.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(sys.argv[1])
