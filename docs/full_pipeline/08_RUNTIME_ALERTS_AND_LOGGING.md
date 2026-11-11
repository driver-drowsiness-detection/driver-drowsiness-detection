# 08. Main runtime, immediate alerts and observability

## Requirement

Connect the individually tested components into one application with explicit lifecycle, one observation clock, independent experiment states, reproducible logs and a warning path that does not wait for generated language.

## Prerequisites

Understand dictionaries, iteration, exceptions, `try/finally`, process ownership and configuration. `try/finally` is essential because a camera or audio process must be released even when inference fails or the user presses Ctrl+C.

## Integration choices

| Choice | Benefit | Limitation |
| --- | --- | --- |
| One long script | Quick first demo | Hard to isolate failures and compare modes |
| Modules coordinated by one main loop | Stable contracts and straightforward profiling | One expensive inference can still delay the loop |
| Multiple inference services | Stronger isolation | More IPC, deployment and timestamp complexity |

The reference uses modules with separate capture and speech workers. The main loop performs perception and cognition sequentially for a readable initial implementation. If profiling finds unbounded model calls, isolate those in a worker as a later engineering change. A capture watchdog cannot run while the main process itself is stuck inside native inference.

## `Pipeline` is the testable core coordinator

1. Construct exactly the requested mode or all three for comparison.
2. Require a model bundle for CNN/fusion.
3. Give each mode its own trigger, rolling window and optional severity state.
4. Validate the two eyes belong to the same observation.
5. Calculate per-eye EAR and, when needed, CNN evidence.
6. Apply calibrated per-eye fusion.
7. Aggregate bilateral state with the same policy for every method.
8. Update each independent temporal history.
9. Return a serializable observation record.

This separation permits fake-estimate tests without a camera and keeps source-of-truth logic out of the overlay.

## Main application walkthrough

`run` reads YAML, validates operating mode and rejects undecided speech configuration if speech is enabled. It creates a unique run directory and saves the runtime configuration. Existing reports are not silently replaced.

Next it loads the optional frozen bundle and the selected inference adapter. It creates perception, capture, audio and speech resources. The run header records settings, clock domain and asset hashes.

Each iteration:

1. Read a frame or receive a bounded live timeout.
2. Use a live watchdog observation when no frame arrives.
3. Reject nonincreasing or stale frames.
4. Run perception and cognition.
5. If live inference exceeds the configured age budget, record an error and stop for reconfiguration instead of actuating a stale prediction as if it were fresh.
6. Log MAR and optional head pose as supporting measurements.
7. Turn selected-mode events into fixed alert requests.
8. Submit optional LLM work only after the fixed request.
9. Cancel stale speech on recovery, track change or restored monitoring.
10. Poll worker/audio status, write measurements, display overlay and accept `q`.

During cleanup, close worker, audio, capture, detector and log; then destroy OpenCV windows.

## Select the mode explicitly

Start with `mode: ear` and `bundle: null`. That uses the illustrative starter EAR threshold. After fitting the bundle:

```yaml
mode: compare
audible_mode: fusion
bundle: models/bundle_v1.json
inference_backend: torch
device: cpu
```

Use `mode: cnn` or `mode: fusion` for individual runs. In comparison mode, all three update from the same perception output, but only `audible_mode` requests sound. There is no automatic fallback from fusion to EAR.

For offline replay:

```yaml
capture:
  kind: video
  source: data/evaluation/session01.mp4
  declared_cfr_fps: null
sound: false
llm:
  enabled: false
  backend: undecided
  model: null
  endpoint: null
  timeout_s: 3.0
  ttl_s: 5.0
```

These are sections to replace in a full runtime configuration, not a complete configuration by themselves. All actual complete files are in the source listings. The runtime forbids LLM speech during video replay so a sped-up evaluation cannot produce a queue of spoken old events.

Run:

```powershell
python -m marvel.main configs\runtime.yaml
```

## Fixed sound policy

The reference creates a short WAV tone once per run. On Windows, `winsound.PlaySound` requests asynchronous playback. On Linux, `aplay` is a subprocess. Confirm the device and volume with a human listening test. “Sound requested” is not proof a physical speaker produced sound.

The first policy plays one bounded tone per new event. It does not beep on every frame. A repeated-warning policy while an event remains active is a separate product decision and should have repeat intervals, limits and tests. Do not let the LLM determine that policy.

GPIO is not wired by the reference. The board revision, physical buzzer type, voltage/current limits and driver circuit are undecided. Do not invent a pin or connect an unknown load directly to a GPIO. A reviewed GPIO adapter can implement the same event interface later.

Speech is a subprocess and can be terminated before a new fixed alarm. The fixed tone gets a short lead time before speech begins. TTS has a ten-second execution cap. Failure status is logged.

## Logs and measurements

The log format is JSONL: one JSON object per line. `JsonLogger` adds UTC wall time, forbids NaN/Infinity and flushes each record. Flushing helps preserve crash evidence but can add latency on slow storage. Profile disk impact; add a bounded logging worker only if needed and explicitly report dropped records.

| Field | Meaning |
| --- | --- |
| `sample_id`, `t`, `track` | Observation identity |
| `evidence` | Raw EAR and calibrated branch scores, validity and reasons |
| `modes` | Observation, trigger state, active alarm, events, window and tier per mode |
| `processing_s` | Main-loop work including capture wait, up to measurement collection |
| `read_to_decision_s` | Live read-completion-to-decision interval |
| `reason` | Missing face, stale frame or related perception status |
| `mar`, `pose` | Supporting measurements, not automatic fatigue diagnoses |
| `type` | Run header, observation, LLM result, audio status, error or shutdown |

The two timing fields exclude some later logging/overlay/output work and do not measure camera exposure latency. Do not present them as full sensor-to-speaker latency. Measure that separately using timestamped instrumentation or a controlled visual/audio recording.

## Failure boundary

This is a research reference, not a fault-tolerant automotive controller. Native landmarker/CNN calls can block; disk writes can stall; processes share hardware resources. A hard real-time guarantee needs additional architecture and target testing.

Unknown current observations preserve an active alarm until recovery or an explicit unresolved track ending. Any stopped run must visibly report monitoring stopped. The reference reports exceptions and logs errors. After an unrecoverable exception, sound cleanup may stop an attempted notification quickly; do not count that exception path as a validated hardware fault alarm.

## Notes and retention

Used one selected audible mode because three simultaneous alarms would distort demonstrations and event counting. Used a run header and hashes because a metric without model/configuration identity cannot be reproduced.

Interview explanation: “We separate decision logic from outputs, save enough evidence to reproduce every event, and bound camera and language-generation work. We distinguish requested audio from verified audible output.”

Retention check: identify which module should own the alarm threshold, which should own speech timeout, and which should record both.

<!-- CODE -->

## Complete source files

### File: `src/marvel/pipeline.py`

[Open the companion file](reference/src/marvel/pipeline.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `Pipeline`: lines 10 to 55.

```python
from dataclasses import asdict
from marvel.contracts import UNKNOWN, bilateral
from marvel.cognition.ear_classifier import measure, probability, state as ear_state
from marvel.cognition.fusion import combine, state as score_state
from marvel.cognition.trigger import Trigger
from marvel.cognition.perclos import ClosureWindow
from marvel.cognition.state_machine import FatigueState


class Pipeline:
    def __init__(self, config, bundle=None, cnn=None):
        self.config, self.bundle, self.cnn = config, bundle, cnn
        self.modes = ("ear", "cnn", "fusion") if config["mode"] == "compare" else (config["mode"],)
        if any(m != "ear" for m in self.modes) and (bundle is None or cnn is None):
            raise ValueError("CNN and fusion need a frozen bundle")
        self.triggers = {m: Trigger(**config["temporal"]) for m in self.modes}
        self.windows = {m: ClosureWindow(**config["window"]) for m in self.modes}
        self.tiers = {m: FatigueState(config["fatigue_policy"]) for m in self.modes}
        self.track = "not_acquired"

    def update(self, sample_id, t, eyes):
        observations = {m: UNKNOWN for m in self.modes}
        evidence = []
        if len(eyes) == 2:
            if len({e.track for e in eyes}) != 1 or {e.side for e in eyes} != {"left", "right"}:
                raise ValueError("Invalid bilateral identity")
            if any(e.sample_id != sample_id or e.t != t for e in eyes):
                raise ValueError("Frame and eye identity mismatch")
            self.track = eyes[0].track
            states = {m: [] for m in self.modes}
            for eye in eyes:
                ear = measure(eye)
                evidence.append(asdict(ear))
                if "ear" in states:
                    threshold = self.bundle["ear_threshold"] if self.bundle else self.config["ear_threshold"]
                    states["ear"].append(ear_state(ear, threshold))
                if self.cnn:
                    cnn = self.cnn.estimate(eye)
                    evidence.append(asdict(cnn))
                    if "cnn" in states:
                        states["cnn"].append(score_state(cnn, self.bundle["cnn_threshold"]))
                    if "fusion" in states:
                        fused = combine(probability(ear, self.bundle["ear_calibration"]), cnn, self.bundle["alpha"])
                        evidence.append(asdict(fused))
                        states["fusion"].append(score_state(fused, self.bundle["fusion_threshold"]))
            observations = {m: bilateral(*values) for m, values in states.items()}
        results = {}
        for mode, observation in observations.items():
            result = self.triggers[mode].update(t, observation, self.track)
            window = self.windows[mode].update(t, observation, self.track)
            tier = self.tiers[mode].update(t, window["closure_fraction"], result["alarm_active"],
                                         observation != UNKNOWN, self.track)
            results[mode] = dict(result, observation=observation, window=window, tier=tier)
        return {"type": "observation", "sample_id": sample_id, "t": t,
                "track": self.track, "evidence": evidence, "modes": results}
```

### File: `src/marvel/main.py`

[Open the companion file](reference/src/marvel/main.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `run`: lines 10 to 144.

```python
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
```

### File: `src/marvel/actuation/audio.py`

[Open the companion file](reference/src/marvel/actuation/audio.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `Audio`: lines 10 to 94.

```python
import math
import os
import shutil
import struct
import subprocess
import time
import wave


class Audio:
    def __init__(self, folder, enabled, tts):
        from pathlib import Path
        self.enabled, self.tts = enabled, tts
        self.speech = self.tone = None
        self.speech_started = 0
        self.tone_until = 0
        self.path = Path(folder)/"fixed_alert.wav"
        if enabled:
            with wave.open(str(self.path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(b"".join(struct.pack("<h", int(7000*math.sin(2*math.pi*900*i/16000)))
                                       for i in range(6400)))

    def stop_speech(self):
        if self.speech and self.speech.poll() is None:
            self.speech.terminate()
            try:
                self.speech.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                self.speech.kill()
        self.speech = None

    def alert(self):
        self.stop_speech()
        if not self.enabled:
            return "sound_disabled"
        self.tone_until = time.monotonic()+0.5
        if os.name == "nt":
            import winsound
            winsound.PlaySound(str(self.path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return "sound_requested"
        if not shutil.which("aplay"):
            return "sound_failed_aplay_missing"
        if self.tone and self.tone.poll() is None:
            self.tone.terminate()
        self.tone = subprocess.Popen(["aplay", "-q", str(self.path)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "sound_requested"

    def speak(self, message):
        if not self.enabled or self.tts == "disabled":
            return "speech_disabled"
        self.stop_speech()
        kwargs = {"stdin": subprocess.PIPE, "stdout": subprocess.DEVNULL,
                  "stderr": subprocess.DEVNULL, "text": True}
        if self.tts == "windows_sapi" and os.name == "nt":
            # Message is input DATA. It is never interpolated into shell code.
            script = ("Add-Type -AssemblyName System.Speech; "
                      "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      "$sentence = [Console]::In.ReadToEnd(); $speaker.Speak($sentence)")
            self.speech = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive",
                                            "-Command", script], creationflags=0x08000000, **kwargs)
        elif self.tts == "espeak" and shutil.which("espeak-ng"):
            self.speech = subprocess.Popen(["espeak-ng", "--stdin"], **kwargs)
        else:
            return "speech_failed_backend_unavailable"
        self.speech.stdin.write(message)
        self.speech.stdin.close()
        self.speech_started = time.monotonic()
        return "speech_requested"

    def poll(self):
        if self.speech and self.speech.poll() is not None:
            result = "speech_completed" if self.speech.returncode == 0 else "speech_failed"
            self.speech = None
            return result
        if self.speech and time.monotonic()-self.speech_started > 10:
            self.stop_speech()
            return "speech_timeout"
        if self.tone and self.tone.poll() is not None:
            result = "sound_completed" if self.tone.returncode == 0 else "sound_failed"
            self.tone = None
            return result
        return None

    def close(self):
        self.stop_speech()
        if self.tone and self.tone.poll() is None:
            self.tone.terminate()
        if self.enabled and os.name == "nt":
            import winsound
            winsound.PlaySound(None, 0)
```

### File: `src/marvel/observability/logger.py`

[Open the companion file](reference/src/marvel/observability/logger.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `JsonLogger`: lines 5 to 15.

```python
import json
from datetime import datetime, timezone


class JsonLogger:
    def __init__(self, path):
        self.file = open(path, "x", encoding="utf-8")

    def write(self, record):
        value = dict(record, wall_utc=datetime.now(timezone.utc).isoformat())
        self.file.write(json.dumps(value, allow_nan=False, separators=(",", ":"))+"\n")
        self.file.flush()

    def close(self):
        self.file.close()
```

### File: `src/marvel/observability/overlay.py`

[Open the companion file](reference/src/marvel/observability/overlay.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `show`: lines 1 to 12.

```python
def show(image, eyes, results, selected):
    import cv2
    view = image.copy()
    for eye in eyes:
        for x, y in eye.points:
            cv2.circle(view, (int(x), int(y)), 2, (0, 200, 0), -1)
    for line, (mode, result) in enumerate(results.items(), start=1):
        text = f"{mode}: {result['state']} active={result['alarm_active']}"
        cv2.putText(view, text, (10, 25*line), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if mode == selected else (255, 255, 255), 1)
    cv2.imshow("MARVEL", view)
    return cv2.waitKey(1) & 0xFF != ord("q")
```

