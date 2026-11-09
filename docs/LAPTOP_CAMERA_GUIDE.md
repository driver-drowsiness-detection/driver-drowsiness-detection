# Laptop camera application

Updated 9 November 2026. Laptop development is the current priority. The user
reports that Nano cooling is unavailable, so board validation is deferred.

## Run the verified EAR path

From the repository root in PowerShell, use a new log name for every run:

```powershell
.\.venv-cognition\Scripts\python.exe scripts/run_cognition.py --camera --display --audio --voice fixed --ear-threshold 0.2 --max-seconds 90 --output reports/local/my_camera_check_01.jsonl
```

The window shows the selected alert mode, state, per-eye values, face status
and time coverage. Press q or Escape to close. The 90 second bound also stops
the run automatically. The process releases the camera at exit.

The 0.2 EAR threshold and default timing are experimental settings. They worked
in the recorded interactive test but have not been validated across users or
conditions. Default sustained closure is 1.5 seconds; recovery requires 0.5
seconds of valid reopening. Do not use this prototype for driving decisions.

The runtime saves measurements and events, not camera frames. Speech creates
local WAV files beside the run log. Logs refuse overwrite. The default camera
is source 0 in configs/runtime.yaml; another device needs an explicit config.

## Current flow

```text
Laptop camera → landmark detection → eye geometry and crops
                                ↓
                       current EAR measurements
                                ↓
                       bilateral eye observation
                                ↓
                      duration and recovery logic
                         ↙                ↘
                   fixed tone       background speech
                                ↓
                      event and performance logs

Optional trained CNN artifact → CNN estimates → its own history
Explicit fusion settings → mapped scores → fusion → its own history
```

The same selected mode supplies the overlay's main state and the alert events.
All available modes remain logged. Comparison does not play three competing
alarms. A slow CNN is checked against the frame freshness limit before its
evidence reaches any timer.

## Optional CNN and fusion

The loader supports a PyTorch checkpoint or ONNX plus its adjacent metadata.
Arushi's trained artifact is now available locally and hash verified. A saved
CEW face has passed through the application with actual CNN inference, and the
baseline ONNX export passed a 100-crop conversion parity check. These checks
verify execution and conversion, not camera accuracy. See
[the local CNN experiment](MRL_LOCAL_CONTINUATION_2026_11_09.md).

```powershell
.\.venv-cognition\Scripts\python.exe scripts/run_cognition.py --camera --display --ear-threshold 0.2 --cnn-checkpoint models/checkpoints/mrl_full_3epoch.pt --cnn-threshold 0.5 --alert-mode ear --max-seconds 90 --output reports/local/camera_cnn_check_01.jsonl
```

This example keeps EAR as the alert source while displaying/logging CNN.
The 0.5 CNN threshold is the experimental baseline, not an approved live rule.
Use --device cuda only with a compatible PyTorch environment. ONNX uses
--cnn-onnx and OpenCV DNN. Logs record the artifact hash and score settings.

Fusion additionally needs --comparison-config pointing to JSON with exactly:
ear_calibration (slope and intercept), fusion_threshold, and alpha. There is
deliberately no production default or fabricated fitted calibration file.
Use --alert-mode fusion only as an explicitly configured experiment. The
comparison guide explains missing evidence and coverage implications.

## Speech choices

| Choice | Behavior | Use |
| :--- | :--- | :--- |
| Omit --voice | Fixed tone only if --audio is supplied | Smallest runtime |
| --voice fixed | Approved fixed sentence through Windows SAPI | Verified laptop demonstration |
| --voice llm | Local model generates wording asynchronously, then SAPI speaks | Experimental local model path |

For LLM mode, use the pinned local folder under models/llm or pass --voice-model.
The existing download utility and environments/windows-voice.txt describe the
optional dependencies. Model assets remain ignored by Git. The runtime logs
the voice manifest hash, and the generator verifies its listed file hashes.

The immediate tone is submitted before speech. An unready or failed generation
worker falls back to the fixed sentence. A timeout falls back only if the same
closure episode is still active. Recovery makes late speech stale; it is
discarded or stopped. Busy workers drop new work rather than building a queue.
The wording filter is conservative and does not prove semantic correctness.

## Read the performance report

```powershell
.\.venv-cognition\Scripts\python.exe scripts/profile_runtime.py reports/local/my_camera_check_01.jsonl reports/local/my_camera_check_01_profile.json
```

The report includes processed frames per loop second, cognition time,
estimated camera read-completion to decision age, face status counts, usable
observations per mode, events and unresolved episodes. Heartbeats are separate
from processed frames. Camera startup time is included in loop throughput.
Inference, overlay, logging and speech scheduling run within that loop.

The age estimate adds elapsed cognition time to the measured age after
perception. It does not measure sensor exposure latency. Cognition time excludes
landmark detection. Memory and temperature are explicitly marked unmeasured.
A completed run_end is required; a failed or truncated log is not summarized
as successful. This report does not calculate accuracy or false alerts per hour.

## Implementation walkthrough

1. runtime.py parses explicit source, model, alert and speech choices. Missing
   artifacts and incompatible options fail before opening the camera.
2. CameraCognition constructs the existing EAR pipeline plus optional CNN and
   comparison histories. Its trigger property exposes only the selected alert
   mode to lifecycle and actuation code.
3. update verifies paired identity and current face status before CNN inference.
   Each returned estimate must preserve the input eye identity and score range.
4. After inference, the freshness check can replace stale evidence with UNKNOWN
   before any closure timer advances. Recorded media does not use this live age gate.
5. EAR still computes MAR and optional calibrated pose. Independent mode records
   are attached to the same observation. No model is trained inside the runtime.
6. SpeechAlerts consumes confirmed events. VoiceWorker owns a separate generation
   process; VoiceOutput owns bounded asynchronous rendering/playback.
7. The loop writes timing, outcomes and artifact identity. Finalization closes
   camera, detector, audio, speech and windows even if execution raises an error.
8. profile_runtime.py summarizes completed evidence without inventing labels.

## Interview explanation

I integrated an EAR camera detector with bounded audio and optional asynchronous
language generation. The model never decides whether to alert. CNN and fusion
share observation contracts but keep separate temporal histories. A second
freshness check prevents slow model inference from turning old images into
current evidence. I verified live closures and recovery with user feedback,
while keeping controlled tests separate from accuracy evaluation.

Retention question: if speech finishes after the eyes have reopened, should it
play? Under the current episode policy, no: it is stale and discarded. The
immediate warning was already requested when closure was confirmed.
