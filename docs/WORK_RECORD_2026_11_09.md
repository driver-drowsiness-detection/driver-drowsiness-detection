# Laptop integration record: 9 November 2026

## User direction and artifact search

The user requested autonomous artifact lookup and completion of laptop camera
work before Nano testing. They report missing Nano cooling. No Nano execution
was attempted.

Fetched origin: cnn remained at 14ae038. Searched project model paths,
Downloads and Desktop including ignored model files. A broader initial
Documents filename search also found no candidate. No eye checkpoint or ONNX
artifact was found. The public GitHub releases query returned no releases;
Actions reported zero artifacts. The tracked CNN model folders contain only
markers and documentation. This does not search Arushi's private laptop.

## Implemented

1. CameraCognition connects current eye observations to optional CNN and all
   three comparison histories; validates returned score identities.
2. CLI options for checkpoint/ONNX, score threshold/temperature, explicit
   fusion settings, device, alert source and bounded run duration.
3. A freshness check after CNN inference, before temporal state mutation.
4. Per-mode overlay values and states, artifact identities, timing logs and
   final per-mode episode disposition.
5. Fixed speech and optional local LLM workers connected to confirmed events;
   current-episode fallback, stale-result rejection and bounded work queues.
6. Runtime profiler replacing the placeholder, plus a standalone background
   voice check using an explicitly synthetic event.
7. Tests and the LAPTOP_CAMERA_GUIDE.md walkthrough and run instructions.

Existing voice drafts were reviewed and integrated. The optional model remains
local and ignored. No real CNN was trained or substituted with random weights.

## Live evidence

The first run processed 450 frames in 26.21 seconds and closed cleanly. Early
frames had no face; 116 were usable for EAR. The camera reopened for the next
run, confirming release/restart in this session.

The second run processed 2,656 frames in 90.02 seconds, approximately 29.50
processed frames per second. There were 2,654 valid face frames and two no-face
frames. EAR made a usable bilateral observation on 2,607 frames. Ten additional
capture-stall heartbeat observations were logged; these are not camera frames.

| Measurement | Observed value |
| :--- | :--- |
| Prolonged closure events | 8 |
| Recovery events | 8 |
| Active episodes at shutdown | None |
| Median cognition time | 0.308 ms |
| P95 cognition time | 0.578 ms |
| Median estimated read-completion to decision age | 16.20 ms |
| P95 estimated read-completion to decision age | 31.25 ms |

The cognition time excludes landmark processing. Timing is from this laptop
session and does not establish sustained thermal performance or Nano speed.
No camera frames were saved. Measurement JSONL and speech WAVs are local.

The user explicitly confirmed: "Warnings matched my eye closures, and I heard
the sound and speech". This is human acceptance evidence for the interactive
session. It is not an independently labelled benchmark or a measured false
alert rate. Logs show completed tone calls and speech playback requests. One
early speech render was discarded after recovery, as required by episode policy.

Artifacts under reports/local:

* laptop_2026_11_09_camera_1.jsonl and its profile JSON
* laptop_2026_11_09_camera_2.jsonl and its profile JSON
* laptop_2026_11_09_camera_2_speech/ contains generated warning WAVs

## Verification

Full repository suite: 203 tests passed in 58.31 seconds with the local face
landmark asset enabled. Focused worker/camera/profiling checks also passed.
pip check reported no broken requirements.

The real background VoiceWorker loaded the pinned local model on CUDA, generated
in approximately 2.20 seconds, and rendered a 4.95 second WAV through Windows
SAPI. Total worker startup, manifest verification, generation and rendering took
55.27 seconds. This check used one explicitly synthetic event and did not play
audio. Its generated wording failed the content filter, so source=fixed_fallback
is recorded. This is evidence of background execution and fallback behavior,
not successful accepted LLM wording. Log:
reports/local/laptop_2026_11_09_voice_worker/worker.jsonl.

The tests cover selected-mode actuation, missing artifacts before camera startup,
stale inference rejection, mismatched CNN identity, worker startup/generation
timeouts, late results, process failure, stale speech, busy queues, malformed
timings and incomplete performance logs. Actual CNN weights were unavailable;
controlled adapters verify that integration boundary.

## Still required

* Obtain the trained eye checkpoint or ONNX and metadata. Its expected hashes
  are in CNN_EXPORT_AND_LIVE_PROBE.md. Actual CNN camera validation remains pending.
* Fit and validate EAR mapping, CNN score settings and fusion on independently
  labelled camera data. The training dataset was infrared; RGB transfer is unproven.
* Add an independent event annotation/evaluation protocol for event recall,
  false alerts per hour and delay. Today's closures were an interactive check.
* Measure camera intrinsics before real head pose evaluation. MAR remains
  telemetry and the rolling metric remains a closure proxy.
* Validate more people, glasses, lighting, head movement and longer sessions.
* Resume Nano compatibility, cooling, camera, audio and resource testing when available.

## Decision

Used the existing EAR path for the live laptop check because it has usable
local assets. Added optional CNN/comparison adapters so the missing artifact
does not require rewriting the application later. Used fixed speech for human
acceptance because it isolates camera/event/audio behavior from generation.
The local LLM is an optional separately tested enhancement.

## Follow-up: prerequisites for meaningful EAR/CNN fusion

The user asked whether integration improves accuracy and authorized prerequisites
where evidence was missing. Implemented the two former placeholders:
calibrate_scores.py now fits the decreasing EAR mapping from calibration eyes;
tune_fusion.py scores validation crops with a supplied real checkpoint and
reports fusion candidates, baseline comparisons and coverage. Added supporting
modules, leakage checks and tests. No camera setting was changed.

Focused suite: 17 tests passed in 12.13 seconds. Tests include the complete CLI
with a controlled synthetic checkpoint and proof that changing validation labels
cannot alter the fitted EAR mapping. The full repository suite passed 220 tests
in 58.13 seconds with the real local landmark asset enabled.

A synthetic demonstration under reports/local/fusion_prerequisites_2026_11_09/
selected zero CNN weight when given deliberately uninformative CNN scores.
Fusion had seven usable rows and EAR eight; this difference is retained in the
report. No synthetic accuracy value is presented as project performance.
The real model and reviewed camera dataset are still needed for real fitting.
See FUSION_CALIBRATION_WORKFLOW.md for commands, algorithm walkthrough and gates.
