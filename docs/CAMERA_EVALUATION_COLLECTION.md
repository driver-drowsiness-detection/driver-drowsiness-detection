# Laptop camera data needed for the EAR/CNN comparison

This is a proposed collection protocol, not completed accuracy evidence.
Use it after the MRL CNN experiment. The earlier laptop demonstration saved
measurements and events but no video, so it cannot be independently relabelled.

## Record inputs that all three methods can share

Record full camera frames, including the face and both eyes, while stationary.
Use the intended laptop camera and preserve the original video. Do not record
only the preview overlay or eye crops: EAR needs landmarks from full frames,
and overlays could show predictions to the reviewer.

For each actual participant, collect separate short clips covering:

1. Looking normally at the camera, with natural blinking and no instructed long
   closures. This supplies negative time for later false-alert evaluation.
2. Deliberate brief and sustained closures followed by reopening, separated by
   several seconds of normal open eyes. Keep the prompt timing separate from
   labels: an instruction to close is not proof of when the eyes closed.
3. The lighting and glasses conditions the participant actually uses. Record
   condition metadata rather than inferring it later from model errors.
4. Moderate head movement, looking away and temporary occlusion. These are
   observation-quality cases; do not label every unavailable eye as closed.

These are staged eye-state and event checks, not genuine fatigue labels.
Keep recordings local under data/raw/camera; this folder is ignored by Git.
The existing run_cognition application does not save video. Use a camera
recording application and close it before trying the live perception application.

## Record real identities and split before tuning

Assign an anonymous, stable subject ID to each actual participant, and unique
session/video IDs to their actual recordings. The same person must keep the
same subject ID across lighting, glasses, devices and recording days.

Assign whole subjects to calibration, validation or final test before comparing
model scores. Keep both eyes and every recording of one subject in the same
partition. Check that calibration and validation each contain both observable
open and observable closed eyes. A small team collection is a pilot; one or
two people per partition cannot establish broad population performance.

Keep final-test recordings and annotations separate from development manifests.
Do not move a difficult person between partitions after seeing their errors.

## Create the independent review queue

Use the existing extractor with the actual paths and identities. The following
is a command template; replace every capitalized placeholder before running:

```powershell
.\.venv-cognition\Scripts\python.exe scripts/extract_frames.py data/raw/camera/ACTUAL_VIDEO.mp4 data/processed/UNIQUE_VIDEO_frames data/processed/UNIQUE_VIDEO_review.csv --subject-id ACTUAL_SUBJECT_ID --session-id ACTUAL_SESSION_ID --video-id ACTUAL_VIDEO_ID --source laptop_camera_pilot --step 5 --limit 600
```

Every fifth decoded frame is a sampling choice for an initial image review,
not a validated temporal sampling rate. Keep the original video for event
annotation. The extractor's limit may truncate a clip; verify the emitted
frame count and run another explicit sampling plan if necessary.

The CSV starts with blank labels and splits. Review the full frame without
consulting EAR/CNN predictions. For each anatomical eye:

| Field | Required meaning |
| :--- | :--- |
| eye_side | The participant's left or right eye, not screen left/right |
| label | 0 = clearly open, 1 = clearly closed; blank when unobservable |
| observable | 1 only when the eye state is independently discernible |
| label_source | human |
| split | The participant's previously assigned partition |
| annotation_status | reviewed after inspection, never before |
| orientation_status | verified after checking actual mirror/orientation behavior |

Agree on the treatment of partial closures before labelling. If a binary label
cannot be justified, keep it unobservable with a blank label and retain it for
coverage reporting. Do not use an EAR cutoff or CNN output to decide its truth.
Have another reviewer resolve disagreements and retain the original annotations
and correction record. A model's rejected frame must stay in the evaluation
inventory so accuracy cannot rise merely by dropping hard cases.

## From labelled frames to measured comparison

```text
Full frames + reviewed labels + fixed subject partitions
                         |
                         v
prepare_dataset.py: common landmarks, eye crops and quality results
                         |
                         v
calibrate_scores.py: fit EAR mapping on calibration subjects
                         |
                         v
tune_fusion.py: compare EAR, CNN and fusion on validation subjects
                         |
                         v
Freeze model, preprocessing, mappings, quality and decision settings
                         |
                         v
Independent final test and separately annotated temporal events
```

Use [the calibration workflow](FUSION_CALIBRATION_WORKFLOW.md) for the implemented
commands. Report recall, open-eye false-positive rate, coverage and subject
breakdowns on the same observations. Keep standalone coverage as well as the
common subset used for direct comparison.

For temporal evaluation, independently annotate closure/reopening intervals and
unscorable intervals in the original video. The extractor derives timestamps
from frame index divided by reported FPS. Verify the recording's timestamp and
frame-rate behavior before treating those values as measured event latency.
Sparse sampled images alone cannot establish false alerts per hour or delay.
