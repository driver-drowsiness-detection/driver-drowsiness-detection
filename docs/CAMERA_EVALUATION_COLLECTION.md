# Laptop camera data needed for the EAR/CNN comparison

Updated 10 November 2026. A first local participant recording is now available;
human annotation and separate participants are still required for meaningful
calibration, validation and final testing. This document is a collection and
review protocol, not completed camera accuracy evidence.

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
The existing run_cognition application does not save video. The new
record_camera_session.py saves full PNG frames, actual camera read-completion
timestamps, capture frame IDs, a session record and a blank review queue.
This avoids deriving time from an assumed constant video FPS. It uses the
existing latest-frame capture worker; some frames can be skipped, and gaps
must remain visible in analysis. Read-completion time is not sensor exposure time.

## Record with a visible start gate

Close other applications using the camera, then use a new session directory
and the actual participant's stable anonymous ID:

```powershell
.\.venv-cognition\Scripts\python.exe scripts/record_camera_session.py data/raw/camera/NEW_SESSION --subject-id ACTUAL_PARTICIPANT_ID --seconds 60
```

The first window is preview only. No image is saved until the participant
presses Space in that window. Q or Escape cancels. An unstarted preview times
out after 120 seconds. During recording, the same keys stop early. A missing
camera frame for ten seconds fails explicitly and preserves partial evidence.
The CLI defaults to this start gate; tests can bypass it with controlled frames.

The sixty-second exercise prompts normal blinking, two brief deliberate
closures and gentle head movement. The participant counts two seconds and
reopens; do not wait for a new prompt with eyes shut. Prompt timing records
instructions, never labels. Saved frames are original input without the
preview text. The preview works on a copy.

Every received image is saved with its timestamp in frames.csv. Approximately
every half second, two blank eye rows are added to review.csv. Original full
frames remain available for denser review and later event annotation. PNGs
use more space than a compressed video: the verified second recording saved
674,388,072 image bytes for 1,739 frames.

## Human review window

For the current confirmed participant, the chosen partition is calibration.
Keep every later recording of this same person in that partition; use other
people for validation and final test. The first capture had no visible prompts
according to the user and is marked unguided in session_feedback.json. Use
the second capture for the intended guided-session review:

```powershell
.\.venv-cognition\Scripts\python.exe scripts/review_camera_session.py data/raw/camera/local_user_001_2026_11_09_session_02/review.csv --reviewer-id local_user_001 --split calibration
```

The window shows 119 full frames, representing 238 eye observations. It does
not load or display EAR/CNN predictions or prompt labels. Verify anatomical
left/right orientation, then use O for both open, C for both closed, or U for
both uncertain. Separate controls support differing eye states. Click the
image for an enlarged inspection view. A partial closure or reflection that
cannot support a clear binary state should remain uncertain.

Each saved decision updates human_review_progress.json and appends to
human_review_actions.jsonl. Closing and reopening resumes saved decisions.
The original review.csv is never edited. A queue hash, reviewer ID and fixed
participant partition prevent resuming the wrong review. Export is blocked
until every sampled frame is reviewed and orientation is confirmed.

Export writes human_reviewed.csv once. Open is label 0, closed is label 1,
and uncertain is observable 0 with a blank label. Only explicit human choices
receive label_source=human and annotation_status=reviewed. The output still
needs both usable classes before fitting, and invalid perception observations
must be retained for coverage reporting.

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
