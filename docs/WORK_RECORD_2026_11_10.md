# Camera recording and independent review preparation

Continuation recorded 10 November 2026. Local session directory names retain
their original 2026_11_09 names. Exact capture timestamps remain in recording.json.

## Requirement and decision

The user confirmed that new camera recordings were needed, that they were the
participant, and that they were ready. Added a local full-frame recorder using
the existing bounded LiveCapture worker. Used PNG frames and explicit camera
read-completion timestamps because constant-FPS video timestamps could hide
capture gaps. PNG storage costs more space, but preserves original pixels and
supports the existing full-frame preparation adapter directly.

Added a local Tkinter review window instead of asking the user to edit hundreds
of CSV cells. Keyboard shortcuts handle paired states, separate controls handle
different eye states, and a zoom view helps inspect eyes behind glasses. Model
predictions and prompt labels are excluded from the review view.

## Observed recording failure and correction

The first recorder invocation saved 1,250 frames spanning 59.938 seconds.
Its file checks passed, but the user reported that no preview window appeared.
The assistant had not verified window visibility before calling it a guided
session. File success did not establish successful participant interaction.

Recorded that feedback in the first session's local session_feedback.json and
classified it as unguided. No closure labels were inferred and no frames were
deleted. Added a Space-key start gate: preview images are not saved until the
participant starts recording in the actual window. Cancellation saves no images.

For the second session, desktop inspection found the MARVEL recording window,
brought it forward, and observed the recording UI. The process log confirmed
the user pressed Space. After completion, the window was absent from the desktop.
This verifies recording interaction, not every requested eye action.

## Second-session integrity evidence

Local folder: data/raw/camera/local_user_001_2026_11_09_session_02/.

| Check | Observed value |
| :--- | ---: |
| Successfully decoded full frames | 1,739 |
| Resolution | 640 by 480 |
| First-to-last saved timestamp span | 59.985 seconds |
| Mean saved-frame rate | 28.974 frames/second |
| Median saved interval | 0.031 seconds |
| Largest saved interval | 0.078 seconds |
| Source frames skipped between saved frames | 62 |
| PNG bytes | 674,388,072 |
| Sampled frames for first review | 119 |
| Blank per-eye review rows | 238 |

Every PNG decoded, saved timestamps increased strictly, capture IDs were unique,
and all review-image paths resolved. Labels, observability and partition fields
in the original queue were still blank. The local integrity.json stores the
counts and hashes of frames.csv, review.csv and recording.json. No model was
used to assign labels. Two visually inspected frames around 16 and 31 seconds
showed open eyes; this is not proof that closures were absent at other times.
The prompt schedule must never substitute for visual annotation.

## Code linkage and behavior

```text
LiveCapture worker
    |
    v
Visible preview + participant Space key
    |
    v
Original PNG frames + frames.csv with read timestamps
    |
    v
Blank two-eye queue sampled approximately every half second
    |
    v
Human review window: image only, separate or paired eye choices
    |
    v
Progress JSON + revision journal + reviewed CSV after complete review
    |
    v
Existing prepare_dataset.py and calibration workflow
```

record_camera_session.py validates arguments, creates a new output directory,
starts the isolated capture worker, waits for a real start key, writes original
images before preview overlays, and flushes timing/queue rows as they are saved.
Its final record distinguishes completion, early stop, cancellation and failure.
Camera ownership is released by the capture context even after a write failure.

review_camera_session.py validates paired frame identity and image paths before
showing the queue. ReviewSession.save requires explicit states and verified
orientation. It preserves revisions in an append-only action journal and writes
a resumable progress file. Export requires a complete review and refuses to
overwrite an existing reviewed CSV. The original queue remains unchanged.

The participant is reserved for calibration. This is one participant, not a
subject-independent calibration/validation/test dataset. Other people and
conditions are still needed before claims of camera generalization or fusion
improvement. Review completion and class balance must be checked before fitting.

## Verification

Twelve focused tests passed across test_camera_recording.py,
test_camera_review.py and test_capture_worker.py. They cover original pixels
without overlays, real timestamp intervals and capture-ID gaps, blank labels,
disk failure with partial evidence, duplicate time rejection, start gating,
zero-image cancellation, incomplete-review rejection, uncertain-label encoding,
resume identity, revision history and overwrite protection.

The actual recording and review windows were inspected on the desktop. The
review window was opened for the user at frame 1 of 119 with zero labels saved.
All recordings and review outputs remain local under ignored data/raw/camera.
No independent validation or fusion accuracy was established by recording alone.
The subsequent human-labelled calibration diagnostic is recorded below.

## Engineering lesson

A process that successfully writes files may still fail its user interaction.
Verify the window and require an observable start action before beginning a
guided capture. Keep instructions, measured observations and independent truth
labels separate. A successful recording is a prerequisite for accuracy work,
not an accuracy result.

## Completed human review, preparation and calibration diagnostic

The user finished all 119 frames and exported the reviewed CSV: 196 open,
32 closed and 10 uncertain eye observations. The manifest validator accepted
238 rows for one participant, all assigned to calibration. Perception produced
238 crops; all 228 observable labels retained usable geometry and crops.

The existing EAR fitter converged on these calibration observations. Its mapping
is provisional and was not installed in runtime. A fixed-cutoff diagnostic
compared EAR and both actual CNN checkpoints on the same 228 eyes. EAR detected
22 of 32 closed eyes with zero open-eye false positives. The original CNN detected
2 closures with 81 false positives; the new CNN detected 1 with 26 false positives.
The fitted EAR mapping detected 26 closures with zero false positives on its own
fitting data. That last result is not independent validation.

Score-order analysis showed that lowering the CNN threshold to catch 31 closures
would misclassify 190 of 196 open eyes. Inspected two failed crops and their
full-frame sources; no specific preprocessing implementation fault was established.
The MRL gains therefore do not justify camera CNN alerts or fusion adoption.

See CAMERA_CALIBRATION_DIAGNOSTIC_2026_11_10.md for metrics, hashes, chart,
limitations and the next target-domain data prerequisite. Source images, labels
and detailed per-eye scores remain ignored local artifacts. No CNN retraining
used this calibration participant, and no final test data was scored.

## Public data and transfer learning continuation

The user selected public datasets instead of more teammate recordings. Verified
the official RT-BENE ResNet18 model checksum, input contracts and anatomical eye
ordering. Added isolated offline inference and public-data download/training
scripts with attribution. Installed TorchVision without replacing the existing
PyTorch version; pip check found no broken requirements.

The official model failed the fixed-cutoff camera and public compatibility checks.
Both original and aligned crops were tested. An archived-training RGB resize
variant also failed. These failures remain recorded, not replaced by later results.

Downloaded five checksum-verified public subjects. A five-epoch ImageNet transfer
pilot trained on 5997 labelled pairs from subjects 3/7/12, selected epoch 2 and
cutoff 0.948702 on 1810 pairs from subject 15, then evaluated 4549 pairs from
subject 16. No exact cross-partition pair duplicates were found. The local camera
participant never supplied gradients or the public cutoff.

Public validation detected 16/42 closures with 7 false positives among 4507 open
pairs. On 114 reviewed camera pairs, the new CNN detected 2/16 closures, EAR
detected 10/16, and exploratory OR fusion detected 11/16. All had zero open-pair
false positives; EAR and fusion each retained two UNKNOWN closed pairs. This is
one additional detected camera closure, not independently demonstrated fusion
reliability. No live alert settings changed and no model was deployed to Nano.

See RT_BENE_PUBLIC_DATA_EXPERIMENT_2026_11_10.md for source attribution, experiment
design, concepts, code map, results and commands. Aggregate evidence is shared in
docs/evidence; images, detailed scores and checkpoints remain ignored locally.
