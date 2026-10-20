# Cognition implementation and evidence plan

Audit date: 2026-10-20. CNN owner: Arushi. EAR owner: Vishruth, confirmed this session.

## Verified GitHub starting state

| Branch | Audited revision | State before this increment |
| --- | --- | --- |
| main | 0dadd58 | Project scaffold; no integrated perception |
| final-perception | 63ddcc1 | Integrated perception, shared Frame/Eye/Estimate |
| perception | d1399d6 | Original capture/landmark contribution |
| perception01 | 154aa85 | Original crop/validity contribution |
| cnn | 730d436 | Initial small CNN, training and inference code |

PR #2 was closed without merging on October 14. The integrated code survives on
`final-perception`. The two earlier perception CI runs succeeded. This does not
prove Jetson execution or completion of the human webcam acceptance checklist.

The current work merges `final-perception` into existing `cnn`, preserving
history, and creates `codex/ear-cognition` from `final-perception`.
No changes to main or either original contributor branch are needed.
Generated local perception handbook files are preserved and excluded from these
commits. Changes are made under the configured Git identity, not attributed to
teammates who did not author them.

## Interfaces and scope

```text
camera -> Frame -> perception -> Eye (same frame ID, time, track, anatomical side)
                                 | points + geometry_ok -> raw EAR -> fitted mapping
                                 | crop + crop_ok       -> CNN logit -> temperature
                                 +---------- same-eye/frame score fusion ----------+
                                                       |
                                    bilateral OPEN / CLOSED / UNKNOWN
                                                       |
                                  time-based closure / PERCLOS / recovery
                                                       |
                                      immediate fixed alert + event telemetry
                                                       |
                                       later optional local LLM and speech
```

This increment establishes independent EAR and CNN measurement paths.
Temporal drowsiness decisions, personal calibration, MAR/pose and LLM output are
later milestones. A closed eye is not itself a drowsiness diagnosis.
Raw EAR is not numerically comparable to a CNN closure score.
Default CNN temperature 1 and classification cutoff .5 are unvalidated baselines.

## Dataset decision and actual availability

The proposal names **NTHU-DDD as the main benchmark**, plus YawDD and UTA-RLDD.
The user confirmed on October 20 that the dataset is **not downloaded**.
The repository contains no training images, eye-state annotations or trained CNN
checkpoint. Its face-landmarker model belongs to perception, not the eye CNN.
The later `05_DATA_AND_CNN.md` guide adds a per-eye open/closed CNN experiment.
That requires independent eye labels and subject-disjoint evaluation; the
proposal's benchmark names alone do not establish a compatible label mapping.

| Source | Verified role / limitation | Next step |
| --- | --- | --- |
| NTHU-DDD | Main benchmark in proposal. Official endpoint timed out during this audit; access terms and exact downloaded annotation schema remain unverified. | Obtain original videos, annotations and README/access terms. Inspect sample labels before writing a dataset-specific importer. |
| YawDD | Yawning-focused in proposal. IEEE DataPort blocked automated access in this audit. | Optional later MAR/yawn evaluation; do not reinterpret yawning as eye closure. |
| UTA-RLDD | Official page describes 180 videos of 60 people; video labels 0/5/10 represent alert/low-vigilant/drowsy. | Useful temporal evaluation candidate; those labels do not directly supply independent binary labels for each eye. |

Sources checked:
[NTHU original endpoint](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/),
[YawDD DataPort](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset),
[UTA-RLDD authors' page](https://sites.google.com/view/utarldd/home).
The UTA page lists a 111.3 GB download and face-publication restrictions, so this
work does not automatically download all three datasets or publish their faces.
No claims about current NTHU license or eye-label definitions are made.

### Data acquisition checklist

1. Follow the original NTHU site/access process. Preserve the version and terms
   delivered with the archive; no third-party repack has been selected.
2. Keep downloaded archives/videos under ignored `data/raw/`, outside Git.
   Give the local folder to the next implementation step.
3. Inspect one video, its actual timestamps, annotation files and README. Record
   whether labels describe drowsiness, visible eye closure, both eyes jointly,
   or anatomical left/right eyes separately.
4. Declare how partly closed and unobservable eyes are handled. Do not convert
   all frames of a drowsy clip to `closed`, or generate ground truth from EAR.
5. If independent eye labels are absent, annotate a reviewed subset of full
   frames. A second reviewer checks a sample and disagreements. Subject/session
   identities must be real dataset metadata; do not guess them from image counts.
6. Assign subjects to train/calibration/validation/test before cropping. Preserve
   official benchmark partitions where supplied; derive development partitions
   only from the allowed development subjects. Do not mix official test into fit.
7. Use `data/manifests/eye_annotations.template.csv`. It is intentionally header
   only; no fabricated labelled subjects have been included.

### Current manifest contract

Required: sample_id, subject_id, session_id, video_id, source, split, frame_path,
timestamp_s, eye_side, label, observable, label_source.

- `eye_side` is anatomical `left`/`right`.
- Label `0` = open, `1` = closed. `observable=0` requires a blank label.
- `label_source` is `human` or `dataset_eye_annotation`, after actual review.
- IDs are globally namespaced. Both eyes retain their source frame identity.
- Image paths are now **relative to the CSV's directory**, or absolute. This
  corrects the old working-directory dependency. Existing older manifests must
  be converted explicitly; the reader does not guess their path base.
- Preparation retains invalid rows and adds `crop_path`, `points_json`,
  `geometry_ok`, `invalid_reason`. Six original pixel points are the handoff to
  EAR. It no longer imports the unfinished EAR classifier or writes invented EAR.
- Detector's offline iteration time only keeps independent image processing
  monotonic. Recording time is retained unchanged in `timestamp_s`.
- Preparation reads the runtime ROI and quality settings, and writes adjacent
  metadata with annotation/model hashes. Blur gate remains disabled by default.
- Audit checks required metadata, image readability, groups, both classes after
  filtering, and exact image-byte duplicates across partitions. Re-encoded or
  near-duplicate images require additional review; this is not perceptual dedup.

## Work completed in this increment

EAR branch: pure six-point geometry, quality-aware identity-preserving estimates,
explicit threshold classification, validated coefficient application, saved-image
inspection, tests and separate CI. See `docs/EAR_COGNITION.md` on that branch.

CNN branch: perception merge, corrected three-value detector adapter,
independent dataset preparation, portable paths, dataset audit, preprocessing
parity, invalid crop/nonfinite score handling, validation-selected checkpoint
training, coverage-aware baseline evaluation and a CPU training/reload test.
CPU tests use tiny generated images solely to verify software behavior. Their
scores are not evidence of eye recognition or project accuracy.

Local verification in this increment: EAR branch **61 passed** with the real
landmarker asset enabled; CNN branch **57 passed** including synthetic CPU
train/reload/evaluation and real landmark blank-image validation. Python 3.11.4,
Windows x86-64; CPU PyTorch 2.14.1+cpu. `pip check` reports no broken requirements.
The isolated `.venv-cognition` environment leaves the existing `.venv` and
`.venv-perception` intact. New GitHub branch checks are separate from these local
results; inspect Actions for their eventual status.

## Completion checkpoints

| Checkpoint | Deliverable | Evidence required |
| --- | --- | --- |
| 1: interfaces | Independent EAR/CNN software and reproducible checks | Passing unit/integration tests, no missing imports, matching identities |
| 2: real data | Versioned original data, reviewed labels and partitions | Actual archive/README, annotation sample, subject sets and audit counts |
| 3: CNN fit | Trained checkpoint and validation history | Real-data run, saved seed/preprocess/class map/manifest hash; learning curves |
| 4: EAR fit | Raw cutoff and independent score mapping | Calibration fit; threshold selection on validation; inspect errors |
| 5: comparison | EAR-only, CNN-only, fusion on paired held-out eyes | Same IDs/validity/bilateral policy, coverage, confusion matrix, precision/recall/F1 by person and condition |
| 6: temporal cognition | Elapsed-time closure/PERCLOS, recovery, missing-data policy | Annotated recordings, event false positives and delay; no stale evidence |
| 7: Jetson | Validated model export and integrated replay/live demo | Actual board/software audit, numeric parity, latency/FPS/memory/thermal measurements |

Train fits CNN weights; calibration fits CNN temperature and EAR score mapping;
validation selects checkpoints/cutoffs/fusion policy; test is used after freezing.
Do not report a successful synthetic smoke test as checkpoint 3 or 5 completion.

Open decisions: training machine (laptop GPU/CPU or Colab), real NTHU access and
annotation schema, partial-eye closure policy, evaluation objective/operating
cutoff, and actual Jetson software versions. There is no promised accuracy/FPS.

## Two-person Git workflow

Arushi works on `cnn`; Vishruth works on `codex/ear-cognition`.
Each person uses their own clone and pulls before editing:

```powershell
git fetch origin
git switch cnn
git pull --ff-only origin cnn
```

Vishruth substitutes `codex/ear-cognition` for `cnn`. If the branch is new,
`git switch --track origin/codex/ear-cognition` creates its local tracking branch.
Review `git status` first; commit or deliberately preserve personal changes before
switching. Stage named source/test/doc files, never raw recordings or environments.
Example commit descriptions: `Add CNN training run provenance`,
`Fit EAR cutoff on held-out validation subjects`.

After each branch passes its own acceptance checks, review diffs against
`final-perception` and combine through a new cognition integration branch.
Do not merge the entire CNN branch into EAR merely to reuse an eye measurement.
Neither branch being pushed means the trained cognition layer is complete.
