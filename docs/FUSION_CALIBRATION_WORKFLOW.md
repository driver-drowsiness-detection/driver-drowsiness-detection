# Calibration before claiming better fusion accuracy

Updated 9 November 2026.

## Current answer

EAR and CNN have an implemented camera integration and weighted fusion path.
The real trained CNN artifact is still unavailable here. No experiment has yet
shown that fusion improves this laptop's camera accuracy. The new tools complete
the calibration and validation analysis mechanisms; real fitting still requires
independently reviewed full-frame data and the trained artifact.

MAR, head pose and the closure proxy are not extra votes added to the score.
MAR measures mouth opening, pose measures orientation when calibrated, and the
closure window summarizes eye history. They need their own labels and quality
protocols before they can justify changing the alert decision.

## Why the ordering matters

```text
Reviewed eye labels + known subjects + full camera frames
                         ↓
          Separate subjects into development and final test
                         ↓
         Perception: landmarks and valid eye crops
                    ↙                 ↘
             EAR measurements      Fixed trained CNN
                    ↓                 ↓
       Calibration subjects fit     Closure scores
       decreasing EAR mapping          │
                    └─────────┬─────────┘
                              ↓
                  Validation subjects compare
                   EAR alone, CNN alone, fusion
                              ↓
             Select a candidate under an explicit objective
                              ↓
                Freeze settings and model identity
                              ↓
                 Final unseen evaluation and event replay
```

A low EAR suggests closure. A high CNN closure score suggests closure. Their
directions and scales differ, so averaging raw EAR and CNN values is wrong.
The mapping is p(EAR) = sigmoid(slope × EAR + intercept), with slope negative.
The experimental fused score is alpha × CNN + (1 − alpha) × p(EAR).

Calibration estimates the mapping. Validation investigates candidate choices.
The final test estimates performance after choices are frozen. Repeatedly
choosing settings using final test results would turn that set into development data.

## Implemented prerequisites

### EAR fitter

`scripts/calibrate_scores.py` now accepts a prepared development manifest and
writes a new experimental report. It fits only labelled, valid EAR measurements
from the calibration partition. It audits all supplied development rows for
group separation and exact duplicate image bytes. Validation labels never enter
the fit. Train and test rows are rejected by this development entry point.

`src/marvel/training/ear_calibration.py` uses a constrained Newton optimization:

1. Validate finite nonnegative EAR, both classes and variation in measurements.
2. Standardize EAR to improve the numerical conditioning of the two-parameter fit.
3. Compute logistic negative log likelihood plus a small L2 regularization term.
4. Calculate its gradient and 2 by 2 Hessian, then solve for a Newton proposal.
5. Constrain the standardized slope below zero. When the constraint is active,
   solve the intercept proposal conditional on that slope.
6. Backtrack along the feasible direction until the objective does not increase.
7. Stop at convergence, or fail explicitly instead of returning an unfinished fit.
8. Convert slope and intercept back to raw EAR units for the existing runtime API.

Regularization is an explicit numerical choice of 0.001, reported in the output;
it keeps separable training fixtures from generating infinite coefficients.
This does not establish optimal real-data calibration. A direction-boundary
flag warns that the constrained solution has almost no decreasing relationship.
The script never replaces a camera configuration automatically.

For N calibration eyes and I iterations, fitting costs O(I × N) with O(N) data
storage. The parameter solve is constant size. File auditing adds image I/O.

### Fusion analysis

`scripts/tune_fusion.py` loads the real checkpoint, checks its subject provenance,
and scores validation crops only. It rejects CNN training subjects in either
development partition and rejects CNN checkpoint-selection subjects in the EAR
calibration partition. It checks that the EAR report identifies the same manifest
and preparation metadata. Checkpoint selection and fusion tuning may both use
validation data; the report remains a development analysis, not an unbiased test.

`src/marvel/training/fusion_analysis.py`:

1. Retains missing estimates and counts independently labelled rows.
2. Finds the common subset with both EAR and CNN evidence.
3. Requires both classes in that subset before comparing settings.
4. Evaluates the caller's alpha and threshold grid, including alpha 0 and 1.
5. Selects the lowest open-eye false-positive rate among candidates meeting an
   explicitly supplied minimum closed-eye recall. Ties prefer higher recall,
   then F1, then lower alpha and cutoff for deterministic output.
6. Returns no candidate if the grid cannot meet the target.
7. Reports fixed EAR/CNN baselines on the same common subset and separately on
   each baseline's full usable subset, plus common-subset results by subject.
8. Records coverage counts, score quality, hashes and the fact that runtime
   settings were not adopted and final-test/event accuracy were not evaluated.

Searching A weights and T cutoffs on N common observations costs O(A × T × N).
Inference is performed once per validation crop, not for every candidate.
Candidate storage is O(A × T); the score arrays require O(N) memory.

The first operating objective follows the previously stated preference for
catching closures while making the false-positive cost visible. There is no
default recall target: the caller must supply one. A chosen recall target and
grid are experimental requirements until the team accepts an operating policy.

## Required data and commands

Use independently reviewed recordings from the intended camera conditions.
Record source and actual subject/session identities. Keep every subject in one
partition. A single person's successful demonstration cannot establish accuracy
on unseen people. MRL eye-only images cannot supply the full facial landmarks
and anatomical pairing required for this workflow.

The existing `extract_frames.py` can sample a recorded video into full frames
and a blank annotation queue. Human reviewers must fill labels, observability,
split, label_source, annotation_status=reviewed and orientation_status=verified.
Never copy EAR decisions or CNN predictions into the ground-truth column.
Keep ambiguous/unobservable labels blank according to the manifest contract.

Create a development annotation manifest containing only calibration and
validation subjects. Keep final-test recordings and annotations separate.
Prepare it through the existing perception adapter:

```powershell
.\.venv-cognition\Scripts\python.exe scripts/prepare_dataset.py data/processed/development_annotations.csv models/face_landmarker.task data/processed/development_crops data/processed/development_prepared.csv
.\.venv-cognition\Scripts\python.exe scripts/calibrate_scores.py data/processed/development_prepared.csv reports/local/ear_calibration_v1.json
```

These are instructions for real reviewed data, which is not yet available here.
Once the trained checkpoint is supplied, investigate fusion. The following
numbers are illustrative analysis choices, not approved runtime settings:

```powershell
.\.venv-cognition\Scripts\python.exe scripts/tune_fusion.py data/processed/development_prepared.csv reports/local/ear_calibration_v1.json models/checkpoints/mrl_full_3epoch.pt reports/local/fusion_validation_v1.json --alphas 0 0.25 0.5 0.75 1 --thresholds 0.3 0.4 0.5 0.6 0.7 --minimum-recall 0.95 --ear-threshold 0.2 --cnn-threshold 0.5
```

Reports refuse overwrite. CNN temperature remains 1 unless explicitly supplied;
the earlier temperature experiment did not justify adopting its fitted value.
No export command automatically turns a candidate into a frozen live bundle.

## Recorded verification

Focused tests verify monotonicity, finite fitting on separable samples, the
direction-boundary diagnostic including imbalanced labels, independence from
validation labels, rejection of test subjects/duplicate images/unreviewed or
generated labels, missing-data coverage, infeasible objectives, endpoint
selection, overwrite protection and the complete CLI with a controlled checkpoint.

The local synthetic demonstration is under
reports/local/fusion_prerequisites_2026_11_09/. It contains generated images,
geometry, an explicit synthetic marker, fitted coefficients and a fusion report.
Its CNN scores are constant controlled values, not real model predictions.
The test selected alpha zero and retained seven of eight observations for fusion,
while EAR alone retained eight. This demonstrates that the code can prefer EAR
and expose coverage loss. Its scores are not evidence of real model accuracy.
Synthetic coefficients were not installed in the camera application.

## Remaining acceptance gates

1. Transfer and verify the actual trained CNN artifact.
2. Obtain independently labelled full-frame camera recordings with suitable
   subject separation. Review lighting, glasses, pose and partial closures.
3. Run real fitting and validation; inspect error examples and subject breakdowns.
4. Decide whether the proposed candidate improves the agreed objective enough
   to adopt. An alpha endpoint may be the honest outcome.
5. Freeze the entire configuration and artifact identities before final testing.
6. Evaluate bilateral temporal events, false alerts per hour and detection delay
   against independent sequence labels. Eye-image FPR is not an alert rate.
7. Validate MAR events, physical PERCLOS and pose separately before incorporating
   them into a richer fatigue policy. Nano remains a later deployment gate.

## Interview explanation and retention

I separated calibration, candidate selection and final testing. The EAR mapping
has the correct monotonic direction, and fusion is evaluated against both
baselines on identical usable observations. I report each method's coverage
separately because rejecting difficult frames can otherwise inflate accuracy.
The pipeline can select no candidate or an endpoint rather than claiming that
combining models must improve results.

Prediction exercise: if fusion makes fewer mistakes but rejects half the inputs,
can we claim it is better? Not from that result alone. Compare on common inputs,
report full coverage, then evaluate missed events and false alerts over time.
