# EAR, CNN and fusion comparison foundation

Date: 7 November 2026.

The user selected comparison and fusion as the next increment. This implements
the existing Phase 1 experimental weighted combination and strict missing-data
policy. It does not select a production operating point.

## Requirement and boundaries

Replay the same timestamped observations through EAR, CNN and fusion, with
separate temporal histories. Preserve missing evidence and expose coverage.
The input consists of already computed estimates. Camera inference and model
artifact loading remain separate integration work. No training was performed.

## Concepts and data flow

```text
One frame, one track, anatomical left and right eyes
       ↓
Validate each branch against the current observation identity
       ↓
Raw EAR                 CNN closure score
   ↓                           ↓
Raw threshold            Score threshold
   ↓                           ↓
EAR observation          CNN observation
   │                           │
   │    EAR logistic mapping   │
   │             ↓             │
   │    Weighted combination ←─┘
   │             ↓
   │     Fusion observation
   ↓             ↓             ↓
Independent temporal controllers and closure windows
       ↓
Events, observation coverage and unresolved episodes
```

EAR is a geometric ratio. A score is a number in [0, 1] pointing toward closure.
Calibration fits the relationship between a score and observed labels; a number
in [0, 1] alone does not establish a trustworthy probability. Alignment means
matching sample ID, timestamp, track and anatomical side. A track is a continuity
epoch, not a person's biometric identity. UNKNOWN means insufficient evidence.

## Approaches and decision

| Approach | Benefit | Limitation |
| :--- | :--- | :--- |
| Agreement rule | Simple to explain | Requiring agreement can miss closures; accepting either can increase false positives |
| Weighted mapped scores | Small and inspectable; specified in the existing design | Needs fitted mappings and validation settings |
| Learned combiner | Can use score interactions and quality features | Adds training requirements and leakage risks |

Used the existing weighted-score experiment because it preserves the agreed
baseline comparison with little computational cost. The user selected the
comparison foundation; the final fusion model, alpha and cutoffs remain open.

For one eye, s = alpha × CNN + (1 − alpha) × mapped EAR.
Illustration: 0.5 × 0.9 + 0.5 × 0.7 = 0.8. These are teaching values.
Raw EAR is never directly averaged with CNN output. The mapped EAR uses the
existing decreasing logistic mapping with externally supplied coefficients.

Both branches are required even when alpha is 0 or 1. Consequently, endpoint
fusion can have lower coverage than a standalone baseline. Missing evidence
never becomes zero, and fallback to another method is never silent.

## Implementation walkthrough

`src/marvel/cognition/fusion.py`:

1. `_unit` validates finite values in [0, 1].
2. `validate_score` checks the method name and score range.
3. `combine` validates alpha, methods and identity before arithmetic.
4. A missing branch produces an invalid estimate with value None.
5. Valid branches produce the explicit weighted score and retain identity.
6. `score_state` converts a valid closure score into OPEN or CLOSED. Equality
   at the cutoff is CLOSED. Raw EAR retains its separate strict less-than rule.

`src/marvel/cognition/comparison.py`:

1. Construction validates all parameters and copies the mapping coefficients.
2. Each mode receives its own Trigger and ClosureWindow using identical timing.
3. `_align` validates method and numerical range, then requires exactly two
   unique sides matching the current observation. Stale or mixed input makes
   that entire branch unavailable. It leaves the other baseline usable.
4. `update` checks the clock and all evidence before advancing any history.
5. Raw EAR supplies its standalone threshold baseline. Mapped EAR is used only
   for fusion. CNN and fusion each use explicit score cutoffs.
6. Bilateral aggregation requires both eyes to agree; otherwise UNKNOWN.
7. Each observation advances only its own mode's controller and rolling window.
8. The returned record contains estimates, decisions, events, coverage and a
   flag indicating whether every mode could make an observation on this sample.

The fusion arithmetic and identity checks use constant work per observation.
Rolling-window work and memory follow the existing ClosureWindow implementation
and scale with retained intervals. Three controllers multiply storage by three.

`scripts/replay_comparison.py` reads JSONL records containing sample_id, t,
track, ear and cnn. Each branch is a list of serialized Estimate records.
Configuration supplies every constructor argument. It logs original input
records alongside normalized results, so rejected evidence remains inspectable.
Input and config SHA256 hashes identify the replay. These hashes do not prove
which model produced the scores; model provenance still needs the inference adapter.
The script refuses existing output paths and requires an explicit synthetic or
recorded provenance flag. Malformed or empty input fails without a summary.
Only a completed summary indicates successful replay completion.

## Reproduce the synthetic demonstration

From the repository root in PowerShell, choose a new output folder:

```powershell
.\.venv-cognition\Scripts\python.exe scripts/demo_comparison.py reports/local/comparison_demo_2026_11_07
```

The folder holds config.json, inputs.jsonl and comparison.jsonl. All data and
parameters are explicitly synthetic. The scenario induces one missing CNN
observation at 2.5 seconds. Expect the first closure event at 1 second and
recovery at 2 seconds in all modes. The next EAR event occurs at 3.5 seconds,
while CNN and fusion wait until 4 seconds because their evidence was missing.
This delay is an induced consequence of the policy, not a measured model error.

For previously recorded estimates:

```powershell
.\.venv-cognition\Scripts\python.exe scripts/replay_comparison.py input.jsonl config.json new_output.jsonl --recorded
```

Do not label generated scores as recorded camera evidence. Ground truth is
not accepted by the prediction API. A future evaluator must join independent
labels to these predictions separately.

## Evidence and limitations

Focused tests exercise alpha endpoints, wrong identities, invalid scores,
missing branches, bilateral disagreement, independent histories, event IDs,
track changes, gaps, clock rejection, deterministic replay, recovery and
overwrite protection. See the dated work record for executed results.

Observation counts are sample counts, not duration or accuracy. Each rolling
window separately reports time coverage and the binary closure proxy. No
false-alert rate, event recall, camera accuracy or fusion improvement has been
measured by these tests. A run ending with an active episode records it as
unresolved. The normal camera application remains EAR only.

## Next acceptance gates

1. Obtain the trained CNN checkpoint or ONNX plus metadata and verify hashes.
2. Connect perception and both estimators to this adapter; log artifact hashes.
3. Fit EAR mapping on independent calibration data and assess CNN calibration.
4. Select alpha and cutoffs on validation recordings, then freeze the bundle.
5. Evaluate each baseline, fusion, their common observable subset, and coverage.
6. Measure event recall, false alerts per hour and detection delay on labelled
   sequences. Validate on the actual camera and Jetson separately.

## Interview explanation and retention

I built an experimental comparison adapter that joins EAR and CNN evidence by
observation identity, preserves missing data, and runs identical temporal logic
with independent histories. The logs expose coverage so rejecting difficult
frames cannot silently inflate performance. Synthetic replay verifies software
behavior; independently labelled recordings are still required to evaluate it.

Prediction exercise: if CNN evidence disappears for one frame while EAR remains
closed, which modes lose closure continuity? Answer: CNN and fusion. EAR keeps
its own continuity if its observation gap remains within the configured limit.
