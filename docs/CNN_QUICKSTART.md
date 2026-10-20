# CNN branch: run, prepare and train

Read [COGNITION_PLAN.md](COGNITION_PLAN.md) for the verified project status,
dataset access decisions, label rules, branch ownership and completion gates.
The pipeline is ready for reviewed data; no trained eye model is supplied.

## Pull and set up

Use your own repository path. Preserve any local edits before switching.

```powershell
git fetch origin
git switch cnn
git pull --ff-only origin cnn
py -3.11 -m venv .venv-cognition
.\.venv-cognition\Scripts\python.exe -m pip install --upgrade pip
.\.venv-cognition\Scripts\python.exe -m pip install -r environments/windows-training.txt
.\.venv-cognition\Scripts\python.exe -m pip check
.\.venv-cognition\Scripts\python.exe scripts/download_landmarker.py
$env:MARVEL_TEST_MODEL = "models/face_landmarker.task"
.\.venv-cognition\Scripts\python.exe -m pytest -q
```

Create the environment only if it does not already exist. This Windows/Python
3.11 setup uses CPU PyTorch so development checks need no GPU. It does not select
the final training machine. Colab/GPU requires a separately verified environment;
do not apply this Windows lock to Jetson Nano.

The tests create small synthetic images in temporary directories, run a real
training step, reload weights, check preprocessing and calculate a report. That
proves code execution only. It is not a trained eye classifier demonstration.

## When the real data and annotations exist

Keep originals under `data/raw/`. Start with the header-only
`data/manifests/eye_annotations.template.csv`; save reviewed rows into
`data/manifests/annotations_v1.csv`. For that CSV, a path such as
`../raw/collection/frames/frame_001.png` is relative to `data/manifests/`.
Choose actual dataset IDs and partition membership; this guide does not invent
an annotation importer before the original NTHU schema can be inspected.

```powershell
.\.venv-cognition\Scripts\python.exe scripts/audit_dataset.py data/manifests/annotations_v1.csv
.\.venv-cognition\Scripts\python.exe scripts/prepare_dataset.py data/manifests/annotations_v1.csv models/face_landmarker.task data/processed/crops_v1 data/processed/prepared_v1.csv --config configs/runtime.yaml
.\.venv-cognition\Scripts\python.exe scripts/audit_dataset.py data/processed/prepared_v1.csv
.\.venv-cognition\Scripts\python.exe scripts/train_cnn.py data/processed/prepared_v1.csv models/checkpoints/eye_v1.pt 10 cpu
.\.venv-cognition\Scripts\python.exe scripts/evaluate.py data/processed/prepared_v1.csv models/checkpoints/eye_v1.pt validation reports/runs/cnn_validation_v1.json cpu
```

These are future commands with explicitly named example output paths. They
cannot train until real annotations and both labels in train/validation exist.
The `10` is an example epoch count, not a selected optimal setting.
Train uses command arguments; `configs/training.yaml` is still a scaffold and is
not read. Python imports are resolved from the scripts' own repository location.

Audit checks the files and partitions before training. Inspect counts after
perception rejection, not only the original labels. Exact duplicate image bytes
cannot cross partitions. This does not detect all re-encoded or near-duplicates.

Preparation uses a new crop directory and output CSV each run. Existing paths
are rejected to prevent accidental replacement. If preparation fails midway,
inspect its partial folder and use a new versioned output path after fixing the
source error. Do not treat a partial folder as a completed dataset.

The preparation sidecar records the model/annotation hashes, ROI and quality
settings. Training records seed 7, class mapping, preprocessing, manifest hash,
subject lists, dataset audit and selected epoch. Validation loss selects the
checkpoint; train loss never substitutes for held-out evaluation.

Evaluate reports the confusion matrix counts, closed-eye precision/recall/F1
and coverage among independently observable eyes. The .5 score cutoff is an
explicit unvalidated baseline. Temperature calibration, cutoff selection,
per-person/condition reporting and paired fusion evaluation remain future
checkpoints. After freezing those decisions, run the **test** split once.

## How Vishruth connects EAR later

The prepared rows retain `points_json` (six ordered original pixel points) and
`geometry_ok`. Original sample_id, timestamp_s and eye_side remain unchanged.
Vishruth's geometry can read these points and produce raw EAR without importing
PyTorch. Preparation does not depend on his branch being merged.

At runtime both methods consume the same `Eye`: EAR uses points/geometry_ok;
CNN uses crop/crop_ok. Match frame ID, timestamp, track and side before combining
outputs. A valid CNN `Estimate.value` is a closure score; it is not yet proven
calibrated. Invalid input or a nonfinite score gives `value=None`, not zero.

Do not average raw EAR with the CNN score. Keep missing evidence UNKNOWN through
the later fusion, bilateral and temporal layers.
