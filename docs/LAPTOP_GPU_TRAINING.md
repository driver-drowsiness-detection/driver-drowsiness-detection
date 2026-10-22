# Laptop CNN training

Decision on 2026-10-22: use the laptop first because its RTX 3050 has 6 GB GPU
memory and the current EyeCNN is a small network. Real data timing remains to be
measured. Colab remains an option if measured local performance is inadequate.

## Verified results on 2026-10-22

Starting CNN revision: `66a446e`.

| Check | Observed result |
| --- | --- |
| Python | 3.11.9 |
| PyTorch | 2.14.1+cu126 |
| CUDA runtime reported by PyTorch | 12.6 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Dependency consistency | pip check passed |
| Actual EyeCNN forward, backward and optimizer steps on CUDA | Passed, batch 32, five synthetic steps |
| Saved checkpoint predictions on CPU and CUDA | Matched within the smoke test tolerance |
| Full suite with local face landmarker enabled | 58 passed in 39.79 seconds |

The GPU smoke run took 0.824 seconds including initialization and reported
30.89 MiB peak PyTorch tensor allocation. These small synthetic inputs do not
predict dataset preparation speed, real training time, or total GPU usage.
Package versions are captured locally in `reports/local/laptop-gpu-packages.txt`.
No real CNN checkpoint was trained, and Jetson execution remains untested here.

## Environment

Use the `cnn` branch and `.venv-cognition`. Keep the perception environment
separate. These instructions are for Windows x86-64 and Python 3.11, not Jetson.
The existing `windows-training.txt` installs CPU PyTorch. For this GPU setup:

```powershell
# Only create the environment if it does not already exist.
py -3.11 -m venv .venv-cognition
.\.venv-cognition\Scripts\python.exe -m pip install --upgrade pip
.\.venv-cognition\Scripts\python.exe -m pip install -r environments/windows-perception-lock.txt
.\.venv-cognition\Scripts\python.exe -m pip install -r environments/windows-training-gpu.txt
.\.venv-cognition\Scripts\python.exe -m pip check
.\.venv-cognition\Scripts\python.exe scripts/check_training_gpu.py
```

On Vishruth's laptop the `py` Store alias currently fails, but the installed
interpreter works. The environment was created using:

```powershell
& 'C:\Users\alwan\AppData\Local\Python\pythoncore-3.11-64\python.exe' -m venv .venv-cognition
```

Do not recreate a working environment just to repeat these instructions.
`python.exe -m pip` installs packages for that exact Python environment.
`cu126` identifies the CUDA 12.6 PyTorch build. A compatible NVIDIA driver is
required; CUDA operation is verified by actual execution, not the package name.
No separate CUDA Toolkit installation was requested for these prebuilt wheels.

## Verify software execution

```powershell
.\.venv-cognition\Scripts\python.exe scripts/check_training_gpu.py
$env:MARVEL_TEST_MODEL = (Resolve-Path models/face_landmarker.task).Path
.\.venv-cognition\Scripts\python.exe -m pytest -q
```

The GPU check imports the actual EyeCNN, creates a batch shaped `(32, 1, 32, 64)`,
performs five forward/backward/optimizer steps on CUDA, verifies that weights
change, and reloads a temporary checkpoint on CPU and GPU to compare predictions.
It deletes its synthetic checkpoint on completion. Its elapsed time includes
initialization and is not a real dataset training benchmark. Its memory figure
is PyTorch tensor allocation, not total device usage.

The pipeline test also runs the actual training and evaluation scripts on both
CPU and CUDA, skipping CUDA when unavailable. All fixtures are synthetic.
Passing proves execution and checkpoint handling, not eye recognition accuracy.

## Dataset gate

At the initial setup check, no real training data was found in this checkout.
Checks of likely download names and older project data folders also found none;
this was not an exhaustive search of every disk. The user did not know whether
any teammate had downloaded it. The proposal names NTHU-DDD as the main benchmark.
Its original site timed out during this session. Subsequent work downloaded and
inspected the original CEW release as a real-image perception candidate. See
[DATASET_ACCESS_AND_INSPECTION.md](DATASET_ACCESS_AND_INSPECTION.md) for evidence
and unresolved identity/label requirements. No dataset is ready for defensible
subject-disjoint training yet.

Before real training, obtain source videos/images, annotations, subject IDs,
and terms; inspect the actual label meaning. Video drowsiness is not a per-eye
open/closed label. EAR-generated labels cannot establish CNN superiority.
Keep subjects separated across train, calibration, validation and test, following
official partitions where supplied. See COGNITION_PLAN.md and CNN_QUICKSTART.md.

Only after reviewed data has passed preparation and audit, a first one-epoch
run would use this command (the paths are future examples):

```powershell
.\.venv-cognition\Scripts\python.exe scripts/train_cnn.py data/processed/prepared_v1.csv models/checkpoints/eye_v1.pt 1 cuda
```

`1` means one pass through the training data, for an initial timing and pipeline
check. `cuda` selects the NVIDIA GPU. Do not use a synthetic checkpoint for the
live demonstration or call this one-epoch run a completed model evaluation.

## Jetson handoff

Transfer trained weights, model code, preprocessing and calibrated settings.
Create a separate compatible Jetson environment; do not copy the Windows venv.
Export/optimization may be necessary after checking JetPack and supported
operators. Retraining is not required merely because the computer changes.
Camera-domain performance, inference parity and latency still need board tests.
