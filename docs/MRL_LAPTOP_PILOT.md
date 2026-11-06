# MRL eye-only CNN pilot on Arushi's laptop

Date: 2026-11-06. This is a supplemental eye-classification experiment, not the
proposal's NTHU-DDD drowsiness benchmark or a live Jetson result.

Source: [MRL Eye Dataset authors' page](https://mrl.cs.vsb.cz/eyedataset.html).
Original archive `mrlEyes_2018_01.zip` was downloaded from the link on that page
to ignored `data/raw/mrl/`; SHA256 is
`17ff89928005cb6e63a53fe020e419060279ff59e57555cfa49693fc1865fdfc`.
The archive CRC, path layout, 84,898 PNG count, 37 subject folders, and
`annotation.txt` label legend were checked before extraction. Source labels are
`0 = closed, 1 = open`; project labels are `0 = open, 1 = closed`.

The full source manifest is `data/processed/mrl_manifest_v1.csv` with adjacent
provenance JSON. It holds 37 subjects in separate train/calibration/validation/test
partitions. MRL does not supply anatomical eye side or image timestamps, so those
fields remain `unspecified` and blank. Session and video IDs conservatively group
all images under each subject; they do not claim to identify actual recordings.
CNN training accepts these eye-only rows; paired EAR comparison rejects them.
Images and manifests remain local and ignored by Git. Do not commit or share
source images, extracted crops, checkpoints or identifying filenames.

To measure CPU execution first, `scripts/sample_mrl_pilot.py` selected at most
50 images per subject and class from train, calibration and validation. Test
subjects were omitted. The resulting 3,023-image manifest passed file decoding,
exact image-byte duplicate, group separation and both-label audit. It contains
2,284 train eyes (25 subjects), 400 calibration eyes (4 subjects), and 339
validation eyes (4 subjects). This pilot does not represent the full dataset.

One CPU epoch took 10.73 seconds including loading and dataset audit. A fresh
10-epoch run took 38.76 seconds. The checkpoint with the lowest validation loss
came from epoch 8 (loss 0.68259). All runs used the repository's EyeCNN and
PyTorch 2.14.1+cpu on Intel i5-1235U, with random seed 7 and 4 CPU threads.
The first validation report used the uncalibrated 0.5 score cutoff and found:

| Validation count | TP | TN | FP | FN | Accuracy | Closed recall | Closed F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 339 | 140 | 47 | 92 | 60 | 0.5516 | 0.7000 | 0.6481 |

These results are weak. The checkpoint is a **pilot artifact only**, stored at
`models/checkpoints/mrl_pilot_10epoch.pt` and excluded from Git. It must not
drive alerts or be presented as a validated model. The 0.5 cutoff was not fit
on calibration subjects. MRL infrared crops may differ substantially from the
project's RGB webcam or Jetson camera imagery.
Per-subject validation accuracy ranged from 0.44 to 0.67; on two subjects the
model correctly identified zero open eyes. The detailed local report is
`reports/runs/mrl_pilot_validation_v2.json`. This variation reinforces the
decision to keep the model out of the live alert path.

Reproduction commands from the repository root:

```powershell
.\.venv-cognition\Scripts\python.exe scripts/prepare_mrl_dataset.py data/raw/mrl/mrlEyes_2018_01.zip data/processed/mrl_manifest_v1.csv
.\.venv-cognition\Scripts\python.exe scripts/sample_mrl_pilot.py data/processed/mrl_manifest_v1.csv data/processed/mrl_pilot_v1.csv
.\.venv-cognition\Scripts\python.exe scripts/audit_dataset.py data/processed/mrl_pilot_v1.csv
.\.venv-cognition\Scripts\python.exe scripts/train_cnn.py data/processed/mrl_pilot_v1.csv models/checkpoints/mrl_pilot_10epoch.pt 10 cpu
.\.venv-cognition\Scripts\python.exe scripts/evaluate.py data/processed/mrl_pilot_v1.csv models/checkpoints/mrl_pilot_10epoch.pt validation reports/runs/mrl_pilot_validation_v2.json cpu
```

Use new versioned paths; scripts refuse to overwrite existing outputs. Before a
full or final model, inspect per-subject error, review label quality and class
balance, choose calibration and selection rules, and obtain NTHU-DDD for the
actual video/EAR/temporal project benchmarks. Keep the test subjects untouched
until model and decision policy are frozen.
