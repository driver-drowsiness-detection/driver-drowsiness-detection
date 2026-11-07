# Full MRL eye CNN training on Arushi's laptop

This development experiment follows the [pilot](MRL_LAPTOP_PILOT.md). It uses
the original [MRL Eye Dataset](https://mrl.cs.vsb.cz/eyedataset.html) archive
downloaded on this laptop. The 84,898-image manifest and images are local and
ignored by Git. A clone contains the code and report, not the trained weights.

## Audit and subject split

Before training, the program verified that all 84,898 PNGs decode, checked
exact file hashes for duplicates across partitions, and confirmed that subject
IDs do not cross partitions. The local checkpoint embeds that audit. Source
labels `0=closed, 1=open` were translated to project labels
`1=closed, 0=open` during import.

| Partition | Subjects | Eye images | Use |
| --- | ---: | ---: | --- |
| Train | 25 | 54,617 | Fit CNN weights |
| Calibration | 4 | 10,317 | Reserved for later score calibration |
| Validation | 4 | 16,263 | Select checkpoint and inspect errors |
| Test | 4 | 3,701 | Unused for fitting or scoring |

Exact hashes do not detect every near-duplicate or mislabeled eye. MRL does
not supply anatomical eye side, video identity or frame time. The manifest
keeps those facts explicit, so these eye-only images cannot evaluate paired
EAR, blink duration, PERCLOS or overall driver drowsiness.

## CPU training and validation

On the Intel i5-1235U laptop, the audit plus three CPU epochs took 713.29
seconds with four CPU threads. Validation loss was 0.4481, 0.5462 and 0.4002
across epochs 1-3. The script saved epoch 3, the lowest validation loss.
Python 3.11 and PyTorch 2.14.1+cpu were used; `pip check` passed. The local
checkpoint `models/checkpoints/mrl_full_3epoch.pt` has SHA256
`f33202048a72f14e0aeb2aa4d5dae622af036d56619c52e815dca534d8bb75bd`.

At the **uncalibrated** 0.5 score cutoff, validation on 16,263 eyes gave:

| Accuracy | Closed precision | Closed recall | Closed F1 | TP | TN | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8268 | 0.7717 | 0.9617 | 0.8563 | 8,392 | 5,054 | 2,483 | 334 |

The local detailed report is `reports/runs/mrl_full_validation_v1.json`.
Despite high aggregate closed-eye recall, 2,483 open eyes were called closed.
Two subjects had no correctly identified open eyes; those subjects have 28 and
11 open examples. Overall accuracy is weighted by subjects with many images.
Review errors by subject and condition before choosing a cutoff or claiming
reliable performance.

The earlier 3,023-image pilot had 0.5516 accuracy on a different-sized sample
from the same validation subjects. The scores are not a controlled architecture
comparison. This full-data checkpoint remains an experimental MRL eye
classifier. Its infrared source imagery differs from the planned RGB webcam
and Jetson camera, so it must not trigger live alerts.

## Reproduce locally

From `cnn`, with the original archive and Windows environment already present:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\.venv-cognition\Scripts\python.exe scripts/train_cnn.py data/processed/mrl_manifest_v1.csv models/checkpoints/mrl_full_3epoch_v2.pt 3 cpu
.\.venv-cognition\Scripts\python.exe scripts/evaluate.py data/processed/mrl_manifest_v1.csv models/checkpoints/mrl_full_3epoch_v2.pt validation reports/runs/mrl_full_validation_v2.json cpu
```

Use new paths; scripts refuse overwrite. The completed calibration experiment,
validation cutoff tradeoff, and false-positive breakdown are in
[MRL calibration and validation](MRL_CALIBRATION_VALIDATION.md). It did not
establish reliable live probabilities or an alert threshold. Obtain the
proposal's NTHU-DDD recordings for paired and temporal project evaluation.
Keep the MRL test subjects untouched until model choice and decision rules
are frozen.
