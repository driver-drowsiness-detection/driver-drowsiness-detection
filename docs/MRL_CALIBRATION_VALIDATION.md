# MRL CNN score calibration and false-positive review

**Run:** 6 November 2026 on Arushi's Intel CPU laptop

**Branch:** `cnn`

**Input checkpoint:** `models/checkpoints/mrl_full_3epoch.pt`

**Detailed local report:** `reports/runs/mrl_calibration_validation_v1.json`

This is a development analysis of **individual eye state** on the original
MRL infrared eye images. It is not a driving-video drowsiness result, a live
webcam/Jetson result or a deployment decision. The MRL test subjects were not
loaded for inference or scored.

## Method and data separation

The script `scripts/calibrate_cnn.py` loads the unchanged epoch-3 checkpoint,
checks that its training-manifest hash matches the input, then scores only the
**calibration** and **validation** partitions in batches. The subject sets do
not overlap. It fits one positive temperature by minimizing binary
negative log likelihood (NLL) on **10,317 calibration eyes from four subjects**.
It also selects two *candidate* cutoffs on that same partition: maximum
closed-eye F1, and the highest closed-eye recall achievable while keeping the
**calibration open-eye false-positive rate at or below 5%**. The 5% target is
an analysis choice for this experiment, not a proven alert policy.

The fixed candidates are then measured on **16,263 validation eyes from four
different subjects**. This validation partition previously chose the CNN
checkpoint epoch, so it is useful for error analysis but is **not a final
unbiased model test**. The original MRL test split (3,701 eyes, four subjects)
remains unscored.

The fitted temperature is **1.7201**. At a 0.5 cutoff, positive temperature
scaling cannot change class predictions: `sigmoid(logit / T) >= 0.5` exactly
when `logit >= 0`. It changes score magnitudes and the behavior of other
cutoffs.

## Did temperature calibration transfer?

| Score set | Calibration NLL | Validation NLL | Calibration Brier | Validation Brier | Validation ECE (10 bins) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw checkpoint | 0.5250 | **0.4002** | **0.1347** | **0.1273** | **0.1173** |
| Temperature 1.7201 | **0.4859** | 0.4093 | 0.1432 | 0.1305 | 0.1249 |

Fitting temperature improved the **calibration NLL**, which was its stated
optimization target. Validation NLL, Brier score and the ten-bin expected
calibration error all got worse. The fitted temperature therefore should **not
be adopted as a validated runtime probability calibration**. Group composition
and image conditions differ, and the four calibration subjects may not
represent the validation or future camera population. ECE depends on binning;
the NLL/Brier direction provides corroborating evidence here.
The class mix also differs: calibration contains **2,646 closed of 10,317**
eyes (25.6%), while validation contains **8,726 closed of 16,263** (53.7%).
This is an observed distribution difference, not proof of a single cause.

## Candidate cutoff performance

Positive class is **closed**. FP means an **open** eye was called closed; FN
means a **closed** eye was missed. FPR divides FP by the number of open eyes.

| Rule | Cutoff | Calibration FP / open (FPR) | Calibration closed recall | Validation TP | Validation TN | Validation FP / open (FPR) | Validation FN | Validation closed recall | Validation accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw baseline | 0.5000 | 1,509 / 7,671 (19.67%) | 84.85% | 8,392 | 5,054 | 2,483 / 7,537 (32.94%) | 334 | 96.17% | 82.68% |
| Temperature + calibration F1 choice | 0.5001 | 1,506 / 7,671 (19.63%) | 84.85% | 8,392 | 5,055 | 2,482 / 7,537 (32.93%) | 334 | 96.17% | 82.68% |
| Temperature + calibration <=5% open-eye FPR choice | 0.7581 | 383 / 7,671 (4.99%) | 40.17% | 6,264 | 7,044 | 493 / 7,537 (6.54%) | 2,462 | 71.79% | 81.83% |

Raising the cutoff reduced validation false-positive **eye classifications**
from 2,483 to 493, but missed **2,462 closed eyes** instead of 334. The
calibration F1 choice stayed effectively at 0.5 and barely changed errors.
Neither candidate is an acceptable live-alert rule on this evidence. An
eye-image FPR cannot be converted to a per-minute driver false-alarm rate
without camera, bilateral and temporal evaluation.

## Where the remaining false positives occur

At cutoff **0.7581**, the 493 validation open-eye false positives break down as
follows. Categories overlap; do not add their counts across tables. Source
condition meanings come from the inspected MRL `annotation.txt`. Small open
sample counts make some rates unstable.

| Validation subject | Open eyes | False positives | Open-eye FPR | Closed eyes missed |
| --- | ---: | ---: | ---: | ---: |
| `s0005` | 28 | 16 | 57.14% | 309 |
| `s0007` | 11 | 0 | 0% | 501 |
| `s0012` | 4,230 | 252 | 5.96% | 990 |
| `s0019` | 3,268 | 225 | 6.88% | 662 |

| Source condition | Open eyes | False positives | Open-eye FPR |
| --- | ---: | ---: | ---: |
| No glasses | 7,455 | 460 | 6.17% |
| Glasses | 82 | 33 | 40.24% |
| No reflection | 5,732 | 349 | 6.09% |
| Low reflection | 1,109 | 10 | 0.90% |
| High reflection | 696 | 134 | 19.25% |
| Bad quality (source flag) | 5,022 | 320 | 6.37% |
| Good quality (source flag) | 2,515 | 173 | 6.88% |
| Sensor 01 | 4,887 | 290 | 5.93% |
| Sensor 02 | 1,080 | 0 | 0% |
| Sensor 03 | 1,570 | 203 | 12.93% |

The rates are descriptive, not causal. For example, glasses, subject and
sensor are correlated in this dataset; the 40.24% glasses rate has only 82
open examples. A few manually opened high-score false positives included
visibly open eyes, a dark/low-contrast eye and one reflective, borderline
appearance. This **small visual spot check** cannot adjudicate labels or
quantify dataset labeling errors. The local ignored review queue
`reports/runs/mrl_calibration_validation_v1.json.false_positives.csv`
contains the 493 paths and scores for a larger human review. No image paths
or source images are committed to Git.

## Reproduce on a licensed local copy

From the repository root on the same CPU environment, after obtaining and
preparing the original MRL archive as described in
[CNN quickstart](CNN_QUICKSTART.md):

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\.venv-cognition\Scripts\python.exe scripts/calibrate_cnn.py data/processed/mrl_manifest_v1.csv models/checkpoints/mrl_full_3epoch.pt reports/runs/mrl_calibration_validation_v2.json cpu
```

Use a new output name because the script refuses overwrite. The JSON includes
checkpoint and manifest hashes, temperature fit, score-quality measures,
candidate cutoff metrics, and validation breakdowns by subject and available
MRL source fields. The adjacent CSV is a **local review aid** that includes
image paths and is excluded by `.gitignore`.

## Decision from this run

The score calibration **experiment was completed**, but it did **not** improve
validation probability quality. The lower-FPR candidate reduced open-eye
errors at a large cost to closed-eye recall, with uneven subject and condition
results. Keep the current checkpoint and temperature **out of live alerts**.
Review more individual errors, establish a task-specific operating objective,
obtain paired/video and representative RGB camera data, and then reevaluate
the model and rule. Keep the MRL test subjects untouched until the model and
decision rule are frozen.
