# Dataset access and real image inspection

Checked 2026-10-22. This is an acquisition and feasibility record, not a training
or accuracy report. NTHU-DDD remains the proposal's main video benchmark.

## Source comparison

| Source | Intended role | Verified access and limitations |
| --- | --- | --- |
| NTHU-DDD | Driving-video benchmark for temporal EAR/CNN evaluation | Original site returned timeout/bad gateway. The 2016 organizers' document requires registration and a signed dataset agreement. That historical process is not confirmed current. No files or agreement were obtained. |
| MRL Eye | Candidate additional CNN eye-classification data | Original authors' paper describes eye-state labels, subject IDs and infrared eye crops. Website and direct official archive timed out. No archive or current license was obtained. Eye crops alone do not establish the full EAR/video benchmark. |
| CEW | Local real-image perception inspection candidate | Authors' original Drive releases downloaded and inspected. Research-only use, no commercial use or redistribution without permission. Both class labels are supplied at face level. Closed-face person identities are unresolved. No training split or model was created from it. |

Primary sources:

1. [NTHU original dataset page](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/)
2. [2016 challenge organizers' access document](https://www2.securecms.com/ICIP2016/DriverDrowsiness_IEEEICIP2016_ChallengeSpecialSession.pdf)
3. [MRL authors' dataset paper](https://mrl.cs.vsb.cz/publications/ISVC18_for_reviews.pdf)
4. [CEW authors' page, description and research terms](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html)

CEW citation: F. Song, X. Tan, X. Liu and S. Chen, Eyes Closeness Detection from
Still Images with Multi-scale Histograms of Principal Oriented Gradients,
Pattern Recognition, 2014.

Raw images, source archives, extracted crops and identifying filenames remain
in ignored local folders. Do not push them or redistribute the download links.
Teammates should consult the authors' terms and obtain their own authorized copy.

## Actual downloaded contents

| Local archive | SHA256 | Observed contents |
| --- | --- | --- |
| `data/raw/cew/faces.rar` | `25c1eb8572a8755b87106a4bfea7f2dd89c1abaab95196eee51d43de9f8c6b48` | Original-resolution release: 1,192 closed-eye face images only |
| `data/raw/cew/faces100.rar` | `0d02c9eb8810cd945b65e6a2e22aca24298e37439b655fc291602cbcf3be7e4a` | Normalized release: 1,192 closed and 1,231 open face images, all 100 by 100 pixels; two eye-coordinate text files; Windows thumbnail files |

Hashes establish the local artifact identity; they were computed here and are
not checksums published by the authors. Archive member paths were checked before
extraction. Image decoding and coordinate-to-image correspondence passed.
There were no identical decoded-pixel image groups within the normalized release;
this does not prove the absence of near duplicates or repeated people.

## Perception feasibility results

The current Detector and runtime quality settings processed all 2,423 normalized
images using the existing face_landmarker.task. No webcam was opened and no
quality thresholds were relaxed. Each still image reset tracking continuity;
adapter timestamps were monotonic indices, not measured video timestamps.

| Face source class | Both eyes pass | No face | Multiple faces | Invalid eye |
| --- | ---: | ---: | ---: | ---: |
| Closed | 1,145 | 32 | 2 | 13 |
| Open | 1,223 | 4 | 2 | 2 |

4,749 usable eye crops were saved: 2,302 from closed-class source faces and 2,447
from open-class source faces. This is 4,749 of 4,846 possible eye observations,
about 98.0 percent. It is pipeline acceptance coverage, not detection correctness
or classification accuracy. Blur rejection is disabled in the existing config;
passing its quality gate does not guarantee a visually correct crop.

Source metadata exposes 1,231 open filename identity hints. Closed filenames
cover 1,134 source-image groups, of which 50 contain multiple extracted faces.
A source-image group is not a subject ID. Cross-class repeated identity is also
unresolved. Never assign image numbers as person IDs just to satisfy the manifest.

## Reproduce the inspection

```powershell
.\.venv-cognition\Scripts\python.exe scripts/inspect_cew_release.py data/raw/cew/normalized/dataset_B_FacialImages reports/local/cew_inspection_v2 --task models/face_landmarker.task
```

Use a fresh output directory. The script rejects overwrite and validates images,
source coordinates and filename structure. It produces:

1. `image_inventory.csv`: file and decoded-pixel hashes, face class and identity hints.
2. `eye_review_queue.csv`: separate anatomical eyes, detected points, crop paths,
   rejection reasons, blank review labels, blank verified subject IDs and splits.
3. `crops/`: valid eye crops for local review.
4. `summary.json`: counts, settings, landmark hash and unresolved blockers.

The first run is under `reports/local/cew_inspection_v1/`. The queue deliberately
does not conform to the training manifest; it must not be fed to train_cnn.py.
Source face class and independently reviewed eye visibility are separate fields.

Validation: four new inspection tests passed, followed by the combined suite
with the real landmark model and CUDA enabled: 62 passed in 45.95 seconds.
One open and one closed source image and their extracted left-eye crops were
visually inspected. This spot check does not review all 4,846 eye observations.

## What is needed before real training

Preferred path: obtain the NTHU original videos, annotations, metadata and current
terms through the authors/university access process. Ask the team supervisor to
help if an institutional registration or signature is required. Inspect a sample
before implementing an importer; do not presume that drowsiness labels are eye labels.

For a supplemental eye-crop source, resolve MRL access and terms, then inspect its
actual subject IDs and label encoding before selecting it. The authors' paper
uses 0 for closed and 1 for open; this project uses the reverse convention.

CEW can already test perception on real images. Using it for reported training
and evaluation still requires reliable grouping metadata and reviewed labels.
No face-recognition system is used to infer identities from these images.

Once grouping and labels are established: split subjects before cropping, keep
both eyes/source frames together, audit leakage, run one CUDA epoch for timing,
then iterate on validation only. Keep the test set untouched until decisions are
frozen. EAR/fusion/temporal and Jetson claims require separate experiments.
