# MARVEL camera and eye perception

The integrated perception demo acquires live BGR frames, detects one face, extracts
six points and a crop for each eye, checks usability and freshness, and shows the
results. It does not classify drowsiness or run EAR/CNN, alarms, or an LLM.

Start here for the current executable endpoint. The older team/environment/design
guides describe the original scaffold and wider roadmap.

- [Branch comparison and verified behavior](docs/PERCEPTION_INTEGRATION.md)
- [Two-person Git ownership, commits, pulls and review](docs/PERCEPTION_TEAM_WORKFLOW.md)
- [Hardware acceptance record](docs/PERCEPTION_ACCEPTANCE.md)
- [Jetson camera and compatibility gates](docs/JETSON_PERCEPTION.md)

## Get the same code on each laptop

Save/commit any existing work on its own branch before switching. From your clone:

```powershell
git status --short
git fetch origin
git switch --track origin/codex/perception-integration
git pull --ff-only origin codex/perception-integration
git rev-parse HEAD
```

If that local branch already exists, use `git switch codex/perception-integration`.
Compare the final commit on all machines. For a new clone:

```powershell
git clone --branch codex/perception-integration https://github.com/driver-drowsiness-detection/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

The integration PR targets main. The existing contributors' branches retain their
history. After the PR is merged, main becomes the shared endpoint.

## Windows setup

Tested here: Windows x86-64, Python 3.11.4. Use a separate local environment for
perception; do not replace an existing CNN environment. Run in the repository root:

```powershell
py -3.11 -V
py -3.11 -m venv .venv-perception
.\.venv-perception\Scripts\python.exe -m pip install -r environments\windows-perception-lock.txt
.\.venv-perception\Scripts\python.exe -m pip check
.\.venv-perception\Scripts\python.exe -m pytest -q
```

Create the environment only once. If it already exists, verify its interpreter
with `.\.venv-perception\Scripts\python.exe -V` before using it. If `py` points
to an inaccessible Store alias, use your actual installed Python 3.11 executable
with `-m venv .venv-perception`. Keep other Python installations intact.

This lock installs one GUI-capable OpenCV distribution: opencv-contrib-python.
Do not add opencv-python or a headless variant alongside it. Each person creates
their own environment. The lock is not a Jetson installation recipe.

## Run camera, then perception

```powershell
.\.venv-perception\Scripts\python.exe scripts\preview_perception.py camera
```

Press q with the preview focused, then repeat to check camera release and restart.
Next download the shared model (versioned URL, hash checked, no overwrite):

```powershell
.\.venv-perception\Scripts\python.exe scripts\download_landmarker.py
.\.venv-perception\Scripts\python.exe scripts\preview_perception.py perception
```

Scripts find the repository's src package automatically; no PYTHONPATH setup is
needed. The default YAML is configs/runtime.yaml. Camera index 0 is a candidate:
copy the YAML to ignored `configs/local-laptop.yaml` and change only the settings
your machine needs. Pass it as the second argument:

```powershell
.\.venv-perception\Scripts\python.exe scripts\preview_perception.py perception configs\local-laptop.yaml
```

The main window shows points, crop rectangles, identity, frame age, processed FPS,
validity and quality scores. Two other windows show current anatomical left/right
eye crops or UNAVAILABLE. VALID is usable evidence; a closed eye can be valid.
No detected face, multiple faces, failed quality and stale frames are unavailable.

Default blur rejection is disabled (`laplacian_min: 0.0`), exactly as in the
reference. Scores are measured so the team can select and test a threshold.
No eye-occlusion guarantee is implied by these simple metrics.

## Saved-image and automated smoke checks

```powershell
.\.venv-perception\Scripts\python.exe scripts\preview_perception.py perception --image C:\path\to\face.png
.\.venv-perception\Scripts\python.exe scripts\preview_perception.py perception --no-display --max-frames 10 --max-seconds 20
$env:MARVEL_TEST_MODEL = (Resolve-Path models\face_landmarker.task).Path
.\.venv-perception\Scripts\python.exe -m pytest -q
```

Without MARVEL_TEST_MODEL, the real-model smoke test is explicitly skipped.
Normal tests use controlled fixtures and do not open a camera. The application
does not save images; its rate-limited terminal observations contain metadata,
coordinates, quality and crop shapes. Keep actual acceptance results separately.

The official model SHA256 is
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
Download and retain the same verified asset on each machine; it is ignored by Git.

## Jetson Nano

Begin with the [Jetson guide](docs/JETSON_PERCEPTION.md). It provides a standalone
Python 3.6+ camera probe using the board's system OpenCV, and a complete candidate
configs/jetson-camera.yaml for a compatible full runtime. Native camera preview,
Python capture and full landmark perception are separate acceptance gates.
No Nano execution has been verified in this session.

## Code ownership and integration

Canonical shared records: src/marvel/contracts.py. Person 1 owns acquisition;
Person 2 owns landmarks, eye cropping and quality. Both own preview integration.
The source branch histories and retained tests are described in the comparison
document. CNN training and the application-wide cognition pipeline remain outside
this perception integration. The older baseline is preserved in legacy/phase1_initial.
