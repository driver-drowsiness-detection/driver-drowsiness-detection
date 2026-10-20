# EAR development handoff

Owner: Vishruth (confirmed by Arushi on 2026-10-20). Reviewer: Arushi.
Branch: `codex/ear-cognition`; base: `final-perception` at `63ddcc1`.

## Implemented endpoint

`Eye` -> `measure(eye)` -> raw `Estimate(method="ear")`.
No camera, MediaPipe, NumPy or CNN is imported by the geometry modules.
Six original-image pixel points use corners 0/3 and vertical pairs 1/5, 2/4.
EAR = (distance(1,5) + distance(2,4)) / (2 * distance(0,3)).
The minimum corner width is 2 pixels. Invalid/degenerate/nonfinite geometry
produces missing evidence. A zero ratio with separated corners remains valid.

The current perception gate couples `geometry_ok` to crop quality. EAR respects
that contract; it does not bypass rejected images. It retains frame ID, timestamp,
track and anatomical side for later same-observation fusion.

`state(estimate, threshold)` requires an explicit positive raw-EAR threshold:
below is CLOSED; equality and above are OPEN; missing evidence is UNKNOWN.
No universal 0.2 cutoff has been silently fitted or accepted.
`probability(estimate, {"slope": a, "intercept": b})` applies a finite negative
slope logistic mapping. Coefficients must eventually come from independent
calibration data. Applying a mapping is implemented; fitting one is pending.

## Run on a laptop

From your own checkout (replace the path), with Python 3.11:

```powershell
git fetch origin
git switch codex/ear-cognition
git pull --ff-only origin codex/ear-cognition
py -3.11 -m venv .venv-perception
.\.venv-perception\Scripts\python.exe -m pip install -r environments/windows-perception-lock.txt
.\.venv-perception\Scripts\python.exe scripts/download_landmarker.py
$env:MARVEL_TEST_MODEL = "models/face_landmarker.task"
.\.venv-perception\Scripts\python.exe -m pytest -q
.\.venv-perception\Scripts\python.exe scripts/inspect_ear.py "path/to/your/full-face-image.png"
```

Reuse an existing working `.venv-perception`; create it only if absent.
The image command reports raw EAR and UNKNOWN state until a threshold is given.
For an explicitly **unvalidated demonstration** add `--threshold 0.20`.
This saved-image tool does not determine drowsiness or estimate durations.
Use the existing perception preview for live camera/quality inspection.

## Next evidence required

1. Obtain the actual NTHU-DDD files/access described in the proposal; inspect the
   accompanying annotation definition. Keep eye closure, drowsiness and yawning
   labels distinct. Do not train the CNN with EAR-generated labels.
2. Review open, closed, partly closed and unobservable eye labels on full frames.
   Pair by sample ID and eye side; group subjects before any frame/crop split.
3. Fit the EAR score mapping on calibration subjects; select the raw-EAR cutoff
   on validation subjects. Report rejected measurements and coverage.
4. Freeze parameters; evaluate held-out subjects with the same bilateral rule
   used for CNN and fusion. No accuracy is claimed by the geometry unit tests.
5. Add live cognition integration, timestamp-based closure/PERCLOS, explicit
   missing-observation handling and recovery before driving alert logic.
6. Audit actual Jetson Python/JetPack/MediaPipe compatibility before deploying.

The proposal's personal open-eye calibration is a later separately evaluated
mode. It must use verified open eyes and must not adapt while eyes are closing.
MAR, head pose and camera intrinsics are outside this first EAR increment.

## Git collaboration

Commit your changes on this branch with descriptions such as
`Fit EAR cutoff on subject-disjoint validation samples`. Pull with `--ff-only`
before working. If it refuses because histories diverge, inspect the commits and
merge deliberately; never force push a teammate's work away.
Review changes against `final-perception`. The CNN branch can proceed separately;
merge both through a reviewed cognition integration branch once each checkpoint
passes. `main` is still a scaffold as of the audit.
