# Work record: 7 November 2026

## Starting point

Local branch: cognition-integration. Starting commit: feb609c, containing the
CNN branch through 14ae038. The preceding merged suite passed 153 tests.
Existing voice files, reference guides and the local .gitignore edit were
already present. They are outside this increment.

The user selected EAR/CNN comparison and fusion foundation. The implementation
and its explanation are in COMPARISON_FOUNDATION.md.

## Implemented

1. Strict per-eye weighted fusion with method, numerical range and identity checks.
2. Three independent temporal histories and closure windows using common timing.
3. Standalone baseline preservation when the other branch is missing or stale.
4. Bilateral UNKNOWN behavior for invalid or disagreeing eyes.
5. Estimate JSONL replay with input/config hashes, original evidence, decisions,
   events, sample coverage counts and unresolved episode IDs.
6. A reproducible synthetic demo that deliberately removes CNN evidence once.
7. Regression tests and notes including design alternatives, implementation
   walkthrough, complexity, interview explanation and a prediction exercise.

## Verification

Focused suite: 32 tests passed in 0.22 seconds. Full repository suite:
185 tests passed in 50.37 seconds with MARVEL_TEST_MODEL pointing to the local
face landmark asset. This includes the previously merged EAR and CNN tests.

Executed demo: reports/local/comparison_demo_2026_11_07/comparison.jsonl.
Its input and settings are stored beside it. All nine observations and all
settings are synthetic. Observed output:

| Event | EAR | CNN | Fusion |
| :--- | :--- | :--- | :--- |
| First prolonged closure | 1.0 s | 1.0 s | 1.0 s |
| Recovery | 2.0 s | 2.0 s | 2.0 s |
| Second prolonged closure | 3.5 s | 4.0 s | 4.0 s |

EAR had nine usable observations; CNN and fusion each had eight. The common
observable count was eight. The deliberately missing CNN evidence at 2.5 s
explains the later CNN/fusion event. The run ends with all three second episodes
unresolved, which is recorded explicitly. No recovery is fabricated at shutdown.

Tests establish software behavior. They do not measure accuracy or prove that
fusion improves detection. No model was trained, camera opened, sound played,
held-out MRL test scored or Jetson accessed in this increment.

## Remaining work and dependencies

| Item | Status and dependency |
| :--- | :--- |
| Model handoff | Need checkpoint or ONNX plus metadata from the training machine |
| Camera comparison adapter | Pending; connect measured EAR and CNN estimates and record model/config identity |
| EAR calibration fitter | Pending; trustworthy fitted coefficients need reviewed calibration samples |
| CNN calibration adoption | Experimental temperature worsened validation quality; no accepted setting |
| Fusion tuning | Pending labelled validation data and agreed operating objective |
| Final MRL test | Remains unscored until model and decision rule are frozen |
| Sequence evaluator | Pending independent event labels, coverage protocol and recordings |
| EAR hardware acceptance | Live threshold behavior, audible tone, stalls and restart checks pending |
| PERCLOS | Closure proxy exists; physical closure protocol and sequence validation pending |
| Head pose | Interface exists; measured camera calibration and real movement validation pending |
| Voice | Local draft workers exist; tests and runtime integration pending |
| Jetson | Compatibility, parity, camera, audio and sustained profiling pending |

No completion percentage is assigned because implementing modules and validating
their behavior on real driving observations are different acceptance gates.
