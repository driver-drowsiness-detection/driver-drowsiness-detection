# 12. Decisions, evidence and references

## Decision register

| Item | Current status | Options / next evidence |
| --- | --- | --- |
| Core experiment | Confirmed | EAR, eye CNN and calibrated fusion, all retained for comparison |
| Team assignments | Intentionally omitted | Assign components after reviewing this package |
| Current perception implementation | Team reports camera/frame work in progress | Compare actual teammate scripts against the shared contract |
| Reference eye CNN | Small custom CNN supplied | Compare with a compact pretrained alternative after data audit |
| Dataset | Undecided | Check candidate licence, subjects, eye labels and temporal annotations |
| Numerical eye/temporal thresholds | Development examples only | Select on development data, freeze before test |
| Severity names and thresholds | Undecided | Independent labels and validation before fatigue claims |
| P80 PERCLOS definition | Undecided | Physical closure/annotation convention and blink treatment |
| Pose facial template/axes | Undecided | Choose documented 3D correspondences and validate orientation |
| Target landmark runtime | Undecided | Saved-image test on inspected Jetson environment |
| Target CNN runtime | Undecided | PyTorch versus OpenCV ONNX versus TensorRT compatibility and measurements |
| LLM model | Undecided by user confirmation | Model card, licence, size, runtime support and event-response tests |
| LLM endpoint | Undecided by user confirmation | On Nano preferred; paired local device fallback from proposal |
| Language policy | Constrained initial reference | Free wording requires factuality evaluation and policy revision |
| Final TTS engine/voice | Undecided | SAPI development, eSpeak adapter, Piper candidate after target audit |
| GPIO wiring/watchdog | Undecided | Actual board, buzzer electrical specification and reviewed circuit |
| Final acceptance targets | Undecided | Agree measured recall, false alerts, coverage, latency and resource limits |

## What was inspected in this run

1. The original proposal: `C:/Users/alwan/Downloads/MARVEL_Proposal_Drowsiness_LLM-proposal final.md`.
2. The original technical guide: `C:/Users/alwan/Downloads/MARVEL_Drowsiness_LLM_Guide.md`.
3. The earlier chat titled “Plan drowsiness detection system,” including the explicit EAR/CNN/fusion scope and proposal attachment paths.
4. The earlier repository setup discussion and stored Phase 1 guide history.
5. Current `docs/PHASE_1_SYSTEM_DESIGN.md`, the four Phase 1 Markdown guides, scaffold source/configuration/tests and the dedicated clone's entry point.
6. Official current documentation for the APIs and compatibility boundaries listed below.

The original proposal supplies the aim. Later user decisions supersede its earlier Phase 1 scheduling: CNN, fusion, data and evaluation are required earlier. The present user instruction supersedes old named ownership: this package contains no junior/senior assignment split.

## Verification performed here

<!-- VERIFIED -->

* Python syntax: all 53 supplied Python files parsed successfully.
* Automated tests: 50 discovered, 48 passed, 2 explicitly skipped.
* Skips: PyTorch forward/backward/checkpoint test and OpenCV preprocessing parity test.
* Runtime used here: bundled Python 3.12 with NumPy/Pillow; OpenCV, MediaPipe and PyTorch unavailable.
* Three rendered diagrams were visually inspected.
* Every source/configuration file is embedded in exactly one numbered guide.
* Relative Markdown links, balanced fences and byte-equivalent source listings were checked.
* [Detailed test output](verification.txt) is included.

<!-- VERIFIED_END -->

The behavior tests include geometry invariants, missing-value handling, fusion alignment and range validation, independent method histories, event timing, recovery, gaps, track changes, window coverage, grouped manifest rejection, JSON logging, constrained speech, hard timeout and stale-event cancellation.

The offline integration test uses artificial predictions and a dummy file solely for hash identity. It exercises calibration, fusion selection and evaluation command contracts. It does not train or execute a neural model. Perfect scores on that fixture are not project accuracy results.

## Not verified here

Camera hardware; actual landmark inference; real image crop quality; CNN forward/backward execution; checkpoint inference on images; training convergence or accuracy; chosen datasets/licences; Windows or Linux audible output; head-pose calibration; ONNX export/runtime parity; TensorRT; Jetson compatibility/performance; a real LLM server; neural TTS; real driving reliability.

Those are acceptance gates in Guides 10 and 11. The reference scripts are supplied for execution and testing, but absent dependencies and hardware are not concealed by successful pure-Python tests.

## Source and API references

References were checked on 4 October 2026. Recheck installation and target-specific compatibility when selecting exact releases. The implementation and project policy are original reference choices; citations support API behavior, not unmeasured project claims.

| Source | What it supports |
| --- | --- |
| [MediaPipe Face Landmarker Python](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python) | Task model asset, running modes, image/landmark contract |
| [MediaPipe Python setup](https://developers.google.com/edge/mediapipe/solutions/setup_python) | Setup and platform/version requirements; not proof of original Nano wheel compatibility |
| [MediaPipe eyelid connections](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py) | Landmark topology reference; overlay checks still required |
| [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) | Camera/video capture interface and backend properties |
| [OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html) | 2D/3D correspondences, camera parameters and pose estimation |
| [PyTorch installation](https://pytorch.org/get-started/locally/) | Select installation for the actual OS and compute platform |
| [PyTorch binary loss with logits](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html) | Stable binary training loss and target convention |
| [PyTorch save/load tutorial](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) | State-dictionary loading and evaluation mode |
| [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx.html) | Version-sensitive export interface; inspect installed exporter |
| [Scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html) | Held-out score calibration and reliability evaluation concepts |
| [NVIDIA JetPack 4.6.6](https://developer.nvidia.com/jetpack-sdk-466) | Original Nano software-generation compatibility context |
| [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano) | Target hardware family |
| [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | Local chat-completions interface and structured response support |
| [Piper maintained repository](https://github.com/OHF-Voice/piper1-gpl) | Current neural TTS project and licence review starting point |
| [Python concurrency documentation](https://docs.python.org/3/library/concurrent.futures.html) | Why waiting with a timeout is not the same as stopping already running work |

No source establishes that the selected CNN, an undecided language model, or a particular dependency combination works on your board. That must come from your own recorded experiments.

## Decisions worth retaining in notes

1. Used EAR as an interpretable baseline because its geometry and errors can be inspected.
2. Used an eye CNN because learned appearance may complement geometry; superiority remains a measured hypothesis.
3. Used separate calibration and validation because score fitting and operating-point selection are different operations.
4. Used the same temporal algorithm with independent instances because fair comparison needs common rules without shared history.
5. Used unknown states and coverage because missing observations must not look like reassuring evidence.
6. Used immediate fixed alerts because generated language cannot be a prerequisite for warning.
7. Used a separate reference package because the team has ongoing perception work that should be integrated deliberately.

## Final learning check

Explain one frame from camera to alert, one training sample from annotation to checkpoint, and one alert from event ID to spoken sentence. At every arrow name the data type, timestamp, validity rule, failure behavior and test. If you can do that without reading variable names aloud, you understand the system's structure.
