# MARVEL: complete implementation handbook

Prepared 4 October 2026. Organized by component, with no personal or senior/junior assignments.

**Goal:** camera input on the Jetson Nano, usable facial observations, separate EAR and CNN estimates, calibrated combination, temporal monitoring, immediate warning, and an independent local LLM and speech path. Compare all three methods and retain measured evidence.

**Read this accurately:** this is a complete instructional reference with full source listings and an executable companion folder. It is not a trained model, a validated embedded release, or proof of reliable driver monitoring. The reference does not overwrite the existing application scaffold or your teammates' perception code.

## Read in this order

| Guide | What it gives you |
| --- | --- |
| [01 Architecture and proposal alignment](01_ARCHITECTURE_AND_SCOPE.md) | Full flow, final aim, corrections to the proposal, confirmed and undecided decisions |
| [02 Repository, environment and contracts](02_REPOSITORY_SETUP_AND_CONTRACTS.md) | Entire reference tree, each folder's purpose, setup, shared records |
| [03 Perception](03_PERCEPTION.md) | Camera/video capture, watchdog, landmarks, crops, quality checks, preview |
| [04 Geometry and head pose](04_EAR_MAR_AND_POSE.md) | EAR, MAR, baseline classifier, camera calibration, optional PnP |
| [05 Data and CNN](05_DATA_AND_CNN.md) | Independent labels, subject splits, frame extraction, preprocessing, training, inference |
| [06 Calibration and fusion](06_CALIBRATION_AND_FUSION.md) | EAR mapping, CNN temperature, weight/threshold selection, frozen bundle |
| [07 Temporal cognition](07_TEMPORAL_COGNITION.md) | Blink versus closure, recovery, missing data, rolling closure, experimental state tiers |
| [08 Runtime, alerts and logs](08_RUNTIME_ALERTS_AND_LOGGING.md) | Main application, three independent modes, overlays, sound, resource cleanup |
| [09 LLM and speech](09_LLM_AND_SPEECH.md) | Provider contract, mock, hard timeout, validation, fallback, speech, endpoint options |
| [10 Execution and testing](10_EXECUTION_AND_ACCEPTANCE.md) | Exact build/run sequence, expected outputs, test matrix, measurements |
| [11 Jetson and deployment](11_JETSON_AND_EXPORT.md) | Board inspection, CSI pipeline, ONNX, parity, profiling, TensorRT decision gate |
| [12 Decisions, evidence and references](12_DECISIONS_EVIDENCE_AND_REFERENCES.md) | Open choices, proposal traceability, verified versus pending, source links |

All code listings marked `File:` are complete files, not fragments with hidden missing functions. The same files exist in [reference](reference). The numbered guides contain the explanations and the complete relevant code. Package `__init__.py` files contain only module docstrings; the source inventory includes them.

## What to do first with your current progress

1. Read Guide 01 together so everyone knows what comes after camera capture.
2. Read the `Frame` and `Eye` contracts in Guide 02. Compare them with the perception scripts your teammates are actually running.
3. Run the dependency-light tests and synthetic replay from Guide 10. These do not need a trained CNN or a camera.
4. Implement or adapt perception so both eyes preserve the same frame ID, observation time and continuity epoch.
5. Run the EAR path. In parallel, prepare independently labelled data and train the CNN.
6. Run CNN alone on the same recordings, then calibration and fusion.
7. Add rolling statistics, inspect the proposed severity policies, and validate them on labelled sequences.
8. Test the speech contract with a mock. Select a real LLM endpoint later, then test it under load.

The simple capture handoff remains small. Most implementation depth is in dataset integrity, CNN training, calibration, fusion, time, evaluation, deployment and LLM integration. You can assign those components later without restructuring this package.

## What is deliberately undecided

The LLM model, exact endpoint, on-device versus companion deployment, production speech engine, dataset choice, validated thresholds, final head-pose template/axes, fatigue severity policy, and Jetson dependency lock remain undecided. Options and decision criteria are in Guides 09, 11 and 12. No credentials, cloud account or paid service is required by the reference.

## Validation boundary

The test report in Guide 12 is the authority for what ran. Syntax checks are not execution tests. Synthetic data verifies behavior, not accuracy. A successful camera preview does not validate facial landmarks. A loadable checkpoint does not validate the CNN. A good Windows run does not establish Nano compatibility.

You can learn from the entire package now and implement one checkpoint at a time. Do not copy all files into the team repository and claim completion before running the acceptance sequence.
