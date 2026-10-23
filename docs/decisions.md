# Decision log

## 2026-10-23: connect EAR and implement observable cognition measurements

Used the reviewed EAR source from `origin/ear-cognition` at `f8cddad` because its geometry and identity behavior had already been checked. Integrated those files without a wholesale branch merge. Used elapsed-time interval accounting with explicit coverage because frame counts and missing observations can distort a rolling closure statistic. The output remains a binary closure proxy until a physical PERCLOS criterion is validated.

Added MAR as telemetry, calibrated PnP as an optional measurement, and a bounded Windows fixed-sound worker. No camera intrinsics, trained CNN weights, calibration coefficients, fatigue cutoffs or LLM model were invented. Voice execution location remains an open user choice. Existing team ownership is preserved.

## 2026-10-22: full proposal for the next review

The user selected "Full proposal: also PERCLOS, head pose and LLM voice" and requested completion "by this week". These are review requirements rather than optional stretch work. The exact review day and Jetson access window are not yet confirmed. Keep implementation, measured detection quality and target-hardware evidence distinct; do not replace missing evidence with synthetic results or assumed compatibility.

Used elapsed-time temporal tracking because the accepted design requires behavior independent of FPS, explicit unknown handling and recovery persistence. The first implementation adds typed events, a strict JSONL logger and synthetic replay. Demonstration durations are not accepted or validated production thresholds. It preserves the existing ownership split and does not select a new CNN or LLM architecture.

## 2026-09-12: scaffold from the revised target tree

Used the existing Phase 1 tree because the team asked to create its listed paths before implementation. Files are explicitly pending, so structure cannot be mistaken for a working detector. Existing documents are retained.

## 2026-09-12: user-directed ownership revision

Used a senior ML pair and a junior runtime pair because Vishruth explicitly requested that he and Arushuuu jointly own most ML implementation and experiments. This supersedes the initial mixed-experience pairing. Vishruth owns geometry, fusion and temporal logic; Arushuuu owns data, CNN training/inference and evaluation, with mutual review. shamaa007 owns capture and overlays; niharika526 owns logging, audio and profiling. Vishruth supervises junior deliverables at checkpoints. Synthetic input contracts let the juniors implement and test independently before trained models exist.

## 2026-09-12: Git setup finding

The inspected workspace inherits a Git root at C:/Users/alwan/OneDrive/Desktop. Vishruth subsequently supplied https://github.com/driver-drowsiness-detection/driver-drowsiness-detection. A read-only remote check found main at 1188bc7663e423364e92e0c4cb4756bbaf31a05d. No repository was initialized, connected, committed or pushed during scaffolding. Use a dedicated clone to preserve remote history before transferring and committing the scaffold.

## 2026-09-12: dedicated Documents clone

Used C:/Users/alwan/Documents/driver-drowsiness-detection because Vishruth requested the Git migration and suggested Documents. The clone preserves remote history. Inspection found actual baseline implementations and tests, so every original tracked file is retained under legacy/phase1_initial instead of overwritten by scaffold placeholders. The root now follows the revised design. Changes are submitted through codex/revised-phase1-scaffold for review; main is not merged automatically.
