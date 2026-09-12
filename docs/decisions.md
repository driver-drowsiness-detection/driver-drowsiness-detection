# Decision log

## 2026-09-12: scaffold from the revised target tree

Used the existing Phase 1 tree because the team asked to create its listed paths before implementation. Files are explicitly pending, so structure cannot be mistaken for a working detector. Existing documents are retained.

## 2026-09-12: user-directed ownership revision

Used a senior ML pair and a junior runtime pair because Vishruth explicitly requested that he and Arushuuu jointly own most ML implementation and experiments. This supersedes the initial mixed-experience pairing. Vishruth owns geometry, fusion and temporal logic; Arushuuu owns data, CNN training/inference and evaluation, with mutual review. shamaa007 owns capture and overlays; niharika526 owns logging, audio and profiling. Vishruth supervises junior deliverables at checkpoints. Synthetic input contracts let the juniors implement and test independently before trained models exist.

## 2026-09-12: Git setup finding

The inspected workspace inherits a Git root at C:/Users/alwan/OneDrive/Desktop. Vishruth subsequently supplied https://github.com/driver-drowsiness-detection/driver-drowsiness-detection. A read-only remote check found main at 1188bc7663e423364e92e0c4cb4756bbaf31a05d. No repository was initialized, connected, committed or pushed during scaffolding. Use a dedicated clone to preserve remote history before transferring and committing the scaffold.

## 2026-09-12: dedicated Documents clone

Used C:/Users/alwan/Documents/driver-drowsiness-detection because Vishruth requested the Git migration and suggested Documents. The clone preserves remote history. Inspection found actual baseline implementations and tests, so every original tracked file is retained under legacy/phase1_initial instead of overwritten by scaffold placeholders. The root now follows the revised design. Changes are submitted through codex/revised-phase1-scaffold for review; main is not merged automatically.
