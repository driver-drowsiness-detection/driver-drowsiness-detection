# Preserved initial baseline

Every tracked file from commit 1188bc7663e423364e92e0c4cb4756bbaf31a05d is preserved at its original relative path beneath this directory, with identical bytes. The old README, ownership, timelines, requirements and TODOs are historical and do not override docs/TEAM_START_HERE.md.

The revised implementation belongs in src/marvel at the repository root. Reuse suitable baseline code through reviewed increments that satisfy the new contracts, timing and validity rules.

The baseline feature and trigger tests were run successfully before migration and again after relocation. Their passing results do not validate the revised scaffold. Capture and detection tests require OpenCV/NumPy/MediaPipe and were not executed in this migration.

From the repository root, using an available Python interpreter:

```powershell
python legacy/phase1_initial/tests/test_features.py
python legacy/phase1_initial/tests/test_trigger.py
```

The legacy buzzer is a print stub and its trigger counts frames. Neither implements the revised sound or timestamp-based requirements.
