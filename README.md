# MARVEL - Driver Drowsiness Detection

Edge-deployed driver drowsiness detection on a Jetson Nano (B01, 4GB),
combining classical CV (EAR/MAR facial landmarks) with a later LLM +
TTS reasoning layer. Full details in the project proposal and technical
guide (not in this repo - keep those in your team drive).

## Phase 1 (current) - CV pipeline v1

4-week MVP: camera -> face + landmark detection -> EAR/MAR -> naive
threshold trigger -> buzzer. Being developed on laptop webcams first,
ported to the Jetson once JetPack is flashed and MediaPipe is installed
there (see project notes on JetPack 4.6 / Python 3.6 constraints).

| Module | File | Owner |
|---|---|---|
| Camera capture + preprocessing | `src/capture/capture.py` | Person 1 |
| Face + landmark detection | `src/detection/detection.py` | Person 2 |
| EAR/MAR feature extraction | `src/features/features.py` | Person 3 |
| Threshold + trigger logic | `src/trigger/trigger.py` | Person 4 |
| Buzzer stub | `src/buzzer/buzzer.py` | Unassigned - team, same as real hardware wiring |
| Integration | `src/main.py` | Built together, Week 4 |

Everything else under `src/` (`pose/`, `perclos/`, `state_machine/`,
`prompt_builder/`, `llm_inference/`, `tts/`, `event_logger/`) is an
empty placeholder for Phases 2-4 - don't build into these yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running things

```bash
# Test one module in isolation (works standalone, no other module needed)
python src/capture/capture.py
python src/detection/detection.py

# Run all unit tests
pytest tests/

# Run the full integrated pipeline (needs all 4 modules working)
python src/main.py
```

## Git workflow

- One `feature/<module-name>` branch per person, PR into `main`.
- At least one other person reviews before merge - even 2 minutes helps,
  and everyone should understand code outside their own module by demo time.
