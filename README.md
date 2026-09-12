# MARVEL drowsiness monitoring

Status: the revised src/marvel layout is scaffold only. The earlier baseline code and tests are preserved in legacy/phase1_initial; they have not been ported into the revised pipeline.

Start with [team setup](docs/TEAM_START_HERE.md). The complete module plan is in the [system design](docs/PHASE_1_SYSTEM_DESIGN.md), section 7. Every path in its revised target tree has been created. Existing design documents and diagrams are retained.

The operational utility in the revised scaffold is the standard library environment inspection:

```powershell
python scripts/inspect_environment.py
```

Use an installed Python interpreter. An unavailable command is a setup blocker to record, not an invitation to install arbitrary packages.

## Planned data flow

```text
Camera or video
    ↓
Face landmarks and eye crops
    ↓
EAR estimate or CNN estimate or calibrated combination
    ↓
Shared duration and recovery logic
    ↓
Fixed sound + display + logs
```

The CNN classifies eye images as open or closed. The temporal controller interprets closure over time.

## Scaffold conventions

1. Python modules contain responsibility notes, not fake predictions.
2. Unimplemented application and script entry points exit with an explicit error.
3. Test files describe pending behavioural checks and currently contain no tests.
4. Configuration files are empty mappings with pending decisions documented in comments.
5. Dependency lists contain comments only. Installing them does not install a working detector.
6. Empty directories contain .gitkeep so they survive Git checkout.
7. Real recordings, model files and run outputs remain local under the ignored directories.

Package metadata establishes the src layout. The build backend requirement is packaging scaffolding, not a validated Windows or Jetson environment. The supported Python range and numerical dependency pins remain pending the environment audit.

After the leads validate and document the laptop environment, install the agreed dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -c "import marvel; print(marvel.__file__)"
```

Editable installation makes imports use src/marvel. It does not implement the application.

## First completion gate

All four people can describe their setup and Git workflow. Vishruth and Arushuuu own the ML pipeline, data decisions and experiments. shamaa007 and niharika526 independently build camera input, overlays, logging, fixed audio and runtime tooling, with Vishruth reviewing their deliverables. First evidence: senior-owned contracts and dataset audit, junior-owned camera preview and synthetic-record logger. See the team guide for individual deliverables.

The existing Word handbook and design ZIP predate this ownership revision. Use docs/TEAM_START_HERE.md and the current Markdown design for assignments; those older exports are not the current team plan.


## Existing baseline and current checkout

The active local clone on Vishruth's machine is C:/Users/alwan/Documents/driver-drowsiness-detection. Open it in your editor for future development. The Desktop scaffold is an older reference copy.

See [the preserved baseline](legacy/phase1_initial/MIGRATION_NOTE.md) and [migration verification](docs/GIT_MIGRATION.md). The legacy requirements are historical pins, not a tested environment for the revised pipeline. Existing feature and trigger tests can be run explicitly from the legacy directory; the revised test files remain placeholders.
