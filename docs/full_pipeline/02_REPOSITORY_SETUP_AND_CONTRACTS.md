# 02. Repository, environment and shared contracts

## Before you copy anything

This chat's Desktop folder still inherits the Git root `C:/Users/alwan/OneDrive/Desktop`. The dedicated development clone exists at `C:/Users/alwan/Documents/driver-drowsiness-detection`. Both inspected application entry points are scaffolds. Your teammates' newer work may exist on another machine or branch; it was not visible here.

The reference is inside `docs/full_pipeline/reference`. It is deliberately separate from the root application's `src`, `scripts`, `configs` and `tests`. All commands in this handbook assume your terminal is at the **reference root**, unless a command explicitly says “on Jetson.”

To integrate later, compare each file with the current team branch, keep the teammates' working capture behavior, and adapt it to the contracts. Do not replace their files blindly. This package makes no Git commits, pushes or branch changes.

## Concepts you need first

1. A **package** is an importable collection of Python modules. `src/marvel` is our package.
2. A **module** is a Python file. `features.py` holds geometry functions.
3. A **virtual environment** selects one Python interpreter and a project-specific package directory. It is recreated on each machine.
4. A **dependency manifest** tells the package installer what libraries are needed. It is not an application configuration.
5. A **configuration** contains camera selection, thresholds, paths and policies. Changing a YAML file does not install a library.
6. An **artifact** is a produced model, manifest, log or report with an identity. Save its version and hash.
7. A **contract** defines the exact fields and meaning exchanged by modules. A contract prevents “working separately but failing together.”

## Entire reference tree

The generated inventory below lists every supplied code/configuration file. At runtime the application also creates `reports/runs/<run_id>/runtime.yaml`, `observations.jsonl` and, when sound is enabled, `fixed_alert.wav`. Training creates a checkpoint, history JSON, calibration JSON, bundle JSON and optional ONNX artifacts.

<!-- TREE -->

```text
reference/
  configs/
    runtime.yaml
  environments/
    windows-runtime.txt
    windows-training.txt
  pyproject.toml
  README.md
  scripts/
    calibrate_camera.py
    calibrate_scores.py
    check_llm.py
    check_onnx.py
    compare_logits.py
    evaluate.py
    evaluate_events.py
    export_onnx.py
    extract_frames.py
    inspect_environment.py
    prepare_dataset.py
    preview_camera.py
    profile_runtime.py
    replay_synthetic.py
    score_dataset.py
    score_onnx.py
    train_cnn.py
    tune_fusion.py
  src/
    marvel/
      __init__.py
      actuation/
        __init__.py
        audio.py
      cognition/
        __init__.py
        cnn_architecture.py
        cnn_inference.py
        ear_classifier.py
        features.py
        fusion.py
        perclos.py
        pose.py
        state_machine.py
        trigger.py
      contracts.py
      main.py
      observability/
        __init__.py
        logger.py
        overlay.py
      perception/
        __init__.py
        capture.py
        crops.py
        detection.py
        quality.py
      pipeline.py
      reasoning/
        __init__.py
        llm_inference.py
        prompt_builder.py
        worker.py
      training/
        __init__.py
        calibration.py
        dataset.py
        transforms.py
  tests/
    test_core.py
    test_integration.py
    test_optional.py
```

<!-- TREE_END -->

## Directory responsibilities

| Directory | Inputs | Outputs |
| --- | --- | --- |
| `src/marvel/perception` | Camera or images | Frames, ordered points, crops, quality reasons |
| `src/marvel/cognition` | Points/crops and observation times | EAR, CNN, fusion, windows and events |
| `src/marvel/training` | Independently labelled manifests | Audited rows, tensors and calibration support |
| `src/marvel/reasoning` | Small event dictionaries | Validated text or fallback |
| `src/marvel/actuation` | Fixed alarm or text request | Sound/speech status |
| `src/marvel/observability` | Runtime measurements | Overlay and JSONL |
| `scripts` | Explicit command arguments | One repeatable development/evaluation operation |
| `configs` | Team-selected settings | Parsed runtime policy |
| `environments` | Candidate package names | A reproducible environment after validation and locking |
| `tests` | Controlled fixtures | Pass/fail evidence for behavior |
| `data` | Create locally for approved recordings | Raw frames, labels and crops; do not commit personal recordings |
| `models` | Create locally for acquired/trained assets | Task model, checkpoint, bundle and exported model |
| `reports` | Create locally | Logs, experiment reports and environment records |

The reference consolidates standalone `training.yaml` and `experiments.yaml` concerns into explicit script arguments and versioned output JSON. The existing root files are not read. Runtime settings remain in `configs/runtime.yaml`. This avoids telling you that a configuration is used when no code reads it.

## Windows setup

Keep Python 3.14 for other projects if installed. The earlier team workflow selected a separate Python 3.11 environment. Verify that interpreter exists first:

```powershell
py -0p
py -3.11 -V
```

Open the reference folder through your editor terminal, then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V
python -c "import sys; print(sys.executable)"
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
New-Item -ItemType Directory -Force data,models,reports
python -m unittest discover -s tests -v
```

Activation is convenient, not mandatory. If script execution policy blocks activation, invoke `.\.venv\Scripts\python.exe` explicitly. Do not weaken the machine-wide policy merely to activate an environment.

`PYTHONPATH` lets Python locate the `src/marvel` package without installation. Set it again in each new terminal. An alternative after reviewing packaging is `python -m pip install -e .`; use one understandable import workflow.

For the camera and image path:

```powershell
python -m pip install -r environments\windows-runtime.txt
python -c "import cv2; print(cv2.__version__); print(cv2.__file__)"
python scripts\inspect_environment.py > environment.json
```

The dependency files are **candidate lists**, not validated version locks. MediaPipe may bring an OpenCV distribution. If `cv2` is absent, install one suitable OpenCV distribution. Do not stack `opencv-python`, `opencv-contrib-python` and headless variants in the same environment.

For CNN training, install PyTorch using the current [official installation selector](https://pytorch.org/get-started/locally/) for the exact operating system and CPU/CUDA choice. A CPU build is enough to test a tiny forward/backward pass. Do not infer CUDA compatibility merely because the laptop has an NVIDIA sticker.

After the required imports and real smoke tests pass:

```powershell
python -m pip freeze > environment-lock.txt
python scripts\inspect_environment.py > environment.json
```

Store these with the run. Pin versions only after a tested combination is known. On Jetson, follow Guide 11 instead of using this Windows installation sequence.

## Contract walkthrough

`@dataclass(frozen=True)` creates a record whose fields cannot be reassigned. The image array inside can still be mutable; do not modify shared images in place. An overlay takes a copy.

`Frame` contains sample ID, observation time and BGR image. It deliberately does not contain a label.

`Eye` adds anatomical side, continuity track, six ordered points, crop and two validity flags. `geometry_ok` means geometry is usable; `crop_ok` means image input is usable. Shared poor-image checks can invalidate both.

`Estimate` preserves the same frame/time/track/eye identity. `value=None` means missing evidence. A missing EAR is not 0: zero would look like a fully closed valid eye.

`bilateral` requires two matching valid states. A wink, one missing eye or disagreement returns unknown. This conservative policy sacrifices coverage; report that coverage.

Never mix live monotonic seconds, video media seconds and UTC wall time in one tracker. The logger adds UTC for human interpretation. Temporal decisions use observation time.

## Implementation choice and handoff test

Used immutable records because they make accidental identity changes visible. Used explicit missing values because absent data cannot establish closure or recovery.

Before integration, pass one valid bilateral frame, one missing face, one invalid crop, one repeated timestamp, and one changed track through the handoff. Both sides should agree on every field without relying on variable names from their own script.

Interview explanation: “A stable observation contract decouples camera and landmark implementations from the classifiers. Frame identity and timestamp checks prevent mixing predictions from different samples.”

Retention question: why is `geometry_ok=True` compatible with `crop_ok=False` in the contract even if this first detector often gates them together?

<!-- CODE -->

## Complete source files

### File: `pyproject.toml`

[Open the companion file](reference/pyproject.toml). Copy the complete block to this path in your reviewed implementation.

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "marvel-reference"
version = "0.1.0"
description = "Instructional reference for MARVEL, hardware acceptance pending"
requires-python = ">=3.9"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
```

### File: `environments/windows-runtime.txt`

[Open the companion file](reference/environments/windows-runtime.txt). Copy the complete block to this path in your reviewed implementation.

```text
# Candidate requirements, not a verified lock. Resolve on a disposable Python 3.11 venv.
numpy
Pillow
PyYAML
mediapipe
# Inspect the MediaPipe installation first. Add only ONE OpenCV distribution
# if cv2 is missing. Do not install opencv-python alongside opencv-contrib-python.
# CNN/fusion also require the training environment's PyTorch installation.
```

### File: `environments/windows-training.txt`

[Open the companion file](reference/environments/windows-training.txt). Copy the complete block to this path in your reviewed implementation.

```text
-r windows-runtime.txt
# Install torch separately with the CPU/CUDA choice from the official selector.
# Then save exact successful versions in reports/environment-lock.txt.
```

### File: `configs/runtime.yaml`

[Open the companion file](reference/configs/runtime.yaml). Copy the complete block to this path in your reviewed implementation.

```yaml
# Illustrative development settings. These are not validated driving thresholds.
mode: ear
audible_mode: fusion
capture:
  kind: camera
  source: 0
  declared_cfr_fps: null
landmarker: models/face_landmarker.task
driver_roi: [0.0, 0.0, 1.0, 1.0]
quality:
  min_crop_pixels: 8
  brightness_min: 10
  brightness_max: 245
  laplacian_min: 0.0
bundle: null
device: cpu
inference_backend: torch
onnx_path: null
ear_threshold: 0.20
temporal:
  close_s: 1.5
  recover_s: 0.5
  max_gap_s: 0.25
  unavailable_s: 2.0
window:
  window_s: 60.0
  max_gap_s: 0.25
  min_coverage: 0.8
fatigue_policy: null
pose_calibration: null
max_frame_age_s: 0.5
max_run_s: 600
display: true
sound: false
tts: disabled
runs_dir: reports/runs
llm:
  enabled: false
  backend: undecided
  model: null
  endpoint: null
  timeout_s: 3.0
  ttl_s: 5.0
```

### File: `src/marvel/contracts.py`

[Open the companion file](reference/src/marvel/contracts.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `Frame`: lines 9 to 12.
* `Eye`: lines 16 to 25.
* `Estimate`: lines 29 to 45.
* `bilateral`: lines 48 to 51.

```python
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import math

OPEN, CLOSED, UNKNOWN = "OPEN", "CLOSED", "UNKNOWN"


@dataclass(frozen=True)
class Frame:
    sample_id: str
    t: float                 # Observation time, seconds in ONE clock domain.
    image: Any               # OpenCV BGR array. Never send this to the LLM.


@dataclass(frozen=True)
class Eye:
    sample_id: str
    t: float
    track: str               # Continuity epoch, not biometric identity.
    side: str                # Subject's anatomical left or right.
    points: tuple
    crop: Any
    geometry_ok: bool
    crop_ok: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Estimate:
    sample_id: str
    t: float
    track: str
    side: str
    method: str
    value: Optional[float]   # Raw EAR or probability-shaped closure score.
    valid: bool
    reason: str = ""

    def __post_init__(self):
        if self.side not in ("left", "right") or not math.isfinite(self.t):
            raise ValueError("Invalid estimate identity or time")
        if self.valid and (self.value is None or not math.isfinite(self.value)):
            raise ValueError("A valid estimate needs a finite value")
        if not self.valid and self.value is not None:
            raise ValueError("Missing evidence must be None")


def bilateral(left: str, right: str) -> str:
    if left == right and left in (OPEN, CLOSED):
        return left
    return UNKNOWN
```

### File: `scripts/inspect_environment.py`

[Open the companion file](reference/scripts/inspect_environment.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 8 to 24.

```python
import importlib
import json
import platform
import shutil
import sys


def main():
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "machine": platform.machine(), "modules": {}}
    for name in ("numpy", "PIL", "yaml", "cv2", "mediapipe", "torch", "onnx", "onnxruntime"):
        try:
            module = importlib.import_module(name)
            result["modules"][name] = {"version": getattr(module, "__version__", "unknown"),
                                       "file": getattr(module, "__file__", None)}
            if name == "cv2":
                result["opencv_build"] = module.getBuildInformation()
            if name == "torch":
                result["cuda_available"] = module.cuda.is_available()
                result["torch_cuda"] = module.version.cuda
        except Exception as exc:
            result["modules"][name] = {"error": str(exc)}
    result["executables"] = {x: shutil.which(x) for x in ("gst-launch-1.0", "aplay", "espeak-ng", "llama-server")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

### File: `src/marvel/__init__.py`

[Open the companion file](reference/src/marvel/__init__.py). Copy the complete block to this path in your reviewed implementation.

```python
"""MARVEL instructional reference, version 0.1."""
```

### File: `src/marvel/actuation/__init__.py`

[Open the companion file](reference/src/marvel/actuation/__init__.py). Copy the complete block to this path in your reviewed implementation.

```python
"""Fixed sound and interruptible speech adapters."""
```

### File: `src/marvel/cognition/__init__.py`

[Open the companion file](reference/src/marvel/cognition/__init__.py). Copy the complete block to this path in your reviewed implementation.

```python
"""Geometry, estimates, combination and time."""
```

### File: `src/marvel/observability/__init__.py`

[Open the companion file](reference/src/marvel/observability/__init__.py). Copy the complete block to this path in your reviewed implementation.

```python
"""Measurement records and overlays."""
```

### File: `src/marvel/perception/__init__.py`

[Open the companion file](reference/src/marvel/perception/__init__.py). Copy the complete block to this path in your reviewed implementation.

```python
"""Camera adapters and landmark observations."""
```

### File: `src/marvel/reasoning/__init__.py`

[Open the companion file](reference/src/marvel/reasoning/__init__.py). Copy the complete block to this path in your reviewed implementation.

```python
"""Optional local telemetry-to-speech path."""
```

### File: `src/marvel/training/__init__.py`

[Open the companion file](reference/src/marvel/training/__init__.py). Copy the complete block to this path in your reviewed implementation.

```python
"""Offline data, training and frozen-model scoring."""
```

