# 11. Jetson Nano, export and target validation

## Requirement

Move the validated Windows/reference behavior to the actual board and measure it. Do not equate portability of Python logic with compatibility of binary packages.

The recorded target is the original Jetson Nano 4 GB. NVIDIA's JetPack 4.6.6 documentation belongs to its older software generation. The actual installed JetPack, Python, OpenCV and camera stack must be inspected on the board. A Windows environment lock is not a Jetson installation recipe.

## First board session: inspect without changing packages

Run on Jetson:

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
cat /etc/os-release
uname -m
python3 -V
free -h
gst-inspect-1.0 nvarguscamerasrc
python3 -c "import cv2; print(cv2.__version__); print(cv2.__file__); print(cv2.getBuildInformation())"
```

Save the full outputs. Check OpenCV's GStreamer support. A virtual environment cannot fix an incompatible CUDA stack. Do not replace system OpenCV with a generic wheel merely because `pip` can download it.

The reference package declares Python 3.9 or newer; the team development interpreter remains 3.11. The original Nano may have an older system interpreter. Keep that mismatch visible. Options are a validated compatible interpreter/runtime, an appropriately tested container/OS route, or a target-specific adapter/port. None is chosen automatically here.

## Camera

The existing project history recorded working Argus camera preview, but this does not verify today's board state. Retest with the actual camera and display. Pi NoIR V2 uses the Jetson CSI/Argus route, not Raspberry Pi-specific camera software.

Candidate application pipeline after confirming supported sensor mode and orientation:

```text
nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true sync=false
```

This is a configuration candidate, not a claim that this exact mode works on your installed stack. Confirm source caps first. Set capture `kind: gstreamer` and `source` to the validated pipeline. Preserve aspect ratio: the proposed 1280 by 720 to 640 by 360 resize does so. A 640 by 480 output would distort geometry unless you implement a documented crop/letterbox transformation.

NoIR sensitivity is not illumination. Night operation needs appropriate lighting, optics and camera tests. Do not claim that CLAHE alone establishes nighttime performance.

## Runtime options

| Route | What is supplied | What remains to validate |
| --- | --- | --- |
| Compatible PyTorch | Checkpoint inference adapter | Board-compatible wheel/build and latency |
| OpenCV DNN with ONNX | Export, inference adapter, parity script | Model parser/operators and target performance |
| TensorRT | Procedure and acceptance gate | Exact installed API, conversion and engine adapter |

First prove one landmarker and one eye CNN prediction on saved inputs. A CNN export does not replace the face/landmark stage. The full application still needs a compatible landmarker on Nano.

## Export on the development machine

Install export dependencies in the development environment only after choosing compatible versions. The supplied script uses an explicit legacy PyTorch exporter and opset 11 as a compatibility candidate.

```powershell
python scripts\export_onnx.py models\eye_v1.pt models\eye_v1.onnx
python scripts\check_onnx.py data\prepared_v1.csv models\eye_v1.pt models\eye_v1.onnx
```

The graph checker validates ONNX structure. The parity script compares logits from PyTorch and OpenCV DNN on up to 100 validation eye crops. A successful graph check alone does not establish numerical equivalence.

Metadata links the ONNX file hash, checkpoint hash and preprocessing policy. To use the alternative runtime, set:

```yaml
inference_backend: opencv_onnx
onnx_path: models/eye_v1.onnx
```

Keep the same bundle, temperature and thresholds. Repeat parity on the board using saved reference logits if PyTorch itself is unavailable there. The first parity script expects both runtimes. The following separate target path does not require target PyTorch:

```bash
export PYTHONPATH=src
python3 scripts/score_onnx.py data/prepared_v1.csv models/eye_v1.onnx reports/target_validation_scores.csv
python3 scripts/compare_logits.py reports/validation_scores.csv reports/target_validation_scores.csv
```

Transfer the exact validation crops, manifest, laptop scores and model metadata. Prefer relative dataset paths from the reference root so both machines can resolve them. `compare_logits` matches frame/eye identities, verifies the checkpoint identity and enforces absolute/relative tolerances. This remains a development parity check, not final-test tuning.

## TensorRT is a measured optimization gate

1. Identify the installed TensorRT version and locate its `trtexec` utility.
2. Read that exact utility's help output. Flags and Python execution APIs vary by version.
3. Parse the selected ONNX graph and preserve the parser log.
4. Build the engine on a compatible target stack and record device/runtime identity.
5. Compare logits and threshold decisions with the reference on held-out development inputs.
6. Measure total pipeline throughput, tail latency, memory, power mode and temperature.
7. Retain the optimization only if it improves the project's measured objective without unacceptable prediction changes.

The package does not supply a fake universal TensorRT engine wrapper. The engine's binding API and supported operators depend on the inspected board. ONNX/OpenCV is a supplied alternative execution path while that choice remains undecided.

FP16/INT8 benefit is not guaranteed on the original Nano. Verify actual acceleration support and performance. Quantization also needs representative calibration inputs and parity/accuracy checks. Never tune quantization using final-test labels.

## Profile the whole system

Run a fixed clip or controlled live scene under four conditions: CV alone, CV with fixed audio, CV with LLM, and CV with LLM plus TTS. Keep camera settings and model bundle identical.

Record:

1. Processed observation rate and dropped-frame behavior.
2. Median, p95 and maximum processing/read-to-decision intervals.
3. Sensor-to-warning timing measured separately if claimed.
4. Peak RAM, swap and memory pressure.
5. CPU/GPU load, power mode and temperature from available board tools such as `tegrastats`.
6. Unknown/invalid coverage and event metrics.
7. LLM completion time, fallback rate, TTS start and interruption behavior.

Averages can hide long pauses. The proposed 15 to 20 FPS and 1 to 3 second LLM latency remain targets, not promised outcomes.

## Prepared board handoff

Bring the selected environment plan, complete reference/adapted code, fixed saved images and video, independently labelled intervals, expected laptop outputs, model hashes and exact run command. Use scarce board time for compatibility and measurements, not first-time geometry coding.

Used adapters because board-specific capture/inference can change while contracts and tests remain stable.

Interview explanation: “We treated deployment as a compatibility and measurement problem. Export success was followed by numerical parity and end-to-end profiling on the actual constrained board.”

Retention check: why can an exported ONNX model pass its graph checker and still fail to run correctly in an older TensorRT parser?

<!-- CODE -->

## Complete source files

### File: `scripts/export_onnx.py`

[Open the companion file](reference/scripts/export_onnx.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 7 to 22.

```python
"""python scripts/export_onnx.py models/eye.pt models/eye.onnx"""
import json
import sys
from pathlib import Path


def main():
    import torch
    import onnx
    from marvel.cognition.cnn_inference import CNN, sha256
    from marvel.training.transforms import PREPROCESS
    source, target = sys.argv[1:]
    if Path(target).exists():
        raise FileExistsError(target)
    model = CNN(source).model
    x = torch.zeros((1, 1, 32, 64))
    # Legacy exporter and opset are explicit compatibility candidates, not Jetson certification.
    torch.onnx.export(model, x, target, input_names=["eye"], output_names=["logit"],
                      opset_version=11, dynamo=False)
    onnx.checker.check_model(onnx.load(target))
    Path(target+".json").write_text(json.dumps({"checkpoint_sha256": sha256(source),
        "onnx_sha256": sha256(target), "preprocess": PREPROCESS, "opset": 11}, indent=2))


if __name__ == "__main__":
    main()
```

### File: `scripts/check_onnx.py`

[Open the companion file](reference/scripts/check_onnx.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 5 to 24.

```python
"""python scripts/check_onnx.py prepared.csv models/eye.pt models/eye.onnx"""
import sys


def main():
    import cv2
    import numpy as np
    from marvel.training.dataset import read_manifest, binary_rows
    from marvel.training.transforms import array_from_bgr
    from marvel.cognition.cnn_inference import CNN
    manifest, checkpoint, onnx = sys.argv[1:]
    reference = CNN(checkpoint)
    net = cv2.dnn.readNetFromONNX(onnx)
    errors = []
    for row in binary_rows(read_manifest(manifest), "validation")[:100]:
        crop = cv2.imread(row["crop_path"])
        if crop is None:
            raise ValueError("Missing crop")
        net.setInput(array_from_bgr(crop)[None, ...])
        result = float(net.forward().reshape(-1)[0])
        expected = reference.logit(crop)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
        errors.append(abs(result-expected))
    print({"examples": len(errors), "max_logit_error": max(errors)})


if __name__ == "__main__":
    main()
```

### File: `scripts/score_onnx.py`

[Open the companion file](reference/scripts/score_onnx.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 8 to 30.

```python
"""python scripts/score_onnx.py prepared.csv models/eye.onnx target_validation_scores.csv"""
import csv
import json
import sys
from pathlib import Path


def main():
    import cv2
    from marvel.training.dataset import read_manifest
    from marvel.cognition.cnn_inference import OpenCVCNN, sha256
    source, onnx_path, target = sys.argv[1:]
    metadata = json.loads(Path(onnx_path+".json").read_text())
    model = OpenCVCNN(onnx_path, metadata["checkpoint_sha256"])
    rows = [dict(r) for r in read_manifest(source) if r["split"] == "validation"]
    if not rows:
        raise ValueError("No validation samples")
    for r in rows:
        r["logit"] = ""
        r["checkpoint_sha256"] = metadata["checkpoint_sha256"]
        r["manifest_sha256"] = sha256(source)
        if r["crop_path"]:
            crop = cv2.imread(r["crop_path"])
            if crop is None:
                raise ValueError("Missing crop; preserve relative dataset paths across machines")
            r["logit"] = model.logit(crop)
    with open(target, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
```

### File: `scripts/compare_logits.py`

[Open the companion file](reference/scripts/compare_logits.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `load`: lines 7 to 13.
* `main`: lines 16 to 34.

```python
"""python scripts/compare_logits.py reference_scores.csv target_scores.csv"""
import csv
import math
import sys


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {(r["sample_id"], r["eye_side"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate score identity")
    return result


def main():
    a, b = (load(p) for p in sys.argv[1:])
    if set(a) != set(b):
        raise ValueError("Different sample sets")
    errors = []
    for key in a:
        x, y = a[key], b[key]
        if x["checkpoint_sha256"] != y["checkpoint_sha256"]:
            raise ValueError("Different checkpoint identity")
        if (x["logit"] == "") != (y["logit"] == ""):
            raise ValueError("Different validity")
        if x["logit"] != "":
            xv, yv = float(x["logit"]), float(y["logit"])
            if not math.isclose(xv, yv, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("Logit parity failed at " + str(key))
            errors.append(abs(xv-yv))
    if not errors:
        raise ValueError("No comparable logits")
    print({"matched": len(errors), "max_absolute_logit_error": max(errors)})


if __name__ == "__main__":
    main()
```

