# 05. Dataset, preprocessing, CNN training and inference

## Requirement

Train a classifier that estimates whether a visible eye is open or closed from its crop. Preserve independent ground truth and compare it fairly with EAR on matching observations.

## Prerequisites and shapes

| Concept | Meaning in this implementation |
| --- | --- |
| Sample | One labelled eye observation |
| Batch | Several samples processed together |
| Tensor | Numeric array used by the model |
| Shape | Axis lengths; input is `(N, 1, 32, 64)` |
| Convolution | Learn local image patterns with small shared filters |
| ReLU | Replace negative activations with zero |
| Pooling | Reduce spatial resolution |
| Logit | Unbounded model output before sigmoid |
| Loss | A differentiable measure of prediction error |
| Backpropagation | Use derivatives to compute parameter gradients |
| Optimizer | Update weights from those gradients |
| Epoch | One pass through the training loader |
| Checkpoint | Saved weights plus interpretation metadata |
| Domain shift | Deployment images differ from training images |

## Dataset choice remains undecided

The proposal mentions NTHU-DDD, YawDD and UTA-RLDD. Treat these as candidates, not interchangeable eye-label datasets. Before selecting one, inspect its current official access conditions, licence, subjects, cameras and annotation schema. A whole-video drowsiness class is not an eye-open/closed label for every frame.

You need three kinds of evidence:

1. Eye labels to train and evaluate the CNN.
2. Paired full-face frames to compute EAR and crop the same eye.
3. Independently annotated temporal recordings to evaluate events and delay.

An eye-only dataset may help pretraining but cannot establish the full EAR comparison if landmarks cannot be obtained. Four teammates can create a useful debugging collection, but small participant counts do not justify broad generalization claims.

## Annotation procedure

1. Obtain permission for recordings and document storage/access.
2. Record source, subject, session, video, camera, lighting and glasses. Namespace IDs globally, such as `collectionA:person03:session02`.
3. Annotate each eye independently as 0 open, 1 closed, or unobservable. Declare how partly closed eyes are handled. Do not force ambiguous images into a clean binary label.
4. Have a second reviewer inspect a sample and disagreements.
5. Record `label_source` as `human` or `dataset_eye_annotation` only after that is true.
6. Do not create ground truth from the EAR threshold that the CNN is supposed to compete against.
7. Assign subjects to train, calibration, validation and test before deriving crops or augmentations.

If the dataset has too few people to populate all partitions with both classes, stop claiming an independent four-way evaluation. Collect more data or design grouped cross-validation with explicit out-of-fold predictions. The reference intentionally refuses empty or single-class fitting partitions.

## Annotation manifest schema

| Field | Example | Purpose |
| --- | --- | --- |
| `sample_id` | `A:v1:00024` | Same frame identity for both eyes |
| `subject_id` | `A:p1` | Keeps a person out of multiple partitions |
| `session_id` | `A:p1:s1` | Keeps a recording session grouped |
| `video_id` | `A:p1:s1:v1` | Keeps a video grouped |
| `source` | `consented_lab_collection` | Provenance |
| `split` | `train` | One declared partition |
| `frame_path` | `data/frames/v1/24.png` | Existing image path relative to command working directory or absolute |
| `timestamp_s` | `0.8` | Observation time in the recording |
| `eye_side` | `left` | Subject's anatomical side |
| `label` | `1` | Closed; 0 means open |
| `observable` | `1` | Human judgement of visibility |
| `label_source` | `human` | Independent annotation source |

Add camera/lighting/glasses columns if available. The preparation script preserves extra columns. It adds `crop_path`, `ear` and `invalid_reason`.

`observable=0` is a ground-truth annotation, not an algorithm's failure flag. Model quality failures are represented separately by missing crop/measurement and reason. This distinction lets evaluation measure how often the algorithm abstains on eyes humans could assess.

## Extract and prepare

```powershell
python scripts\extract_frames.py data\raw\session01.mp4 data\frames\session01 A:p1:s1:v1 0.2
```

This creates frames and an annotation template. It deliberately writes undecided groups/splits and pending labels. Fill them with actual information. Do not simply mark every frame observable or copy a video-level state into both eyes.

Merge reviewed rows into `data/annotations.csv`. Then:

```powershell
python scripts\prepare_dataset.py data\annotations.csv models\face_landmarker.task data\crops_v1 data\prepared_v1.csv
```

Use a new output folder per preprocessing version. Preserve invalid rows to measure coverage. Inspect saved open/closed crops visually. Inspect split subject sets and class counts **after quality filtering**, not just before.

The preparation defaults match the reference crop policy and default quality settings. If runtime quality, ROI or crop policy changes, update preparation accordingly, version both and repeat validation. The manifest hash identifies content, not all external files; preserve source images and their provenance too.

## Approach comparison for the CNN

| Option | Benefit | Drawback | Use in this guide |
| --- | --- | --- | --- |
| Small custom CNN | Short, inspectable and few operators | Learns features from your data | Complete runnable reference |
| Pretrained MobileNetV3-Small | Can reuse learned visual features | Requires its own preprocessing/export audit | Strong later candidate if data is limited |
| Larger CNN | More capacity | More memory, latency and overfitting risk | Not the initial Nano reference |

The small CNN is a reference implementation choice, not the final experimentally selected model. It is consistent with the earlier learning guide. Keep the alternative comparison in the report if you later change architecture.

## Network walkthrough

1. Input: batch of grayscale eye crops, 32 pixels high and 64 wide, values from 0 to 1.
2. First convolution: 1 channel becomes 16 feature maps. Padding preserves spatial dimensions.
3. ReLU introduces nonlinearity. Max pooling halves height and width.
4. Second convolution: 16 becomes 32 maps, followed by another pooling step.
5. Third convolution: 32 becomes 64 maps.
6. Adaptive average pooling produces one value per map, independent of the remaining spatial size.
7. `flatten(1)` preserves the batch axis and converts each sample to 64 values.
8. Linear head produces one logit per eye.
9. `squeeze(1)` removes only the final singleton output axis, preserving a batch of size one.

The training loss is binary cross entropy with logits. It combines sigmoid and binary cross entropy internally for numerical stability. Do not add a sigmoid to the training head and then feed its output to `BCEWithLogitsLoss`.

\[
L = -\frac{1}{N}\sum_i[y_i\log(p_i)+(1-y_i)\log(1-p_i)],\quad p_i=\sigma(z_i).
\]

## Preprocessing parity

Training loads PNG through Pillow, converts to grayscale, resizes with Pillow bilinear interpolation, adds a channel dimension and divides by 255. Live inference first converts BGR to RGB, then uses the same Pillow procedure.

Why share the implementation? Different colour order, interpolation, image sizes or normalization can change predictions even with identical weights. The optional preprocessing test compares a saved PNG with the live BGR path exactly.

The baseline has no augmentation. Add modest augmentation only after this path is correct and measure validation impact. Augmented descendants stay in their parent's training partition.

## Training loop explained

```powershell
python scripts\train_cnn.py data\prepared_v1.csv models\eye_v1.pt 10 cpu
```

`10` is a development run length. It is not an optimized epoch count. Change `cpu` to `cuda` only after the environment inspection confirms working CUDA.

Each batch follows:

1. Move images and labels to the selected device.
2. Compute logits.
3. Compute the scalar loss against independently labelled eyes.
4. Clear previous gradients with `zero_grad`.
5. Call `backward` to apply the chain rule through the network.
6. Call `step` to update weights.
7. In validation, disable gradient recording and skip optimizer updates.
8. Save the lowest validation-loss checkpoint, including label mapping, preprocessing policy and manifest hash.

The saved checkpoint contains CPU tensors for portability. It does not mean the training used the CPU. The seed improves reproducibility but does not promise bit-for-bit deterministic CUDA behavior across environments.

Class weighting, balanced sampling and augmentation are optional experiments. Changing the sampling distribution can affect score calibration. Refit calibration after changing the trained model.

## Inference and handoff

`CNN` checks the class mapping and preprocessing metadata, loads weights strictly, calls `eval`, and uses `inference_mode`. It divides a logit by positive temperature and applies sigmoid. An invalid crop produces a missing estimate.

The model output is an eye-state score. The temporal controller, not the CNN, decides whether the sampled sequence warrants an event.

## Acceptance checks

1. Train one short smoke run and confirm finite loss.
2. Confirm both labels survive filtering in fitting partitions.
3. Load the saved checkpoint in a fresh process.
4. Verify the output shape and that label 1 means closed.
5. Plot training/validation losses. A decreasing training curve alone does not establish generalization.
6. Inspect mistakes by person, camera, glasses, lighting and pose.
7. Save environment, manifest, weights, run history and preprocessing version.

Used grouped partitions because adjacent frames and paired eyes are strongly related. Used shared preprocessing because training and inference must describe the same input problem.

Interview explanation: “We trained an eye-state CNN with explicit class and preprocessing metadata, kept participants separate across development and test sets, and evaluated its score alongside a geometric baseline.”

Retention exercise: explain what changes during `loss.backward`, what changes during `optimizer.step`, and why neither runs during live inference.

<!-- CODE -->

## Complete source files

### File: `src/marvel/training/dataset.py`

[Open the companion file](reference/src/marvel/training/dataset.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `read_manifest`: lines 9 to 46.
* `binary_rows`: lines 49 to 54.
* `EyeDataset`: lines 57 to 71.

```python
import csv
from pathlib import Path

SPLITS = {"train", "calibration", "validation", "test"}
REQUIRED = {"sample_id", "subject_id", "session_id", "video_id", "source", "split",
            "frame_path", "timestamp_s", "eye_side", "label", "observable", "label_source"}


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError("Manifest missing: " + str(REQUIRED-set(reader.fieldnames or [])))
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    seen, ownership, frame_splits = set(), {}, {}
    for row in rows:
        key = (row["sample_id"], row["eye_side"])
        if key in seen or row["eye_side"] not in ("left", "right"):
            raise ValueError("Duplicate sample/eye or wrong eye side")
        seen.add(key)
        if row["split"] not in SPLITS or row["observable"] not in ("0", "1"):
            raise ValueError("Invalid split or observable flag")
        import math
        t = float(row["timestamp_s"])
        if not math.isfinite(t) or t < 0:
            raise ValueError("Invalid timestamp")
        if row["observable"] == "1" and row["label"] not in ("0", "1"):
            raise ValueError("Observable samples need independent 0/1 eye labels")
        if row["label_source"] not in ("human", "dataset_eye_annotation"):
            raise ValueError("EAR-generated labels are not independent ground truth")
        for field in ("source", "subject_id", "session_id", "video_id"):
            if not row[field]:
                raise ValueError("Missing grouping identity: " + field)
        # IDs must be globally namespaced, e.g. datasetA:person12:session2.
        for field in ("subject_id", "session_id", "video_id"):
            key = (field, row[field])
            if key in ownership and ownership[key] != row["split"]:
                raise ValueError("Group leakage: " + str(key))
            ownership[key] = row["split"]
        frame = str(Path(row["frame_path"]).resolve())
        if frame in frame_splits and frame_splits[frame] != row["split"]:
            raise ValueError("Same frame in different partitions")
        frame_splits[frame] = row["split"]
    return rows


def binary_rows(rows, split, paired=False):
    selected = [r for r in rows if r["split"] == split and r["observable"] == "1"
                and r.get("crop_path") and (not paired or r.get("ear", "") != "")]
    if {r["label"] for r in selected} != {"0", "1"}:
        raise ValueError("Partition needs both labels after quality filtering: " + split)
    return selected


class EyeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        from .transforms import array_from_image
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            x = array_from_image(image)
        return torch.from_numpy(x), torch.tensor(float(row["label"]), dtype=torch.float32)
```

### File: `src/marvel/training/transforms.py`

[Open the companion file](reference/src/marvel/training/transforms.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `array_from_image`: lines 8 to 10.
* `array_from_bgr`: lines 13 to 14.

```python
import numpy as np
from PIL import Image

PREPROCESS = {"width": 64, "height": 32, "channels": 1,
              "scale": "0_to_1", "resize": "PIL_BILINEAR", "crop": "width_1.6_height_0.8"}


def array_from_image(image):
    image = image.convert("L").resize((64, 32), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)[None, :, :] / 255.0


def array_from_bgr(crop):
    return array_from_image(Image.fromarray(crop[:, :, ::-1].copy()))
```

### File: `src/marvel/cognition/cnn_architecture.py`

[Open the companion file](reference/src/marvel/cognition/cnn_architecture.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `EyeCNN`: lines 4 to 14.

```python
from torch import nn


class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.features(x).flatten(1)).squeeze(1)
```

### File: `src/marvel/cognition/cnn_inference.py`

[Open the companion file](reference/src/marvel/cognition/cnn_inference.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `sha256`: lines 7 to 8.
* `CNN`: lines 11 to 36.
* `load_bundle`: lines 39 to 47.
* `OpenCVCNN`: lines 50 to 66.

```python
import json
import hashlib
from pathlib import Path
from marvel.contracts import Estimate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CNN:
    def __init__(self, checkpoint, temperature=1.0, device="cpu"):
        import torch
        from .cnn_architecture import EyeCNN
        from marvel.training.transforms import PREPROCESS
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.torch, self.device, self.temperature = torch, device, temperature
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["class_map"] != {"open": 0, "closed": 1} or saved["preprocess"] != PREPROCESS:
            raise ValueError("Checkpoint preprocessing or labels do not match")
        self.model = EyeCNN().to(device)
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()

    def logit(self, crop):
        from marvel.training.transforms import array_from_bgr
        x = self.torch.from_numpy(array_from_bgr(crop)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return float(self.model(x).item())

    def estimate(self, eye):
        from .features import sigmoid
        value = sigmoid(self.logit(eye.crop)/self.temperature) if eye.crop_ok else None
        return Estimate(eye.sample_id, eye.t, eye.track, eye.side, "cnn", value,
                        value is not None, "" if value is not None else "crop_quality")


def load_bundle(path):
    p = Path(path)
    bundle = json.loads(p.read_text(encoding="utf-8"))
    if bundle["status"] != "frozen" or bundle["schema"] != 1:
        raise ValueError("Bundle is not frozen")
    checkpoint = p.parent / bundle["checkpoint"]
    if sha256(checkpoint) != bundle["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    return bundle, checkpoint


class OpenCVCNN(CNN):
    def __init__(self, onnx_path, checkpoint_hash, temperature=1.0):
        import cv2
        from marvel.training.transforms import PREPROCESS
        metadata = json.loads(Path(str(onnx_path)+".json").read_text())
        if (metadata["checkpoint_sha256"] != checkpoint_hash or
                metadata["onnx_sha256"] != sha256(onnx_path) or metadata["preprocess"] != PREPROCESS):
            raise ValueError("ONNX artifact identity mismatch")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))

    def logit(self, crop):
        from marvel.training.transforms import array_from_bgr
        self.net.setInput(array_from_bgr(crop)[None, ...])
        return float(self.net.forward().reshape(-1)[0])
```

### File: `scripts/extract_frames.py`

[Open the companion file](reference/scripts/extract_frames.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 7 to 43.

```python
"""python scripts/extract_frames.py video.mp4 new_output_folder video_id 0.2"""
import csv
import sys
from pathlib import Path


def main():
    import cv2
    from marvel.perception.capture import VideoCapture
    source, folder, video_id, spacing = sys.argv[1:]
    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError("Sample spacing must be positive")
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=False)
    capture = VideoCapture({"source": source, "declared_cfr_fps": None})
    rows, previous = [], -float("inf")
    try:
        while True:
            frame = capture.read()
            if frame is None:
                break
            if frame.t-previous < spacing:
                continue
            path = out/(frame.sample_id+".png")
            if not cv2.imwrite(str(path), frame.image):
                raise RuntimeError("Frame write failed")
            for side in ("left", "right"):
                rows.append({"sample_id": video_id+":"+frame.sample_id,
                    "subject_id": "undecided", "session_id": "undecided", "video_id": video_id,
                    "source": "undecided", "split": "undecided", "frame_path": str(path),
                    "timestamp_s": frame.t, "eye_side": side, "label": "", "observable": "0",
                    "label_source": "pending_annotation"})
            previous = frame.t
    finally:
        capture.close()
    if not rows:
        raise ValueError("No frames extracted")
    with (out/"annotation_template.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Fill independent labels, visibility and group/split metadata before preparation.")


if __name__ == "__main__":
    main()
```

### File: `scripts/prepare_dataset.py`

[Open the companion file](reference/scripts/prepare_dataset.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 10 to 51.

```python
"""python scripts/prepare_dataset.py annotations.csv detector.task crops prepared.csv"""
import csv
import hashlib
import sys
from pathlib import Path
from marvel.contracts import Frame
from marvel.training.dataset import read_manifest


def main():
    import cv2
    from marvel.perception.detection import Detector
    from marvel.cognition.ear_classifier import measure
    source, task, folder, target = sys.argv[1:]
    rows = read_manifest(source)
    Path(folder).mkdir(parents=True, exist_ok=True)
    quality = {"min_crop_pixels": 8, "brightness_min": 10, "brightness_max": 245,
               "laplacian_min": 0.0}
    detector = Detector(task, [0, 0, 1, 1], quality)
    output, cache = [], {}
    try:
        for row in rows:
            key = (row["sample_id"], row["frame_path"], row["timestamp_s"])
            if key not in cache:
                image = cv2.imread(row["frame_path"])
                if image is None:
                    raise ValueError("Unreadable frame: " + row["frame_path"])
                detector.invalidate()
                eyes, _, _, reason = detector.process(Frame(row["sample_id"], float(row["timestamp_s"]), image))
                cache = {key: ({e.side: e for e in eyes}, reason)}
            eyes, reason = cache[key]
            eye = eyes.get(row["eye_side"])
            result = dict(row, crop_path="", ear="", invalid_reason=reason)
            if eye:
                ear = measure(eye)
                result["ear"] = ear.value if ear.valid else ""
                result["invalid_reason"] = ",".join(eye.reasons)
                if eye.crop_ok:
                    name = hashlib.sha256((row["sample_id"]+":"+eye.side).encode()).hexdigest()+".png"
                    path = Path(folder)/name
                    if not cv2.imwrite(str(path), eye.crop):
                        raise RuntimeError("Crop write failed")
                    result["crop_path"] = str(path)
            output.append(result)
    finally:
        detector.close()
    with open(target, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print({"rows": len(output), "usable_crops": sum(bool(r["crop_path"]) for r in output)})


if __name__ == "__main__":
    main()
```

### File: `scripts/train_cnn.py`

[Open the companion file](reference/scripts/train_cnn.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 8 to 56.

```python
"""python scripts/train_cnn.py prepared.csv models/eye.pt 10 cpu"""
import random
import sys
from pathlib import Path
from marvel.training.dataset import read_manifest, binary_rows, EyeDataset


def main():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from marvel.cognition.cnn_architecture import EyeCNN
    from marvel.cognition.cnn_inference import sha256
    from marvel.training.transforms import PREPROCESS
    manifest, output, epochs, device = sys.argv[1:]
    if int(epochs) <= 0:
        raise ValueError("Epoch count must be positive")
    if Path(output).exists():
        raise FileExistsError(output)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    rows = read_manifest(manifest)
    loaders = {s: DataLoader(EyeDataset(binary_rows(rows, s)), batch_size=32,
                           shuffle=s == "train", num_workers=0) for s in ("train", "validation")}
    model = EyeCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, history = float("inf"), []
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(epochs)):
        metrics = {"epoch": epoch+1}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, count = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.set_grad_enabled(split == "train"):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += loss.item()*len(y)
                count += len(y)
            metrics[split+"_loss"] = total/count
        history.append(metrics)
        print(metrics)
        if metrics["validation_loss"] < best:
            best = metrics["validation_loss"]
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "class_map": {"open": 0, "closed": 1}, "preprocess": PREPROCESS,
                        "manifest_sha256": sha256(manifest), "epoch": epoch+1}, output)
    import json
    Path(output+".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

### File: `scripts/score_dataset.py`

[Open the companion file](reference/scripts/score_dataset.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 7 to 29.

```python
"""python scripts/score_dataset.py prepared.csv models/eye.pt calibration scores.csv"""
import csv
import sys
from marvel.training.dataset import read_manifest


def main():
    import cv2
    from marvel.cognition.cnn_inference import CNN, sha256
    manifest, checkpoint, split, target = sys.argv[1:]
    if split not in ("calibration", "validation", "test"):
        raise ValueError("Never calibrate on training predictions")
    model = CNN(checkpoint)
    rows = [dict(r) for r in read_manifest(manifest) if r["split"] == split]
    if not rows:
        raise ValueError("Empty split")
    for row in rows:
        row["logit"] = ""
        row["checkpoint_sha256"] = sha256(checkpoint)
        row["manifest_sha256"] = sha256(manifest)
        if row["crop_path"]:
            crop = cv2.imread(row["crop_path"])
            if crop is None:
                raise ValueError("Missing prepared crop")
            row["logit"] = model.logit(crop)
    with open(target, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
```

