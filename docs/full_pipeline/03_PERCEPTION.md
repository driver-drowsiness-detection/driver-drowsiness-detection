# 03. Camera, frames, landmarks, crops and quality

## Requirement and build order

Produce trustworthy observations before interpreting eye state. Begin with camera preview, then landmark overlays, then crops and quality checks. The existing team camera work can be reused by adapting its output.

## Choose the capture approach

| Approach | Mechanism | Appropriate use | Limitation |
| --- | --- | --- | --- |
| Synchronous read | Main loop calls `read()` | Offline video replay | A stalled live read can freeze the whole loop |
| Latest-frame worker | Separate process owns camera; queue holds one frame | Live reference | Drops obsolete frames, adds process/IPC cost |
| Native timestamped multimedia pipeline | GStreamer/Argus capture with validated timestamps | Target optimization | More platform-specific work |

The reference uses synchronous read for replay and a separate live process. It keeps one queued frame, so inference does not consume a growing backlog. This alone does not establish the age of a frame inside a camera driver's own buffer.

## What `capture.py` does

1. `_live_worker` imports OpenCV in its own process, opens the selected camera and reads frames.
2. Each successful read creates a `Frame`. The timestamp is **read completion**, not sensor exposure time. Log this limitation.
3. `publish` attempts a nonblocking queue write. If full, it tries to replace the older frame. Under a race, dropping the new frame is acceptable; unbounded waiting is not.
4. `LiveCapture.read` waits at most 0.1 seconds. Returning `None` means “no new frame yet,” not a recovered driver.
5. The main process can then advance its monitoring-unavailable watchdog.
6. `close` requests cooperative stop, waits, and terminates the dedicated capture process if a native read is stuck.
7. `VideoCapture` uses media timestamps. A missing or nonincreasing timestamp fails clearly. A declared CFR fallback may be used only after checking that the recording really has a constant frame rate.

Frames may be dropped in live mode. Do not drop frames in offline evaluation unless the experiment explicitly measures subsampling. A playback delay does not alter the recorded observation timestamp.

## Run camera preview first

```powershell
python scripts\preview_camera.py configs\runtime.yaml
```

Expected: a window, printed frame IDs, increasing timestamps, and a frame shape. Press `q` to close. If another application owns the camera, close that application and retry. If index 0 is wrong, identify the correct camera index; do not label a failed device read as “no face.”

## Landmarks and coordinate systems

The MediaPipe Tasks adapter uses `FaceLandmarker` in image mode for a simple explicit per-frame implementation. A `.task` model asset is required at `models/face_landmarker.task`; obtain it from the model link in the official Face Landmarker documentation and record its hash. Installing the Python library does not download your chosen asset automatically.

Image mode is easy to reason about but can cost more than video tracking. A later video-mode adapter must handle strictly increasing millisecond timestamps and preserve the same output contract. Do not silently switch APIs and claim identical runtime behavior.

The detector returns normalized x and y coordinates. Convert to pixel space using `x*width` and `y*height` before Euclidean distances. Computing EAR in normalized coordinates distorts the axes when width and height differ.

The six points for each eye must follow corners 0 and 3, vertical pairs 1/5 and 2/4. The supplied index map is an explicit MediaPipe eyelid mapping. Visually check it on your selected model asset with open eyes, closed eyes and head movement. Array positions alone are not a visual validation.

The adapter rejects multiple visible faces and a face outside `driver_roi`. It starts a new continuity epoch after loss or a large centroid displacement. This heuristic is adequate for a controlled demonstration, not robust identity tracking. No face-recognition model is required.

## Why eye crops use eye width

Suppose a closed eyelid has almost zero vertical separation. Cropping with the eyelid's minimum and maximum y coordinates would create a thin or empty image precisely when the CNN needs it most.

`eye_crop` uses the distance between corners:

* Crop width = 1.6 times eye width.
* Crop height = 0.8 times eye width.
* Crop center = midpoint of the eye corners.

These are a declared reference preprocessing policy, not experimentally optimal constants. Changing them creates a new dataset and model contract. The function rejects out-of-frame crops instead of silently clipping away part of the eye, and `.copy()` detaches the crop from the original image.

## Quality: what the checks do and do not prove

| Check | What it catches | What it does not prove |
| --- | --- | --- |
| Finite in-frame points | NaN, infinity and out-of-frame coordinates | Correct eyelid localization |
| Minimum eye/crop size | Insufficient spatial resolution | Good classification |
| Mean brightness | Extreme under/overexposure | Absence of glare or local saturation |
| Laplacian variance | A simple blur indicator | A universal cross-camera quality score |
| Unique face and driver region | Common passenger ambiguity | Perfect identity continuity |
| Frame age | Delay after read completion | Sensor-to-read delay |

The default `laplacian_min: 0` leaves blur rejection disabled until you measure an operating threshold. The other numerical defaults are illustrative development settings. Do not describe them as validated.

NoIR imagery requires actual target-camera testing. Converting RGB to grayscale does not reproduce infrared reflectance. Record lighting, camera distance, exposure, glasses and head pose when collecting examples.

## Tests to perform before handoff

1. Camera unavailable: explicit error and cleanup.
2. No face: unknown, no fabricated coordinates.
3. One face: each plotted point matches the intended eyelid.
4. Two visible faces: unknown rather than selecting whichever is largest.
5. Eyes closed: crop still has area.
6. Face near the edge: invalid crop, not an unrelated background patch.
7. Cover one eye: inspect whether the current quality gates detect it. If not, record a known limitation and do not claim occlusion robustness.
8. Stalled capture: main loop continues watchdog updates and reports monitoring unavailable.
9. Video replay at different wall-clock speeds: event observation times remain identical.

## Learning notes

Used a latest-frame queue because freshness matters more than processing every obsolete frame in live monitoring. Used media timestamps for replay because CPU speed must not change the meaning of “two seconds of closure.”

Interview explanation: “We separate sensor acquisition from inference, preserve observation timestamps and reject unusable observations before classification. We measure both classification quality and monitoring coverage.”

Retention exercise: draw the BGR image shape, a six-point eye polygon, and the crop rectangle on one saved frame. Explain why resizing the crop does not affect the EAR computed from original-frame points.

<!-- CODE -->

## Complete source files

### File: `src/marvel/perception/capture.py`

[Open the companion file](reference/src/marvel/perception/capture.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `_live_worker`: lines 7 to 37.
* `LiveCapture`: lines 40 to 64.
* `VideoCapture`: lines 67 to 93.

```python
import multiprocessing as mp
import queue
import time
from marvel.contracts import Frame


def _live_worker(config, output, stop):
    import cv2
    source = config["source"]
    api = cv2.CAP_GSTREAMER if config["kind"] == "gstreamer" else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, api)
    def publish(item):
        try:
            output.put_nowait(item)
        except queue.Full:
            try:
                output.get_nowait()
            except queue.Empty:
                pass
            try:
                output.put_nowait(item)
            except queue.Full:
                pass
    try:
        if not cap.isOpened():
            publish(("error", "camera_open_failed"))
            return
        index = 0
        while not stop.is_set():
            ok, image = cap.read()
            if not ok:
                publish(("error", "camera_read_failed"))
                return
            publish(("frame", Frame(str(index), time.monotonic(), image)))
            index += 1
    finally:
        cap.release()


class LiveCapture:
    def __init__(self, config):
        ctx = mp.get_context("spawn")
        self.queue, self.stop = ctx.Queue(maxsize=1), ctx.Event()
        self.process = ctx.Process(target=_live_worker, args=(config, self.queue, self.stop))
        self.process.start()

    def read(self, timeout=0.1):
        try:
            kind, value = self.queue.get(timeout=timeout)
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError("Capture worker exited")
            return None
        if kind == "error":
            raise RuntimeError(value)
        return value

    def close(self):
        self.stop.set()
        self.process.join(1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1)
        self.queue.close()


class VideoCapture:
    def __init__(self, config):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(config["source"])
        if not self.cap.isOpened():
            raise RuntimeError("video_open_failed")
        self.index, self.last = 0, None
        self.fps = config.get("declared_cfr_fps")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("Declared constant frame rate must be positive")

    def read(self):
        import math
        ok, image = self.cap.read()
        if not ok:
            return None
        t = (self.index / self.fps if self.fps else
             self.cap.get(self.cv2.CAP_PROP_POS_MSEC) / 1000.0)
        if not math.isfinite(t) or t < 0 or (self.last is not None and t <= self.last):
            raise ValueError("Bad media timestamps: supply verified CFR fps or re-encode with audited timing")
        packet = Frame(str(self.index), t, image)
        self.index, self.last = self.index + 1, t
        return packet

    def close(self):
        self.cap.release()
```

### File: `src/marvel/perception/quality.py`

[Open the companion file](reference/src/marvel/perception/quality.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `points_ok`: lines 4 to 7.
* `image_quality`: lines 10 to 24.

```python
import math


def points_ok(points, width, height):
    return len(points) == 6 and all(
        math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height
        for x, y in points)


def image_quality(crop, config):
    import cv2
    if crop is None or crop.size == 0:
        return False, ("crop_bounds",)
    if min(crop.shape[:2]) < config["min_crop_pixels"]:
        return False, ("crop_small",)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if not config["brightness_min"] <= mean <= config["brightness_max"]:
        reasons.append("exposure")
    if blur < config["laplacian_min"]:
        reasons.append("blur")
    return not reasons, tuple(reasons)
```

### File: `src/marvel/perception/crops.py`

[Open the companion file](reference/src/marvel/perception/crops.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `eye_crop`: lines 4 to 18. Height follows eye WIDTH, so a closed eyelid does not create a zero-height crop.

```python
import math


def eye_crop(image, points, width_factor=1.6, height_factor=0.8):
    """Height follows eye WIDTH, so a closed eyelid does not create a zero-height crop."""
    if len(points) != 6:
        return None
    w = math.dist(points[0], points[3])
    if not math.isfinite(w) or w < 2:
        return None
    cx = (points[0][0] + points[3][0]) / 2
    cy = (points[0][1] + points[3][1]) / 2
    x1, x2 = math.floor(cx - width_factor*w/2), math.ceil(cx + width_factor*w/2)
    y1, y2 = math.floor(cy - height_factor*w/2), math.ceil(cy + height_factor*w/2)
    h, iw = image.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > iw or y2 > h or x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()
```

### File: `src/marvel/perception/detection.py`

[Open the companion file](reference/src/marvel/perception/detection.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `Detector`: lines 12 to 63. Controlled single-driver ROI. Continuity tracking is NOT face recognition.

```python
import math
from marvel.contracts import Eye
from .quality import points_ok, image_quality
from .crops import eye_crop

EYES = {"left": (362, 385, 387, 263, 373, 380),
        "right": (33, 160, 158, 133, 153, 144)}
MOUTH = (78, 81, 311, 308, 402, 178)
POSE = (1, 152, 33, 263, 61, 291)


class Detector:
    """Controlled single-driver ROI. Continuity tracking is NOT face recognition."""
    def __init__(self, model_path, roi, quality):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE, num_faces=2)
        self.model = vision.FaceLandmarker.create_from_options(options)
        self.roi, self.quality = roi, quality
        self.epoch, self.previous = 0, None

    def invalidate(self):
        self.previous = None

    def process(self, frame):
        import cv2
        h, w = frame.image.shape[:2]
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        result = self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        # Reject multiple visible faces rather than silently picking a passenger.
        if len(result.face_landmarks) != 1:
            self.invalidate()
            return [], (), (), "no_unique_face"
        lm = result.face_landmarks[0]
        x = sum(p.x for p in lm) / len(lm)
        y = sum(p.y for p in lm) / len(lm)
        x1, y1, x2, y2 = self.roi
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            self.invalidate()
            return [], (), (), "outside_driver_roi"
        if self.previous is None or math.dist((x, y), self.previous) > 0.12:
            self.epoch += 1
        self.previous = (x, y)
        track = "track_" + str(self.epoch)
        def coordinates(indices):
            return tuple((lm[i].x*w, lm[i].y*h) for i in indices)
        eyes = []
        for side, indices in EYES.items():
            points = coordinates(indices)
            geometry = points_ok(points, w, h)
            crop = eye_crop(frame.image, points) if geometry else None
            crop_valid, reasons = image_quality(crop, self.quality)
            # Shared exposure/blur gates apply to geometry too; shape checks remain separate.
            eyes.append(Eye(frame.sample_id, frame.t, track, side, points, crop,
                            geometry and crop_valid, crop_valid, reasons))
        return eyes, coordinates(MOUTH), coordinates(POSE), ""

    def close(self):
        self.model.close()
```

### File: `scripts/preview_camera.py`

[Open the companion file](reference/scripts/preview_camera.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 6 to 28.

```python
"""python scripts/preview_camera.py configs/runtime.yaml"""
import sys
from pathlib import Path


def main():
    import cv2
    import yaml
    from marvel.perception.capture import LiveCapture, VideoCapture
    config = yaml.safe_load(Path(sys.argv[1]).read_text())["capture"]
    live = config["kind"] != "video"
    capture = LiveCapture(config) if live else VideoCapture(config)
    try:
        while True:
            frame = capture.read()
            if frame is None:
                if not live:
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            cv2.imshow("Camera preview", frame.image)
            print(frame.sample_id, round(frame.t, 3), frame.image.shape)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

