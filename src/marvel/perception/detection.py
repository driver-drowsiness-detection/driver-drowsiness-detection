import math
from pathlib import Path

from marvel.contracts import Eye
from marvel.perception.crops import eye_crop
from marvel.perception.quality import frame_problem, points_ok, image_quality, validate_quality


EYES = {
    "left": (362, 385, 387, 263, 373, 380),
    "right": (33, 160, 158, 133, 153, 144),
}


class MediaPipeBackend:
    """One IMAGE-mode model instance, loaded once and explicitly closed."""

    def __init__(self, model_path):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Missing landmark model: {model_path}")
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
        )
        self.model = vision.FaceLandmarker.create_from_options(options)

    def detect_bgr(self, image_bgr):
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        return self.model.detect(image).face_landmarks

    def close(self):
        self.model.close()


class Detector:
    """Return (eyes, face_status, metrics) using the reference selection rule.

    The optional backend makes selection and quality behavior testable without
    a camera. Production uses MediaPipeBackend, never synthetic landmarks.
    """

    def __init__(self, model_path, roi, quality, *, backend=None):
        validate_quality(quality)
        if (len(roi) != 4 or not all(math.isfinite(v) for v in roi)
                or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)):
            raise ValueError("driver_roi must be normalized [left, top, right, bottom]")
        self.backend = backend if backend is not None else MediaPipeBackend(model_path)
        self.roi = roi
        self.quality = quality
        self.epoch = 0
        self.previous = None
        self.previous_frame = None

    def invalidate(self):
        self.previous = None

    def process(self, frame):
        problem = frame_problem(frame, self.previous_frame)
        if problem:
            self.invalidate()
            return [], problem, {}
        self.previous_frame = frame
        faces = self.backend.detect_bgr(frame.image)
        if len(faces) != 1:
            self.invalidate()
            reason = "no_face" if len(faces) == 0 else "multiple_faces"
            return [], reason, {}

        landmarks = faces[0]
        if len(landmarks) < 468 or not all(
                math.isfinite(p.x) and math.isfinite(p.y) for p in landmarks):
            self.invalidate()
            return [], "invalid_landmarks", {}
        cx = sum(point.x for point in landmarks) / len(landmarks)
        cy = sum(point.y for point in landmarks) / len(landmarks)
        x1, y1, x2, y2 = self.roi
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            self.invalidate()
            return [], "outside_driver_roi", {}

        if self.previous is None or math.dist((cx, cy), self.previous) > 0.12:
            self.epoch += 1
        self.previous = (cx, cy)
        track = f"track_{self.epoch}"
        height, width = frame.image.shape[:2]
        eyes = []
        metrics = {}

        for side, indices in EYES.items():
            points = tuple(
                (landmarks[i].x * width, landmarks[i].y * height)
                for i in indices
            )
            geometry = points_ok(points, width, height)
            crop = eye_crop(frame.image, points) if geometry else None
            valid, reasons, scores = image_quality(crop, self.quality)
            if not geometry:
                reasons = ("invalid_points",) + reasons
            metrics[side] = scores
            eyes.append(Eye(
                sample_id=frame.sample_id,
                t=frame.t,
                track=track,
                side=side,
                points=points,
                crop=crop if valid else None,
                geometry_ok=geometry and valid,
                crop_ok=valid,
                reasons=reasons,
            ))

        status = "valid" if all(eye.crop_ok for eye in eyes) else "invalid_eye"
        return eyes, status, metrics

    def close(self):
        self.backend.close()
