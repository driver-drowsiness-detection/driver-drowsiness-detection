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
