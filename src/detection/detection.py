"""
Module: detection
Owner: Person 2

PURPOSE
    Turns a raw frame into structured face geometry. Everyone downstream
    just calls .detect(frame) and gets landmarks back - nobody else needs
    to touch the MediaPipe API directly.

TERMINOLOGY (for your notes)
    - BlazeFace: the lightweight CNN MediaPipe uses internally to find the
      face bounding box before it looks for landmarks.
    - FaceMesh: MediaPipe's 468-point dense facial landmark model.
    - Normalized coordinates: MediaPipe returns landmark x/y as 0.0-1.0
      FRACTIONS of frame width/height, NOT pixel coordinates. You must
      multiply by frame width/height to get pixels - features.py does
      this conversion, not here.
    - Detection confidence vs tracking confidence: detection confidence
      gates the initial face "lock"; tracking confidence gates whether a
      face stays locked across frames without re-running full detection.
      This distinction is what makes MediaPipe fast enough for real-time
      video instead of re-detecting from scratch every single frame.

WHY THIS MODULE EXISTS ON ITS OWN
    If MediaPipe's API were called from multiple places, an upgrade or
    library-version change would mean editing code everywhere. This way,
    it's one file, and Person 3/4's code never breaks even if this file's
    internals change.

INTERFACE CONTRACT
    FaceLandmarkDetector(max_faces=1, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
        .detect(frame) -> list[landmark] | None
            Returns MediaPipe's raw list of 468 landmarks (each has .x, .y
            in 0.0-1.0 normalized coordinates), or None if no face found.
"""

import cv2
import mediapipe as mp


class FaceLandmarkDetector:
    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe needs RGB, OpenCV gives BGR
        results = self._face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark


if __name__ == "__main__":
    # Standalone demo: overlays landmark dots on the live feed.
    # Depends only on capture.py to get frames - proves detection works
    # in isolation before features.py/trigger.py exist.
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "capture"))
    from capture import CameraCapture

    cam = CameraCapture(source=0)
    detector = FaceLandmarkDetector()
    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                break
            h, w = frame.shape[:2]
            landmarks = detector.detect(frame)
            if landmarks:
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
            else:
                cv2.putText(frame, "No face detected", (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Detection module demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
