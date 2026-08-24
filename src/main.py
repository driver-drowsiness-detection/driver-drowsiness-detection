"""
main.py - wires all four Phase 1 modules together.

Owner: built together as a team (Week 4 integration), same principle as
the buzzer wiring - nobody owns this alone. This minimal version exists
now so anyone can sanity-check the whole chain early once all 4 modules
are ready, instead of only finding out it works in Week 4.
"""

import os
import sys

BASE_DIR = os.path.dirname(__file__)
for module_dir in ("capture", "detection", "features", "trigger", "buzzer"):
    sys.path.append(os.path.join(BASE_DIR, module_dir))

import cv2
from capture import CameraCapture
from detection import FaceLandmarkDetector
from features import compute_features
from trigger import DrowsinessTrigger
from buzzer import trigger_alert


def main():
    cam = CameraCapture(source=0)
    detector = FaceLandmarkDetector()
    trigger = DrowsinessTrigger()

    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                break
            h, w = frame.shape[:2]
            landmarks = detector.detect(frame)

            if landmarks:
                features = compute_features(landmarks, w, h)
                ear = features["ear"]
                alert = trigger.update(ear)
                if alert:
                    trigger_alert()
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                trigger.reset()
                cv2.putText(frame, "No face detected", (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("MARVEL - Phase 1", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
