"""
Module: capture
Owner: Person 1

PURPOSE
    Abstracts "where frames come from" so nobody else on the team needs to
    know or care whether we're reading from a laptop webcam (now) or a
    Jetson CSI camera via GStreamer (later). Only this file changes when
    we port to the Jetson - everyone else's code never touches a camera
    API directly.

TERMINOLOGY (for your notes)
    - Frame: one still image pulled from the video stream. In OpenCV it's
      a NumPy array of shape (height, width, 3), channel order BGR.
    - FPS (Frames Per Second): how many frames you process per second.
      The single most important real-time performance number in this
      whole project - benchmark it here first, before anything else.
    - Preprocessing: any transform applied to a frame before it goes to
      face detection (resize, contrast correction, denoising, etc).
      Keeping this separate from detection means detection.py never
      has to know HOW a frame was prepared, only that it's ready.

WHY THIS MODULE EXISTS ON ITS OWN
    If capture logic were scattered across every file that needs a frame,
    porting to the Jetson later would mean hunting through the whole
    codebase. Isolating it here means the Jetson port is a one-file change.

INTERFACE CONTRACT (do not change the shape of this without telling the team)
    CameraCapture(source=0, width=640, height=480)
        .read_frame() -> np.ndarray | None   # BGR frame, or None on failure
        .release()                            # always call on shutdown
"""

import cv2
import time


class CameraCapture:
    def __init__(self, source=0, width=640, height=480):
        self.cap = cv2.VideoCapture(source)
        self.width = width
        self.height = height
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")

    def read_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.resize(frame, (self.width, self.height))
        # TODO (Week 2, Person 1): add CLAHE contrast correction here for
        # low-light / night-driving robustness - see guide section 1.1.9.
        return frame

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    # Standalone demo: proves the capture module works completely on its
    # own, with zero dependency on detection/features/trigger.
    cam = CameraCapture(source=0)
    prev_time = time.time()
    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                break
            now = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Capture module demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
