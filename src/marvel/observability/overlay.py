import math

import cv2
import numpy as np
from marvel.perception.crops import eye_box


def draw_preview(frame, eyes, status, fps, age, metrics=None):
    canvas = frame.image.copy()
    lines = [
        f"frame={frame.sample_id}  processed FPS={fps:.1f}",
        f"status={status}  read age={age:.3f}s  q=quit",
    ]
    for eye in eyes:
        color = (0, 200, 0) if eye.crop_ok else (0, 0, 255)
        box = eye_box(eye.points)
        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, 1)
        for x, y in eye.points:
            if math.isfinite(x) and math.isfinite(y):
                if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                    cv2.circle(canvas, (round(x), round(y)), 2, color, -1)
        state = "VALID" if eye.crop_ok else "INVALID"
        lines.append(f"{eye.side}: {state} {' '.join(eye.reasons)}")
        scores = (metrics or {}).get(eye.side, {})
        if scores:
            lines.append(f"  mean={scores['brightness']:.1f} sharpness={scores['sharpness']:.1f}")
    if not eyes and status != "camera_only":
        lines.append("left: INVALID   right: INVALID")

    for index, text in enumerate(lines):
        position = (10, 25 + 25 * index)
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3)
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
    return canvas


def eye_panels(eyes):
    """Pure panel builder so clearing stale crops is covered by tests."""
    panels = {}
    lookup = {eye.side: eye for eye in eyes}
    for side in ("left", "right"):
        panel = np.zeros((160, 320, 3), dtype=np.uint8)
        eye = lookup.get(side)
        if eye is not None and eye.crop_ok and eye.crop is not None:
            height, width = eye.crop.shape[:2]
            scale = min(320 / width, 160 / height)
            small = cv2.resize(eye.crop, (max(1, int(width * scale)),
                                         max(1, int(height * scale))))
            h, w = small.shape[:2]
            y, x = (160 - h) // 2, (320 - w) // 2
            panel[y:y+h, x:x+w] = small
        else:
            cv2.putText(panel, "UNAVAILABLE", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        panels[side] = panel
    return panels


def show_eyes(eyes):
    for side, panel in eye_panels(eyes).items():
        cv2.imshow(f"Subject {side} eye", panel)
