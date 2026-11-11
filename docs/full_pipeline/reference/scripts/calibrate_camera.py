"""python scripts/calibrate_camera.py checkerboard_images 9 6 25 intrinsics.json"""
import json
import sys
from pathlib import Path


def main():
    import cv2
    import numpy as np
    folder, columns, rows, square_mm, target = sys.argv[1:]
    columns, rows, square_mm = int(columns), int(rows), float(square_mm)
    if columns < 3 or rows < 3 or square_mm <= 0:
        raise ValueError("Specify actual INNER corners and measured square size")
    objects = np.zeros((columns*rows, 3), np.float32)
    objects[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)*square_mm
    object_points, image_points, size = [], [], None
    for p in sorted(Path(folder).glob("*.png")):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Unreadable calibration image")
        current = (gray.shape[1], gray.shape[0])
        if size is not None and current != size:
            raise ValueError("Calibration images must share one resolution")
        size = current
        found, corners = cv2.findChessboardCorners(gray, (columns, rows))
        if found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, .001))
            object_points.append(objects.copy())
            image_points.append(refined)
    if len(image_points) < 10:
        raise ValueError("Need at least 10 varied successful board views; count alone is not quality")
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, size, None, None)
    result = {"image_size": list(size), "camera_matrix": matrix.tolist(),
              "distortion": distortion.ravel().tolist(), "rms_px": rms,
              "views": len(image_points), "square_mm": square_mm,
              "pose_object_points": "undecided", "pose_axis_convention": "undecided"}
    with open(target, "x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
