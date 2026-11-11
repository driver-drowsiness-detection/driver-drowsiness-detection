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
