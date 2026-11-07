"""Sample a video into full frames and a blank two-eye annotation queue.

This NEVER assigns eye labels or a dataset split. A human reviewer must fill
the queue, verify anatomical sides/orientation and decide observable status.
"""

import argparse
import csv
import math
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("frames", type=Path, help="New output folder for sampled full frames")
    parser.add_argument("manifest", type=Path, help="New CSV annotation queue")
    parser.add_argument("--subject-id", required=True, help="Verified subject ID; do not invent")
    parser.add_argument("--session-id", required=True, help="Verified recording session ID")
    parser.add_argument("--video-id", required=True, help="Verified globally unique video ID")
    parser.add_argument("--source", required=True, help="Actual source collection name")
    parser.add_argument("--step", type=int, default=15, help="Save every Nth decoded frame")
    parser.add_argument("--limit", type=int, default=300, help="Maximum saved frames")
    args = parser.parse_args(argv)
    if args.step <= 0 or args.limit <= 0:
        parser.error("--step and --limit must be positive")
    if not all(value.strip() for value in (args.subject_id, args.session_id,
                                          args.video_id, args.source)):
        parser.error("Verified identity fields must not be blank")
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.frames.exists() or args.manifest.exists():
        raise FileExistsError("Use new output paths; existing data will not be replaced")

    import cv2
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Cannot open input video: " + str(args.video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS is unavailable; timestamp derivation would be unreliable")
        args.frames.mkdir(parents=True, exist_ok=False)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        frame_index = saved = 0
        while saved < args.limit:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.size == 0:
                raise ValueError(f"Empty decoded frame at index {frame_index}")
            if frame_index % args.step == 0:
                path = args.frames / f"frame_{frame_index:08d}.png"
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError("Could not write frame: " + str(path))
                sample_id = f"{args.video_id}:frame_{frame_index:08d}"
                for side in ("left", "right"):
                    rows.append(dict(sample_id=sample_id, subject_id=args.subject_id,
                                     session_id=args.session_id, video_id=args.video_id,
                                     source=args.source, split="", frame_path=os.path.relpath(
                                         path.resolve(), args.manifest.parent.resolve()),
                                     timestamp_s=f"{frame_index / fps:.6f}", eye_side=side,
                                     label="", observable="", label_source="",
                                     timestamp_method="frame_index_over_reported_fps",
                                     annotation_status="needs_human_review",
                                     orientation_status="verify_anatomical_sides"))
                saved += 1
            frame_index += 1
        if saved == 0:
            raise ValueError("Video decoded no frames")
        with args.manifest.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    finally:
        capture.release()
    print({"saved_frames": saved, "review_rows": len(rows), "reported_fps": fps,
           "timestamp_method": "frame_index_over_reported_fps",
           "status": "requires_human_labels_and_subject_split"})


if __name__ == "__main__":
    main()
