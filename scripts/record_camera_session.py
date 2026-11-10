"""Record original full frames and read-completion times; never infer labels."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from marvel.perception.capture import LiveCapture

PROMPTS = [(0, 'Look at the camera and blink normally'),
           (15, 'Close both eyes for 2 seconds, then reopen'),
           (20, 'Look at the camera and blink normally'),
           (30, 'Close both eyes for 2 seconds, then reopen'),
           (35, 'Look at the camera and blink normally'),
           (45, 'Gently turn your head, then face the camera'),
           (55, 'Look at the camera and blink normally')]


def prompt_at(seconds):
    return next(text for start, text in reversed(PROMPTS) if seconds >= start)


def review_rows(subject, session, index, timestamp, path):
    for side in ('left', 'right'):
        yield dict(sample_id=f'{session}:saved:{index}', subject_id=subject,
                   session_id=session, video_id=session, source='laptop_camera_pilot',
                   split='', frame_path=path, timestamp_s=f'{timestamp:.9f}',
                   eye_side=side, label='', observable='', label_source='',
                   timestamp_method='camera_read_completion_monotonic_relative',
                   annotation_status='needs_human_review',
                   orientation_status='verify_anatomical_sides')


def record(output, subject, *, seconds=60, source=0, display=True,
           capture_factory=LiveCapture, clock=time.monotonic, wait_for_start=True):
    import cv2
    if not subject.strip() or not math.isfinite(seconds) or not 0 < seconds <= 300:
        raise ValueError('A subject ID and duration in (0, 300] seconds are required')
    if wait_for_start and not display:
        raise ValueError('Start confirmation requires a visible preview')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    session = 'camera_' + uuid.uuid4().hex
    metadata = dict(schema=1, session_id=session, subject_id=subject,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    source=source, requested_seconds=seconds,
                    status='waiting_for_start' if wait_for_start else 'recording',
                    start_key_required=wait_for_start, start_key_received=False,
                    timestamp_method='camera_read_completion_monotonic_relative',
                    sensor_exposure_time_measured=False,
                    capture_policy='latest_frame_queue_may_drop_frames',
                    predictions_used=False, labels_assigned=False,
                    stored_images='original_unmirrored_input_without_overlays',
                    prompt_schedule_is_ground_truth=False, prompts=PROMPTS)
    meta_path = output / 'recording.json'
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    count, reviews, next_review, first_t, last_t = 0, 0, 0.0, None, None
    start = last_received = clock()
    armed = not wait_for_start
    status = 'failed'
    try:
        with (output / 'frames.csv').open('x', newline='', encoding='utf-8') as fh, \
                (output / 'review.csv').open('x', newline='', encoding='utf-8') as rh:
            frames = csv.DictWriter(fh, fieldnames=['saved_index', 'capture_sample_id',
                'timestamp_s', 'read_completion_monotonic_s', 'frame_path', 'prompt'])
            frames.writeheader()
            fields = list(next(review_rows(subject, session, 0, 0, '')))
            review = csv.DictWriter(rh, fieldnames=fields)
            review.writeheader()
            with capture_factory(dict(kind='camera', source=source)) as capture:
                while True:
                    now = clock()
                    if not armed and now - start >= 120:
                        status = 'not_started_timeout'
                        break
                    if first_t is not None and now - first_t >= seconds:
                        status = 'completed'
                        break
                    if now - last_received > 10:
                        raise RuntimeError('No camera frame for 10 seconds')
                    frame = capture.read(.1)
                    if frame is not None:
                        last_received = clock()
                        if not armed:
                            preview = frame.image.copy()
                            cv2.putText(preview, 'PREVIEW ONLY: press SPACE to start recording',
                                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 1)
                            cv2.putText(preview, 'No frames saved yet. Q or Esc cancels.',
                                        (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1)
                            cv2.imshow('MARVEL recording', preview)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (27, ord('q')):
                                status = 'cancelled_before_start'
                                break
                            if key == 32:
                                armed = True
                                metadata['start_key_received'] = True
                                print('User pressed SPACE; recording begins with the next frame.', flush=True)
                            continue
                        if not math.isfinite(frame.t) or (last_t is not None and frame.t <= last_t):
                            raise ValueError('Camera timestamps must be finite and strictly increasing')
                        if first_t is None:
                            first_t = frame.t
                        relative = frame.t - first_t
                        if relative >= seconds:
                            status = 'completed'
                            break
                        path = f'frames/frame_{count:06d}.png'
                        if not cv2.imwrite(str(output / path), frame.image,
                                           [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                            raise RuntimeError('Frame write failed')
                        prompt = prompt_at(relative)
                        frames.writerow(dict(saved_index=count, capture_sample_id=frame.sample_id,
                            timestamp_s=f'{relative:.9f}', read_completion_monotonic_s=f'{frame.t:.9f}',
                            frame_path=path, prompt=prompt))
                        fh.flush()
                        if relative >= next_review:
                            review.writerows(review_rows(subject, session, count, relative, path))
                            rh.flush()
                            reviews += 2
                            next_review = relative + .5
                        count += 1
                        last_t = frame.t
                        if display:
                            preview = frame.image.copy()
                            cv2.putText(preview, f'{relative:04.1f}/{seconds:g}s: {prompt}',
                                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1)
                            cv2.putText(preview, 'Recording full frames locally. Q or Esc stops.',
                                        (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                            cv2.imshow('MARVEL recording', preview)
                    if display and cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        status = 'stopped_by_user'
                        break
        if count == 0 and status not in ('not_started_timeout', 'cancelled_before_start', 'stopped_by_user'):
            raise RuntimeError('No frames recorded')
    except BaseException as error:
        status = 'failed'
        metadata['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if display:
            cv2.destroyAllWindows()
        metadata.update(status=status, saved_frames=count, review_rows=reviews,
                        first_frame_monotonic_s=first_t, last_frame_monotonic_s=last_t,
                        saved_time_span_s=None if last_t is None else last_t-first_t,
                        wall_seconds=clock()-start,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--subject-id', required=True)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--source', type=int, default=0)
    args = parser.parse_args()
    record(args.output, args.subject_id, seconds=args.seconds, source=args.source, wait_for_start=True)


if __name__ == '__main__':
    main()
