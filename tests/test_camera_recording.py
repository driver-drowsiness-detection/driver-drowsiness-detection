"""Exercise recording timing, independent review queues and failure evidence."""
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from marvel.contracts import Frame

spec = importlib.util.spec_from_file_location('record_camera_session',
    Path(__file__).resolve().parents[1] / 'scripts/record_camera_session.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class FakeCamera:
    def __init__(self):
        self.now = 0.
        self.closed = False
        self.images = []
    def __call__(self, config):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.closed = True
    def read(self, timeout):
        self.now += .25
        image = np.full((80, 640, 3), len(self.images) + 1, dtype=np.uint8)
        self.images.append(image)
        return Frame(f'camera:{len(self.images)*2}', self.now, image)


def test_original_frames_timing_gaps_and_blank_review_queue(tmp_path, monkeypatch):
    camera = FakeCamera()
    monkeypatch.setattr(cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda *a: -1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    output = tmp_path / 'session'
    result = recorder.record(output, 'confirmed_person', seconds=.75,
                             capture_factory=camera, clock=lambda: camera.now, wait_for_start=False)
    assert result['status'] == 'completed' and camera.closed
    assert result['saved_frames'] == 3 and result['saved_time_span_s'] == .5
    frames = list(csv.DictReader((output / 'frames.csv').open()))
    assert [float(row['timestamp_s']) for row in frames] == [0, .25, .5]
    assert [row['capture_sample_id'] for row in frames] == ['camera:2', 'camera:4', 'camera:6']
    for index, row in enumerate(frames):
        saved = cv2.imread(str(output / row['frame_path']))
        assert np.all(saved == index + 1)  # No preview text in saved images.
        assert np.all(camera.images[index] == index + 1)
    review = list(csv.DictReader((output / 'review.csv').open()))
    assert len(review) == 4
    assert {row['eye_side'] for row in review} == {'left', 'right'}
    for row in review:
        assert all(row[key] == '' for key in ('label', 'observable', 'label_source', 'split'))
        assert row['annotation_status'] == 'needs_human_review'
    with pytest.raises(FileExistsError):
        recorder.record(output, 'confirmed_person', capture_factory=camera)


def test_disk_failure_records_partial_status_and_closes_camera(tmp_path, monkeypatch):
    camera = FakeCamera()
    original = cv2.imwrite
    calls = []
    def fails_second(*args):
        calls.append(args[0])
        return original(*args) if len(calls) == 1 else False
    monkeypatch.setattr(cv2, 'imwrite', fails_second)
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='Frame write failed'):
        recorder.record(output, 'confirmed_person', seconds=1, display=False,
                        capture_factory=camera, clock=lambda: camera.now, wait_for_start=False)
    result = json.loads((output / 'recording.json').read_text())
    assert result['status'] == 'failed' and result['saved_frames'] == 1
    assert camera.closed


def test_duplicate_camera_time_is_rejected(tmp_path):
    camera = FakeCamera()
    original = camera.read
    def repeated(timeout):
        frame = original(timeout)
        return Frame(frame.sample_id, .25, frame.image)
    camera.read = repeated
    with pytest.raises(ValueError, match='strictly increasing'):
        recorder.record(tmp_path / 'duplicate', 'confirmed_person', seconds=1,
                        display=False, capture_factory=camera, clock=lambda: camera.now,
                        wait_for_start=False)
    assert camera.closed


def test_preview_does_not_record_until_user_presses_space(tmp_path, monkeypatch):
    camera = FakeCamera()
    keys = iter([-1, 32])
    monkeypatch.setattr(cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda *a: next(keys, -1))
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    output = tmp_path / 'confirmed'
    result = recorder.record(output, 'confirmed_person', seconds=.75,
        capture_factory=camera, clock=lambda: camera.now, wait_for_start=True)
    assert result['start_key_received'] and result['saved_frames'] == 3
    assert np.all(cv2.imread(str(output / 'frames/frame_000000.png')) == 3)


def test_cancel_before_start_saves_no_images(tmp_path, monkeypatch):
    camera = FakeCamera()
    monkeypatch.setattr(cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda *a: ord('q'))
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    output = tmp_path / 'cancelled'
    result = recorder.record(output, 'confirmed_person', capture_factory=camera,
        clock=lambda: camera.now, wait_for_start=True)
    assert result['status'] == 'cancelled_before_start'
    assert result['saved_frames'] == 0 and not list((output / 'frames').iterdir())
    assert camera.closed
