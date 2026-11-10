"""Human-review persistence and independent-label provenance boundaries."""
import csv
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest
from marvel.training.dataset import read_manifest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('review_camera_session', ROOT / 'scripts/review_camera_session.py')
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)
spec2 = importlib.util.spec_from_file_location('record_camera_session', ROOT / 'scripts/record_camera_session.py')
recorder = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(recorder)


def make_queue(tmp_path):
    rows = []
    for i in range(2):
        name = f'frame{i}.png'
        Image.new('RGB', (20, 20), (50*i, 20, 10)).save(tmp_path / name)
        rows.extend(recorder.review_rows('actual_participant', 'actual_session', i, i*.5, name))
    path = tmp_path / 'review.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    return path


def test_labels_require_explicit_choices_orientation_and_complete_review(tmp_path):
    session = reviewer.ReviewSession(make_queue(tmp_path), 'test_human', 'calibration')
    with pytest.raises(ValueError, match='every frame'):
        session.export()
    with pytest.raises(ValueError, match='orientation'):
        session.save(session.samples[0], 'open', 'open', False)
    with pytest.raises(ValueError, match='Choose a state'):
        session.save(session.samples[0], 'open', '', True)
    assert not session.progress_path.exists()
    session.save(session.samples[0], 'open', 'open', True)
    with pytest.raises(ValueError, match='every frame'):
        session.export()
    session.save(session.samples[1], 'closed', 'uncertain', True)
    rows = read_manifest(session.export())
    assert [(r['label'], r['observable']) for r in rows] == [('0','1'),('0','1'),('1','1'),('','0')]
    assert {r['split'] for r in rows} == {'calibration'}
    assert {r['label_source'] for r in rows} == {'human'}
    assert all(r['reviewer_id'] == 'test_human' for r in rows)
    with pytest.raises(FileExistsError):
        session.export()


def test_resume_preserves_choices_and_rejects_changed_provenance(tmp_path):
    queue = make_queue(tmp_path)
    session = reviewer.ReviewSession(queue, 'test_human', 'calibration')
    session.save(session.samples[0], 'open', 'closed', True)
    resumed = reviewer.ReviewSession(queue, 'test_human', 'calibration')
    assert resumed.answers == session.answers and resumed.orientation_verified
    for person, split in [('someone_else','calibration'),('test_human','test')]:
        with pytest.raises(ValueError, match='another queue'):
            reviewer.ReviewSession(queue, person, split)
    queue.write_text(queue.read_text()+'\n')
    with pytest.raises(ValueError, match='another queue'):
        reviewer.ReviewSession(queue, 'test_human', 'calibration')


def test_revisions_are_logged_without_editing_source_queue(tmp_path):
    queue = make_queue(tmp_path)
    original = queue.read_bytes()
    session = reviewer.ReviewSession(queue, 'test_human', 'calibration')
    session.save(session.samples[0], 'open', 'open', True)
    session.save(session.samples[0], 'closed', 'uncertain', True)
    assert queue.read_bytes() == original
    actions = [json.loads(line) for line in session.journal_path.read_text().splitlines()]
    assert [r['left'] for r in actions] == ['open', 'closed']
    assert session.answers[session.samples[0]]['right'] == 'uncertain'
