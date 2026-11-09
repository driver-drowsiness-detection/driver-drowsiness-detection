import json
from pathlib import Path
import runpy
import pytest

summarize = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/profile_runtime.py"))["summarize"]

def test_incomplete_log_cannot_be_reported_as_completed(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps(dict(type="run")))
    with pytest.raises(ValueError, match="completed"): summarize(log)

def test_profile_counts_heartbeats_separately_from_camera_frames(tmp_path):
    rows = [dict(type="run", synthetic=True, source_kind="camera"),
            dict(type="observation", face_status="valid", cognition_ms=2, decision_age_s=.1),
            dict(type="observation", face_status="capture_stall"),
            dict(type="run_end", loop_seconds=2)]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows))
    result = summarize(log)
    assert result["frames"] == 1 and result["processed_frames_per_second"] == .5
    assert result["observations_by_status"]["capture_stall"] == 1
    assert not result["accuracy_evaluated"]
