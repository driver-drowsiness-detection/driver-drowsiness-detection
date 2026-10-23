"""Behavioral checks for temporal evidence, lifecycle and reproducible replay."""

import importlib.util
import json
from pathlib import Path

import pytest

from marvel.cognition.trigger import Trigger
from marvel.observability.logger import JsonLogger


def tracker(mode="ear"):
    return Trigger(1.0, 0.5, 0.5, 1.0, mode=mode, run_id="test")


def update(controller, t, state, track="driver"):
    return controller.update(t, state, track, sample_id=f"s:{t}")


def alarm(controller):
    for t in (0.0, 0.5, 1.0):
        result = update(controller, t, "CLOSED")
    return result.events[0]


def test_blink_cancels_then_continued_closure_emits_once():
    c = tracker()
    update(c, 0.0, "CLOSED")
    assert update(c, 0.25, "OPEN").state == "OPEN"
    results = [update(c, t, "CLOSED") for t in (0.5, 1.0, 1.5, 2.0)]
    assert [e.kind for r in results for e in r.events] == ["PROLONGED_CLOSURE"]
    assert results[2].events[0].t == 1.5
    assert results[2].events[0].closure_s == 1.0


def test_unknown_without_track_preserves_alarm_and_recovery_restarts():
    c = tracker()
    event = alarm(c)
    unknown = update(c, 1.25, "UNKNOWN", None)
    assert unknown.state == "UNKNOWN" and unknown.alarm_active
    assert unknown.episode_id == event.event_id
    assert unknown.track == "driver"
    update(c, 1.5, "OPEN")
    update(c, 1.75, "UNKNOWN", None)
    assert update(c, 2.0, "OPEN").alarm_active
    result = update(c, 2.5, "OPEN")
    assert not result.alarm_active
    assert result.events[0].kind == "RECOVERED"
    assert result.events[0].episode_id == event.event_id
    assert result.events[0].event_id != event.event_id


@pytest.mark.parametrize("breaker", ["unknown", "gap", "track"])
def test_continuity_breaks_require_fresh_closure(breaker):
    c = tracker()
    update(c, 0.0, "CLOSED")
    update(c, 0.5, "CLOSED")
    if breaker == "unknown":
        update(c, 0.75, "UNKNOWN", None)
        t, track = 1.0, "driver"
    elif breaker == "gap":
        t, track = 1.25, "driver"
    else:
        t, track = 1.0, "new_driver"
    result = update(c, t, "CLOSED", track)
    assert result.closure_s == 0.0 and not result.events
    update(c, t + 0.5, "CLOSED", track)
    assert update(c, t + 1.0, "CLOSED", track).alarm_active


def test_long_gap_does_not_clear_alarm_or_count_as_recovery():
    c = tracker()
    alarm(c)
    result = update(c, 10.0, "OPEN")
    assert result.alarm_active and not result.events
    assert update(c, 10.5, "OPEN").events[0].kind == "RECOVERED"


def test_track_change_records_old_episode_as_unresolved():
    c = tracker()
    event = alarm(c)
    result = update(c, 1.25, "CLOSED", "other")
    ended = result.events[0]
    assert ended.kind == "TRACK_ENDED_UNRESOLVED"
    assert ended.track == "driver" and ended.episode_id == event.event_id
    assert result.track == "other" and not result.alarm_active
    assert result.state == "CLOSURE_PENDING" and result.closure_s == 0.0


def test_unavailable_bounded_per_episode_even_when_track_changes():
    c = tracker()
    results = [update(c, t, "UNKNOWN", str(t)) for t in (0, 0.5, 1, 1.5, 2)]
    assert [e.kind for r in results for e in r.events] == ["MONITORING_UNAVAILABLE"]
    update(c, 2.5, "OPEN")
    update(c, 3.0, "UNKNOWN", None)
    update(c, 3.5, "UNKNOWN", None)
    assert update(c, 4.0, "UNKNOWN", None).events[0].kind == "MONITORING_UNAVAILABLE"


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_durations_rejected(index, value):
    durations = [1.0, 0.5, 0.5, 1.0]
    durations[index] = value
    with pytest.raises(ValueError):
        Trigger(*durations, mode="ear", run_id="test")


@pytest.mark.parametrize("t,state,track,sample", [
    (float("nan"), "OPEN", "a", "s"), (float("inf"), "OPEN", "a", "s"),
    (0.0, "OPEN", "a", "s"), (-1.0, "OPEN", "a", "s"),
    (0.5, "BAD", "a", "s"), (0.5, "CLOSED", None, "s"),
    (0.5, "OPEN", "", "s"), (0.5, "OPEN", "a", ""),
])
def test_bad_update_does_not_mutate_controller(t, state, track, sample):
    c = tracker()
    update(c, 0.0, "CLOSED")
    with pytest.raises(ValueError):
        c.update(t, state, track, sample_id=sample)
    assert update(c, 0.5, "CLOSED").closure_s == 0.5


def test_instances_have_independent_histories_and_event_ids():
    ear, cnn = tracker("ear"), tracker("cnn")
    first = alarm(ear)
    assert not update(cnn, 1.0, "CLOSED").alarm_active
    update(cnn, 1.5, "CLOSED")
    second = update(cnn, 2.0, "CLOSED").events[0]
    assert first.event_id != second.event_id


def test_replay_events_and_provenance_are_deterministic(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "replay_synthetic.py"
    spec = importlib.util.spec_from_file_location("replay_synthetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "first.jsonl"
    first = module.replay(output, "repeatable")
    second = module.replay(tmp_path / "second.jsonl", "repeatable")
    assert first == second
    assert [(e["kind"], e["t"]) for e in first] == [
        ("PROLONGED_CLOSURE", 2.5), ("RECOVERED", 3.5), ("MONITORING_UNAVAILABLE", 6.0)]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 82 and all(r["synthetic"] for r in records)
    assert records[0]["config"]["close_s"] == 1.5
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.replay(output, "another")
    assert output.read_bytes() == before


def test_logger_rejects_nonfinite_json_without_partial_line(tmp_path):
    path = tmp_path / "log.jsonl"
    with JsonLogger(path) as log:
        with pytest.raises(ValueError):
            log.write({"value": float("nan")})
        log.write({"value": None})
    assert json.loads(path.read_text()) == {"value": None}
