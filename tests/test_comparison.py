from dataclasses import replace
import pytest
from marvel.contracts import Estimate, OPEN, CLOSED, UNKNOWN
from marvel.cognition.comparison import Comparison

def config():
    return dict(run_id="synthetic_test", ear_threshold=.2,
                ear_calibration={"slope": -20, "intercept": 4},
                cnn_threshold=.5, fusion_threshold=.5, alpha=.5,
                close_s=1, recover_s=.5, max_gap_s=.6, unavailable_s=1,
                window_s=1, min_coverage=.5)

def pair(method, value, t):
    return [Estimate(str(t), t, "driver", side, method, value, value is not None)
            for side in ("left", "right")]

def update(controller, t, ear=.1, cnn=.9):
    return controller.update(sample_id=str(t), t=t, track="driver",
                             ear=pair("ear", ear, t), cnn=pair("cnn", cnn, t))

def test_histories_are_independent_and_event_ids_are_unique():
    c = Comparison(**config())
    for t in (0, .5, 1): result = update(c, t, ear=.1, cnn=.1)
    assert result["modes"]["ear"]["decision"]["alarm_active"]
    assert result["modes"]["cnn"]["decision"]["state"] == OPEN
    assert not result["modes"]["fusion"]["decision"]["alarm_active"]
    c = Comparison(**config())
    for t in (0, .5, 1): result = update(c, t)
    ids = [m["decision"]["events"][0]["event_id"] for m in result["modes"].values()]
    assert len(set(ids)) == 3

@pytest.mark.parametrize("damage", ["missing", "duplicate", "sample", "time", "track", "side"])
def test_bad_cnn_alignment_preserves_ear(damage):
    c = Comparison(**config())
    cnn = pair("cnn", .9, 0)
    if damage == "missing": cnn = []
    elif damage == "duplicate": cnn.append(cnn[0])
    else:
        field, value = {"sample": ("sample_id", "old"), "time": ("t", -1),
                        "track": ("track", "other"), "side": ("side", "right")}[damage]
        cnn[0] = replace(cnn[0], **{field: value})
    result = c.update(sample_id="0", t=0, track="driver", ear=pair("ear", .1, 0), cnn=cnn)
    assert result["modes"]["ear"]["decision"]["observation"] == CLOSED
    for mode in ("cnn", "fusion"):
        assert result["modes"][mode]["decision"]["observation"] == UNKNOWN
    assert not result["common_observable"]

def test_missing_ear_does_not_remove_cnn_baseline():
    result = update(Comparison(**config()), 0, ear=None)
    assert result["modes"]["cnn"]["decision"]["observation"] == CLOSED
    assert result["modes"]["fusion"]["decision"]["observation"] == UNKNOWN

def test_invalid_input_does_not_advance_any_history():
    c = Comparison(**config())
    with pytest.raises(ValueError): update(c, 0, cnn=2)
    assert c.last is None and all(t.last is None for t in c.triggers.values())
    update(c, 0)
    with pytest.raises(ValueError): update(c, 0)
    assert c.last == 0

def test_disagreement_breaks_candidate_and_coverage():
    c = Comparison(**config())
    update(c, 0)
    cnn = pair("cnn", .9, .5)
    cnn[1] = replace(cnn[1], value=.1)
    result = c.update(sample_id="0.5", t=.5, track="driver", ear=pair("ear", .1, .5), cnn=cnn)
    assert result["modes"]["cnn"]["decision"]["observation"] == UNKNOWN
    result = update(c, 1)
    assert result["modes"]["ear"]["decision"]["alarm_active"]
    assert not result["modes"]["cnn"]["decision"]["alarm_active"]
    assert result["modes"]["cnn"]["rolling"]["coverage"] == 0

def test_heartbeat_and_track_switch_do_not_fabricate_recovery():
    c = Comparison(**config())
    for t in (0, .5, 1): update(c, t)
    result = c.update(sample_id="lost", t=1.5, track=None, ear=[], cnn=[])
    assert all(m["decision"]["alarm_active"] for m in result["modes"].values())
    result = c.update(sample_id="new", t=2, track="new_driver", ear=[], cnn=[])
    assert all(m["decision"]["events"][0]["kind"] == "TRACK_ENDED_UNRESOLVED"
               for m in result["modes"].values())

def test_long_gap_breaks_all_candidates():
    c = Comparison(**config())
    update(c, 0)
    result = update(c, 2)
    assert all(not m["decision"]["alarm_active"] for m in result["modes"].values())
