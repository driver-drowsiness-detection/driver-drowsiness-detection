from threading import Event

from marvel.actuation.audio import FixedAlert


def event(index=1, kind="PROLONGED_CLOSURE"):
    return {"event_id": str(index), "kind": kind}


def test_submit_returns_while_sound_is_blocked_and_no_backlog():
    started, release = Event(), Event()
    def play():
        started.set()
        assert release.wait(timeout=3)
    audio = FixedAlert(play)
    try:
        assert audio.submit(event()) == "sound_requested"
        assert started.wait(timeout=2)
        assert audio.poll() is None
        assert audio.submit(event()) == "duplicate_ignored"
        assert audio.submit(event(2)) == "busy_dropped"
        assert audio.submit(event(3, "MONITORING_UNAVAILABLE")) == "not_a_closure_alert"
    finally:
        release.set()
        audio.close()
    assert audio.poll() == {"event_id": "1", "status": "playback_call_completed"}


def test_playback_failure_is_visible():
    def fail():
        raise RuntimeError("fake output failure")
    audio = FixedAlert(fail)
    audio.submit(event())
    audio.close()
    assert audio.poll()["status"] == "playback_failed"
