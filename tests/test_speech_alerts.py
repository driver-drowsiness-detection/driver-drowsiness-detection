from types import SimpleNamespace
from threading import Event
from marvel.actuation.speech_alerts import SpeechAlerts
from marvel.actuation.voice import VoiceOutput
from marvel.reasoning.local_model import FALLBACK, checked_wording


def event():
    return dict(kind="PROLONGED_CLOSURE", event_id="e", episode_id="episode", closure_s=1.5)

def test_fixed_speech_nonblocking_busy_and_stale_discard(tmp_path):
    entered, release = Event(), Event()
    def renderer(text, path):
        assert text == FALLBACK
        entered.set()
        assert release.wait(3)
        return dict(audio_path=str(path))
    output = VoiceOutput(tmp_path, renderer=renderer)
    speech = SpeechAlerts(tmp_path, output=output)
    try:
        assert speech.submit(event()) == "synthesis_submitted"
        assert entered.wait(2)
        assert speech.submit(event()) == "busy_dropped"
    finally:
        release.set()
        speech.close()
    assert speech.poll(None)[0]["status"] == "speech_stale_discarded"

def test_timeout_falls_back_only_for_current_episode(tmp_path):
    submitted = []
    output = SimpleNamespace(submit=lambda r: submitted.append(r) or "queued", poll=lambda e: None, close=lambda: None)
    worker = SimpleNamespace(submit=lambda e: "submitted", poll=lambda e: {"status": "generation_timeout"}, close=lambda: None)
    speech = SpeechAlerts(tmp_path, output=output, worker=worker)
    speech.submit(event())
    speech.poll(None)
    assert not submitted
    speech.submit(event())
    speech.poll("episode")
    assert submitted[0]["text"] == FALLBACK

def test_wording_filter_and_worker_not_ready_fallback(tmp_path):
    assert checked_wording("Continue driving.")[1] == "fixed_fallback"
    submitted = []
    output = SimpleNamespace(submit=lambda r: submitted.append(r) or "queued")
    worker = SimpleNamespace(submit=lambda e: "not_ready")
    assert SpeechAlerts(tmp_path, output=output, worker=worker).submit(event()) == "fallback:queued"
    assert submitted[0]["source"] == "fixed_fallback"
