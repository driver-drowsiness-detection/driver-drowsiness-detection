"""Exercise background LLM and speech with one explicitly synthetic event."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marvel.reasoning.worker import VoiceWorker
from marvel.actuation.voice import VoiceOutput
from marvel.observability.logger import JsonLogger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=False)
    worker = VoiceWorker(args.model)
    output = VoiceOutput(args.folder, speak=False)
    event = dict(kind="PROLONGED_CLOSURE", event_id="synthetic:1", episode_id="synthetic:1", closure_s=1.5)
    submitted = False
    started = time.monotonic()
    try:
        with JsonLogger(args.folder / "worker.jsonl") as log:
            log.write(dict(type="run", synthetic=True, audible_playback=False))
            while time.monotonic() - started < 120:
                result = worker.poll(event["episode_id"])
                if result:
                    log.write(dict(type="generation", **result))
                    if result["status"] == "generated":
                        output.submit(result)
                    elif result["status"] not in ("ready",):
                        raise RuntimeError(result)
                if worker.ready and not submitted:
                    submitted = worker.submit(event) == "submitted"
                speech = output.poll(event["episode_id"])
                if speech:
                    log.write(dict(type="speech", **speech))
                    if speech["status"] != "speech_rendered":
                        raise RuntimeError(speech)
                    log.write(dict(type="run_end", elapsed_s=time.monotonic() - started))
                    print(json.dumps(speech))
                    return
                time.sleep(.02)
            raise TimeoutError("Background voice check did not complete")
    finally:
        worker.close()
        output.close()


if __name__ == "__main__": main()
