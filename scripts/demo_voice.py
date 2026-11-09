"""Generate a local LLM warning and offline WAV from a labelled synthetic event."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from marvel.reasoning.local_model import LocalGenerator
from marvel.actuation.speech import render_speech


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/llm/SmolLM2-360M-Instruct")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    generator = LocalGenerator(args.model)
    load_s = time.perf_counter() - started
    event = dict(event_id="synthetic_voice:1", episode_id="synthetic_voice:1",
                 kind="PROLONGED_CLOSURE", closure_s=1.5)
    first = generator.generate(event)
    warm = generator.generate(event)
    speech = render_speech(first["text"], args.output / "warning.wav")
    report = dict(synthetic=True, event=event, load_s=load_s, first=first, warm=warm, **speech)
    (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
