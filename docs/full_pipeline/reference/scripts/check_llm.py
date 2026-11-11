"""python scripts/check_llm.py configs/runtime.yaml"""
import json
import sys
import time
from pathlib import Path
from marvel.reasoning.worker import SpeechWorker


def main():
    import yaml
    config = yaml.safe_load(Path(sys.argv[1]).read_text())["llm"]
    event = {"event_id": "contract-test:1", "kind": "PROLONGED_CLOSURE", "method": "fusion", "closure_s": 2.0}
    worker = SpeechWorker(config)
    try:
        worker.submit(event)
        while True:
            result = worker.poll(event["event_id"])
            if result:
                print(json.dumps(result, indent=2))
                break
            time.sleep(0.02)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
