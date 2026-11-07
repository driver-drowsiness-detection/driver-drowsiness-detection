"""Compare recorded estimates without a camera, model execution or audio."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marvel.contracts import Estimate
from marvel.cognition.comparison import Comparison
from marvel.cognition.cnn_inference import sha256
from marvel.observability.logger import JsonLogger

def replay(source, config_path, output, *, synthetic):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    controller = Comparison(**config)
    counts = {mode: 0 for mode in controller.triggers}
    total = common = 0
    with Path(source).open(encoding="utf-8") as inputs, JsonLogger(output) as log:
        log.write(dict(type="run", schema=1, experimental=True, synthetic=synthetic,
                       config=config, input_sha256=sha256(source),
                       config_sha256=sha256(config_path)))
        for line in inputs:
            row = json.loads(line)
            row["ear"] = [Estimate(**e) for e in row["ear"]]
            row["cnn"] = [Estimate(**e) for e in row["cnn"]]
            result = controller.update(**row)
            log.write(dict(type="observation", synthetic=synthetic,
                           source_estimates=json.loads(line), **result))
            total += 1
            common += int(result["common_observable"])
            for mode, record in result["modes"].items():
                counts[mode] += int(record["decision"]["observation"] != "UNKNOWN")
        if not total:
            raise ValueError("Input contains no observations")
        summary = dict(type="summary", observations=total, observable_counts=counts,
                       common_observable_count=common, accuracy_evaluated=False,
                       unresolved_episode_ids={m: t.episode_id for m, t in controller.triggers.items()})
        log.write(summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    provenance = parser.add_mutually_exclusive_group(required=True)
    provenance.add_argument("--synthetic", action="store_true")
    provenance.add_argument("--recorded", action="store_true")
    args = parser.parse_args()
    print(json.dumps(replay(args.source, args.config, args.output, synthetic=args.synthetic)))

if __name__ == "__main__":
    main()
