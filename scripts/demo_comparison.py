"""Generate an explicitly synthetic comparison run in a new directory."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marvel.contracts import Estimate
from replay_comparison import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=False)
    # Illustrations only: none of these parameters was fitted to real data.
    config = dict(run_id="synthetic_comparison", ear_threshold=.2,
                  ear_calibration={"slope": -20, "intercept": 4},
                  cnn_threshold=.5, fusion_threshold=.5, alpha=.5,
                  close_s=1, recover_s=.5, max_gap_s=.6, unavailable_s=1,
                  window_s=1, min_coverage=.5)
    settings, source = args.folder / "config.json", args.folder / "inputs.jsonl"
    settings.write_text(json.dumps(config, indent=2), encoding="utf-8")
    with source.open("x", encoding="utf-8") as handle:
        for index in range(9):
            t = index / 2
            closed = t <= 1 or t >= 2.5
            values = {"ear": .1 if closed else .3, "cnn": .9 if closed else .1}
            row = dict(sample_id=f"synthetic:{index}", t=t, track="synthetic_driver")
            for method, value in values.items():
                if method == "cnn" and t == 2.5:
                    value = None
                row[method] = [asdict(Estimate(row["sample_id"], t, row["track"], side,
                                              method, value, value is not None,
                                              "induced_missing_crop" if value is None else ""))
                               for side in ("left", "right")]
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(replay(source, settings, args.folder / "comparison.jsonl", synthetic=True)))


if __name__ == "__main__":
    main()
