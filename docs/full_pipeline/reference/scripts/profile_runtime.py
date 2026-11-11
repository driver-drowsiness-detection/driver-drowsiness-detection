"""python scripts/profile_runtime.py observations.jsonl"""
import json
import statistics
import sys
from pathlib import Path


def main():
    records = [json.loads(s) for s in Path(sys.argv[1]).read_text().splitlines() if s]
    samples = [r for r in records if r.get("type") == "observation" and "processing_s" in r]
    values = sorted(r["processing_s"] for r in samples)
    if not values:
        raise ValueError("No measured runtime samples")
    def percentile(p):
        return values[min(len(values)-1, int(p*(len(values)-1)))]
    # This includes waiting for capture; label it honestly, not model-only latency.
    report = {"records": len(samples), "loop_median_s": statistics.median(values),
              "loop_p95_s": percentile(0.95), "loop_p99_s": percentile(0.99),
              "loop_max_s": max(values), "loop_throughput_hz": len(values)/sum(values)}
    age = [r["read_to_decision_s"] for r in samples if r.get("read_to_decision_s") is not None]
    if age:
        report["read_to_decision_max_s"] = max(age)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
