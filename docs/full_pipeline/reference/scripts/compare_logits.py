"""python scripts/compare_logits.py reference_scores.csv target_scores.csv"""
import csv
import math
import sys


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {(r["sample_id"], r["eye_side"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate score identity")
    return result


def main():
    a, b = (load(p) for p in sys.argv[1:])
    if set(a) != set(b):
        raise ValueError("Different sample sets")
    errors = []
    for key in a:
        x, y = a[key], b[key]
        if x["checkpoint_sha256"] != y["checkpoint_sha256"]:
            raise ValueError("Different checkpoint identity")
        if (x["logit"] == "") != (y["logit"] == ""):
            raise ValueError("Different validity")
        if x["logit"] != "":
            xv, yv = float(x["logit"]), float(y["logit"])
            if not math.isclose(xv, yv, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("Logit parity failed at " + str(key))
            errors.append(abs(xv-yv))
    if not errors:
        raise ValueError("No comparable logits")
    print({"matched": len(errors), "max_absolute_logit_error": max(errors)})


if __name__ == "__main__":
    main()
