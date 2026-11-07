import json
from pathlib import Path
import runpy

import pytest
from test_comparison import config, pair
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
replay = runpy.run_path(str(ROOT / "scripts/replay_comparison.py"))["replay"]

def inputs(tmp_path):
    source, settings = tmp_path / "inputs.jsonl", tmp_path / "config.json"
    settings.write_text(json.dumps(config()), encoding="utf-8")
    rows = []
    for t in (0, .5, 1, 1.5, 2):
        rows.append(dict(sample_id=str(t), t=t, track="driver",
                         ear=[asdict(e) for e in pair("ear", .1 if t <= 1 else .3, t)],
                         cnn=[asdict(e) for e in pair("cnn", .9 if t <= 1 else .1, t)]))
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return source, settings

def test_replay_is_deterministic_preserves_provenance_and_refuses_overwrite(tmp_path):
    source, settings = inputs(tmp_path)
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    summary = replay(source, settings, first, synthetic=True)
    replay(source, settings, second, synthetic=True)
    assert first.read_bytes() == second.read_bytes()
    records = [json.loads(line) for line in first.read_text().splitlines()]
    assert records[0]["synthetic"] and len(records[0]["input_sha256"]) == 64
    assert records[1]["source_estimates"]["ear"][0]["value"] == .1
    for mode in ("ear", "cnn", "fusion"):
        events = [e for r in records[1:-1] for e in r["modes"][mode]["decision"]["events"]]
        assert [(e["kind"], e["t"]) for e in events] == [("PROLONGED_CLOSURE", 1), ("RECOVERED", 2)]
    assert summary["common_observable_count"] == 5
    assert not summary["accuracy_evaluated"]
    with pytest.raises(FileExistsError): replay(source, settings, first, synthetic=True)

def test_empty_input_cannot_produce_completed_report(tmp_path):
    source, settings = inputs(tmp_path)
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no observations"):
        replay(source, settings, tmp_path / "empty.jsonl", synthetic=True)
