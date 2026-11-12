#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.replay_failures import load_probes, replay


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    probes_dir = root / "probes"
    probes_dir.mkdir()
    probe = {
        "probe_id": "math.demo.L1.001",
        "canonical_id": "math.demo",
        "category": "math",
        "ground_truth": "4",
        "scorer": {"type": "exact_match_numeric", "expected": 4},
    }
    (probes_dir / "math.jsonl").write_text(json.dumps(probe) + "\n")
    loaded = load_probes(probes_dir)
    assert set(loaded) == {"math.demo.L1.001"}

    # Probe JSONL files may contain human-readable comment lines.  The replay
    # loader must match the production loader and ignore those records.
    (probes_dir / "commented.jsonl").write_text(
        "# release-note comment\n\n  "
        + json.dumps({**probe, "probe_id": "math.demo.L1.002"})
        + "\n"
    )
    loaded = load_probes(probes_dir)
    assert set(loaded) == {"math.demo.L1.001", "math.demo.L1.002"}
    assert loaded["math.demo.L1.002"]["_source"] == "commented.jsonl:3"

    audit = {
        "items": [{
            "probe_id": "math.demo.L1.001",
            "occurrences": [{
                "run": "demo",
                "passed": False,
                "excluded_from_score": False,
                "deferred": False,
                "strategy": "wrong_number",
                "response": "Answer: 4",
            }],
        }],
    }
    result = replay(audit, loaded, {})
    assert result["summary"]["input_rows"] == 1
    assert result["summary"]["replayed_now_pass"] == 1
    assert result["rows"][0]["new_passed"] is True

print("replay_failures tests: PASS")
