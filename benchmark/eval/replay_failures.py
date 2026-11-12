#!/usr/bin/env python3
"""Replay cached ACE deterministic failures against the current scorer.

This is a release diagnostic, not a score migration tool.  It never calls a
candidate endpoint or LLM judge.  Judge-rubric rows are reported as skipped;
all other cached responses are scored against the current materialized probe.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.dispatch import load_policy, score_probe


def load_probes(probes_dir: Path) -> dict[str, dict]:
    probes: dict[str, dict] = {}
    for path in sorted(probes_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.lstrip()
            if not stripped or not stripped.startswith("{"):
                continue
            probe = json.loads(stripped)
            probe_id = probe["probe_id"]
            if probe_id in probes:
                raise ValueError(f"duplicate probe_id {probe_id!r}")
            probe["_source"] = f"{path.name}:{line_number}"
            probes[probe_id] = probe
    return probes


def replay(audit: dict, probes: dict[str, dict], policy: dict) -> dict:
    rows = []
    for item in audit["items"]:
        probe_id = item["probe_id"]
        probe = probes.get(probe_id)
        if probe is None:
            raise KeyError(f"current probe missing: {probe_id}")
        for occurrence in item["occurrences"]:
            if occurrence.get("passed"):
                continue
            if occurrence.get("excluded_from_score") or occurrence.get("deferred"):
                continue
            scorer_type = probe.get("scorer", {}).get("type")
            row = {
                "probe_id": probe_id,
                "run": occurrence["run"],
                "category": probe.get("category"),
                "scorer_type": scorer_type,
                "old_strategy": occurrence.get("strategy"),
                "source": probe.get("_source"),
            }
            if scorer_type == "judge_rubric":
                row.update({"status": "judge_skipped", "new_passed": None})
            else:
                result, denominator = score_probe(
                    probe, occurrence.get("response") or "", policy,
                )
                row.update({
                    "status": "replayed",
                    "new_passed": result.passed,
                    "new_strategy": result.strategy,
                    "new_note": result.note,
                    "new_extracted": result.extracted,
                    "denominator": denominator,
                })
            rows.append(row)

    status_counts = Counter(row["status"] for row in rows)
    replayed = [row for row in rows if row["status"] == "replayed"]
    by_run = {}
    for run in sorted({row["run"] for row in rows}):
        run_rows = [row for row in replayed if row["run"] == run]
        by_run[run] = {
            "replayed": len(run_rows),
            "now_pass": sum(row["new_passed"] is True for row in run_rows),
            "still_fail": sum(row["new_passed"] is False for row in run_rows),
        }
    return {
        "schema_version": "ace-cached-failure-replay-v1",
        "warning": (
            "Diagnostic only. Cached responses are not a new official run, "
            "and judge_rubric probes are never replayed locally."
        ),
        "summary": {
            "input_rows": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "replayed_now_pass": sum(
                row["new_passed"] is True for row in replayed
            ),
            "replayed_still_fail": sum(
                row["new_passed"] is False for row in replayed
            ),
            "by_run": by_run,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--probes-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = replay(
        json.loads(args.audit.read_text()),
        load_probes(args.probes_dir),
        load_policy(args.policy),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
