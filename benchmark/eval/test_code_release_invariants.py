#!/usr/bin/env python3
"""Release-wide ACE code scorer invariants."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))

from eval.dispatch import load_policy, score_probe  # noqa: E402


probes = [
    json.loads(raw)
    for raw in (ACE / "probes" / "code.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if raw.lstrip().startswith("{")
]
policy = load_policy(ACE / "policy" / "atom_intents.yaml")

assert len(probes) == 390
assert not [
    probe["probe_id"]
    for probe in probes
    if probe["scorer"]["type"] == "substring_all"
], "ACE code probes must not use keyword-only substring_all scoring"

failures = []
deterministic_enabled = 0
judge_enabled = 0
disabled = 0
for probe in probes:
    if not probe.get("enabled", True):
        disabled += 1
        result, denominator = score_probe(probe, probe["ground_truth"], policy)
        if denominator != 0 or not result.deferred:
            failures.append((probe["probe_id"], denominator, result.to_dict()))
        continue
    if probe["scorer"]["type"] == "judge_rubric":
        judge_enabled += 1
        continue
    deterministic_enabled += 1
    result, denominator = score_probe(probe, probe["ground_truth"], policy)
    if denominator != 1 or not result.passed:
        failures.append((probe["probe_id"], denominator, result.to_dict()))

assert deterministic_enabled == 374
assert judge_enabled == 1
assert disabled == 15
assert not failures, failures

print(
    "ACE code release invariants: PASS "
    "(374 deterministic GT, 1 Judge contract, 15 excluded security probes, "
    "0 substring_all)"
)
