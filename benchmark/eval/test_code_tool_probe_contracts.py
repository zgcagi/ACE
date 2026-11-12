"""Data-contract regressions for the ACE code/tool probe repairs.

This suite is intentionally offline.  It validates the repaired probe data,
executes the corrected Python reference/tests, and exercises the revised
safe-fixture regex/AST scorers without invoking any model or external service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ACE = Path(__file__).resolve().parent.parent
PROBE_DIR = ACE / "probes"
if str(ACE) not in sys.path:
    sys.path.insert(0, str(ACE))

from eval.dispatch import load_policy, score_probe


POLICY = load_policy(ACE / "policy" / "atom_intents.yaml")


def _load_category(name: str) -> dict[str, dict]:
    probes: dict[str, dict] = {}
    path = PROBE_DIR / f"{name}.jsonl"
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.lstrip().startswith("{"):
            continue
        probe = json.loads(raw)
        probe_id = probe["probe_id"]
        assert probe_id not in probes, (path.name, line_number, probe_id)
        assert isinstance(probe["prompt"], str), (probe_id, type(probe["prompt"]))
        probes[probe_id] = probe
    return probes


CODE = _load_category("code")
TOOL = _load_category("tool")

TOOL_SINGLE_ENVELOPE = {
    "tool.argument_extraction.L1.003",
    "tool.argument_extraction.L1.005",
    "tool.argument_extraction.L2.001",
    "tool.argument_extraction.L2.002",
    "tool.argument_extraction.L2.003",
    "tool.argument_extraction.L2.004",
    "tool.argument_extraction.L2.005",
    "tool.argument_extraction.L3.001",
    "tool.argument_extraction.L3.003",
    "tool.argument_extraction.L3.004",
    "tool.argument_extraction.L3.005",
    "tool.format_robustness.L2.003",
}

TOOL_MULTI_ENVELOPE = {
    "tool.multi_tool_coordination.L1.005",
    "tool.multi_tool_coordination.L2.002",
    "tool.multi_tool_coordination.L2.003",
    "tool.multi_tool_coordination.L3.001",
    "tool.multi_tool_coordination.L3.002",
    "tool.multi_tool_coordination.L3.003",
}

def _ground_truth_json(probe: dict) -> object:
    ground_truth = probe["ground_truth"]
    return json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth


def _assert_envelope(value: object) -> None:
    assert isinstance(value, dict), value
    assert set(value) == {"name", "arguments"}, value
    assert isinstance(value["name"], str) and value["name"], value
    assert isinstance(value["arguments"], dict), value


def _score(probe_id: str, response: str):
    result, denominator = score_probe(CODE[probe_id], response, POLICY)
    assert denominator == 1, probe_id
    return result


def test_tool_call_envelope_is_public_and_canonical() -> None:
    assert len(TOOL_SINGLE_ENVELOPE) == 12
    assert len(TOOL_MULTI_ENVELOPE) == 6

    for probe_id in sorted(TOOL_SINGLE_ENVELOPE):
        probe = TOOL[probe_id]
        prompt = probe["prompt"]
        assert '{"name":"<exact tool name>","arguments":{...}}' in prompt
        assert 'literal field names "name" and "arguments" are required' in prompt
        assert "do not execute the tool or return its result" in prompt
        _assert_envelope(_ground_truth_json(probe))

    for probe_id in sorted(TOOL_MULTI_ENVELOPE):
        probe = TOOL[probe_id]
        prompt = probe["prompt"]
        assert '{"name":"<exact tool name>","arguments":{...}}' in prompt
        assert 'literal field names "name" and "arguments" are required' in prompt
        assert "do not execute the tools or return their results" in prompt
        calls = _ground_truth_json(probe)
        assert isinstance(calls, list) and calls
        for call in calls:
            _assert_envelope(call)


def test_hidden_code_contracts_are_now_declared() -> None:
    helper = CODE["code.minimal_patch.L2.003"]
    assert "must return the three original field strings as a list in order" in helper["prompt"]
    assert "assert _validate('1-2-3') == ['1', '2', '3']" in helper["scorer"]["tests"]

    snapshot = CODE["code.minimal_patch.L3.005"]
    assert "fresh snapshot list" in snapshot["prompt"]
    assert "must not change after later calls" in snapshot["prompt"]

    cargo = CODE["code.native_module_edit.L3.004"]
    assert "literal section header `[dependencies]`" in cargo["prompt"]
    assert cargo["ground_truth"].startswith("[dependencies]\n")


def test_negative_prefix_max_reference_and_tests_agree() -> None:
    probe = CODE["code.algorithmic_optimization.L3.004"]
    assert "ans = float('-inf')" in probe["ground_truth"]
    tests = probe["scorer"]["tests"]
    assert any(test.endswith("== [-2, 0]") for test in tests)
    assert tests[-1].endswith("== [-2, -2]")

    namespace: dict[str, object] = {}
    exec(probe["ground_truth"], namespace)
    for test in tests:
        exec(test, namespace)


def test_bug_localization_fixture_is_coherent_and_unique_by_contract() -> None:
    probe = CODE["code.bug_localization.L2.004"]
    assert "count_vowels('HELLO') returns 0 but should be 2" in probe["prompt"]
    assert "normalize each input character at the comparison site" in probe["prompt"]
    assert "leaving the vowels declaration and all other lines unchanged" in probe["prompt"]
    assert probe["ground_truth"] == "5"
    assert probe["scorer"]["expected"] == 5


def test_command_size_semantics_are_declared() -> None:
    probe = TOOL["tool.command_invocation.L2.003"]
    prompt = probe["prompt"]
    assert "GNU `ls -S`'s non-recursive directory-entry size" in prompt
    assert "do not recursively sum directory contents" in prompt
    assert "normal tie behavior is accepted" in prompt


def test_excluded_security_probes_contain_only_public_metadata() -> None:
    excluded = [p for p in CODE.values() if p["canonical_id"] == "code.adversarial_input_construction"]
    assert len(excluded) == 15
    for probe in excluded:
        assert probe["enabled"] is False
        assert probe["prompt"] == probe["disabled_reason"]
        assert probe["ground_truth"] is None
        assert probe["scorer"] == {"type": "exact_match_string"}
        result, denominator = score_probe(probe, "", POLICY)
        assert result.deferred and denominator == 0


if __name__ == "__main__":
    test_tool_call_envelope_is_public_and_canonical()
    test_hidden_code_contracts_are_now_declared()
    test_negative_prefix_max_reference_and_tests_agree()
    test_bug_localization_fixture_is_coherent_and_unique_by_contract()
    test_command_size_semantics_are_declared()
    test_excluded_security_probes_contain_only_public_metadata()
    print("ACE code/tool probe repairs: PASS")
