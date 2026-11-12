"""End-to-end regressions for the hardened deterministic scorer paths.

This test intentionally stays offline: judge_rubric probes are inspected as
data contracts but never sent to an external model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))

from eval.dispatch import (  # noqa: E402
    _simple_yaml_load,
    default_route,
    load_policy,
    resolve_intent,
    score_probe,
)
from eval.scorer import (  # noqa: E402
    code_structural,
    extract_numeric,
    extract_string,
    format_lenient_regex,
    semantic_keywords_all,
)


POLICY_PATH = ACE / "policy/atom_intents.yaml"
POLICY = load_policy(POLICY_PATH)


def load_probes() -> list[dict]:
    probes = []
    for path in sorted((ACE / "probes").glob("*.jsonl")):
        for raw in path.read_text().splitlines():
            if raw.lstrip().startswith("{"):
                probes.append(json.loads(raw))
    return probes


PROBES = load_probes()
ENABLED = [probe for probe in PROBES if probe.get("enabled", True)]
BY_ID = {probe["probe_id"]: probe for probe in PROBES}


def scored(probe: dict, response: str):
    result, denominator = score_probe(probe, response, POLICY)
    assert denominator == 1, (probe["probe_id"], result.to_dict())
    return result


# ---------------------------------------------------------------------------
# YAML fallback: comments, quoted '#', and block lists must survive no-PyYAML.
# ---------------------------------------------------------------------------
fallback_policy = _simple_yaml_load(POLICY_PATH.read_text())
for atom, expected_intent in {
    "arith.numeric_sort": "sequence_match",
    "arith.base_conversion": "extract_string",
    "arith.unit_conversion": "extract_numeric",
    "arith.version_compare": "extract_string",
    "know.brand_canonical_case": "extract_string",
    "arith.exact_integer_op": "exact_integer",
}.items():
    assert fallback_policy["prefixes"][atom]["intent"] == expected_intent
assert fallback_policy["excluded_from_main_score"]["atom_prefixes"] == [
    "multimodal.image_understanding",
    "multimodal.audio_understanding",
    "multimodal.video_understanding",
]
assert _simple_yaml_load('root:\n  value: "kept # literally" # comment\n')["root"]["value"] == "kept # literally"


# ---------------------------------------------------------------------------
# Ordered sequences and exact large integers.
# ---------------------------------------------------------------------------
sequences = [p for p in ENABLED if p.get("scorer", {}).get("type") == "sequence_equal"]
assert len(sequences) == 3
for probe in sequences:
    assert scored(probe, probe["ground_truth"]).passed
    reversed_answer = ", ".join(reversed([x.strip() for x in probe["ground_truth"].split(",")]))
    assert not scored(probe, reversed_answer).passed

sort_probe = BY_ID["arith.numeric_sort.L1.001"]
assert scored(
    sort_probe,
    "Sorted ascending, comma-separated: 28, 35, 61, 73, 89",
).passed
assert not scored(
    sort_probe,
    "Sorted ascending: 0, 28, 35, 61, 73, 89",
).passed

set_probe = BY_ID["plan.subgoal_decomposition.L1.002"]
assert scored(
    set_probe,
    "Required subgoals: print_document; sign_document",
).passed
assert not scored(
    set_probe,
    "Required subgoals: print_document, sign_document, restart_router",
).passed

integers = [p for p in ENABLED if p.get("scorer", {}).get("type") == "exact_integer"]
assert len(integers) == 15
for probe in integers:
    expected = str(probe["scorer"]["expected"])
    assert scored(probe, expected).passed
    assert scored(probe, f"The answer is {expected}").passed
    assert scored(probe, f"Reasoning\n\\boxed{{{expected}}}").passed
    assert not scored(probe, f"{expected}.0").passed
    assert not scored(probe, f"{expected}e0").passed
    assert not scored(probe, f"The answer is not {expected}").passed

# At least one adjacent pair is indistinguishable as a float; text comparison
# must still reject the adjacent wrong integer.
collision_checked = False
for probe in integers:
    expected = str(probe["scorer"]["expected"])
    wrong = str(int(expected) + 1)
    if float(expected) == float(wrong):
        collision_checked = True
        assert not scored(probe, wrong).passed
        break
assert collision_checked


# ---------------------------------------------------------------------------
# Numeric and MCQ negation false-positive protection, across every probe.
# ---------------------------------------------------------------------------
numeric = [p for p in ENABLED if p.get("scorer", {}).get("type") == "exact_match_numeric"]
assert len(numeric) >= 340
for probe in numeric:
    expected = str(probe["scorer"].get("expected", probe.get("ground_truth")))
    assert scored(probe, expected).passed, probe["probe_id"]
    assert scored(probe, f"The answer is {expected}").passed, probe["probe_id"]
    assert not scored(probe, f"The answer is not {expected}").passed, probe["probe_id"]
assert extract_numeric("not 4, answer 5", 5).passed
assert not extract_numeric("答案不是5", 5).passed

mcq = [p for p in ENABLED if p.get("scorer", {}).get("type") == "mcq_letter"]
assert len(mcq) >= 340
for probe in mcq:
    expected = str(probe["scorer"].get("expected", probe.get("ground_truth")))
    assert scored(probe, expected).passed, probe["probe_id"]
    assert scored(probe, f"**Answer:** {expected}").passed, probe["probe_id"]
    assert scored(probe, f"{expected}) selected option").passed, probe["probe_id"]
    assert not scored(probe, f"Not {expected}").passed, probe["probe_id"]
    assert not scored(probe, f"I reject {expected}").passed, probe["probe_id"]

option_list = "A) first\nB) second\nC) third\nD) fourth"
assert not scored(mcq[0], option_list).passed


# ---------------------------------------------------------------------------
# Exact strings: safe non-instr extraction; strict whole-output for instr.
# ---------------------------------------------------------------------------
short_no = {
    "probe_id": "synthetic.no",
    "canonical_id": "synthetic.no",
    "ground_truth": "no",
    "scorer": {"type": "exact_match_string", "expected": "no"},
}
assert default_route(short_no, "no", {}).passed
assert default_route(short_no, "No, because the evidence is insufficient.", {}).passed
assert not default_route(short_no, "None", {}).passed
assert not default_route(short_no, "I don't know", {}).passed
assert not default_route(short_no, "The answer is not no.", {}).passed

short_a = {
    "probe_id": "synthetic.a",
    "canonical_id": "synthetic.a",
    "ground_truth": "A",
    "scorer": {"type": "exact_match_string", "expected": "A"},
}
assert not default_route(short_a, "pass", {}).passed
assert not default_route(short_a, "UNAVAILABLE", {}).passed
assert not default_route(short_a, "A useful plan", {}).passed
assert default_route(short_a, "Answer: A", {}).passed

banana = {
    "probe_id": "synthetic.banana",
    "canonical_id": "synthetic.banana",
    "category": "know",
    "ground_truth": "banana",
    "scorer": {"type": "exact_match_string", "expected": "banana"},
}
assert not default_route(
    banana, "banana may be possible. Answer: apple", {},
).passed
assert not default_route(banana, "The answer is not banana", {}).passed

marigold = BY_ID["instr.system_prompt_adherence.L2.002"]
assert scored(marigold, "Marigold.").passed
assert not scored(marigold, "Marigold").passed
assert not scored(marigold, "Marigold..").passed
assert not scored(marigold, "Answer: Marigold.").passed

# A scorer-level output-only flag is enforced for instr, but ignored for a
# semantic capability probe where the answer can be safely extracted.
numeric_format = {
    "probe_id": "synthetic.numeric",
    "canonical_id": "synthetic.numeric",
    "category": "arith",
    "ground_truth": 42,
    "scorer": {
        "type": "exact_match_numeric",
        "expected": 42,
        "require_only_numeric_output": True,
    },
}
assert default_route(numeric_format, "The answer is 42", {}).passed
numeric_format["category"] = "instr"
assert not default_route(numeric_format, "The answer is 42", {}).passed
assert default_route(numeric_format, "42", {}).passed

strict_mcq = {
    "probe_id": "synthetic.mcq",
    "canonical_id": "synthetic.mcq",
    "category": "instr",
    "ground_truth": "B",
    "scorer": {"type": "mcq_letter", "expected": "B"},
}
assert not default_route(strict_mcq, "Answer: B", {}).passed
assert default_route(strict_mcq, "B", {}).passed

budget = BY_ID["agent.compute_budget_management.L3.005"]
assert scored(budget, "Explanation omitted.\nAnswer: 150").passed
assert not scored(budget, "Answer: 100.5").passed
assert not scored(budget, "150 was considered.\nAnswer: 250").passed

# Non-instr regex probes may remove an explicit answer wrapper, but the full
# extracted payload must still satisfy the original regex.
assignment = BY_ID["logic.constraint_satisfaction.L1.004"]
assert scored(assignment, "Reasoning omitted.\nAnswer: x=9, y=5, z=7").passed
colors = BY_ID["logic.set_operations.L1.004"]
assert not scored(colors, "Answer: orange, blue, green").passed

command = BY_ID["tool.command_invocation.L1.001"]
assert scored(command, "```sh\ncat /etc/hostname\n```").passed
assert not scored(command, "```sh\ncat /etc/hosts\n```").passed

# A policy-level format_lenient override must still route through the same
# non-instr regex answer extraction path.
override_probe = {
    "probe_id": "synthetic.format_override",
    "canonical_id": "synthetic.format_override",
    "category": "dialog",
    "ground_truth": "blue, green",
    "scorer": {
        "type": "regex",
        "pattern": r"^\s*blue\s*,\s*green\s*$",
        "flags": ["i"],
    },
}
override_policy = {"prefixes": {
    "synthetic.format_override": {"intent": "format_lenient"},
}}
override_result, override_denom = score_probe(
    override_probe, "Answer: blue, green", override_policy,
)
assert override_denom == 1 and override_result.passed
override_result, _ = score_probe(
    override_probe, "Answer: orange, blue, green", override_policy,
)
assert not override_result.passed

# One outer JSON fence is presentation scaffolding outside instr; prose around
# the fence and schema-invalid data still fail.
state = BY_ID["plan.entity_state_tracking.L3.001"]
state_json = json.dumps(state["ground_truth"], ensure_ascii=False)
assert scored(state, f"```json\n{state_json}\n```").passed
assert not scored(state, f"Here it is:\n```json\n{state_json}\n```").passed
wrong_state = json.loads(state_json)
wrong_state["A"]["downloaded_mb"] = 601
assert not scored(
    state, f"```json\n{json.dumps(wrong_state)}\n```",
).passed

preserve = BY_ID["instr.preserve_constraint.L2.005"]
assert scored(preserve, preserve["ground_truth"]).passed
assert not scored(preserve, preserve["ground_truth"].rstrip("\n") + "\n").passed
assert not scored(preserve, preserve["ground_truth"] + "\n").passed

punctuation_opt_in = [
    p for p in ENABLED
    if p.get("scorer", {}).get("type") == "exact_match_string"
    and p["scorer"].get("allow_terminal_punctuation") is True
]
assert len(punctuation_opt_in) == 119
for probe in punctuation_opt_in:
    expected = str(probe["scorer"].get("expected", probe.get("ground_truth")))
    assert scored(probe, expected + ".").passed, probe["probe_id"]


# ---------------------------------------------------------------------------
# Structural code checks only count calls/use-sites, not assignments/comments.
# ---------------------------------------------------------------------------
assert not code_structural("x = np.zeros", must_call_any=["np.zeros"]).passed
assert code_structural("x = np.zeros(3)", must_call_any=["np.zeros"]).passed
assert not code_structural("x = 1", must_use_any=["x"]).passed
assert code_structural("x = 1\nprint(x)", must_use_any=["x"]).passed
assert semantic_keywords_all(
    "model.fit(X, y)", required=[".fit(X, y)"], case_sensitive=True,
).passed
assert not semantic_keywords_all(
    "alpha beta", required=["alpha", "beta", "gamma"],
).passed

# ---------------------------------------------------------------------------
# Format regex scope: anchors apply to the whole response unless flags say so.
# ---------------------------------------------------------------------------
for probe_id in ("instr.confirmation_gate.L3.002", "instr.confirmation_gate.L3.004"):
    probe = BY_ID[probe_id]
    assert scored(probe, probe["ground_truth"]).passed
    assert not scored(probe, "INVALID\n" + probe["ground_truth"]).passed
    assert not scored(probe, probe["ground_truth"] + "\nINVALID").passed

non_latin_ids = (
    "lang.non_latin_script_robustness.L2.001",
    "lang.non_latin_script_robustness.L2.003",
    "lang.non_latin_script_robustness.L2.004",
    "lang.non_latin_script_robustness.L3.001",
    "lang.non_latin_script_robustness.L3.003",
    "lang.non_latin_script_robustness.L3.004",
    "lang.non_latin_script_robustness.L3.005",
)
for probe_id in non_latin_ids:
    probe = BY_ID[probe_id]
    assert scored(probe, probe["ground_truth"]).passed
    assert not scored(probe, probe["ground_truth"] + "\nLatin").passed

seed = BY_ID["world.plausible_seed_data.L3.005"]
assert scored(seed, seed["ground_truth"]).passed
assert not scored(seed, seed["ground_truth"].replace("Severity: High\n", "")).passed


# ---------------------------------------------------------------------------
# Output-category answers use safe extraction; JSON still validates one value.
# ---------------------------------------------------------------------------
strict_output = [
    p for p in ENABLED
    if p.get("canonical_id") in {"output.no_extra_text", "output.unit_correct_final"}
]
assert len(strict_output) == 30
for probe in strict_output:
    answer = str(probe["ground_truth"])
    assert scored(probe, answer).passed
    wrapped = scored(probe, "Answer: " + answer)
    if probe.get("canonical_id") == "output.no_extra_text":
        assert not wrapped.passed
    else:
        assert wrapped.passed


def json_answer(probe: dict) -> str:
    value = probe["ground_truth"]
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            pass
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


strict_json = [p for p in ENABLED if p.get("scorer", {}).get("type") == "strict_json"]
assert len(strict_json) >= 14
for probe in strict_json:
    answer = json_answer(probe)
    assert scored(probe, answer).passed, probe["probe_id"]
    assert scored(probe, "```json\n" + answer + "\n```").passed, probe["probe_id"]
    assert not scored(probe, "EXTRA " + answer).passed
    assert not scored(probe, answer + " EXTRA").passed


# ---------------------------------------------------------------------------
# Specific repaired false positives and policy-override regressions.
# ---------------------------------------------------------------------------
longctx = BY_ID["longctx.position_invariant_recall.L2.001"]
assert scored(longctx, "1.8").passed
assert scored(longctx, "Answer: 1.8").passed

for probe_id, answer in {
    "logic.liar_puzzle.L3.001": "2",
    "read.table_lookup.L1.001": "7",
    "logic.rule_induction.L1.001": "21",
    "logic.rule_induction.L2.001": "48",
    "logic.rule_induction.L3.001": "18",
}.items():
    assert scored(BY_ID[probe_id], answer).passed

truth = BY_ID["truth.calibrated_abstention.L3.005"]
assert scored(truth, "No one.").passed
assert not scored(truth, "No one knows; Neil Armstrong.").passed

complex_axis = BY_ID["math.complex_number_geometry.L1.005"]
assert scored(complex_axis, "imaginary axis").passed
assert not scored(complex_axis, "not imaginary axis").passed

proper_name = BY_ID["know.parametric_recall.L2.004"]
assert scored(proper_name, "Anduril").passed
assert scored(proper_name, "Andúril").passed
assert not scored(proper_name, "not Anduril").passed

formula = BY_ID["know.formula_recall.L2.004"]
assert scored(formula, formula["ground_truth"]).passed
assert scored(formula, "Viète's formulas").passed

assert not format_lenient_regex(
    "a = numpy.zeros(100)",
    pattern=r"^\s*a\s*=\s*np\.zeros\(\s*100\s*\)\s*$",
    flags=["m"],
    allow_stem_fallback=True,
).passed


# ---------------------------------------------------------------------------
# Real response-shape regressions from the two July evaluation runs.
# ---------------------------------------------------------------------------
boolean_eval = BY_ID["logic.boolean_eval.L3.005"]
assert scored(boolean_eval, "**Final Answer:**  \n**False**").passed

numeric_sort = BY_ID["arith.numeric_sort.L1.001"]
assert scored(
    numeric_sort,
    "**Final Answer:**  \n28, 35, 61, 73, 89",
).passed

series_limit = BY_ID["math.series_limit_eval.L3.002"]
assert scored(series_limit, r"\boxed{e^{2}}").passed
assert scored(series_limit, "\\[\n\\boxed{e^{2}}\n\\]").passed

complex_center = BY_ID["math.complex_number_geometry.L1.001"]
assert scored(complex_center, "\\[\n\\boxed{0 + 3i}\n\\]").passed

root_pair = BY_ID["math.polynomial_root_theory.L2.005"]
assert scored(root_pair, "\\boxed{(-2,\\ 5)}").passed

sqrt_probe = {
    "probe_id": "synthetic.sqrt_regex",
    "canonical_id": "synthetic.sqrt_regex",
    "category": "math",
    "ground_truth": "sqrt(2)",
    "scorer": {
        "type": "regex",
        "pattern": r"^\s*sqrt\s*\(\s*2\s*\)\s*$",
    },
}
assert default_route(sqrt_probe, r"\boxed{\sqrt{2}}", {}).passed

classifier = BY_ID["lang.classifier_concord.L3.001"]
assert scored(classifier, "The standard answer is **piece**.").passed
assert not scored(
    classifier,
    "The answer is **a bit of advice**.\n"
    "While *a piece of advice* is possible, bit is the preferred answer.",
).passed

punctuated_output = BY_ID["output.no_extra_text.L3.001"]
assert not scored(
    punctuated_output, "Answer: " + punctuated_output["ground_truth"],
).passed
multiline_output = BY_ID["output.no_extra_text.L3.004"]
assert not scored(
    multiline_output, "Answer: " + multiline_output["ground_truth"],
).passed
assert not scored(
    multiline_output, "Line one.  \nLine two.  \nLine three.",
).passed


# Every LLM rubric has the exact runtime enum-like pass rule. No network call.
judge_probes = [p for p in ENABLED if p.get("scorer", {}).get("type") == "judge_rubric"]
assert judge_probes
assert all(
    p["scorer"]["rubric"].get("pass_rule")
    == "all pass criteria and no fail criteria"
    for p in judge_probes
)


summary = {
    "enabled_probes_loaded": len(ENABLED),
    "numeric_negation_checked": len(numeric),
    "mcq_negation_checked": len(mcq),
    "terminal_punctuation_checked": len(punctuation_opt_in),
    "strict_output_checked": len(strict_output),
    "strict_json_checked": len(strict_json),
    "judge_pass_rule_checked": len(judge_probes),
}
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
