"""Regression tests for non-instruction answer normalization."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))

import eval.dispatch as dispatch  # noqa: E402
from eval.dispatch import default_route, load_policy, score_probe  # noqa: E402
from eval.scorer import (  # noqa: E402
    ScoreResult, _tagged_answer_candidates,
)


POLICY = load_policy(ACE / "policy" / "atom_intents.yaml")
PROBES = {}
for path in (ACE / "probes").glob("*.jsonl"):
    for raw in path.read_text().splitlines():
        if raw.lstrip().startswith("{"):
            probe = json.loads(raw)
            PROBES[probe["probe_id"]] = probe


def scored(probe: dict, response: str):
    result, denominator = score_probe(probe, response, POLICY)
    assert denominator == 1, (probe.get("probe_id"), result.to_dict())
    return result


def passes(result):
    assert result.passed, result.to_dict()


def fails(result):
    assert not result.passed, result.to_dict()


# Non-instruction transport cleanup: API line endings, boundary controls,
# Markdown hard-break spaces, and one uniquely unclosed outer fence.
plain = {
    "probe_id": "synthetic.noninstr.plain",
    "canonical_id": "synthetic.noninstr.plain",
    "category": "truth",
    "ground_truth": "Berlin",
    "scorer": {"type": "exact_match_string", "expected": "Berlin"},
}
passes(scored(plain, "\ufeff\u200bBerlin\u2060"))

multiline = {
    "probe_id": "synthetic.noninstr.lines",
    "canonical_id": "synthetic.noninstr.lines",
    "category": "agent",
    "ground_truth": "one\ntwo",
    "scorer": {"type": "regex", "pattern": r"\Aone\ntwo\Z"},
}
passes(scored(multiline, "one  \r\ntwo"))

numeric = {
    "probe_id": "synthetic.noninstr.numeric",
    "canonical_id": "synthetic.noninstr.numeric",
    "category": "arith",
    "ground_truth": "10",
    "scorer": {"type": "exact_match_numeric", "expected": 10},
}
passes(scored(numeric, "```text\n10"))
fails(scored(numeric, "```text\n10\n```\n```text\n11"))


# The same bytes remain meaningful in instr: no BOM/ZWSP cleanup, no missing
# fence repair, no line-end Markdown-space cleanup, and no explanatory prose.
strict = {
    "probe_id": "synthetic.instr.strict",
    "canonical_id": "synthetic.instr.strict",
    "category": "instr",
    "ground_truth": "A",
    "scorer": {
        "type": "exact_match_string", "expected": "A",
        "case_sensitive": True, "strip_whitespace": True,
    },
}
passes(scored(strict, "A"))
for response in ("\ufeffA\u200b", "```text\nA", "A\nExplanation"):
    fails(scored(strict, response))
strict_lines = dict(multiline, category="instr")
fails(scored(strict_lines, "one  \r\ntwo"))

tf_probe = dict(
    plain,
    probe_id="synthetic.noninstr.tf",
    canonical_id="synthetic.noninstr.tf",
    ground_truth="TF",
    scorer={"type": "exact_match_string", "expected": "TF"},
)
fails(scored(tf_probe, "TF\nFT"))


# Paired answer tags are allowlisted and their payload still has to satisfy
# the original scorer. Distinct tag values are ambiguous rather than last-win.
integer = PROBES["arith.exact_integer_op.L2.005"]
expected_integer = str(integer["scorer"]["expected"])
passes(scored(integer, f"Worked result:\n<result>{expected_integer}</result>"))
fails(scored(
    integer,
    f"<result>{expected_integer}</result><answer>1</answer>",
))


# A compact first-line answer may introduce a consistent explanation, but a
# later correction or explicit answer remains authoritative.
choice = PROBES["arith.compare_quantities.L3.004"]
passes(scored(choice, "A\n\nA dropped farther than B."))
fails(scored(choice, "A\n\nActually: B"))
fails(scored(choice, "A\n\nAnswer: B"))


# A marker-declared scalar equation uses its terminal RHS; no expression is
# evaluated, and a wrong RHS remains a wrong answer.
modular = PROBES["arith.modular_op.L1.002"]
passes(scored(modular, "Final result: $67 \\mod 11 = 1$"))
fails(scored(modular, "Final result: $67 \\mod 11 = 2$"))
passes(scored(numeric, "Output:\n```\n10\n```"))
fails(scored(numeric, "Output:\n```\n1 + 9\n```"))
fails(scored(numeric, "Final answer: 10\n```\nforty-three\n```"))
passes(scored(numeric, "Final answer: 10\nExample:\n```\n11\n```"))
fails(scored(numeric, "Final answer: 11\nExample:\n```\n10\n```"))
fails(scored(
    modular,
    "Final result: 42. Example: 67 mod 11 = 1",
))


# Separator whitespace and the weekday comma are proven presentation-only
# equivalences; values/order/delimiter identity remain strict.
inventory = PROBES["plan.long_horizon_persistence.L3.004"]
passes(scored(inventory, "apple,bread,cheese,dates,eggs"))
fails(scored(inventory, "apple,bread,dates,cheese,eggs"))
fails(scored(inventory, "apple/bread/cheese/dates/eggs"))

date_probe = PROBES["world.time_calendar.L3.004"]
passes(scored(date_probe, "Thus, the answer is **Monday, January 1, 2024**."))
passes(scored(
    date_probe,
    "Thus, the film ends on **Monday, January 1, 2024**.",
))
fails(scored(date_probe, "Thus, the answer is **Monday, January 2, 2024**."))


# Legacy anchored semantic regexes can reuse answer-position extraction only
# when the regex itself accepts the declared reference. Boundaries and later
# explicit answers prevent substring and override false positives.
berlin = PROBES["dialog.intent_inference.L1.002"]
passes(scored(berlin, "The capital of Germany is Berlin."))
fails(scored(berlin, "Berlin may be possible. Answer: Paris"))
fails(scored(berlin, "Final answer: Paris.\nReference: **Berlin**."))
fails(scored(
    berlin,
    "Final answer: Paris.\nExample XML: <output>Berlin</output>",
))
fails(scored(berlin, "Final answer: Paris <result>Berlin</result>"))
fails(scored(berlin, "not <result>Berlin</result>"))
fails(scored(berlin, "<result>Berlin</result>junk</result>"))
fails(scored(berlin, "Berlin\nReasoning.\nParis"))
path_probe = {
    "probe_id": "synthetic.truth.path",
    "canonical_id": "synthetic.truth.path",
    "category": "truth",
    "ground_truth": "/tmp/ace",
    "scorer": {"type": "regex", "pattern": r"^/tmp/ace$"},
}
fails(scored(path_probe, "The path is /tmp/ace-extra."))

# Paired tags are authoritative only as a complete response, standalone final
# line, or the complete payload of a strong marker. Negation/malformed markup
# must not leak back into ordinary numeric/string fallbacks.
passes(scored(berlin, "<result>Berlin</result>"))
for response in (
    "not <result>10</result>",
    "The answer is not <result>10</result>.",
    "Final answer: not <result>10</result>",
    "Final answer: 7 <result>10</result>",
):
    fails(scored(numeric, response))

started = time.perf_counter()
assert not _tagged_answer_candidates("<output>" * 12_500)
assert time.perf_counter() - started < 0.5


# Real Qwen regression shapes: final completed partitive, Oxford comma,
# explicit build flags, balanced LaTeX boxes/display math, and harmless prose
# after a proven answer.
classifier = PROBES["lang.classifier_concord.L3.001"]
passes(scored(
    classifier,
    "The correct fill-in is:\n**'a piece of advice'**.",
))
quotation = PROBES["truth.verbatim_quotation.L2.003"]
passes(scored(quotation, "blood, toil, tears, and sweat"))

build = PROBES["code.build_system_use.L2.003"]
for command in (
    "python -m build", "python -m build --sdist --wheel",
    "python3 -m build --wheel --sdist", "py -m build -s -w",
):
    passes(scored(build, command))
for command in (
    "python -m build --wheel", "python -m build --sdist",
    "python -m build --sdist --wheel dist/", "setup.py sdist bdist_wheel",
):
    fails(scored(build, command))

# Artifact/output atoms remain byte-exact: global non-instr transport cleanup
# may not repair fences, trim hard-break spaces, or canonicalize line endings.
artifact = PROBES["output.artifact_specification_compliance.L1.001"]
passes(scored(artifact, artifact["ground_truth"]))
fails(scored(artifact, "```text\nhello world"))
fails(scored(artifact, "```text\nhello world  \n```"))
byte_exact = PROBES["output.artifact_specification_compliance.L3.001"]
passes(scored(byte_exact, byte_exact["ground_truth"]))
fails(scored(byte_exact, byte_exact["ground_truth"].replace("\n", "\r\n")))

# Static/executing/structured scorers receive the original response bytes.
captured = []
original_code_parse_ast = dispatch.code_parse_ast
try:
    dispatch.code_parse_ast = lambda response, scorer: (
        captured.append(response) or ScoreResult(True, "captured")
    )
    static_probe = {
        "probe_id": "synthetic.code.bytes",
        "canonical_id": "synthetic.code.bytes",
        "category": "code",
        "ground_truth": "",
        "scorer": {"type": "code_parse_ast", "parser": "python"},
    }
    raw_static = 'value = "line  \\r\\n"\r\n'
    passes(scored(static_probe, raw_static))
    assert captured == [raw_static]
finally:
    dispatch.code_parse_ast = original_code_parse_ast

steinbeck = PROBES["truth.hallucination_resistance.L2.002"]
for answer in ("Steinbeck", "John Steinbeck", "I don't know"):
    passes(scored(steinbeck, answer))
for answer in ("John Ernst Steinbeck", "Probably Steinbeck", "Hemingway"):
    fails(scored(steinbeck, answer))

math_cases = {
    "math.synthetic_geometry.L3.003": r"""### Final Answer:
$$
\boxed{4\sqrt{3}}
$$
This is the exact distance requested.""",
    "math.equation_solve.L3.002": r"""### Final Answer:
$$
\boxed{\frac{7\pi}{6}}
$$""",
    "math.polynomial_root_theory.L1.003": r"""### Final Answer:
$$
1, -1, 1/2, -1/2
$$""",
    "math.series_limit_eval.L3.003": r"""### Final Answer:
$$
\boxed{\frac{\pi^2}{6}}
$$""",
}
for probe_id, response in math_cases.items():
    passes(scored(PROBES[probe_id], response))

# Algebraic equality must not erase an explicitly tested representation
# constraint: this probe requires the denominator to remain factored.
factored = PROBES["math.symbolic_manipulation.L2.004"]
passes(scored(factored, "Final Answer: 2/((x-1)(x+1))"))
passes(scored(factored, r"Final Answer: \frac{2}{(x-1)(x+1)}"))
fails(scored(factored, r"Final Answer: \boxed{\frac{2}{x^2-1}}"))
fails(scored(factored, "2/(x^2-1) + 0*(x-1)*(x+1)"))

# Algebraic equality cannot bypass an explicitly requested representation.
for probe_id, wrong_form in {
    "math.symbolic_manipulation.L1.001": "(x-3)(x+3)",
    "math.symbolic_manipulation.L1.004": "x^2-7x+12",
    "math.symbolic_manipulation.L2.005": "cos(x)*2",
    "math.symbolic_manipulation.L3.001": "-4.5",
    "math.synthetic_geometry.L1.002": "0.25",
}.items():
    fails(scored(PROBES[probe_id], wrong_form))

ratio = PROBES["math.synthetic_geometry.L1.002"]
fails(scored(
    ratio,
    r"\boxed{1/4}" "\nThis is the actual final value 1/2.",
))

# Qwen uncertainty answers use endpoint labels inside a box and a two-line
# structured answer section followed by explanatory prose.
uncertainty_range = PROBES["meta.uncertainty_communication.L3.004"]
passes(scored(
    uncertainty_range,
    r"Final Answer: \boxed{\$14,000 \text{ (minimum)} "
    r"\text{ to } \$28,800 \text{ (maximum)}}"
    "\nThis is the propagated range requested.",
))
uncertainty_power = PROBES["meta.uncertainty_communication.L3.005"]
passes(scored(
    uncertainty_power,
    "### Final Answer:\n"
    r"- $P_{\min} = 28.08$ W" "\n"
    r"- $P_{\max} = 31.98$ W" "\n\n"
    "Both values are reported to two decimal places.",
))

# Six audited Qwen answers have a correct, explicitly headed boxed answer and
# then one descriptive sentence which happens to repeat problem-side numbers
# or formulas. The opt-in is probe-local and exercised below.
boxed_tail_probe_ids = {
    "math.synthetic_geometry.L1.002",
    "math.synthetic_geometry.L3.003",
    "math.optimization_extremum.L2.005",
    "math.optimization_extremum.L3.001",
    "math.complex_number_geometry.L2.001",
    "math.complex_number_geometry.L3.004",
}
assert {
    probe_id for probe_id, probe in PROBES.items()
    if probe.get("scorer", {}).get(
        "allow_boxed_final_descriptive_math_tail"
    ) is True
} == boxed_tail_probe_ids

# The opt-in does not make a later answer/correction/value harmless. It also
# requires the box itself to satisfy the original probe regex.
boxed_ratio_prefix = (
    "### Final Answer:\n\n"
    "$$\n"
    r"\boxed{\frac{1}{4}}" "\n"
    "$$\n\n"
)
passes(scored(
    ratio,
    boxed_ratio_prefix
    + "This is the reduced fraction of the area requested.",
))
for conflicting_tail in (
    "This is the actual final value 1/2.",
    "Final answer: 1/2",
    "Actually, the answer is 1/2.",
    "1/2",
    "This is the correct number 1/2.",
    "<result>1/2</result>.",
    r"This is \boxed{1/2}.",
):
    fails(scored(ratio, boxed_ratio_prefix + conflicting_tail))
fails(scored(
    ratio,
    "### Final Answer:\n\n$$\n"
    r"\boxed{\frac{1}{2}}" "\n$$\n\n"
    "This is the reduced fraction of the area requested.",
))

# A non-opted-in probe retains the default rule that math-bearing trailing
# prose is answer-bearing rather than silently ignored.
non_opted_series = PROBES["math.series_limit_eval.L3.003"]
fails(scored(
    non_opted_series,
    "### Final Answer:\n\n$$\n"
    r"\boxed{\frac{\pi^2}{6}}" "\n$$\n\n"
    "This is the exact value for the first 12 terms.",
))

# A terminal balanced box is the final answer presentation even when ordinary
# prose earlier contains a descriptive ``result is`` phrase. Reference/example
# boxes are not answers, and a later conflicting terminal box still wins.
passes(scored(
    numeric,
    "The result is an integer and has been verified.\n\\boxed{10}",
))
passes(scored(numeric, "Thus the value is ready.\n\\boxed{\\text{10}}"))
fails(scored(numeric, "Final answer: 10\n\\boxed{11}"))
fails(scored(numeric, "Final answer: 11\nReference:\n\\boxed{10}"))

boxed_string = PROBES["lang.ordinal_indexing.L3.002"]
passes(scored(boxed_string, "The result is verified.\n\\boxed{\\text{B}}"))

# A final JSON fence may transport one numeric scalar leaf. Multiple numeric
# leaves remain ambiguous and no arithmetic expression is evaluated.
passes(scored(numeric, '```json\n{"output": 10}\n```'))
fails(scored(numeric, '```json\n{"output": 10, "example": 9}\n```'))
fails(scored(numeric, '```json\n{"output": "10"}\n```'))

thousand = dict(numeric, ground_truth="1000")
thousand["scorer"] = {"type": "exact_match_numeric", "expected": 1000}
passes(scored(thousand, r"\boxed{1,000}"))
fails(scored(numeric, "Final answer: 1,2"))

# tool.format_robustness tolerates only one whole-response Markdown wrapper or
# one terminal sentence mark; prose/multiple fences/combined wrappers remain
# format violations. Output artifacts and no-extra-text stay strict.
tool_call = PROBES["tool.format_robustness.L2.005"]
for response in (
    '`lookup_zip(zip_code="94110")`',
    '```python\nlookup_zip(zip_code="94110")\n```',
    'lookup_zip(zip_code="94110")!',
):
    passes(scored(tool_call, response))
for response in (
    'Call: `lookup_zip(zip_code="94110")`',
    '`lookup_zip(zip_code="94110")`!',
    '```python\nlookup_zip(zip_code="94110")\n```\nextra',
    '```python\nlookup_zip(zip_code="94110")\n```\n```python\nlookup_zip(zip_code="94110")\n```',
):
    fails(scored(tool_call, response))

for probe_id in (
    "logic.set_operations.L1.001",
    "logic.set_operations.L1.004",
):
    set_probe = PROBES[probe_id]
    expected = set_probe["ground_truth"]
    passes(scored(set_probe, f"Thus: {expected}"))
extra_lists = {
    "logic.set_operations.L1.001": "1, 7, 18",
    "logic.set_operations.L1.004": "orange, blue, green",
}
for probe_id, response in extra_lists.items():
    fails(scored(PROBES[probe_id], response))

no_extra = PROBES["output.no_extra_text.L3.004"]
passes(scored(no_extra, no_extra["ground_truth"]))
fails(scored(no_extra, "Line one.  \nLine two.  \nLine three."))
fails(scored(no_extra, "Answer: " + no_extra["ground_truth"]))


print("ACE non-instr output normalization regressions: PASS")
