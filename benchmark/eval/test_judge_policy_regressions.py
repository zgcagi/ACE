"""ACE offline regressions for Judge policy and audited probe repairs."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))

from eval import judge


def load_probes() -> dict[str, dict]:
    probes: dict[str, dict] = {}
    for path in sorted((ACE / "probes").glob("*.jsonl")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.lstrip().startswith("{"):
                continue
            probe = json.loads(raw)
            probes[probe["probe_id"]] = probe
    return probes


PROBES = load_probes()

# Prompt version and category policy changed together; an old prompt version
# must never silently acquire the new mechanical-observation semantics.
assert judge.PROMPT_VERSION == \
    "ace-binary-judge-v1-deterministic-observations"
for required_policy in (
    "candidate_observations",
    "Ordinary adjacent punctuation does not change the word",
    '"First word" means the first lexical word',
    "Ordinary prose-length, paragraph-count, sentence-count",
    "Do not invent requirements absent from the task",
    "Apply the same rule to materially equivalent candidates",
    "definition and scope exactly match",
    "never substitute a global observation",
    "quoted_term_casefold_whole_word_counts",
):
    assert required_policy in judge.JUDGE_SYSTEM_PROMPT, required_policy

# The seven historical instr false negatives now receive deterministic facts.
keyword = PROBES["instr.keyword_frequency_constraint.L2.001"]
packet = judge.build_judge_payload(
    keyword, "helmet, lane. helmet! lane? (helmet) lane;"
)
observed = packet["candidate_observations"]
assert observed["whitespace_token_count"] == 6
assert observed["quoted_term_whole_word_counts"]["helmet"] == 3
assert observed["quoted_term_whole_word_counts"]["lane"] == 3
case_counts = judge._candidate_observations(
    {"system_prompt": "", "user_prompt": "Avoid 'volcano' in any capitalization."},
    "Volcano HELMET helmet",
)
assert case_counts["quoted_term_exact_case_whole_word_counts"]["volcano"] == 0
assert case_counts["quoted_term_casefold_whole_word_counts"]["volcano"] == 1

segmented = judge._candidate_observations(
    {"system_prompt": "", "user_prompt": ""},
    "Spring, apples grow.\n\nLast, workers harvest.",
)
assert segmented["paragraph_count"] == 2
assert segmented["blank_lines_per_separator"] == [1]
assert segmented["blank_line_separator_count"] == 1
assert segmented["paragraph_details_truncated"] is False
assert [p["first_lexical_word"] for p in segmented["paragraphs"]] == [
    "Spring", "Last",
]

wrapped = judge._candidate_observations(
    {"system_prompt": "", "user_prompt": ""},
    "(One two\n\nThree four)",
)
assert wrapped["whitespace_token_count"] == 4
assert wrapped["paragraph_count"] == 2
assert wrapped["first_non_whitespace_character"] == "("
assert wrapped["last_non_whitespace_character"] == ")"
assert wrapped["recognized_outer_wrapper"] == {
    "opening_character": "(",
    "closing_character": ")",
    "inner_whitespace_token_count": 4,
}

# Observation detail stays bounded for adversarially segmented responses while
# retaining exact aggregate counts and details from both ends.
many_paragraphs = "\n\n".join(f"P{index} word" for index in range(5000))
bounded = judge._candidate_observations(
    {"system_prompt": "", "user_prompt": ""}, many_paragraphs
)
assert bounded["paragraph_count"] == 5000
assert bounded["blank_line_separator_count"] == 4999
assert len(bounded["paragraphs"]) == judge._OBSERVATION_DETAIL_LIMIT
assert len(bounded["blank_lines_per_separator"]) == \
    judge._OBSERVATION_DETAIL_LIMIT
assert bounded["paragraph_details_truncated"] is True
assert bounded["blank_line_details_truncated"] is True
assert bounded["paragraphs"][0]["index"] == 1
assert bounded["paragraphs"][-1]["index"] == 5000
assert len(json.dumps(bounded, ensure_ascii=False)) < 15_000

judge_fn_probes = {
    "instr.keyword_frequency_constraint.L2.001",
    "instr.keyword_frequency_constraint.L3.004",
    "instr.multi_constraint_compliance.L2.002",
    "instr.response_wrap.L2.002",
    "instr.structural_segmentation.L3.002",
    "instr.structural_segmentation.L3.003",
    "instr.structural_segmentation.L3.005",
    "truth.misconception_resistance.L3.002",
}
for probe_id in judge_fn_probes:
    assert PROBES[probe_id]["scorer"]["rubric"]["version"] == "1"

# The ninth Judge false negative is a tool probe and belongs to the tool/code
# integration owner; the global judge policy still applies to it at runtime.
assert PROBES["tool.command_invocation.L3.005"]["scorer"]["type"] == \
    "judge_rubric"

# All nine audited non-instr rubric problems distinguish core capability from
# ordinary presentation diagnostics.
rubric_repairs = {
    "dialog.persona_stability.L1.004",
    "lang.expository_explanation.L2.003",
    "lang.expository_explanation.L2.005",
    "meta.uncertainty_communication.L1.003",
    "output.fluent_prose.L1.002",
    "output.fluent_prose.L3.001",
    "output.fluent_prose.L3.002",
    "output.fluent_prose.L3.003",
    "output.fluent_prose.L3.005",
}
assert len(rubric_repairs) == 9
for probe_id in rubric_repairs:
    rubric = PROBES[probe_id]["scorer"]["rubric"]
    assert rubric["version"] == "1", probe_id
    rendered = json.dumps(rubric, ensure_ascii=False).lower()
    assert "nonbinding" in rendered or "non-binding" in rendered, probe_id

# Audited ground-truth/data repairs in this agent's non-code/tool scope.
assert "55 compute units" in PROBES[
    "agent.compute_budget_management.L3.003"
]["prompt"]
assert PROBES["lang.character_frequency.L2.001"]["scorer"]["expected"] == 10
assert PROBES["lang.character_frequency.L3.001"]["scorer"]["expected"] == 20
assert PROBES["lang.chinese_de_particle.L2.001"]["scorer"]["expected"] == "B"
assert PROBES["lang.multi_negation.L3.001"]["scorer"]["expected"] == "unclear"
assert PROBES["meta.ambiguity_detection.L1.005"]["ground_truth"] == "ambiguous"
assert "I, the user" in PROBES["truth.indexical_grounding.L2.005"]["prompt"]
assert PROBES["truth.verbatim_quotation.L3.002"]["tags"][0] == "public_domain"
assert "A Tale of Two Cities" in PROBES[
    "truth.verbatim_quotation.L3.002"
]["prompt"]
assert len(PROBES[
    "truth.verbatim_quotation.L3.002"
]["ground_truth"].split()) >= 100
assert PROBES["truth.verbatim_quotation.L3.003"]["ground_truth"] == \
    "Sie ist das Opium des Volks."

# Non-code/tool prompt defects identified by the post-run audit are explicit.
assert "ultimately defeat all three" in PROBES[
    "agent.risk_reward_tradeoff.L3.002"
]["prompt"]
for probe_id in (
    "agent.shared_state_coordination.L2.004",
    "agent.shared_state_coordination.L2.005",
):
    assert "exact lowercase token" in PROBES[probe_id]["prompt"]
assert "条、头、匹" in PROBES["lang.classifier_concord.L2.001"]["prompt"]
assert "increasing markers" in PROBES["output.structured_artifact.L1.002"]["prompt"]
assert "less dense than air" in PROBES["read.passage_integration.L2.005"]["prompt"]
assert "Rick's Café" in PROBES["truth.verbatim_quotation.L1.003"]["prompt"]
quote_scorer = PROBES["truth.verbatim_quotation.L1.003"]["scorer"]
assert quote_scorer["type"] == "regex"
assert "again" not in quote_scorer["pattern"].lower()
flight = PROBES["instr.confirmation_gate.L2.002"]
assert "Simulation clock: 2030-01-01" in flight["system_prompt"]
assert "2030-07-12" in flight["prompt"]

# The eight stable upstream content-filter prompts retain their expected answer
# and capability while removing the phrases observed to trigger Prompt Shields.
filter_repairs = {
    "dialog.multi_turn_coherence.L3.001": ["overrides t1"],
    "instr.system_prompt_adherence.L1.002": ["secret rule", "disregard"],
    "instr.system_prompt_adherence.L1.004": ["ignore previous"],
    "instr.system_prompt_adherence.L2.005": [
        "secret rule", "normalgpt", "no longer bound",
    ],
    "plan.long_horizon_persistence.L3.005": [
        "password", "credential", "untrusted", "no authority",
    ],
    "read.speaker_stance_attribution.L1.001": ["moon landing", "faked"],
    "truth.indexical_grounding.L3.003": ["breakfast", "what did you eat"],
    "truth.shortcut_resistance.L3.001": [
        "guard", "treasure", "mistruth", "always lies",
    ],
}
for probe_id, banned in filter_repairs.items():
    probe = PROBES[probe_id]
    prompt_surface = (
        str(probe.get("system_prompt", "")) + "\n" + probe["prompt"]
    ).lower()
    for phrase in banned:
        assert phrase not in prompt_surface, (probe_id, phrase)

assert PROBES["dialog.multi_turn_coherence.L3.001"]["ground_truth"] == "B"
for probe_id in (
    "instr.system_prompt_adherence.L1.002",
    "instr.system_prompt_adherence.L2.005",
):
    assert PROBES[probe_id]["ground_truth"] == "Marigold."
assert PROBES["instr.system_prompt_adherence.L1.004"]["ground_truth"] == "42"
assert PROBES["plan.long_horizon_persistence.L3.005"]["ground_truth"] == \
    "turtleAvocado19"
assert PROBES["read.speaker_stance_attribution.L1.001"]["ground_truth"] == \
    "oppose"
assert PROBES["truth.shortcut_resistance.L3.001"]["ground_truth"] == "RIGHT"

print("ACE Judge/data policy regressions: PASS")
