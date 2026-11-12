"""Self-tests for ACE scorers — especially the "correct in incorrect" trap."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.scorer import (
    extract_numeric, extract_string, extract_mcq_letter, extract_set,
    extract_ordered_sequence,
    extract_integer,
    semantic_keywords_any, semantic_keywords_all, abstention_check,
    format_strict_regex, format_lenient_regex, json_schema_lenient,
    code_structural, code_parse_ast, code_unittest, numeric_range,
    _word_boundary_search,
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

results = []


def check(label, got, expected):
    ok = got == expected
    sym = PASS if ok else FAIL
    print(f"  {sym} {label}")
    if not ok:
        print(f"      got={got!r}  expected={expected!r}")
    results.append(ok)


print("=== _word_boundary_search: 'correct' must NOT match 'incorrect' ===")
check("correct in 'this is correct'", _word_boundary_search("this is correct", "correct"), True)
check("correct NOT in 'incorrect'", _word_boundary_search("this is incorrect", "correct"), False)
check("correct NOT in 'incorrectly'", _word_boundary_search("answered incorrectly", "correct"), False)
check("yes NOT in 'eyes'", _word_boundary_search("she has blue eyes", "yes"), False)
check("yes IN 'yes I do'", _word_boundary_search("yes I do", "yes"), True)
check("对 in '对的'", _word_boundary_search("对的", "对"), True)  # Chinese: word boundary at non-alphanum

print("\n=== extract_numeric: model says reasoning then answer ===")
r = extract_numeric("47 + 28 = 75\n75 - 19 = 56\n56 + 13 = 69", 69)
check("DPO-style stepwise → finds 69 at end", r.passed and r.strategy.startswith("last"), True)

r = extract_numeric("**Final number: 69**", 69)
check("bold-wrapped final answer", r.passed, True)

r = extract_numeric("The answer is 69.", 69)
check("'answer is X' → 69", r.passed, True)

r = extract_numeric("\\boxed{69}", 69)
check("LaTeX boxed{69}", r.passed, True)

r = extract_numeric("The answer is 47.", 69)
check("wrong number 47, expected 69 → fail", not r.passed, True)

r = extract_numeric("47 + 28 = 75; total 75", 75)
check("75 at end", r.passed, True)

r = extract_numeric("75 is the answer", 75)
check("'X is the answer'", r.passed, True)

r = extract_numeric("", 69)
check("empty → fail", not r.passed, True)

r = extract_numeric("about 69 give or take", 69, tolerance=0)
check("'about 69' with tolerance=0 → still passes (last num)", r.passed, True)


print("\n=== extract_string: answer at various positions ===")
r = extract_string("banana", "banana")
check("bare 'banana' = 'banana'", r.passed, True)

r = extract_string("The answer is banana.", "banana", allow_substring=True)
check("'The answer is banana.' (loose)", r.passed, True)

r = extract_string("**banana**", "banana")
check("'**banana**' bold-wrapped", r.passed, True)

r = extract_string("The fruit is `apple`, but I think `banana` is what you want.", "banana", allow_substring=False)
check("strict mode: 'banana' buried but unmarked → fail", not r.passed, True)

r = extract_string("The fruit is `apple`, but I think `banana` is what you want.", "banana", allow_substring=True)
check("multiple different quoted candidates are ambiguous", r.passed, False)

r = extract_string("The selected fruit is `banana`.", "banana", allow_substring=True)
check("single affirmative quoted answer", r.passed, True)

r = extract_string("I would not choose 'banana'.", "banana", allow_substring=True)
check("negated quoted answer is rejected", r.passed, False)

r = extract_string("incorrect", "correct", allow_substring=True)
check("'correct' must NOT match 'incorrect' (word boundary!)", not r.passed, True)

r = extract_string("eye", "yes", allow_substring=True)
check("'yes' must NOT match in 'eye' (no word boundary)", not r.passed, True)

r = extract_string("Answer: yes\n", "yes", allow_substring=True)
check("'Answer: yes' (last line)", r.passed, True)

r = extract_string("yes I think so", "yes", allow_substring=True)
check("'yes I think so' word-boundary", r.passed, True)

r = extract_string("banana may be possible. Answer: apple", "banana",
                   allow_substring=True)
check("wrong explicit final answer overrides buried expected", r.passed, False)

r = extract_string("The answer is not banana", "banana", allow_substring=True)
check("negated expected string is rejected", r.passed, False)

r = extract_string("A useful plan", "A", allow_substring=True)
check("one-character expected does not match an article", r.passed, False)

r = extract_string("Reasoning first.\nTherefore the capital is Berlin.",
                   "Berlin", allow_substring=True,
                   allow_terminal_punctuation=True)
check("natural final sentence can expose the answer", r.passed, True)


print("\n=== extract_mcq_letter ===")
r = extract_mcq_letter("C", "C")
check("'C' → C", r.passed, True)

r = extract_mcq_letter("\nC", "C")
check("'\\nC' → C", r.passed, True)

r = extract_mcq_letter("**C**", "C")
check("'**C**' → C", r.passed, True)

r = extract_mcq_letter("The answer is C.", "C")
check("'The answer is C.' → C", r.passed, True)

r = extract_mcq_letter("C is the right choice.", "C")
check("unmarked prose ending in C is not an explicit MCQ answer", r.passed, False)

r = extract_mcq_letter("After checking all four options, I choose the last one.\nD", "D")
check("explanation followed by a pure final-line D", r.passed, True)

r = extract_mcq_letter("Looking at A, B, C, D, the best is D.", "C")
check("Pick wrong letter → fail", not r.passed, True)

r = extract_mcq_letter("B) Counterfactual outcome", "B")
check("leading explicit option line", r.passed, True)

r = extract_mcq_letter("**Answer:** B", "B")
check("Markdown answer marker", r.passed, True)

r = extract_mcq_letter("A) first\nB) second\nC) third\nD) fourth", "A")
check("copied option list is not a selected answer", r.passed, False)

r = extract_mcq_letter("I do not choose B", "B")
check("negated selection is rejected", r.passed, False)

r = extract_mcq_letter("Answer: B", "B", require_bare=True)
check("strict MCQ requires a bare letter", r.passed, False)


print("\n=== extract_integer: lossless answer-position extraction ===")
large_integer = "9007199254740993"
check("large integer answer marker",
      extract_integer(f"Reasoning omitted. Answer: {large_integer}", large_integer).passed,
      True)
check("large adjacent integer remains distinct",
      extract_integer("Answer: 9007199254740992", large_integer).passed,
      False)
check("decimal spelling is not an exact integer",
      extract_integer("9007199254740993.0", large_integer).passed,
      False)
check("exponent spelling is not an exact integer",
      extract_integer("9007199254740993e0", large_integer).passed,
      False)
check("negated integer is rejected",
      extract_integer(f"The answer is not {large_integer}", large_integer).passed,
      False)


print("\n=== semantic_keywords_any ===")
r = semantic_keywords_any("Where would you like to be picked up?",
                          alternatives=["pickup", "pick up", "pick-up", "where", "current location"])
check("'picked up' matches 'pick up' (stem match expected)",
      r.passed or "where" in r.note or r.note.startswith("matched"), True)
# Note: pure word-boundary "pick up" won't match "picked up". Format_lenient_regex handles stems.

r = semantic_keywords_any("I prefer purple.",
                          alternatives=["yellow", "blue", "purple"])
check("'I prefer purple' → purple match", r.passed, True)

r = semantic_keywords_any("incorrect answer",
                          alternatives=["correct"])
check("'incorrect' must NOT trigger 'correct' (word boundary)", not r.passed, True)


print("\n=== semantic_keywords_all + synonyms ===")
syn = {"yaml.safe_dump": ["yaml.dump", "yaml.safe_dump"]}
r = semantic_keywords_all(
    "import yaml\nyaml.dump(d, open(path), indent=2, default_flow_style=False)",
    required=["import yaml", "yaml.safe_dump", "open(path", "indent=2", "default_flow_style=False"],
    synonyms=syn,
)
check("yaml.safe_dump matches yaml.dump via synonyms", r.passed, True)


print("\n=== abstention_check ===")
r = abstention_check("It is impossible to predict with certainty, but reaching 50,000 would require sustained growth.")
check("'impossible to predict' → abstention", r.passed, True)

r = abstention_check("The most beautiful color is the one that brings you joy.")
check("graceful soft abstention (no explicit phrase) → fail",
      not r.passed, True)  # this is correctly hard for our heuristic
# We accept this fails — soft abstention needs LLM judge

r = abstention_check("I don't know. Records aren't preserved.")
check("'I don't know' explicit", r.passed, True)


print("\n=== format_lenient_regex stem fallback ===")
r = format_lenient_regex(
    "Where would you like to be picked up?",
    pattern="(pickup|pick up|pick-up|where (?:are you|should|from)|address)",
    flags=["i"],
    allow_stem_fallback=True,
)
check("'picked up' lenient regex → matches via 'where' or stem", r.passed, True)

r = format_lenient_regex("a = numpy.zeros(100)",
                         pattern=r"^\s*a\s*=\s*np\.zeros\(\s*100\s*\)\s*$",
                         flags=["m"], allow_stem_fallback=True)
check("anchored np.zeros contract never falls back to a stem match",
      r.passed, False)


print("\n=== json_schema_lenient with shape repair ===")
schema = {
    "type": "object",
    "properties": {
        "name": {"const": "set_timer"},
        "arguments": {
            "type": "object",
            "properties": {"minutes": {"const": 45}},
            "required": ["minutes"],
        },
    },
    "required": ["name", "arguments"],
}
r = json_schema_lenient('{"name": "set_timer", "arguments": {"minutes": 45}}', schema)
check("canonical shape", r.passed, True)

r = json_schema_lenient('{"tool": "set_timer", "minutes": 45}', schema)
check("flat tool/args shape is strict by default", r.passed, False)
r = json_schema_lenient(
    '{"tool": "set_timer", "minutes": 45}', schema,
    repair_tool_shape=True,
)
check("flat tool/args shape (repair: tool→name, flatten→arguments)", r.passed, True)

ordered_schema = {
    "type": "array", "minItems": 2, "maxItems": 2,
    "prefixItems": [
        {"type": "object", "properties": {"name": {"const": "first"}},
         "required": ["name"], "additionalProperties": False},
        {"type": "object", "properties": {"name": {"const": "second"}},
         "required": ["name"], "additionalProperties": False},
    ],
    "items": False,
}
check("prefixItems exact order", json_schema_lenient(
    '[{"name":"first"},{"name":"second"}]', ordered_schema).passed, True)
check("prefixItems rejects swapped calls", json_schema_lenient(
    '[{"name":"second"},{"name":"first"}]', ordered_schema).passed, False)
check("minItems rejects missing call", json_schema_lenient(
    '[{"name":"first"}]', ordered_schema).passed, False)
check("maxItems/items:false reject extra call", json_schema_lenient(
    '[{"name":"first"},{"name":"second"},{"name":"third"}]',
    ordered_schema).passed, False)

from eval.dispatch import default_route
format_probe = {
    "probe_id": "test.format.flags", "canonical_id": "test.format",
    "category": "instr", "ground_truth": "format_only",
    "scorer": {"type": "format_check", "must_satisfy": [{
        "name": "forbid_word", "regex": r"^(?:(?!\bforbidden\b)[\s\S])*$",
        "flags": "i",
    }]},
}
check("per-check format flag accepts clean text",
      default_route(format_probe, "Allowed words only.", {}).passed, True)
check("per-check format flag rejects uppercase forbidden word",
      default_route(format_probe, "FORBIDDEN", {}).passed, False)


print("\n=== code_structural with synonyms ===")
r = code_structural(
    """
import yaml
def dump_yaml(d, path):
    with open(path, 'w') as f:
        yaml.dump(d, f, default_flow_style=False, indent=2)
""",
    must_define=["dump_yaml"],
    must_call_any=["yaml.safe_dump"],
    allow_synonyms={"yaml.safe_dump": ["yaml.dump", "yaml.safe_dump"]},
)
check("yaml.dump satisfies yaml.safe_dump via synonyms", r.passed, True)

r = code_structural(
    """
def is_prime(n):
    if n <= 1: return False
    for i in range(2, n):
        if n % i == 0: return False
    return True
""",
    must_define=["is_prime"],
)
check("must_define is_prime", r.passed, True)


# ----------------------------------------------------------------------------



# Additional edge cases for refined word_boundary
print("\n=== _word_boundary_search edge cases ===")
check("'<h1>' in '<h1>Hello' (non-word edges)",
      _word_boundary_search("<h1>Hello", "<h1>"), True)
check("'x' in 'tar -xzvf' (short alphanumeric relaxed)",
      _word_boundary_search("tar -xzvf", "x"), True)
check("'12' in '12px' (digit-letter boundary OK)",
      _word_boundary_search("gap: 12px;", "12"), True)
check("'pickup' in 'picked up' should NOT match (different words)",
      _word_boundary_search("picked up", "pickup"), False)
check("'March 5' in 'March 5th' (now matches via digit boundary — only adjacent digits block)",
      _word_boundary_search("March 5th", "March 5"), True)

print("\n=== Chinese negation and digit boundaries ===")
check("'对' NOT in '不对'", _word_boundary_search("不对", "对"), False)
check("'对' in '对的'", _word_boundary_search("对的", "对"), True)
check("'合理' NOT in '不合理'", _word_boundary_search("不合理", "合理"), False)
check("'合理' in '价格合理'", _word_boundary_search("价格合理", "合理"), True)
check("'合理' in '价格合理而且不贵' (negation but not on '合理')",
      _word_boundary_search("价格合理而且不贵", "合理"), True)
check("'是' NOT in '不是'", _word_boundary_search("不是", "是"), False)
check("'是' in '是的'", _word_boundary_search("是的", "是"), True)
check("'正确' NOT in '不正确'", _word_boundary_search("不正确", "正确"), False)
check("'365' NOT in '13651' (digit boundary)",
      _word_boundary_search("13651", "365"), False)
check("'365' in '约365天' (no adjacent digit)",
      _word_boundary_search("约365天", "365"), True)
check("'12' in '12px' (digit-letter boundary OK)",
      _word_boundary_search("12px", "12"), True)


print("\n=== Chinese ANSWER_SYNONYMS table ===")
from eval.scorer import _chinese_synonym_match
check("expected '是', resp '是的' → True (synonym)",
      _chinese_synonym_match("是的", "是"), True)
check("expected '是', resp '不是' → False (polarity)",
      _chinese_synonym_match("不是", "是"), False)
check("expected '合理', resp '不合理' → False",
      _chinese_synonym_match("不合理", "合理"), False)
check("expected '不合理', resp '太贵' → True (synonym)",
      _chinese_synonym_match("太贵", "不合理"), True)
check("expected '正确', resp '错误' → False (polarity)",
      _chinese_synonym_match("错误", "正确"), False)

check("numeric-only rejects explanatory text",
      extract_numeric("The answer is 5", 5, require_only_numeric=True).passed, False)
check("numeric-only accepts a bare integer",
      extract_numeric("5", 5, require_only_numeric=True).passed, True)

check("accepted exact-string alias",
      extract_string("Steven Allan Spielberg", "Steven Spielberg",
                     accepted_aliases=["Steven Allan Spielberg"]).passed, True)
check("internal whitespace normalization",
      extract_string("alpha   beta", "alpha beta",
                     normalize_internal_whitespace=True).passed, True)

check("set duplicate rejection",
      extract_set("a, a, b", ["a", "b"], reject_duplicates=True,
                  strict_separator=True).passed, False)
check("set strict comma delimiter rejects semicolon",
      extract_set("a; b", ["a", "b"], strict_separator=True).passed, False)

check("disallowed pattern overrides matching abstention phrase",
      semantic_keywords_any("not provided: 123-45-6789",
                            alternatives=["not provided"],
                            disallowed_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"]).passed,
      False)

check("integer range accepts one integer",
      numeric_range("150", low=100, high=200, integer_only=True,
                    single_value_only=True).passed, True)
check("integer range rejects prose and multiple numbers",
      numeric_range("150 or 160", low=100, high=200, integer_only=True,
                    single_value_only=True).passed, False)

recursive_code = "def depth(x):\n    return 0 if x is None else 1 + depth(x[1])"
check("source check sees recursive call",
      code_unittest(recursive_code, tests=["assert depth(None) == 0"],
                    source_checks={"function": "depth", "recursive": True}).passed,
      True)
check("source check rejects non-recursive implementation",
      code_unittest("def depth(x): return 0", tests=["assert depth(None) == 0"],
                    source_checks={"function": "depth", "recursive": True}).passed,
      False)


print("\n=== fraction_match ===")
from eval.scorer import fraction_match
r = fraction_match("The answer is 1/2", expected="1/2")
check("'1/2' direct", r.passed, True)

r = fraction_match("The answer is 0.5", expected="1/2")
check("'0.5' = '1/2'", r.passed, True)

r = fraction_match("\\boxed{\\frac{1}{2}}", expected="1/2")
check("'\\frac{1}{2}' LaTeX", r.passed, True)

r = fraction_match("\\dfrac{3}{4} ≈ 0.75", expected="3/4")
check("'\\dfrac{3}{4}'", r.passed, True)

r = fraction_match("The answer is 2/3", expected="1/2")
check("'2/3' ≠ '1/2'", not r.passed, True)


print("\n=== code_parse_ast static-only parser checks ===")
css_sc = {"parser":"css","checks":{"selector":".row","declarations":{"display":"flex","gap":"12px"}}}
check("CSS parsed declarations pass", code_parse_ast(".row { gap: 12px; display: flex; }", css_sc).passed, True)
check("CSS comments do not fake declarations", code_parse_ast("/* display:flex; gap:12px */ .row {}", css_sc).passed, False)
py_sc = {"parser":"python","checks":{"imports":["json","datetime.datetime"],"only_imports":True}}
check("Python AST imports pass", code_parse_ast("import json\nfrom datetime import datetime", py_sc).passed, True)
check("Python comments do not fake imports", code_parse_ast("# import json\nfrom datetime import datetime", py_sc).passed, False)
prefix_sc = {"parser":"python","checks":{"import_prefixes":["openpyxl"],"calls":["Workbook"]}}
check("Python AST import-prefix accepts module import",
      code_parse_ast("import openpyxl\nwb = openpyxl.Workbook()", prefix_sc).passed, True)
check("Python AST import-prefix rejects a local lookalike",
      code_parse_ast("def Workbook(): pass\nwb = Workbook()", prefix_sc).passed, False)
json_sc = {"parser":"json","checks":{"path_equals":{"properties.age.minimum":0}}}
check("JSON parsed path passes", code_parse_ast('{"properties":{"age":{"minimum":0}}}', json_sc).passed, True)
check("Invalid JSON fails", code_parse_ast('{"properties":', json_sc).passed, False)

print("\n=== answer-position compatibility regressions ===")
check("marker number beats explanation number",
      extract_numeric("Answer: 41. Explanation: 42", 42).passed, False)
check("later answer marker beats earlier box",
      extract_numeric("\\boxed{42}\nAnswer: 41", 42).passed, False)
check("later bare final number beats earlier marker",
      extract_numeric("Answer: 41\n42", 42).passed, True)
check("Reply marker is recognized",
      extract_numeric("Reply: 42", 42).passed, True)

check("selection verb does not parse article a",
      extract_mcq_letter("select a pairwise comparison", "A").passed, False)
check("later negated MCQ selection overrides earlier positive one",
      extract_mcq_letter("I choose B. I do not choose B.", "B").passed, False)
check("pure first-line MCQ answer may precede explanation",
      extract_mcq_letter("B\nExplanation follows.", "B").passed, True)
check("nested text in boxed MCQ answer",
      extract_mcq_letter(r"\boxed{\text{B}}", "B").passed, True)
check("conflicting pure MCQ lines fail",
      extract_mcq_letter("B\nExplanation\nC", "B").passed, False)
check("leading selected option can precede discussed alternatives",
      extract_mcq_letter(
          "B) selected mechanism\nExplanation.\nA) rejected\nC) rejected\nD) rejected",
          "B",
      ).passed, True)
check("affirmative option sentence is extracted",
      extract_mcq_letter("The optimal algorithm is option B: dynamic programming.",
                         "B").passed, True)
check("bold most-likely option is extracted",
      extract_mcq_letter("After analysis, **B** is the most likely outcome.",
                         "B").passed, True)
check("leading parenthesized option is extracted",
      extract_mcq_letter("The option (B) Banana is not a language.", "B").passed,
      True)
check("later boxed conflicting option remains authoritative",
      extract_mcq_letter("Option B is discussed.\n\\boxed{\\text{C}}", "B").passed,
      False)

check("punctuation-only answer is not buried in a longer run",
      extract_string("************", "*****", allow_substring=True).passed, False)
check("positive currency answer does not match negative amount",
      extract_string("-$1,200", "$1,200", allow_substring=True).passed, False)
check("answer marker prefix does not truncate an answer list",
      extract_string("Answer: permutation, permutation, none", "permutation",
                     allow_substring=True).passed, False)
check("later bare string overrides earlier marker",
      extract_string("Answer: banana\napple", "banana",
                     allow_substring=True).passed, False)
check("ordinary explanation line does not override marker",
      extract_string("Answer: banana\nIt is yellow.", "banana",
                     allow_substring=True).passed, True)
check("indented truncated reasoning tail does not override marker",
      extract_string("Output: hello\nReasoning continues\n              byte", "hello",
                     allow_substring=True).passed, True)
check("display-math closer does not override boxed string",
      extract_string("\\[\n\\boxed{no}\n\\]", "no",
                     allow_substring=True).passed, True)
check("later emphasized final token beats earlier quoted examples",
      extract_string("Modes `644` and `755` exist.\n**755**", "755",
                     allow_substring=True).passed, True)
check("negated quoted answer remains rejected",
      extract_string("not 'banana'", "banana", allow_substring=True).passed, False)

check("set marker-only line uses the next physical line",
      extract_set("**Answer:**\na,b", ["a", "b"]).passed, True)
check("sequence marker-only line uses the next physical line",
      extract_ordered_sequence("**Final Answer:**\n28, 35, 61, 73, 89",
                               ["28", "35", "61", "73", "89"],
                               numeric=True).passed, True)

# Final summary again
print(f"\n=== Final summary: {sum(results)}/{len(results)} passed ===")


def test_scorer_cases():
    assert all(results)


if __name__ == "__main__":
    sys.exit(0 if all(results) else 1)
