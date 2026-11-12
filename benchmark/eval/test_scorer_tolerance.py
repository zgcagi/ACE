"""ACE regressions for tolerant non-instr scoring and static parsers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))

from eval.dispatch import default_route  # noqa: E402
from eval.scorer import (  # noqa: E402
    code_parse_ast,
    extract_mcq_letter,
    extract_numeric,
    extract_string,
    json_schema_lenient,
    python_ast_contains_reference,
    semantic_keywords_any,
    _symbolic_expressions_equivalent,
)


def passes(result):
    assert result.passed, result.to_dict()


def fails(result):
    assert not result.passed, result.to_dict()


# Final answer transport wrappers are tolerated only on the semantic path.
passes(extract_string(
    "Reasoning.\n**Final answer:**\n```text\nutils/csv.py\n```",
    "utils/csv.py", allow_substring=True, allow_terminal_punctuation=True,
    allow_semantic_equivalence=True,
))
fails(extract_string(
    "**Final answer:**\n```text\nutils/csv.py\n```\nActually: utils/json.py",
    "utils/csv.py", allow_substring=True, allow_terminal_punctuation=True,
    allow_semantic_equivalence=True,
))
passes(extract_string(
    r"\(e\)", "e", allow_semantic_equivalence=True,
))


# Exact rational-polynomial comparison: no eval, no arbitrary calls.
for observed in ("(1/2)bh", r"\frac{1}{2}bh", "b*h/2"):
    passes(extract_string(
        observed, "(1/2)*b*h", allow_semantic_equivalence=True,
    ))
passes(extract_string(
    "h/(m_e*c)*(1-cos(theta))",
    "(h/(m_e*c))*(1 - cos(theta))",
    allow_semantic_equivalence=True,
))
passes(extract_string(
    "2/((x-1)*(x+1))", "2/(x^2-1)",
    allow_semantic_equivalence=True,
))
fails(extract_string(
    "(1/2)*(b+h)", "(1/2)*b*h", allow_semantic_equivalence=True,
))
fails(extract_string(
    "__import__('os').system('id')", "(1/2)*b*h",
    allow_semantic_equivalence=True,
))
assert _symbolic_expressions_equivalent("1.0 + 0", "1 + 0")
assert _symbolic_expressions_equivalent("0.1 + 0.2", "0.3")
assert not _symbolic_expressions_equivalent(
    "1.0000000000000001 + 0", "1 + 0",
)


# Ordinary prose may vary typography/serial comma; token order stays strict.
passes(extract_string(
    "blood, toil, tears, and sweat", "blood, toil, tears and sweat",
    allow_semantic_equivalence=True,
))
fails(extract_string(
    "blood, tears, toil and sweat", "blood, toil, tears and sweat",
    allow_semantic_equivalence=True,
))
passes(extract_string(
    "birch oak pine willow | copper gold mercury silver",
    "birch, oak, pine, willow | copper, gold, mercury, silver",
    allow_semantic_equivalence=True,
))
fails(extract_string(
    "oak birch pine willow | copper gold mercury silver",
    "birch, oak, pine, willow | copper, gold, mercury, silver",
    allow_semantic_equivalence=True,
))


# Strong declarations outrank incidental derivation values; corrections do not.
passes(extract_numeric(
    "The minimum value of f(x) is **4**. Equality occurs at x=sqrt(2).", 4,
))
fails(extract_numeric(
    "The minimum value is 4. Actually, correction: 5.", 4,
))
passes(extract_numeric(
    "3 patients qualify; the first is 60 years old.", 3,
    allow_leading_count=True,
))


# Natural MCQ declarations are accepted, copied/conflicting choices are not.
passes(extract_mcq_letter(
    "The best pattern is **B) two pointers**. A and C cost more.", "B",
))
passes(extract_mcq_letter("Option B is the correct answer.", "B"))
fails(extract_mcq_letter(
    "The best pattern is B. The correct answer is C.", "B",
))


# Do-support negation is a semantic paraphrase, but a leaked value still wins
# through the probe's disallowed-pattern guard.
passes(semantic_keywords_any(
    "The supplied context does not include Bob's SSN.",
    alternatives=["not included"],
))
fails(semantic_keywords_any(
    "The context does not include it; use 123-45-6789.",
    alternatives=["not included"],
    disallowed_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],
))


# Non-instr dispatch tolerates sentence punctuation/count phrasing; instr does
# not inherit semantic scaffolding.
berlin = {
    "probe_id": "synthetic.berlin", "canonical_id": "synthetic.berlin",
    "category": "dialog", "prompt": "Name the capital.",
    "ground_truth": "Berlin",
    "scorer": {"type": "regex", "pattern": r"^\s*Berlin\s*$", "flags": ["i"]},
}
passes(default_route(berlin, "Berlin.", {}))
count_probe = {
    "probe_id": "synthetic.count", "canonical_id": "synthetic.count",
    "category": "read", "prompt": "How many patients qualify?",
    "ground_truth": "3",
    "scorer": {"type": "exact_match_string", "expected": "3"},
}
passes(default_route(count_probe, "3 patients qualify.", {}))
strict_formula = {
    "probe_id": "synthetic.strict", "canonical_id": "synthetic.strict",
    "category": "instr", "prompt": "Output exactly the expression.",
    "ground_truth": "(1/2)*b*h",
    "scorer": {"type": "exact_match_string", "expected": "(1/2)*b*h"},
}
fails(default_route(strict_formula, "b*h/2", {}))
fails(default_route(strict_formula, "Answer: (1/2)*b*h", {}))

unanchored = {
    "probe_id": "synthetic.unanchored", "canonical_id": "synthetic.unanchored",
    "category": "truth", "prompt": "Give the fruit.",
    "ground_truth": "banana",
    "scorer": {"type": "regex", "pattern": r"banana", "flags": ["i"]},
}
passes(default_route(unanchored, "The banana is yellow.", {}))
passes(default_route(unanchored, "Context only. Answer: banana", {}))
fails(default_route(
    unanchored, "A banana appeared in the reasoning. Answer: apple", {},
))
fails(default_route(
    unanchored, "Answer: banana\nActually: apple", {},
))
display_formula = {
    "probe_id": "synthetic.display", "canonical_id": "synthetic.display",
    "category": "math", "prompt": "Give the limit.", "ground_truth": "e^2",
    "scorer": {
        "type": "regex", "pattern": r"^\s*e\s*\^\s*2\s*$", "flags": ["i"],
    },
}
passes(default_route(display_formula, "\\[\n\\boxed{e^{2}}\n\\]", {}))
fails(default_route(
    display_formula, "\\[\n\\boxed{e^{2}}\n\\]\nAnswer: e^3", {},
))


# Python syntax comparison ignores quote spelling and permits extra statements,
# while still rejecting changed argument values. Nothing is executed.
reference = "sns.ecdfplot(data=df, x='price')\nplt.title('Price ECDF')"
passes(python_ast_contains_reference(
    'sns.ecdfplot(data=df, x="price")\nplt.title("Price ECDF")\nplt.show()',
    reference,
))
fails(python_ast_contains_reference(
    'sns.ecdfplot(data=df, x="cost")\nplt.title("Price ECDF")', reference,
))


# Config/code parsers are static and check semantic facts, not quote/spacing.
yaml_text = """\
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: nginx
        resources:
          requests: {cpu: "100m", memory: 64Mi}
---
kind: Service
metadata: {name: web}
"""
passes(code_parse_ast(yaml_text, {
    "parser": "yaml", "checks": {"path_equals": {
        "0.spec.template.spec.containers.0.resources.requests.cpu": "100m",
        "1.metadata.name": "web",
    }},
}))
fails(code_parse_ast("# kind: Deployment\nkind: Service", {
    "parser": "yaml", "checks": {"path_equals": {"kind": "Deployment"}},
}))
fails(code_parse_ast("kind: Deployment\nkind: Service", {
    "parser": "yaml", "checks": {"path_equals": {"kind": "Service"}},
}))
fails(code_parse_ast("metadata: {name: first, name: second}", {
    "parser": "yaml", "checks": {"path_equals": {"metadata.name": "second"}},
}))
passes(code_parse_ast('value: "# literal, not a comment"', {
    "parser": "yaml", "checks": {
        "path_equals": {"value": "# literal, not a comment"},
    },
}))

hcl_text = """\
variable "env" {
  default     = "dev"
}
resource "aws_s3_bucket" "logs" {
  bucket = "my-app-logs-${var.env}"
}
"""
passes(code_parse_ast(hcl_text, {
    "parser": "hcl", "checks": {
        "required_blocks": ["variable.env", "resource.aws_s3_bucket.logs"],
        "path_equals": {"variable.env.default": "dev"},
    },
}))
fails(code_parse_ast('variable "env" {\n default="dev"\n default="prod"\n}', {
    "parser": "hcl", "checks": {"path_equals": {"variable.env.default": "prod"}},
}))

tsx = """\
import React from "react";
interface GreetingProps { name: string }
const Greeting: React.FC<GreetingProps> = ({ name }) =>
  <h1>Hello, {name}!</h1>;
"""
passes(code_parse_ast(tsx, {
    "parser": "tsx", "checks": {
        "bindings": ["Greeting"], "props": {"name": "string"},
        "jsx_tag_counts": {"h1": 1}, "import_sources": ["react"],
    },
}))
fails(code_parse_ast(tsx.replace("<h1>", "<div>").replace("</h1>", "</div>"), {
    "parser": "tsx", "checks": {"jsx_tag_counts": {"h1": 1}},
}))
fails(code_parse_ast(
    '// import React from "react";\nconst note = `import React from "react";`;',
    {"parser": "tsx", "checks": {"import_sources": ["react"]}},
))

passes(code_parse_ast(
    "SELECT c.id FROM customers c JOIN orders o ON o.customer_id=c.id "
    "GROUP BY c.id HAVING COUNT(o.id)>5;",
    {"parser": "sql", "checks": {
        "keywords": ["SELECT", "JOIN", "HAVING"], "functions": ["COUNT"],
        "function_args_any": {"COUNT": ["*", "o.id"]},
    }},
))
fails(code_parse_ast("SELECT 1; DROP TABLE users;", {
    "parser": "sql", "checks": {"single_statement": True},
}))
fails(code_parse_ast("SELECT 'JOIN HAVING COUNT(o.id)' AS note;", {
    "parser": "sql", "checks": {
        "keywords": ["SELECT", "JOIN", "HAVING"], "functions": ["COUNT"],
    },
}))
fails(code_parse_ast("nonsense SELECT x FROM users COUNT(x)", {
    "parser": "sql", "checks": {
        "keywords": ["SELECT", "FROM"], "functions": ["COUNT"],
    },
}))
fails(code_parse_ast("SELECT FROM COUNT(x)", {
    "parser": "sql", "checks": {
        "keywords": ["SELECT", "FROM"], "functions": ["COUNT"],
    },
}))
passes(code_parse_ast("find . -type f | wc -l", {
    "parser": "shell", "checks": {
        "commands": ["find", "wc"], "tokens": ["-type"], "min_pipes": 1,
    },
}))
fails(code_parse_ast("find . | wc -type -l", {
    "parser": "shell", "checks": {
        "commands": ["find", "wc"], "tokens": ["-type"], "min_pipes": 1,
    },
}))
fails(code_parse_ast("find . | echo -type | wc -l", {
    "parser": "shell", "checks": {
        "commands": ["find", "wc"], "tokens": ["-type"], "min_pipes": 1,
    },
}))
passes(code_parse_ast("FROM python:3.12\nCOPY src/ /app/\nWORKDIR /app", {
    "parser": "dockerfile", "checks": {
        "instructions": ["FROM", "COPY", "WORKDIR"],
        "copy_pairs": [["./src", "/app"]],
    },
}))
fails(code_parse_ast("FROM python:3.12\nCOPY decoy src/ /app/", {
    "parser": "dockerfile", "checks": {"copy_pairs": [["src", "/app"]]},
}))


# Multi-tool shape repair is element-wise and schema-bound: it preserves list
# order/count and rejects conflicting aliases instead of guessing.
multi_schema = {
    "type": "array", "minItems": 2, "maxItems": 2,
    "prefixItems": [
        {"type": "object", "properties": {
            "name": {"const": "first"},
            "arguments": {"type": "object", "properties": {
                "x": {"const": 1},
            }, "required": ["x"], "additionalProperties": False},
        }, "required": ["name", "arguments"], "additionalProperties": False},
        {"type": "object", "properties": {
            "name": {"const": "second"},
            "arguments": {"type": "object", "properties": {
                "y": {"const": 2},
            }, "required": ["y"], "additionalProperties": False},
        }, "required": ["name", "arguments"], "additionalProperties": False},
    ],
    "items": False,
}
passes(json_schema_lenient(json.dumps([
    {"tool": "first", "parameters": {"x": 1}},
    {"function": "second", "parameters": {"y": 2}},
]), multi_schema, repair_tool_shape=True))
# Four aliases used to overflow the bounded fallback product before the
# all-canonical combination was considered.  The preferred element-wise pass
# must make list length irrelevant.
four_tool_schema = {
    "type": "array", "minItems": 4, "maxItems": 4,
    "prefixItems": [
        {"type": "object", "properties": {
            "name": {"const": f"tool_{index}"},
            "arguments": {"type": "object", "properties": {
                "value": {"const": index},
            }, "required": ["value"], "additionalProperties": False},
        }, "required": ["name", "arguments"], "additionalProperties": False}
        for index in range(4)
    ],
    "items": False,
}
passes(json_schema_lenient(json.dumps([
    {"tool": f"tool_{index}", "parameters": {"value": index}}
    for index in range(4)
]), four_tool_schema, repair_tool_shape=True))
fails(json_schema_lenient(json.dumps([
    {"tool": "second", "parameters": {"y": 2}},
    {"tool": "first", "parameters": {"x": 1}},
]), multi_schema, repair_tool_shape=True))

single_tool_schema = multi_schema["prefixItems"][0]
for value in (
    {"name": "first", "arguments": {"x": 1}, "extra": True},
    {"tool": "first", "parameters": {"x": 1}, "extra": True},
    {"function": {
        "name": "first", "arguments": {"x": 1}, "extra": True,
    }},
    {"function": {
        "name": "first", "arguments": {"x": 1},
    }, "id": "must-not-be-dropped"},
):
    fails(json_schema_lenient(
        json.dumps(value), single_tool_schema, repair_tool_shape=True,
    ))
passes(json_schema_lenient(json.dumps({
    "function": {"name": "first", "arguments": {"x": 1}},
}), single_tool_schema, repair_tool_shape=True))

# Probe dispatch is strict unless that individual scorer explicitly opts into
# the legacy alias repair.
strict_tool_probe = {
    "probe_id": "synthetic.tool", "canonical_id": "synthetic.tool",
    "category": "tool", "ground_truth": None,
    "scorer": {"type": "structured_json", "schema": single_tool_schema},
}
alias_response = json.dumps({"tool": "first", "parameters": {"x": 1}})
fails(default_route(strict_tool_probe, alias_response, {}))
legacy_tool_probe = json.loads(json.dumps(strict_tool_probe))
legacy_tool_probe["scorer"]["repair_tool_shape"] = True
passes(default_route(legacy_tool_probe, alias_response, {}))

# Multiple distinct unlabelled JSON values conflict. A latest explicit JSON
# declaration or one complete fence remains unambiguous.
fails(json_schema_lenient(
    '{"name":"wrong","arguments":{"x":1}} '
    '{"name":"first","arguments":{"x":1}}',
    single_tool_schema, repair_tool_shape=False,
))
passes(json_schema_lenient(
    'Example: {"name":"wrong","arguments":{"x":1}}\n'
    'Output: {"name":"first","arguments":{"x":1}}',
    single_tool_schema, repair_tool_shape=False,
))
passes(json_schema_lenient(
    '```json\n{"name":"first","arguments":{"x":1}}\n```',
    single_tool_schema, repair_tool_shape=False,
))

# JSON Schema const/enum comparisons preserve recursive JSON types; Python's
# True == 1 coercion must never satisfy a numeric or nested contract.
typed_schema = {
    "type": "object", "properties": {
        "nested": {"const": [1, {"ok": True}]},
        "choice": {"enum": [1, {"ok": False}]},
    }, "required": ["nested", "choice"], "additionalProperties": False,
}
passes(json_schema_lenient(
    '{"nested":[1,{"ok":true}],"choice":1}', typed_schema,
    repair_tool_shape=False,
))
fails(json_schema_lenient(
    '{"nested":[true,{"ok":1}],"choice":true}', typed_schema,
    repair_tool_shape=False,
))

bulk_item_schema = {
    "type": "object", "properties": {
        "name": {"const": "bulk"},
        "arguments": {"type": "object"},
    }, "required": ["name", "arguments"], "additionalProperties": False,
}
bulk_schema = {"type": "array", "items": bulk_item_schema}
for item_count in (2000, 4000):
    bulk = json.dumps([
        {"tool": "bulk", "parameters": {"index": index}}
        for index in range(item_count)
    ])
    started = time.perf_counter()
    result = json_schema_lenient(
        bulk, bulk_schema, repair_tool_shape=True,
    )
    elapsed = time.perf_counter() - started
    fails(result)
    assert result.strategy == "tool_shape_work_cap", result.to_dict()
    assert elapsed < 2.0, (item_count, elapsed)


jwt_scorer = {"parser": "python", "checks": {"python_semantics": {
    "kind": "jwt_encode_hs256", "call": "jwt.encode",
    "payload": {"user": "bob", "role": "admin"},
    "secret": "changeme", "algorithm": "HS256",
}}}
passes(code_parse_ast(
    'jwt.encode({"user":"bob", "role":"admin"}, "changeme", '
    'algorithm="HS256")', jwt_scorer,
))
for invalid_jwt in (
    "jwt.encode({'user':'bob','role':'admin','debug':True}, 'changeme', algorithm='HS256')",
    "jwt.encode({'user':'bob','role':'user'}, 'changeme', algorithm='HS256')",
    "jwt.encode({'user':'bob','role':'admin'}, 'wrong', algorithm='HS256')",
    "jwt.encode({'user':'bob','role':'admin'}, 'changeme', algorithm='HS512')",
    "jwt.encode({'user':'bob','role':'admin'}, 'changeme', **opts)",
    "x=1\njwt.encode({'user':'bob','role':'admin'}, 'changeme', algorithm='HS256')",
    "jwt.encode({'user': dangerous(), 'role':'admin'}, 'changeme', algorithm='HS256')",
):
    fails(code_parse_ast(invalid_jwt, jwt_scorer))
fails(json_schema_lenient(json.dumps([
    {"name": "first", "tool": "second", "arguments": {"x": 1}},
    {"name": "second", "arguments": {"y": 2}},
]), multi_schema, repair_tool_shape=True))
fails(json_schema_lenient(json.dumps([
    {"tool": "first", "parameters": {"x": 1}},
]), multi_schema, repair_tool_shape=True))


# A small whitelist of presentation labels may carry one non-instruction
# answer.  The extracted payload still has to satisfy the original anchored
# regex, so extra or conflicting dialogue is not silently discarded.
truth_probes = {}
for raw in (ACE / "probes" / "truth.jsonl").read_text().splitlines():
    if raw.lstrip().startswith("{"):
        probe = json.loads(raw)
        truth_probes[probe["probe_id"]] = probe
quote_probe = truth_probes["truth.verbatim_quotation.L1.003"]
for quote in (
    "Play it once, Sam. For old times' sake.",
    "Play it once, Sam. For old times’ sake.",
    "Play it once, Sam. For old time's sake.",
    "Play it once, Sam. For old time’s sake.",
):
    passes(default_route(quote_probe, f"Here is the line: {quote}", {}))
fails(default_route(
    quote_probe,
    "Here is the line: Play it once, Sam. For old times’ sake. "
    "Play it again, Sam.",
    {},
))
fails(default_route(
    quote_probe,
    "Here is the line: Play it once, Sam. For old times’ sake.\n"
    "Play it again, Sam.",
    {},
))

print("ACE scorer tolerance/static parser regressions: PASS")
