"""
ACE atom-aware scorer suite.

Design principles:

1. **Intent over format**. Each scorer is named by what it MEASURES (capability),
   not what regex it runs. An atom like `arith.percent_of` has intent
   `extract_numeric` — we look for the answer wherever the model put it,
   instead of pretending the model must echo the answer alone.

2. **No naive substring traps**. We never use `"correct" in resp` style checks
   that match `incorrect`. Token-aware matching with word boundaries.

3. **IF (instruction following) is opt-in**. Strict format scoring is reserved
   for atoms in the `instr.*` category and probes explicitly tagged
   `tests_format=true`. Non-IF atoms tolerate preamble/CoT/reasoning chatter.

4. **Multi-strategy fallback**. Each smart scorer tries strategies in order
   (most-strict to most-lenient) and returns the strategy that matched.
   Reviewers can audit the strategy list to see WHY a probe passed.

5. **No SIGALRM in worker threads**. Caller must run in main thread or use
   process-level timeouts.

6. **Defensive caps**. Inputs longer than 1MB are truncated for the scorer
   (scoring should not depend on the long tail). Pathological regex inputs
   are bounded to 200KB before regex.

Scorers exposed:
    extract_numeric(resp, expected, tolerance=0)        → (passed, info)
    extract_string(resp, expected, *, case_sensitive=False, allow_substring=False)
    extract_mcq_letter(resp, expected, *, require_bare=False)
    extract_set(resp, expected_set, *, separator_hint=",")
    extract_code(resp)                                  → returns the code only
    extract_json(resp)                                  → returns parsed dict or None
    
    semantic_keywords_any(resp, alternatives)           → for abstention/refusal
    semantic_keywords_all(resp, required, *, allow_synonyms_table=None)
    semantic_format(resp, scorer)                       → strict, IF-only

    code_structural(resp, scorer)                       → AST + tests
    json_schema_lenient(resp, scorer)                   → schema with shape repair

The runner picks the right scorer based on atom intent (see
policy/atom_intents.yaml) and the scorer type declared in the probe.
"""
from __future__ import annotations
import json, re, ast, subprocess, sys, tempfile, os, signal, resource, shlex, io, tokenize, math
from dataclasses import dataclass, field
from pathlib import Path
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any
import unicodedata

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 deployment compatibility
    import tomli as tomllib

MAX_RESP_BYTES = 200_000  # used for regex/string operations
MAX_RESP_BYTES_HARD = 1_000_000  # hard truncation


# ----------------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------------
@dataclass
class ScoreResult:
    passed: bool
    strategy: str = ""           # which fallback strategy succeeded
    extracted: Any = None        # what we extracted from the response
    expected: Any = None
    note: str = ""               # human-readable explanation
    deferred: bool = False       # e.g. unavailable external judge

    def to_dict(self):
        d = {"passed": self.passed, "strategy": self.strategy}
        if self.extracted is not None: d["extracted"] = self.extracted
        if self.expected is not None: d["expected"] = self.expected
        if self.note: d["note"] = self.note
        if self.deferred: d["deferred"] = True
        return d


def code_parse_ast(resp: str, scorer: dict) -> ScoreResult:
    """Static-only parsing checks. Never compiles or executes submitted code."""
    parser = scorer.get("parser")
    checks = scorer.get("checks", {})
    raw_text = _sanitize(resp).strip()
    if checks.get("require_single_fenced_code"):
        fence = re.fullmatch(
            r"\x60{3}python[ \t]*\r?\n([\s\S]+?)\r?\n\x60{3}",
            raw_text,
            flags=re.IGNORECASE,
        )
        if not fence:
            return ScoreResult(False, "ast_requirement_failed", note="expected_single_python_fence")
        text = fence.group(1)
    else:
        fenced = _fenced_block_candidates(raw_text)
        # Non-instruction code tasks tolerate one clearly delimited answer
        # block plus harmless prose.  Multiple blocks stay unresolved so a
        # copied example cannot silently outrank a later correction.
        if len(fenced) == 1:
            text = fenced[0][3]
        else:
            text = re.sub(
                r"^\s*\x60{3}[A-Za-z0-9_+-]*\s*\n?|\n?\s*\x60{3}\s*$", "", raw_text
            )
        # A single inline-code wrapper is a transport convention, not Python
        # syntax.  Strip it only when it encloses the entire one-line answer;
        # embedded/backtick-rich prose remains unmodified and cannot become
        # executable-looking code accidentally.
        inline_code = re.fullmatch(r"`([^`\r\n]+)`", text.strip())
        if inline_code:
            text = inline_code.group(1).strip()
    # Static parser probes ask for compact source/configuration artifacts.  A
    # very large selected artifact is both non-responsive and an avoidable
    # parser DoS surface.  Apply this after safe fenced-answer extraction so
    # non-instruction scaffolding outside one unambiguous code block remains
    # tolerated, as it is elsewhere in ACE.
    if len(text) > MAX_RESP_BYTES:
        return ScoreResult(
            False,
            "parse_input_too_long",
            note=f"source_length={len(text)}, max={MAX_RESP_BYTES}",
        )
    try:
        if parser == "python":
            tree = ast.parse(text)
            imports, calls, constants = set(), set(), set()
            functions, classes = {}, {}
            def dotted(n):
                if isinstance(n, ast.Name): return n.id
                if isinstance(n, ast.Attribute):
                    p = dotted(n.value)
                    return f"{p}.{n.attr}" if p else n.attr
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    base = "." * node.level + (node.module or "")
                    imports.update(f"{base}.{n.name}" for n in node.names)
                elif isinstance(node, ast.Call) and dotted(node.func):
                    calls.add(dotted(node.func))
                elif isinstance(node, ast.Constant):
                    try: constants.add(node.value)
                    except TypeError: pass
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions[node.name] = node
                elif isinstance(node, ast.ClassDef):
                    classes[node.name] = node
            expected_imports = set(checks.get("imports", []))
            if not expected_imports.issubset(imports):
                return ScoreResult(False, "ast_requirement_failed", note=f"missing_imports={sorted(expected_imports-imports)}")
            for prefix in checks.get("import_prefixes", []):
                if not any(name == prefix or name.startswith(prefix + ".") for name in imports):
                    return ScoreResult(False, "ast_requirement_failed",
                                       note=f"missing_import_prefix={prefix}")
            if checks.get("only_imports") and any(not isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body):
                return ScoreResult(False, "ast_requirement_failed", note="non_import_top_level_statement")
            fn = checks.get("function")
            scoped_nodes = ast.walk(functions[fn]) if fn in functions else ast.walk(tree)
            scoped_calls, scoped_constants = set(), set()
            for node in scoped_nodes:
                if isinstance(node, ast.Call) and dotted(node.func):
                    scoped_calls.add(dotted(node.func))
                elif isinstance(node, ast.Constant):
                    try: scoped_constants.add(node.value)
                    except TypeError: pass
            expected_calls = set(checks.get("calls", []))
            missing_calls = {name for name in expected_calls
                             if not any(actual == name or actual.endswith("." + name) for actual in scoped_calls)}
            if missing_calls:
                return ScoreResult(False, "ast_requirement_failed", note=f"missing_calls={sorted(missing_calls)}")
            if not set(checks.get("constants", [])).issubset(scoped_constants):
                return ScoreResult(False, "ast_requirement_failed", note="missing_constants")
            if fn and fn not in functions:
                return ScoreResult(False, "ast_requirement_failed", note=f"missing_function={fn}")
            cls = checks.get("class")
            if cls:
                node = classes.get(cls)
                if not node: return ScoreResult(False, "ast_requirement_failed", note=f"missing_class={cls}")
                bases = {ast.unparse(x) for x in node.bases}
                if checks.get("base") and checks["base"] not in bases:
                    return ScoreResult(False, "ast_requirement_failed", note=f"missing_base={checks['base']}")
                fields = {n.target.id: ast.unparse(n.annotation) for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
                for name, ann in checks.get("annotated_fields", {}).items():
                    if fields.get(name) != ann:
                        return ScoreResult(False, "ast_requirement_failed", note=f"field={name}, expected={ann}, got={fields.get(name)}")
            assign = checks.get("assignment_call")
            if assign:
                if checks.get("only_top_level_statement") and len(tree.body) != 1:
                    return ScoreResult(False, "ast_requirement_failed",
                                       note="expected_one_top_level_statement")
                ok = False
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == assign["target"] for t in node.targets):
                        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == assign["call"]:
                            args = [ast.literal_eval(a) for a in node.value.args]
                            ok = args == assign.get("args", args)
                if not ok: return ScoreResult(False, "ast_requirement_failed", note="assignment_call_mismatch")
            returned_object = checks.get("returned_object")
            if returned_object:
                scope = functions.get(fn) if fn else None
                returned_names = {
                    node.value.id for node in ast.walk(scope or tree)
                    if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
                }
                constructor = returned_object.get("constructor")
                initialized = set()
                called_methods: dict[str, set[str]] = {}
                for node in ast.walk(scope or tree):
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                        call_name = dotted(node.value.func)
                        if call_name and (call_name == constructor or call_name.endswith("." + constructor)):
                            initialized.update(
                                target.id for target in node.targets if isinstance(target, ast.Name)
                            )
                    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                          and isinstance(node.func.value, ast.Name)):
                        called_methods.setdefault(node.func.value.id, set()).add(node.func.attr)
                required_methods = set(returned_object.get("method_calls", []))
                candidates = returned_names & initialized
                if not any(required_methods.issubset(called_methods.get(name, set())) for name in candidates):
                    return ScoreResult(False, "ast_requirement_failed", note="returned_object_mismatch")
            if checks.get("python_semantics"):
                semantic_spec = checks["python_semantics"]
                if semantic_spec.get("kind") == "jwt_encode_hs256":
                    ok, note = _validate_jwt_encode_hs256_ast(tree, semantic_spec)
                else:
                    from eval.static_python_semantics import validate_python_semantics_ast
                    ok, note = validate_python_semantics_ast(tree, semantic_spec)
                if not ok:
                    return ScoreResult(False, "python_semantics_requirement_failed", note=note)
                return ScoreResult(True, "python_semantics_passed")
            if checks.get("ooxml"):
                from eval.static_ooxml import validate_ooxml_ast
                ok, note, extracted = validate_ooxml_ast(tree, fn or "", checks["ooxml"])
                if not ok:
                    return ScoreResult(False, "ooxml_static_requirement_failed", note=note)
                return ScoreResult(True, "ooxml_static_passed", extracted=extracted)
            return ScoreResult(True, "python_ast_passed")
        if parser == "regex":
            if len(text) > int(checks.get("max_pattern_length", 500)):
                return ScoreResult(False, "parse_requirement_failed", note="regex_pattern_too_long")
            pattern = re.compile(text)
            for sample in checks.get("fullmatch", []):
                if pattern.fullmatch(sample) is None:
                    return ScoreResult(False, "parse_requirement_failed",
                                       note=f"regex_should_fullmatch={sample!r}")
            for sample in checks.get("reject", []):
                if pattern.fullmatch(sample) is not None:
                    return ScoreResult(False, "parse_requirement_failed",
                                       note=f"regex_should_reject={sample!r}")
            return ScoreResult(True, "regex_behavior_passed")
        if parser in {"json", "toml", "yaml"}:
            if parser == "json":
                obj = json.loads(text)
            elif parser == "toml":
                obj = tomllib.loads(text)
            else:
                obj = _parse_yaml_subset(text)
            for path, expected in checks.get("path_equals", {}).items():
                cur = _static_path_get(obj, path)
                if cur != expected: return ScoreResult(False, "parse_requirement_failed", note=f"path={path}")
            for path, expected in checks.get("path_set_equals", {}).items():
                cur = _static_path_get(obj, path)
                if not isinstance(cur, list) or set(cur) != set(expected):
                    return ScoreResult(False, "parse_requirement_failed", note=f"path_set={path}")
            return ScoreResult(True, f"{parser}_parse_passed")
        if parser == "hcl":
            obj = _parse_hcl_subset(text)
            for path, expected in checks.get("path_equals", {}).items():
                if _static_path_get(obj, path) != expected:
                    return ScoreResult(False, "parse_requirement_failed", note=f"path={path}")
            for path, expected in checks.get("path_any_equals", {}).items():
                if expected not in _static_path_values(obj, path):
                    return ScoreResult(False, "parse_requirement_failed",
                                       note=f"path_any={path}")
            required_blocks = set(checks.get("required_blocks", []))
            missing_blocks = required_blocks - set(obj.get("__blocks__", []))
            if missing_blocks:
                return ScoreResult(False, "parse_requirement_failed",
                                   note=f"missing_blocks={sorted(missing_blocks)}")
            return ScoreResult(True, "hcl_parse_passed")
        if parser in {"tsx", "typescript", "javascript"}:
            return _validate_ecmascript_structure(text, checks, parser=parser)
        if parser == "source_contract":
            contract_text = text
            literal_text = text
            string_literals: list[str] = []
            source_language = checks.get("source_language")
            if source_language == "python":
                # Parsing here is still static-only.  Besides rejecting comment
                # and string stuffing, it gives this explicitly Python contract
                # the same basic syntax guarantee as parser="python" probes.
                ast.parse(text)
                contract_text, literal_text, string_literals = _mask_python_contract(text)
            elif source_language in {"c", "cpp", "qsharp", "c_like"}:
                contract_text, string_literals = _mask_ecmascript(text)
                literal_text = _mask_ecmascript_comments(text)
            elif source_language == "rust":
                # Rust lifetimes such as ``Option<&'a Node>`` are not quoted
                # character literals.  A JS-style masker therefore mistakes
                # the apostrophe for an unterminated string.  Rust contracts
                # only need comments and double-quoted strings masked.
                contract_text, literal_text, string_literals = _mask_delimited_contract(
                    text, line_markers=("//",),
                    block_markers=(("/*", "*/"),), quote_chars='"',
                )
            elif source_language == "haskell":
                contract_text, literal_text, string_literals = _mask_delimited_contract(
                    text, line_markers=("--",), block_markers=(("{-", "-}"),),
                    quote_chars='"',
                )
            elif source_language == "ocaml":
                contract_text, literal_text, string_literals = _mask_delimited_contract(
                    text, block_markers=(("(*", "*)"),), quote_chars='"',
                )
            elif source_language in {"shell", "docker_shell"}:
                contract_text, literal_text, string_literals = _mask_delimited_contract(
                    text, line_markers=("#",), quote_chars="'\"",
                    preserve_shebang=True,
                )
            if checks.get("balanced_delimiters"):
                if not _balanced_static_delimiters(contract_text):
                    return ScoreResult(False, "parse_failed", note="unbalanced_delimiters")
            if checks.get("csharp_max_contract"):
                error = _validate_csharp_max_contract(contract_text)
                if error:
                    return ScoreResult(False, "source_contract_failed", note=error)
            for literal in checks.get("string_literals", []):
                if literal not in string_literals:
                    return ScoreResult(False, "source_contract_failed",
                                       note=f"missing_string_literal={literal!r}")
            return _validate_static_source_contract(
                text, checks, code_text=contract_text, literal_text=literal_text,
            )
        if parser == "sql":
            return _validate_sql_structure(text, checks)
        if parser == "shell":
            return _validate_shell_structure(text, checks)
        if parser == "dockerfile":
            return _validate_dockerfile_structure(text, checks)
        if parser == "css":
            m = re.fullmatch(r"\s*([^{}]+)\{([^{}]*)\}\s*", text, re.S)
            if not m: return ScoreResult(False, "parse_failed", note="expected_one_css_rule")
            selector, declarations = m.group(1).strip(), {}
            for raw in m.group(2).split(";"):
                if not raw.strip(): continue
                if ":" not in raw: return ScoreResult(False, "parse_failed", note="malformed_declaration")
                k, v = raw.split(":", 1); declarations[k.strip().lower()] = re.sub(r"\s+", " ", v.strip().lower())
            if selector != checks.get("selector"): return ScoreResult(False, "parse_requirement_failed", note="selector")
            for k, v in checks.get("declarations", {}).items():
                if declarations.get(k) != v: return ScoreResult(False, "parse_requirement_failed", note=f"declaration={k}")
            return ScoreResult(True, "css_parse_passed")
        return ScoreResult(False, "unsupported_parser", note=str(parser))
    except Exception as e:
        return ScoreResult(False, "parse_failed", note=f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------------
def _sanitize(resp: str) -> str:
    if not isinstance(resp, str):
        resp = str(resp)
    if len(resp) > MAX_RESP_BYTES_HARD:
        resp = resp[:MAX_RESP_BYTES_HARD]
    return resp


_NONINSTR_EDGE_CONTROLS = "\ufeff\u200b\u2060"


def normalize_noninstr_transport(
        resp: str, *, repair_unclosed_outer_fence: bool = True) -> str:
    """Normalize transport-only differences for non-instruction scorers.

    The helper is intentionally smaller than a semantic text normalizer.  It
    canonicalizes API/OS newline spellings and removes boundary BOM/zero-width
    transport characters, but it does not collapse internal whitespace,
    case-fold text, or unwrap Markdown.  Those operations can change code,
    structured data, or an instruction-following contract and therefore stay
    in the individual answer extractors below.

    ``score_probe`` calls this helper only outside the ``instr`` category.
    Direct strict scorer calls consequently retain their existing byte/format
    behaviour as well.
    """
    text = _sanitize(resp).replace("\r\n", "\n").replace("\r", "\n")
    # ``str.strip`` handles Unicode whitespace but deliberately does not
    # remove BOM, ZERO WIDTH SPACE, or WORD JOINER.  Strip the two boundary
    # classes to a fixed point without touching the same characters inside an
    # answer, where they can be semantically significant.
    while True:
        previous = text
        text = text.strip().strip(_NONINSTR_EDGE_CONTROLS).strip()
        if text == previous:
            break
    # Recover one uniquely identifiable outer Markdown fence whose closing
    # delimiter alone was truncated.  Completing it here keeps the behaviour
    # non-instruction-only; shared wrapper helpers and strict scorers retain
    # their prior treatment of malformed fences.
    if repair_unclosed_outer_fence:
        unclosed = re.fullmatch(
            r"(?P<fence>```|~~~)[A-Za-z0-9_+.-]*[ \t]*\n(?P<body>[\s\S]+)",
            text,
        )
        if (unclosed and not re.search(
                r"(?m)^[ \t]*(?:```|~~~)", unclosed.group("body"))):
            text = text + "\n" + unclosed.group("fence")
    return text


def _strip_markdown_wrappers(s: str, *,
                             strip_terminal_punctuation: bool = True) -> str:
    """Strip outer code-fences, bold, italics, quotes, backticks."""
    s = s.strip()
    # Outer code fence
    m = re.match(r"^```(?:\w+)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL)
    if m: s = m.group(1).strip()
    # Outer bold/italic
    while True:
        m2 = re.match(r"^\*\*(.+)\*\*$", s, re.DOTALL) or \
             re.match(r"^__(.+)__$", s, re.DOTALL) or \
             re.match(r"^\*(.+)\*$", s, re.DOTALL) or \
             re.match(r"^_(.+)_$", s, re.DOTALL)
        if m2:
            s = m2.group(1).strip()
        else:
            break
    # Outer quotes / backticks. Terminal punctuation is a separate opt-in
    # semantic normalization; exact-string callers must not silently turn
    # `token.` into `token`.
    s = re.sub(r"^[`\"'\u201c\u2018\u300c\u300e]+", "", s)
    trailing = r"[`\"'\u201d\u2019\u300d\u300f.!?,;:]+$" if strip_terminal_punctuation \
        else r"[`\"'\u201d\u2019\u300d\u300f]+$"
    s = re.sub(trailing, "", s)
    return s.strip()


def _last_meaningful_line(resp: str, *,
                          strip_terminal_punctuation: bool = True) -> str:
    """Last non-empty line, with markdown wrappers stripped."""
    lines = [l.strip() for l in resp.strip().split("\n") if l.strip()]
    if not lines: return ""
    return _strip_markdown_wrappers(
        lines[-1], strip_terminal_punctuation=strip_terminal_punctuation,
    )


def _nonempty_lines_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """Return non-empty physical lines as ``(start, end, raw_line)`` tuples."""
    out = []
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        if raw.strip():
            out.append((offset, offset + len(raw), raw))
        offset += len(line)
    # ``splitlines(keepends=True)`` returns nothing for an empty string and
    # already includes a final unterminated line when present.
    return out


def _strip_answer_candidate_wrappers(text: str, *,
                                     strip_terminal_punctuation: bool = False) -> str:
    """Strip harmless Markdown that may be shared with an answer label.

    ``**Answer: False**`` leaves only a trailing ``**`` after the marker; the
    general wrapper stripper cannot remove that unmatched suffix, so answer
    candidates need this small additional normalization.
    """
    value = text.strip()
    value = re.sub(r"^[*_`]+[ \t]*", "", value)
    value = re.sub(r"[ \t]*[*_`]+$", "", value)
    value = _strip_markdown_wrappers(
        value, strip_terminal_punctuation=strip_terminal_punctuation,
    )
    # In ``**TF**.`` the terminal period initially prevents the outer bold
    # wrapper from being recognized.  Once punctuation is removed, make one
    # more bounded wrapper pass; do not recursively edit internal Markdown.
    value = re.sub(r"^[*_`]+[ \t]*", "", value)
    value = re.sub(r"[ \t]*[*_`]+$", "", value)
    value = _strip_markdown_wrappers(
        value, strip_terminal_punctuation=strip_terminal_punctuation,
    )
    latex_text = re.fullmatch(
        r"\\(?:text|mathrm|operatorname)[ \t]*\{([\s\S]*)\}", value,
    )
    if latex_text:
        value = _strip_markdown_wrappers(
            latex_text.group(1),
            strip_terminal_punctuation=strip_terminal_punctuation,
        )
    return _unwrap_math_presentation(value)


_ANSWER_MARKER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[*_`]*[ \t]*)?"
    r"(?:function[ \t]+to[ \t]+use|selected[ \t]+function|"
    r"here[ \t]+is[ \t]+(?:the[ \t]+)?(?:answer|line|quote|translation|result|response|output)|"
    r"(?:the[ \t]+)?correct[ \t]+(?:answer|choice|evaluation|result)|"
    r"final[ \t]+(?:choice|answer|evaluation|result)|the[ \t]+answer|"
    r"answer|choice|option|response|reply|output|result)"
    r"(?:[ \t]*[*_`]*)[ \t]*(?:is[ \t]*:|is|:|=|[-\u2013\u2014])[ \t]*",
    re.IGNORECASE,
)


_ANSWER_HEADING_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:[*_`]+[ \t]*)?"
    r"(?:final[ \t]+(?:answer|choice|evaluation|result)|"
    r"(?:the[ \t]+)?correct[ \t]+(?:answer|choice|evaluation|result)|"
    r"answer|result|response|output)"
    r"(?:[ \t]*[*_`]+)?[ \t]*:?[ \t]*(?:#{1,6})?[ \t]*$",
)


_TAGGED_ANSWER_OPEN_RE = re.compile(
    r"<(answer|final_answer|final|result|response|output)[ \t]*>",
    re.IGNORECASE,
)
_TAGGED_ANSWER_TOKEN_RE = re.compile(
    r"<(/?)(answer|final_answer|final|result|response|output)[ \t]*>",
    re.IGNORECASE,
)


def _answer_marker_candidates(text: str) -> list[tuple[int, str]]:
    """Return explicit answer candidates without letting whitespace cross lines.

    A marker with same-line payload yields that payload.  A marker-only line
    (including ``**Final Answer:**``) uses the next non-empty physical line.
    Positions point to the payload and let callers choose the latest explicit
    answer relative to a later boxed or bare final-line answer.
    """
    text = _sanitize(text)
    lines = _nonempty_lines_with_offsets(text)
    candidates = []
    for marker in _ANSWER_MARKER_RE.finditer(text):
        line_end = text.find("\n", marker.end())
        if line_end < 0:
            line_end = len(text)
        tail = text[marker.end():line_end].rstrip("\r")
        if tail.strip(" \t*_`"):
            leading = len(tail) - len(tail.lstrip())
            candidates.append((marker.end() + leading, tail.strip()))
            continue
        for start, _, raw_line in lines:
            if start >= line_end + (1 if line_end < len(text) else 0):
                leading = len(raw_line) - len(raw_line.lstrip())
                candidates.append((start + leading, raw_line.strip()))
                break
    # Markdown headings such as ``### Final Answer`` are a common transport
    # convention even when no colon follows the label.  Requiring the entire
    # physical line to be the heading prevents prose mentions of "the final
    # answer" from promoting the next line to an answer.
    for marker in _ANSWER_HEADING_LINE_RE.finditer(text):
        line_end = marker.end()
        for start, _, raw_line in lines:
            if start >= line_end + (1 if line_end < len(text) else 0):
                leading = len(raw_line) - len(raw_line.lstrip())
                candidates.append((start + leading, raw_line.strip()))
                break
    # A decorated marker can be recognized by both patterns.  Keep one
    # candidate per exact payload position so precedence remains deterministic.
    return sorted(set(candidates), key=lambda item: item[0])


def _tagged_answer_candidates(text: str) -> list[tuple[int, str]]:
    """Return payloads from a small allowlist of paired answer tags.

    Arbitrary XML/HTML is not parsed.  A tag must be paired, attribute-free,
    and explicitly answer-like, which covers model transport such as
    ``<result>42</result>`` without treating an arbitrary element in prose or
    code as an answer.  Callers still validate the payload with their original
    numeric/string/regex contract.
    """
    text = _sanitize(text)
    candidates = []
    opening: tuple[str, int] | None = None
    nested = False
    # One token scan is linear even for a response made entirely of unmatched
    # ``<output>`` openers.  Nested/mismatched answer tags are deliberately not
    # parsed recursively; the complete outer region is treated as ambiguous.
    for token in _TAGGED_ANSWER_TOKEN_RE.finditer(text):
        closing = bool(token.group(1))
        tag = token.group(2).lower()
        if not closing:
            if opening is None:
                opening = (tag, token.end())
                nested = False
            else:
                nested = True
            continue
        if opening is None:
            continue
        opening_tag, body_start = opening
        if tag != opening_tag:
            nested = True
            continue
        if not nested:
            candidates.append((body_start, text[body_start:token.start()].strip()))
        opening = None
        nested = False
    return candidates


def _mask_answer_tag_regions(text: str) -> str:
    """Mask answer-tag regions while preserving offsets and physical lines.

    Ordinary number/string fallbacks must not rediscover a tag payload after
    tag authority or negation has rejected it.  Unclosed/nested allowlisted
    tags are masked through the end because they are malformed transport, not
    bare answer text.
    """
    text = _sanitize(text)
    chars = list(text)
    opening_start: int | None = None
    opening_tag: str | None = None
    for token in _TAGGED_ANSWER_TOKEN_RE.finditer(text):
        closing = bool(token.group(1))
        tag = token.group(2).lower()
        if not closing:
            if opening_start is None:
                opening_start = token.start()
                opening_tag = tag
            continue
        if opening_start is None or tag != opening_tag:
            continue
        for index in range(opening_start, token.end()):
            if chars[index] not in "\r\n":
                chars[index] = " "
        opening_start = None
        opening_tag = None
    if opening_start is not None:
        for index in range(opening_start, len(chars)):
            if chars[index] not in "\r\n":
                chars[index] = " "
    return "".join(chars)


def _tagged_answer_is_negated(text: str, body_start: int) -> bool:
    """Detect a local negator immediately before an allowlisted answer tag."""
    prefix = _sanitize(text)[max(0, body_start - 96):body_start]
    return bool(re.search(
        r"(?i)(?:\bnot|\bno|\bnever|\bincorrect(?:ly)?|"
        r"不是|并非|非)[ \t]*"
        r"<(?:answer|final_answer|final|result|response|output)[ \t]*>[ \t]*$",
        prefix,
    ))


def _is_complete_tagged_answer(text: str) -> bool:
    """Whether ``text`` consists of exactly one non-nested paired answer tag."""
    stripped = _sanitize(text).strip()
    tokens = list(_TAGGED_ANSWER_TOKEN_RE.finditer(stripped))
    if len(tokens) != 2:
        return False
    opening, closing = tokens
    return bool(
        opening.start() == 0
        and not opening.group(1)
        and closing.group(1) == "/"
        and opening.group(2).lower() == closing.group(2).lower()
        and closing.end() == len(stripped)
    )


_NONANSWER_LABEL_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:reference|"
    r"example(?:[ \t]+(?:xml|answer|output|result))?|"
    r"expected(?:[ \t]+answer)?|"
    r"ground[ \t]+truth|candidate)[ \t]*:",
)


def _position_has_nonanswer_label(text: str, position: int) -> bool:
    """Whether the closest same-line label before ``position`` is non-answer.

    This prevents presentation in ``Reference: **Berlin**`` or
    ``Example: <result>42</result>`` from overriding an earlier declared
    answer.  If a later affirmative answer marker occurs on the same line, it
    regains authority.
    """
    text = _sanitize(text)
    line_start = text.rfind("\n", 0, max(0, position)) + 1
    prefix = text[line_start:position]
    nonanswer = list(_NONANSWER_LABEL_RE.finditer(prefix))
    if not nonanswer:
        return False
    answer = list(_ANSWER_MARKER_RE.finditer(prefix))
    return not answer or nonanswer[-1].start() > answer[-1].start()


def _position_has_nonanswer_block_label(text: str, position: int) -> bool:
    """Recognize ``Example:``/``Reference:`` immediately above a block."""
    if _position_has_nonanswer_label(text, position):
        return True
    prefix_lines = _sanitize(text)[:max(0, position)].splitlines()
    # The body position of a fenced candidate follows its opening-fence line.
    while prefix_lines and (
            not prefix_lines[-1].strip()
            or re.fullmatch(
                r"[ \t]*(?:```|~~~)[A-Za-z0-9_+.-]*[ \t]*",
                prefix_lines[-1],
            )):
        prefix_lines.pop()
    if not prefix_lines:
        return False
    line = prefix_lines[-1].strip()
    line = re.sub(r"^(?:#{1,6}|[-*>])[ \t]*", "", line)
    return bool(_NONANSWER_LABEL_RE.fullmatch(line))


def _weak_presentation_allowed(
        text: str, position: int,
        strong_candidates: list[tuple[int, str]] | None) -> bool:
    """Whether a box/math wrapper belongs to the latest strong answer marker."""
    strong = sorted(strong_candidates or [], key=lambda item: item[0])
    if not strong:
        return True
    payload_start, payload = strong[-1]
    line_end = _sanitize(text).find("\n", payload_start)
    if line_end < 0:
        line_end = len(_sanitize(text))
    if payload_start <= position <= line_end:
        return True
    # A marker-only heading points to its first presentation opener; wrappers
    # inside that immediately-declared display are part of the same payload.
    return _answer_payload_is_presentation_opener(payload)


def _answer_payload_is_presentation_opener(payload: str) -> bool:
    """Whether a marker payload introduces, rather than states, its answer."""
    value = str(payload).strip()
    if re.fullmatch(
            r"(?:\$\$?|\\\[|\\\(|```[A-Za-z0-9_+.-]*|"
            r"~~~[A-Za-z0-9_+.-]*)",
            value):
        return True
    return bool(
        len(value) <= 200
        and len(re.findall(r"\S+", value)) <= 24
        and re.search(
            r"(?i)\b(?:is|are|follows?|below)[ \t]*:[ \t]*$", value,
        )
    )


def _authoritative_tagged_answer_candidates(
        text: str,
        strong_candidates: list[tuple[int, str]] | None = None,
        ) -> list[tuple[int, str]]:
    """Return only tags that are themselves in an answer position.

    Answer-like XML is common model transport, but a tag in arbitrary prose is
    not an answer declaration.  With an existing strong marker/correction,
    only a tag inside that latest marker's physical payload is eligible.  With
    no strong marker, one whole-response tag or a standalone final-line tag is
    eligible.  Local negation is retained for callers to reject explicitly.
    """
    text = _sanitize(text)
    raw = _tagged_answer_candidates(text)
    if not raw:
        return []

    strong = sorted(strong_candidates or [], key=lambda item: item[0])
    if strong:
        payload_start, _ = strong[-1]
        payload_line_end = text.find("\n", payload_start)
        if payload_line_end < 0:
            payload_line_end = len(text)
        payload_text = text[payload_start:payload_line_end].strip()
        payload_text = payload_text.strip("*_` \t")
        payload_text = payload_text.rstrip(".!?,;: \t")
        # A tag embedded after another answer token is not the marker payload:
        # ``Final answer: Paris <result>Berlin</result>`` remains Paris.
        if not _is_complete_tagged_answer(payload_text):
            return []
        return [
            (position, payload) for position, payload in raw
            if payload_start <= position <= payload_line_end
            and not _position_has_nonanswer_label(text, position)
        ]

    if _is_complete_tagged_answer(text):
        return raw
    lines = _nonempty_lines_with_offsets(text)
    if not lines:
        return []
    last_start, last_end, last_line = lines[-1]
    if not _is_complete_tagged_answer(last_line):
        return []
    return [
        (position, payload) for position, payload in raw
        if last_start <= position <= last_end
        and not _position_has_nonanswer_label(text, position)
    ]


def _tagged_answers_conflict(
        candidates: list[tuple[int, str]]) -> bool:
    """Reject distinct tagged answers instead of guessing which one wins."""
    normalized = {
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", payload)).strip()
        for _, payload in candidates
    }
    return len(normalized) > 1


def _conclusion_candidates(text: str) -> list[tuple[int, str]]:
    """Return same-line payloads after an explicit reasoning conclusion cue."""
    text = _sanitize(text)
    pattern = re.compile(
        r"\b(?:thus|therefore|hence|consequently|in[ \t]+conclusion)\b"
        r"[ \t]*[,;:]?[ \t]*(?:the[ \t]+)?"
        r"(?:answer|output|result|relation|comparison|choice)?[ \t]*"
        r"(?:(?:is|equals?|should[ \t]+be)[ \t]*|[:=][ \t]*)?",
        re.IGNORECASE,
    )
    candidates = []
    for marker in pattern.finditer(text):
        line_end = text.find("\n", marker.end())
        if line_end < 0:
            line_end = len(text)
        payload = text[marker.end():line_end].strip()
        if (payload and not payload.endswith(":")
                and len(re.findall(r"\S+", payload)) <= 16):
            candidates.append((marker.end(), payload))
    return candidates


def _correction_candidates(text: str) -> list[tuple[int, str]]:
    pattern = re.compile(
        r"(?im)^[ \t]*(?:actually|correction|corrected[ \t]+answer|"
        r"on[ \t]+second[ \t]+thought|rather|instead)[ \t]*[:,][ \t]*(.+)$",
    )
    return [(match.start(1), match.group(1).strip())
            for match in pattern.finditer(_sanitize(text))]


def _fenced_block_candidates(text: str) -> list[tuple[int, int, str, str]]:
    """Return complete Markdown fences as ``(start, end, language, body)``.

    This is deliberately a transport-layer extractor, not a Markdown parser.
    Only paired, line-delimited fences count, so an isolated backtick run or a
    copied inline example cannot become an answer by accident.
    """
    text = _sanitize(text)
    pattern = re.compile(
        r"(?m)^[ \t]*(?P<fence>`{3}|~{3})"
        r"(?P<lang>[A-Za-z0-9_+.-]*)[ \t]*\r?\n"
        r"(?P<body>[\s\S]*?)\r?\n[ \t]*(?P=fence)[ \t]*(?=\r?$)",
    )
    return [
        (match.start("body"), match.end(), match.group("lang"), match.group("body").strip())
        for match in pattern.finditer(text)
    ]


def _unwrap_math_presentation(value: str) -> str:
    """Remove one or more harmless *outer* math presentation wrappers."""
    value = value.strip()
    while True:
        previous = value
        for pattern in (
            r"\\\(([\s\S]*)\\\)",
            r"\\\[([\s\S]*)\\\]",
            r"\$\$([\s\S]*)\$\$",
            r"(?<!\$)\$([^$]*)\$(?!\$)",
        ):
            match = re.fullmatch(pattern, value)
            if match:
                value = match.group(1).strip()
                break
        if value == previous:
            return value


def _balanced_latex_arguments(text: str, command: str = "boxed") \
        -> list[tuple[int, int, int, int, str]]:
    """Extract balanced ``\\command{...}`` arguments, including nested braces.

    Tuples are ``(command_start, command_end, arg_start, arg_end, content)``.
    Escaped braces do not affect nesting depth.
    """
    pattern = re.compile(r"\\" + re.escape(command) + r"[ \t]*\{")
    found = []
    for match in pattern.finditer(text):
        open_brace = match.end() - 1
        depth = 1
        i = open_brace + 1
        while i < len(text):
            ch = text[i]
            slash_count = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                slash_count += 1
                j -= 1
            escaped = slash_count % 2 == 1
            if not escaped:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        found.append((
                            match.start(), i + 1, open_brace + 1, i,
                            text[open_brace + 1:i],
                        ))
                        break
            i += 1
    return found


def _is_terminal_latex_box(text: str, command_end: int) -> bool:
    """Whether a balanced box is the response's final answer presentation.

    Only display-math closers and one optional sentence terminator may follow
    the box.  This narrow definition lets a final ``\\boxed{...}`` outrank an
    earlier descriptive use of words such as ``result is`` while preserving a
    later correction/answer as authoritative.
    """
    suffix = _sanitize(text)[max(0, command_end):].strip()
    return bool(re.fullmatch(
        r"(?:(?:\\\]|\\\)|\$\$?|\\end\s*\{(?:equation\*?|align\*?)\})"
        r"[ \t\r\n]*)*[.!?。！？]?",
        suffix,
    ))


_THOUSANDS_GROUP_RE = re.compile(
    r"(?<![\d,])([+-]?\d{1,3}(?:,\d{3})+)(?![\d,])"
)


def _remove_numeric_thousands_separators(text: str) -> str:
    """Remove commas only from canonical grouped decimal numerals.

    Commas in lists, tuples, prose, code, or malformed groupings are retained.
    This replaces the old global ``replace(',', '')`` numeric normalization.
    """
    return _THOUSANDS_GROUP_RE.sub(
        lambda match: match.group(1).replace(",", ""), _sanitize(text),
    )


def _numeric_presentation_scalar(value: str) -> str | None:
    """Extract one literal scalar from a box/text/fraction presentation.

    No arithmetic expression is evaluated.  Besides an ordinary numeric
    literal, only a single LaTeX ``frac``/``dfrac`` with numeric literal
    numerator and denominator is accepted.
    """
    clean = _remove_numeric_thousands_separators(
        _strip_answer_candidate_wrappers(value),
    ).strip()
    if NUMBER_RE.fullmatch(clean):
        return clean
    fraction = re.fullmatch(
        r"([+-]?)[ \t]*\\d?frac[ \t]*"
        r"\{[ \t]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[ \t]*\}"
        r"[ \t]*\{[ \t]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[ \t]*\}",
        clean,
    )
    if not fraction:
        return None
    try:
        numerator = Fraction(Decimal(fraction.group(2)))
        denominator = Fraction(Decimal(fraction.group(3)))
        if denominator == 0:
            return None
        result = numerator / denominator
        if fraction.group(1) == "-":
            result = -result
        return repr(float(result))
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None


def _single_json_numeric_leaf(value: str) -> str | None:
    """Return the sole finite JSON numeric leaf, rejecting multi-number data."""
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    numbers: list[int | float] = []
    stack = [parsed]
    while stack:
        item = stack.pop()
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            if not math.isfinite(float(item)):
                return None
            numbers.append(item)
            if len(numbers) > 1:
                return None
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return repr(numbers[0]) if len(numbers) == 1 else None


def _normalize_for_match(s: str) -> str:
    """Lowercase, collapse whitespace, normalize unicode quotes/dashes."""
    s = s.lower()
    s = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", s)
    s = re.sub(r"[\u2013\u2014]", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------------------------------------------------------
# Chinese answer synonyms and negation pairs
# ----------------------------------------------------------------------------
# When the expected answer is one of these canonical forms, accept any
# listed synonym. Important: the table is queried EXCLUSIVELY — if the
# response matches a *different* canonical's synonym list, that's an
# explicit non-match (so "不是" doesn't pass when expected is "是").
CHINESE_ANSWER_SYNONYMS = {
    "是": ["是", "是的", "对", "对的", "会", "能", "可以", "有", "有的", "成立", "正确", "为真"],
    "否": ["否", "不", "不是", "不一定", "不能", "不对", "不会", "不可以", "否定",
          "没有", "未必", "不正确", "错误", "为假"],
    "对": ["对", "对的", "是", "是的", "正确"],
    "不对": ["不对", "错误", "不正确", "不是"],
    "正确": ["正确", "正确的", "对", "对的"],
    "错误": ["错误", "错误的", "不对", "不正确"],
    "合理": ["合理", "合理的", "算合理"],
    "不合理": ["不合理", "不合理的", "不算合理", "太贵", "太高", "太低", "偏高", "偏低"],
    "能": ["能", "能够", "可以", "会"],
    "不能": ["不能", "不可以", "不会"],
}


def _chinese_synonym_match(resp: str, expected: str) -> bool | None:
    """Returns True if response is a synonym of `expected`,
    False if response is a synonym of a DIFFERENT canonical answer
    (so we explicitly reject the wrong polarity), or None if neither."""
    if expected not in CHINESE_ANSWER_SYNONYMS: return None
    s = resp.strip().rstrip("\u3002.\uff0c, \n\u201c\u201d\"'")
    if not s: return None
    # Direct match
    if s == expected: return True
    if s in CHINESE_ANSWER_SYNONYMS[expected]: return True
    # Starts with synonym
    for syn in CHINESE_ANSWER_SYNONYMS[expected]:
        if s.startswith(syn):
            # But also check it's not a longer wrong-polarity word
            for canon, syns in CHINESE_ANSWER_SYNONYMS.items():
                if canon == expected: continue
                for wrong in syns:
                    if len(wrong) > len(syn) and s.startswith(wrong):
                        return False
            return True
    # Response matches a DIFFERENT canonical → explicit reject
    for canon, syns in CHINESE_ANSWER_SYNONYMS.items():
        if canon == expected: continue
        if s == canon or s in syns or any(s.startswith(syn) for syn in syns):
            return False
    return None


def _text_span_is_negated(text: str, start: int, end: int) -> bool:
    """Return whether an otherwise-matching answer span is explicitly rejected.

    This is deliberately local.  It prevents tolerant answer extraction from
    accepting text such as ``not banana`` or ``banana is wrong`` while still
    allowing a later affirmative answer in the same response.
    """
    before = text[max(0, start - 72):start].lower()
    after = text[end:min(len(text), end + 56)].lower()
    before_pattern = (
        r"(?:\bnot(?:\s+(?:choose|select|pick))?|\bno|\bnever|\bneither|"
        r"\breject(?:ed|ing|s)?|"
        r"\bexclude(?:d|s|ing)?|\bexcept(?:\s+for)?|"
        r"\bis\s+not|\bisn't|\bisnt|\bain't|"
        r"不是|并非|不选|排除|拒绝|非)"
        r"[ \t]*(?:the[ \t]+answer[ \t]+(?:is|:|=)[ \t]*)?"
        r"[\"'`“‘「『]*[ \t]*$"
    )
    after_pattern = (
        r"^\s*(?:(?:is|was|would\s+be)\s+)?"
        r"(?:not\b|incorrect\b|wrong\b|false\b|rejected\b|excluded\b|"
        r"不正确|错误|不对|不是)"
    )
    locally_negated = bool(
        re.search(before_pattern, before, re.IGNORECASE)
        or re.search(after_pattern, after, re.IGNORECASE)
    )
    if locally_negated:
        return True

    # In a contrastive subordinate clause, the mentioned value is normally
    # the alternative being set aside: ``While piece is possible, bit is the
    # answer``.  Do not turn that mention into an affirmative answer.  The
    # comma guard keeps the main clause in ``While bit is informal, piece is
    # standard`` eligible.
    line_before = text[text.rfind("\n", 0, start) + 1:start].lower()
    if re.search(r"\b(?:while|although|though)\b[^,;]*$", line_before):
        return True
    return False


def _word_boundary_search(haystack: str, needle: str,
                          case_sensitive: bool = False, *,
                          allow_short_substring: bool = True,
                          reject_negated: bool = False) -> bool:
    """Find `needle` as a whole-token in haystack. Avoids false positives from
    naive substring matches.

    Per-edge rules (decided by the needle's first/last char):
      - ASCII letter / underscore: require no adjacent ASCII letter/digit/_
        (so `correct` doesn't match in `incorrect`)
      - ASCII digit: require no adjacent ASCII digit
        (so `365` doesn't match in `13651`, but DOES match in `约365天`)
      - CJK character: NO general adjacent-char constraint (Chinese has no word
        spaces). Negation safety applied separately.
      - Non-word (punctuation, symbol): no boundary constraint
        (so `<h1>` matches in `<h1>Hello`)

    Special cases:
      - Length ≤ 2 ASCII alphanumeric needle → raw substring (CLI flag chars)
      - CJK-leading needle → reject occurrence preceded by Chinese negation
        prefix (不/没/非/无/未/否/勿/别/甭/莫); if no clean occurrence, return False

    Examples:
      _word_boundary_search("incorrect",       "correct") → False
      _word_boundary_search("不合理",            "合理")    → False  (negation)
      _word_boundary_search("价格合理",           "合理")    → True
      _word_boundary_search("13651",            "365")    → False  (digit-digit)
      _word_boundary_search("约365天",           "365")    → True   (no adjacent digit)
      _word_boundary_search("<h1>Hello",        "<h1>")   → True
      _word_boundary_search("tar -xzvf",        "x")      → True
      _word_boundary_search("是的",              "是")     → True
      _word_boundary_search("不是",              "是")     → False
    """
    if not needle: return False
    flags = 0 if case_sensitive else re.IGNORECASE

    # Some keyword/CLI scorers intentionally search for one-character flags.
    # Answer extraction opts out so expected ``A`` never matches ``pass`` and
    # expected ``no`` never matches a larger token.
    if (allow_short_substring and len(needle) <= 2
            and needle.isascii() and needle.isalnum()):
        target = haystack if case_sensitive else haystack.lower()
        n = needle if case_sensitive else needle.lower()
        return n in target

    first = needle[0]
    last = needle[-1]

    def _edge_class(ch):
        """Return the regex char-class fragment that must NOT be adjacent on
        this side (or empty string if no constraint)."""
        if not ch.isascii():
            # CJK / Latin-1 / etc. — no automatic boundary;
            # CJK negation handled below.
            return ""
        if ch.isalpha() or ch == "_":
            return "[A-Za-z0-9_]"
        if ch.isdigit():
            return "[0-9]"
        # punctuation / symbol — no boundary needed
        return ""

    left_class = _edge_class(first)
    right_class = _edge_class(last)

    pat = ""
    if left_class:
        pat += r"(?<!" + left_class + r")"
    pat += re.escape(needle)
    if right_class:
        pat += r"(?!" + right_class + r")"

    matches = list(re.finditer(pat, haystack, flags))
    if not matches:
        return False

    # A comma-list reference must not be recovered as the suffix of a larger
    # list (``7, 18`` inside ``1, 7, 18``).  Keep discourse prefixes such as
    # ``Thus: Ben, Carl`` valid: only an immediately preceding comma-delimited
    # scalar token establishes the extra list member.
    if "," in needle:
        safe_matches = []
        for match in matches:
            line_start = haystack.rfind("\n", 0, match.start()) + 1
            prefix = haystack[line_start:match.start()]
            extra_item = re.search(
                r"(?i)(?<![A-Za-z0-9_])"
                r"(?!thus\b|therefore\b|hence\b)"
                r"[A-Za-z0-9_.+\-À-ɏ]+[ \t]*,[ \t]*$",
                prefix,
            )
            if not extra_item:
                safe_matches.append(match)
        matches = safe_matches
        if not matches:
            return False

    # A positive currency answer must not be recovered from an explicitly
    # signed negative amount (``$1,200`` from ``-$1,200``).  Currency symbols
    # are punctuation to the generic boundary logic, so handle polarity here.
    if reject_negated and first in "$\u00a3\u20ac\u00a5":
        matches = [
            match for match in matches
            if not (
                match.start() > 0
                and haystack[match.start() - 1] in "+-\u2212\u2013\u2014"
            )
        ]
        if not matches:
            return False

    # Chinese negation safety: when needle starts with CJK, reject occurrences
    # preceded by a negation prefix UNLESS another non-negated occurrence exists.
    if "\u4e00" <= first <= "\u9fff" or "\u3400" <= first <= "\u4dbf":
        NEG_PREFIXES = "不没非无未否勿别甭莫"
        for m in matches:
            pos = m.start()
            preceded_by_neg = pos > 0 and haystack[pos - 1] in NEG_PREFIXES
            if (not preceded_by_neg
                    and not (reject_negated
                             and _text_span_is_negated(haystack, m.start(), m.end()))):
                return True
        return False

    if reject_negated:
        return any(
            not _text_span_is_negated(haystack, m.start(), m.end())
            for m in matches
        )
    return True


# ----------------------------------------------------------------------------
# extract_numeric — robust last-number / boxed / answer-marker fallback
# ----------------------------------------------------------------------------
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _numeric_token_is_negated(text: str, start: int, end: int) -> bool:
    """Detect local, explicit negation of one numeric token.

    The check is intentionally local so ``not 4, answer 5`` can still select
    the affirmative 5. It covers common English/Chinese negators and symbolic
    inequality forms without treating the minus sign of a negative number as
    semantic negation.
    """
    before = text[max(0, start - 64):start].lower()
    after = text[end:min(len(text), end + 48)].lower()
    before_pattern = (
        r"(?:\bnot|\bno|\bnever|\bis\s+not|\bisn't|\bisnt|\bain't|"
        r"\bexcept(?:\s+for)?|\bexcluding|!=|<>|\u2260|"
        r"不是|并非|不等于|不为|非)"
        r"\s*(?:equal\s+to\s*)?$"
    )
    after_pattern = (
        r"^\s*(?:(?:is\s+not|isn't|isnt|is\s+incorrect)\b|"
        r"!=|<>|\u2260|不是|并非|不等于|不为)"
    )
    return bool(re.search(before_pattern, before, re.IGNORECASE)
                or re.search(after_pattern, after, re.IGNORECASE))


def _non_negated_number_matches(text: str):
    return [
        match for match in NUMBER_RE.finditer(text)
        if not _numeric_token_is_negated(text, match.start(), match.end())
    ]


def _declared_numeric_candidates(text: str) -> list[tuple[int, str]]:
    """Numbers attached to a strong value declaration, not incidental math.

    This lets ``the minimum value of f(x) is 4`` outrank a later equality-case
    variable such as ``x=sqrt(2)`` on the same line. A later correction cue
    invalidates the earlier declaration unless it introduces its own strong
    declaration.
    """
    wrappers = r"[ \t*_`$\\()\[\]{}]*"
    patterns = (
        re.compile(
            r"\b(?:minimum|maximum|min(?:imum)?|max(?:imum)?)"
            r"(?:[ \t]+(?:possible[ \t]+)?value)?\b"
            r"[^.?!\n]{0,240}?\b(?:is|equals?)\s*" + wrappers
            + r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the[ \t]+)?(?:answer|result|total|count)"
            r"[ \t]*(?:is|equals?|:|=)\s*" + wrappers
            + r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            re.IGNORECASE,
        ),
    )
    found = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            tail = text[match.end():]
            correction = re.search(
                r"\b(?:actually|correction|corrected|rather|instead|on[ \t]+second[ \t]+thought)\b",
                tail, re.IGNORECASE,
            )
            if correction and not re.search(
                r"\b(?:answer|result|minimum|maximum|min|max|total|count)"
                r"[^.?!\n]{0,120}?\b(?:is|equals?)\b",
                tail[correction.start():], re.IGNORECASE,
            ):
                continue
            found.append((match.start(1), match.group(1)))
    return found


def extract_numeric(resp: str, expected, tolerance: float = 0,
                    require_only_numeric: bool = False,
                    allow_leading_count: bool = False) -> ScoreResult:
    """Find a number in `resp` that equals `expected` (within tolerance).
    Strategy order:
      1. Strip markdown, exact equality
      2. \\boxed{X} match
      3. Last number on the last meaningful line
      4. Last number anywhere in the response
      5. "answer: X" / "= X" patterns at end
      6. Any number in the response (last-resort, marked "buried")
    """
    resp = _sanitize(resp)
    try:
        exp = float(expected)
    except (TypeError, ValueError):
        return ScoreResult(False, "expected_not_numeric", expected=expected,
                           note=f"expected={expected!r} not numeric")

    def matches(x):
        return abs(x - exp) <= tolerance

    if not resp.strip():
        return ScoreResult(False, "empty", expected=exp)

    # 1. Stripped exact
    s = _remove_numeric_thousands_separators(_strip_markdown_wrappers(resp))
    try:
        if matches(float(s)):
            return ScoreResult(True, "stripped_exact", extracted=float(s), expected=exp)
    except ValueError: pass
    if require_only_numeric:
        return ScoreResult(
            False, "numeric_only_format_violation", expected=exp,
            note="response must contain only the requested numeric value",
        )

    # Outside instruction-following probes, extract the latest answer-position
    # declaration by *text position*.  A later Answer/Reply/Output marker can
    # override an earlier box, and a later bare final line can override either.
    # Within a marker payload, the first number is the declared answer; numbers
    # in a following ``Explanation: ...`` clause are not competing answers.
    normalized_resp = _remove_numeric_thousands_separators(resp)
    scan_resp = _mask_answer_tag_regions(normalized_resp)
    positioned: list[tuple[int, str | None, str]] = []

    marker_candidates = _answer_marker_candidates(normalized_resp)
    for position, payload in _answer_marker_candidates(scan_resp):
        number_matches = _non_negated_number_matches(payload)
        if number_matches:
            # A declared equation such as ``Final result: 67 mod 11 = 1``
            # carries its scalar answer on the final RHS.  Select it only
            # when the RHS contains exactly one numeric literal plus harmless
            # math/Markdown closers; comparisons and multi-value RHS text do
            # not enter this fallback.
            rhs_match = None
            equals = list(re.finditer(r"(?<![<>=!])=(?!=)", payload))
            lhs_is_one_expression = bool(
                len(equals) == 1
                and not re.search(
                    r"[.;:!?]|\b(?:example|check|reference|answer|result)\b",
                    payload[:equals[0].start()], re.IGNORECASE,
                )
            )
            if lhs_is_one_expression:
                rhs = payload[equals[-1].end():]
                rhs_numbers = _non_negated_number_matches(rhs)
                if len(rhs_numbers) == 1:
                    candidate = rhs_numbers[0]
                    before = rhs[:candidate.start()]
                    after = rhs[candidate.end():]
                    if (not before.strip(" \t*_`$\\()[]{}")
                            and not after.strip(" \t*_`$\\()[]{}.!?,;:")):
                        rhs_match = candidate
            match = rhs_match or number_matches[0]
            local_start = (
                equals[-1].end() + match.start()
                if rhs_match is not None else match.start()
            )
            positioned.append((
                position + local_start, match.group(0),
                "answer_marker_rhs" if rhs_match is not None else "answer_marker",
            ))
        else:
            positioned.append((position, None, "answer_marker"))

    tagged_candidates = _authoritative_tagged_answer_candidates(
        normalized_resp, marker_candidates,
    )
    if _tagged_answers_conflict(tagged_candidates):
        return ScoreResult(
            False, "ambiguous_tagged_answers", expected=exp,
            note="multiple distinct paired answer-tag payloads",
        )
    for position, payload in tagged_candidates:
        if _tagged_answer_is_negated(normalized_resp, position):
            positioned.append((position, None, "negated_tagged_answer"))
            continue
        clean = _remove_numeric_thousands_separators(
            _strip_answer_candidate_wrappers(payload),
        )
        if NUMBER_RE.fullmatch(clean):
            positioned.append((position, clean, "tagged_answer"))
        else:
            positioned.append((position, None, "tagged_answer"))

    boxes = _balanced_latex_arguments(scan_resp)
    for command_start, command_end, arg_start, arg_end, content in boxes:
        terminal_box = _is_terminal_latex_box(scan_resp, command_end)
        if (_position_has_nonanswer_block_label(scan_resp, command_start)
                or (not terminal_box and not _weak_presentation_allowed(
                    scan_resp, arg_start, marker_candidates))):
            continue
        clean = _numeric_presentation_scalar(content)
        if clean is not None and not _numeric_token_is_negated(
                scan_resp, arg_start, arg_end):
            positioned.append((arg_start, clean, "boxed"))
        else:
            positioned.append((arg_start, None, "boxed"))

    # A scalar inside one fenced block is another transport convention.  Each
    # payload must still be exactly numeric; a worked expression fence is not
    # reduced or evaluated.  Text-position ordering lets a later scalar fence
    # override an earlier derivation fence without guessing among values.
    for fence_start, fence_end, language, content in _fenced_block_candidates(scan_resp):
        if _position_has_nonanswer_block_label(scan_resp, fence_start):
            continue
        clean = _remove_numeric_thousands_separators(
            _strip_answer_candidate_wrappers(content),
        )
        if (not NUMBER_RE.fullmatch(clean)
                and language.casefold() == "json"
                and not scan_resp[fence_end:].strip()):
            clean = _single_json_numeric_leaf(content)
        if clean is not None and NUMBER_RE.fullmatch(clean):
            positioned.append((fence_start, clean, "fenced_numeric_answer"))
        else:
            positioned.append((fence_start, None, "fenced_numeric_answer"))

    declared_numbers = _declared_numeric_candidates(scan_resp)
    if allow_leading_count:
        leading_count = re.match(
            r"^[ \t*_`]*(\d+)[ \t]+[^\W\d_]+\b", scan_resp,
            re.UNICODE,
        )
        if leading_count:
            declared_numbers.append((leading_count.start(1), leading_count.group(1)))
    for position, raw_number in declared_numbers:
        if not _numeric_token_is_negated(
                scan_resp, position, position + len(raw_number)):
            positioned.append((position, raw_number, "declared_value"))

    lines = _nonempty_lines_with_offsets(scan_resp)
    if lines:
        line_start, line_end, raw_last = lines[-1]
        explicit_on_last = any(
            line_start <= position <= line_end for position, _ in marker_candidates
        ) or any(
            line_start <= command_start <= line_end
            for command_start, _, _, _, _ in boxes
        )
        bare_numeric_line = bool(NUMBER_RE.fullmatch(
            _remove_numeric_thousands_separators(
                _strip_answer_candidate_wrappers(raw_last),
            ),
        ))
        if not explicit_on_last and (not declared_numbers or bare_numeric_line):
            nums_in_last = _non_negated_number_matches(raw_last)
            if nums_in_last:
                match = nums_in_last[-1]
                positioned.append((
                    line_start + match.start(), match.group(0),
                    "last_line_last_num",
                ))

    if positioned:
        _, raw_value, strategy = max(positioned, key=lambda item: item[0])
        if raw_value is None:
            return ScoreResult(
                False, f"{strategy}_not_numeric", expected=exp,
                note="latest explicit answer is not an affirmative numeric value",
            )
        value = float(raw_value)
        return ScoreResult(
            matches(value), strategy if matches(value) else "wrong_number",
            extracted=value, expected=exp,
            note="latest answer-position candidate" if not matches(value) else "",
        )

    # No explicit answer position was available.  Keep the old safe fallback:
    # only the final affirmative number may count; an earlier occurrence is
    # reported as buried rather than silently accepted.
    nums = _non_negated_number_matches(scan_resp)
    if nums:
        x = float(nums[-1].group(0))
        if matches(x):
            return ScoreResult(True, "last_num", extracted=x, expected=exp)

    # Any number — but distinguish "right-but-buried" from outright wrong.
    if nums:
        for number_match in nums:
            raw_number = number_match.group(0)
            x = float(raw_number)
            if matches(x):
                return ScoreResult(False, "buried_only_correct_number_present",
                                   extracted=x, expected=exp,
                                   note="correct number appears but not at answer position")
        last_number = nums[-1].group(0)
        return ScoreResult(False, "wrong_numbers", extracted=float(last_number),
                           expected=exp, note=f"got {last_number} expected {exp}")

    if NUMBER_RE.search(scan_resp):
        return ScoreResult(False, "only_negated_numbers", expected=exp,
                           note="all numeric candidates were explicitly negated")
    return ScoreResult(False, "no_number", expected=exp, note="no number in response")


# ----------------------------------------------------------------------------
# exact_integer_output — byte-exact integer scoring without float conversion
# ----------------------------------------------------------------------------
INTEGER_LITERAL_RE = re.compile(r"[+-]?(?:0|[1-9]\d*)")


def exact_integer_output(resp: str, expected) -> ScoreResult:
    """Require one canonical decimal integer and compare it as text.

    This intentionally does not call ``float`` (or accept scientific notation):
    large adjacent integers can collapse to the same IEEE-754 value. Outer
    transport whitespace is ignored, but prose, commas, decimal points,
    exponent notation, Markdown wrappers, and leading-zero variants fail.
    """
    expected_text = str(expected).strip()
    if not INTEGER_LITERAL_RE.fullmatch(expected_text):
        return ScoreResult(False, "invalid_expected_integer", expected=expected_text)
    candidate = _sanitize(resp).strip()
    if not INTEGER_LITERAL_RE.fullmatch(candidate):
        return ScoreResult(
            False,
            "integer_only_format_violation",
            extracted=candidate[:200],
            expected=expected_text,
            note="expected one canonical base-10 integer with no prose, decimal point, comma, or exponent",
        )
    return ScoreResult(
        candidate == expected_text,
        "exact_integer_output" if candidate == expected_text else "wrong_integer",
        extracted=candidate,
        expected=expected_text,
    )


def extract_integer(resp: str, expected) -> ScoreResult:
    """Safely extract a canonical decimal integer from an answer position.

    Unlike ``extract_numeric`` this never converts through ``float``: adjacent
    large integers must remain distinguishable.  It accepts only an exact
    integer, ``\\boxed{...}``, an explicit final-answer marker, or a canonical
    integer at the end of the final meaningful line.  Decimal and exponent
    spellings (``1.0`` / ``1e0``) remain invalid integer representations.
    """
    expected_text = str(expected).strip()
    if not INTEGER_LITERAL_RE.fullmatch(expected_text):
        return ScoreResult(False, "invalid_expected_integer", expected=expected_text)

    text = _sanitize(resp)
    if not text.strip():
        return ScoreResult(False, "empty", expected=expected_text)
    scan_text = _mask_answer_tag_regions(text)

    def result_for(raw: str, strategy: str, *, span=None) -> ScoreResult | None:
        candidate = raw.strip()
        if not INTEGER_LITERAL_RE.fullmatch(candidate):
            return None
        if span is not None and _text_span_is_negated(text, span[0], span[1]):
            return None
        return ScoreResult(
            candidate == expected_text,
            strategy if candidate == expected_text else "wrong_integer",
            extracted=candidate,
            expected=expected_text,
        )

    # Exact whole answer after harmless transport/Markdown wrapping.
    stripped = _strip_markdown_wrappers(scan_text)
    exact = result_for(stripped, "stripped_exact_integer")
    if exact is not None:
        return exact

    positioned = []
    for command_start, command_end, arg_start, arg_end, content in \
            _balanced_latex_arguments(scan_text):
        terminal_box = _is_terminal_latex_box(scan_text, command_end)
        if (_position_has_nonanswer_block_label(scan_text, command_start)
                or (not terminal_box and not _weak_presentation_allowed(
                    scan_text, arg_start, _answer_marker_candidates(text)))):
            continue
        positioned.append((arg_start, content, "boxed_integer", (arg_start, arg_end)))
    marker_candidates = _answer_marker_candidates(text)
    for position, payload in _answer_marker_candidates(scan_text):
        cleaned = _strip_answer_candidate_wrappers(payload)
        positioned.append((position, cleaned, "answer_marker_integer",
                           (position, position + len(payload))))
    for fence_start, _, _, content in _fenced_block_candidates(scan_text):
        if _position_has_nonanswer_block_label(scan_text, fence_start):
            continue
        cleaned = _strip_answer_candidate_wrappers(content)
        positioned.append((fence_start, cleaned, "fenced_integer_answer",
                           (fence_start, fence_start + len(content))))
    tagged_candidates = _authoritative_tagged_answer_candidates(
        text, marker_candidates,
    )
    if _tagged_answers_conflict(tagged_candidates):
        return ScoreResult(
            False, "ambiguous_tagged_answers", expected=expected_text,
            note="multiple distinct paired answer-tag payloads",
        )
    for position, payload in tagged_candidates:
        negated = _tagged_answer_is_negated(text, position)
        cleaned = ("\0NEGATED_TAGGED_ANSWER\0" if negated
                   else _strip_answer_candidate_wrappers(payload))
        strategy = ("negated_tagged_answer_integer" if negated
                    else "tagged_answer_integer")
        positioned.append((position, cleaned, strategy,
                           (position, position + len(payload))))

    # A canonical integer at the end of the final meaningful line.  The
    # alphanumeric/dot guards prevent extracting the trailing 0 from 1.0/e0.
    lines = _nonempty_lines_with_offsets(scan_text)
    if lines:
        line_start, _, last_line = lines[-1]
        tail = re.search(
            r"(?<![0-9A-Za-z_.])([+-]?(?:0|[1-9]\d*))"
            r"(?![0-9A-Za-z_.])[ \t]*[.!?]?[ \t]*$",
            last_line,
        )
        if tail:
            span = (line_start + tail.start(1), line_start + tail.end(1))
            positioned.append((span[0], tail.group(1), "last_line_integer", span))

    if positioned:
        _, raw, strategy, span = max(positioned, key=lambda item: item[0])
        candidate = result_for(raw, strategy, span=span)
        if candidate is not None:
            return candidate
        return ScoreResult(
            False, f"{strategy}_invalid", extracted=str(raw)[:200],
            expected=expected_text,
            note="latest explicit integer answer is invalid or negated",
        )

    return ScoreResult(False, "no_explicit_integer_answer", expected=expected_text)


# ----------------------------------------------------------------------------
# extract_string — robust exact-or-answer-position match
# ----------------------------------------------------------------------------
def _safe_answer_marker_suffix(candidate: str, answer_end: int) -> bool:
    """Whether text after a marker answer is clearly explanatory, not a list.

    ``Answer: banana, banana, none`` is an answer set/list, so accepting its
    first token as the complete answer would be a false positive.  A sentence
    such as ``Answer: banana. It is the only fruit ...`` remains tolerable.
    """
    suffix = candidate[answer_end:].lstrip(" \t*_`")
    if not suffix:
        return True
    if re.match(r"^(?:[,;/|]|(?:or|and|versus|vs)\b)", suffix, re.IGNORECASE):
        return False
    return True


def _allows_buried_string_fallback(expected: str) -> bool:
    """Restrict prose fallback to answers with semantic token characters.

    Punctuation-only answers need an exact/explicit answer position; otherwise
    ``*****`` would match inside a longer run of stars.
    """
    return any(ch.isalnum() for ch in expected)


def _looks_like_bare_answer_line(candidate: str, expected: str, *,
                                 case_sensitive: bool,
                                 raw_line: str | None = None) -> bool:
    """Conservatively distinguish a later bare answer from an explanation."""
    clean = candidate.strip()
    got = clean if case_sensitive else clean.lower()
    want = expected if case_sensitive else expected.lower()
    if got == want:
        return True
    if re.fullmatch(r"(?:\\[\])]|\$+|`{3,})", clean):
        return False
    if raw_line is not None and raw_line[:1].isspace():
        return False
    if len(re.findall(r"\S+", clean)) == 1:
        return True
    return bool(re.search(r"[,;|]", clean) and len(clean) <= 240)


def _mask_quoted_and_emphasized_spans(text: str) -> str:
    """Blank non-answer mentions before whole-response boundary fallback."""
    chars = list(text)
    patterns = (
        re.compile(r'"[^"\n]+"'),
        re.compile(r"(?<!\w)'[^'\n]+'(?!\w)"),
        re.compile(r"`[^`\n]+`"),
        re.compile(r"\*\*[^*\n]+\*\*"),
        re.compile(r"__[^_\n]+__"),
        re.compile(r"(?<!\*)\*[^*\n]+\*(?!\*)"),
        re.compile(r"(?<!_)_[^_\n]+_(?!_)"),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            for index in range(match.start(), match.end()):
                chars[index] = " "
    return "".join(chars)


def _safe_leading_answer_with_explanation(
        text: str, expected: str, *, case_sensitive: bool,
        allow_semantic_equivalence: bool) -> tuple[bool, str]:
    """Recognize one compact first-line answer followed by explanation.

    ACE 2.5 accepted the common ``A\n\nExplanation...`` shape.  A blanket
    first-token rule is unsafe, however, because a model may correct itself
    later.  This fallback therefore requires a complete compact first line and
    rejects every explicit later answer/correction/tag/box plus common
    contradiction openers and compact alternative labels.
    """
    lines = _nonempty_lines_with_offsets(text)
    if len(lines) < 2:
        return False, ""
    _, first_end, raw_first = lines[0]
    first = _strip_answer_candidate_wrappers(
        raw_first, strip_terminal_punctuation=True,
    )
    got = first if case_sensitive else first.casefold()
    want = expected if case_sensitive else expected.casefold()
    equivalent = got == want
    if not equivalent and allow_semantic_equivalence:
        equivalent = bool(_safe_noninstr_equivalence(first, expected))
    if not equivalent:
        return False, ""
    if len(first) > 64 or len(re.findall(r"\S+", first)) > 8:
        return False, ""

    remainder = text[first_end:].lstrip("\r\n")
    if (not remainder.strip()
            or _answer_marker_candidates(remainder)
            or _tagged_answer_candidates(remainder)
            or _correction_candidates(remainder)
            or _balanced_latex_arguments(remainder)):
        return False, ""
    if re.search(
            r"(?im)^[ \t]*(?:[*_`#>-]+[ \t]*)?"
            r"(?:actually|however|but|rather|instead|correction|"
            r"corrected[ \t]+answer|on[ \t]+second[ \t]+thought)\b",
            remainder):
        return False, ""

    # Reject a later standalone alternative for the compact closed answer
    # vocabularies that occur most often in deterministic probes.
    tf_length = None
    if re.fullmatch(r"[A-D]", first, re.IGNORECASE):
        alternatives = {"A", "B", "C", "D", "EQUAL"}
    elif re.fullmatch(r"[TF]+", first, re.IGNORECASE):
        alternatives = set()
        tf_length = len(first)
    else:
        opposites = {
            "yes": {"no"}, "no": {"yes"},
            "true": {"false"}, "false": {"true"},
            "correct": {"incorrect", "wrong"},
            "incorrect": {"correct"}, "ambiguous": {"unambiguous"},
        }
        if first.casefold() not in opposites:
            # Open-vocabulary entities have no enumerable contradiction set:
            # ``Berlin\nReasoning\nParis`` cannot safely select the first line.
            return False, ""
        alternatives = opposites[first.casefold()]
    if alternatives or tf_length is not None:
        first_key = first.casefold()
        for _, _, raw_line in lines[1:]:
            candidate = _strip_answer_candidate_wrappers(
                raw_line, strip_terminal_punctuation=True,
            ).casefold()
            if (tf_length is not None
                    and len(candidate) == tf_length
                    and re.fullmatch(r"[tf]+", candidate)
                    and candidate != first_key):
                return False, ""
            if candidate in {item.casefold() for item in alternatives} \
                    and candidate != first_key:
                return False, ""
    return True, first


# ----------------------------------------------------------------------------
# Safe non-instruction answer equivalence
# ----------------------------------------------------------------------------
_SYMBOLIC_FUNCTIONS = {
    "abs", "acos", "asin", "atan", "cos", "cosh", "exp", "ln", "log",
    "max", "min", "sin", "sinh", "sqrt", "tan", "tanh",
}
_Poly = dict[tuple[str, ...], Fraction]
_RatPoly = tuple[_Poly, _Poly]


def _poly_clean(poly: _Poly) -> _Poly:
    return {monomial: coefficient for monomial, coefficient in poly.items()
            if coefficient}


def _poly_add(left: _Poly, right: _Poly, sign: int = 1) -> _Poly:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, Fraction(0)) + sign * coefficient
    result = _poly_clean(result)
    if len(result) > 4096:
        raise ValueError("symbolic term cap exceeded")
    return result


def _poly_mul(left: _Poly, right: _Poly) -> _Poly:
    result: _Poly = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = tuple(sorted((*left_monomial, *right_monomial)))
            result[monomial] = (
                result.get(monomial, Fraction(0))
                + left_coefficient * right_coefficient
            )
            if len(result) > 4096:
                raise ValueError("symbolic term cap exceeded")
    return _poly_clean(result)


def _poly_pow(poly: _Poly, exponent: int) -> _Poly:
    if not 0 <= exponent <= 32:
        raise ValueError("unsupported symbolic exponent")
    result: _Poly = {(): Fraction(1)}
    base = poly
    power = exponent
    while power:
        if power & 1:
            result = _poly_mul(result, base)
        power >>= 1
        if power:
            base = _poly_mul(base, base)
    return result


def _rat_key(value: _RatPoly) -> str:
    def poly_key(poly: _Poly):
        return tuple(sorted((monomial, str(coefficient))
                            for monomial, coefficient in poly.items()))
    return repr((poly_key(value[0]), poly_key(value[1])))


def _latex_to_plain_expression(value: str) -> str:
    value = _unwrap_math_presentation(value)
    value = re.sub(r"\\(?:left|right)\b", "", value)
    value = re.sub(r"\\(?:cdot|times)\b", "*", value)
    value = re.sub(r"\\[,;!]", " ", value)
    greek = {
        "alpha": "alpha", "beta": "beta", "gamma": "gamma",
        "delta": "delta", "epsilon": "epsilon", "theta": "theta",
        "lambda": "lambda_", "mu": "mu", "pi": "pi",
        "rho": "rho", "sigma": "sigma", "phi": "phi", "omega": "omega",
    }
    for command, name in greek.items():
        value = re.sub(r"\\" + command + r"\b", name, value)

    # Resolve innermost LaTeX fractions/roots without evaluating anything.
    for _ in range(32):
        previous = value
        value = re.sub(
            r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
            r"((\1)/(\2))", value,
        )
        value = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", value)
        value = re.sub(
            r"\\(?:mathrm|operatorname|text)\s*\{([^{}]*)\}", r"\1", value,
        )
        if value == previous:
            break
    value = re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", value)
    value = value.translate(str.maketrans({
        "²": "^2", "³": "^3", "−": "-", "×": "*", "·": "*",
    }))
    value = re.sub(r"√\s*\(?\s*([A-Za-z0-9_.]+)\s*\)?", r"sqrt(\1)", value)
    return value.strip()


def _tokenize_symbolic_expression(
        value: str, known_variables: set[str] | None = None,
) -> list[tuple[str, str]]:
    value = _latex_to_plain_expression(value)
    token_re = re.compile(
        r"(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
        r"(?P<name>[^\W\d]\w*)|(?P<op>\*\*|[()+\-*/,^])",
        re.UNICODE,
    )
    tokens = []
    position = 0
    while position < len(value):
        if value[position].isspace():
            position += 1
            continue
        match = token_re.match(value, position)
        if not match:
            raise ValueError("unsupported symbolic token")
        kind = match.lastgroup or ""
        token = match.group(0)
        tokens.append((kind, "**" if token == "^" else token))
        position = match.end()
        if len(tokens) > 256:
            raise ValueError("symbolic token cap exceeded")

    if known_variables:
        expanded_names = []
        variables = sorted(known_variables, key=lambda item: (-len(item), item))

        def segment(name: str) -> list[str] | None:
            if name.lower() in {item.lower() for item in known_variables}:
                return None
            solutions: list[list[str]] = []

            def visit(position: int, parts: list[str]):
                if len(solutions) > 1:
                    return
                if position == len(name):
                    if len(parts) > 1:
                        solutions.append(parts[:])
                    return
                for variable in variables:
                    if name.lower().startswith(variable.lower(), position):
                        visit(position + len(variable), [*parts, variable])

            visit(0, [])
            return solutions[0] if len(solutions) == 1 else None

        for kind, token in tokens:
            pieces = segment(token) if kind == "name" else None
            if pieces:
                for index, piece in enumerate(pieces):
                    if index:
                        expanded_names.append(("op", "*"))
                    expanded_names.append(("name", piece))
            else:
                expanded_names.append((kind, token))
        tokens = expanded_names

    # Insert conventional implicit multiplication: ``(1/2)bh``, ``4sqrt(3)``,
    # and ``(x-1)(x+1)``. A whitelisted function name followed by ``(`` remains
    # a call; arbitrary calls are never accepted by the AST evaluator below.
    expanded = []
    for token in tokens:
        if expanded:
            previous_kind, previous_value = expanded[-1]
            kind, current_value = token
            left_value = previous_kind in {"number", "name"} or previous_value == ")"
            right_value = kind == "name" or current_value == "("
            function_call = (
                previous_kind == "name" and current_value == "("
                and previous_value.lower() in _SYMBOLIC_FUNCTIONS
            )
            if left_value and right_value and not function_call:
                expanded.append(("op", "*"))
        expanded.append(token)
    return expanded


def _symbolic_ast(value: str, known_variables: set[str] | None = None) -> ast.Expression:
    source_parts = []
    exact_numbers: dict[str, Fraction] = {}
    for kind, token in _tokenize_symbolic_expression(value, known_variables):
        if kind == "number":
            if len(token) > 128:
                raise ValueError("symbolic numeric literal too long")
            decimal_value = Decimal(token)
            if (not decimal_value.is_finite()
                    or abs(decimal_value.as_tuple().exponent) > 10_000
                    or len(decimal_value.as_tuple().digits) > 128):
                raise ValueError("symbolic numeric literal out of bounds")
            placeholder = f"_ace_exact_number_{len(exact_numbers)}"
            exact_numbers[placeholder] = Fraction(decimal_value)
            source_parts.append(placeholder)
        else:
            source_parts.append(token)
    source = "".join(source_parts)
    parsed = ast.parse(source, mode="eval")
    if len(list(ast.walk(parsed))) > 256:
        raise ValueError("symbolic AST cap exceeded")
    for node in ast.walk(parsed):
        if isinstance(node, ast.Name) and node.id in exact_numbers:
            node._ace_exact_fraction = exact_numbers[node.id]
    return parsed


def _symbolic_rational(node: ast.AST) -> _RatPoly:
    one: _Poly = {(): Fraction(1)}
    if isinstance(node, ast.Expression):
        return _symbolic_rational(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return ({(): Fraction(str(node.value))}, one)
    if isinstance(node, ast.Name) and hasattr(node, "_ace_exact_fraction"):
        return ({(): node._ace_exact_fraction}, one)
    if isinstance(node, ast.Name):
        return ({(node.id.lower(),): Fraction(1)}, one)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        numerator, denominator = _symbolic_rational(node.operand)
        if isinstance(node.op, ast.USub):
            numerator = {monomial: -coefficient
                         for monomial, coefficient in numerator.items()}
        return numerator, denominator
    if isinstance(node, ast.BinOp):
        left_num, left_den = _symbolic_rational(node.left)
        right_num, right_den = _symbolic_rational(node.right)
        if isinstance(node.op, ast.Add):
            return (_poly_add(_poly_mul(left_num, right_den),
                              _poly_mul(right_num, left_den)),
                    _poly_mul(left_den, right_den))
        if isinstance(node.op, ast.Sub):
            return (_poly_add(_poly_mul(left_num, right_den),
                              _poly_mul(right_num, left_den), -1),
                    _poly_mul(left_den, right_den))
        if isinstance(node.op, ast.Mult):
            return _poly_mul(left_num, right_num), _poly_mul(left_den, right_den)
        if isinstance(node.op, ast.Div):
            if not right_num:
                raise ValueError("division by zero")
            return _poly_mul(left_num, right_den), _poly_mul(left_den, right_num)
        if isinstance(node.op, ast.Pow):
            exponent_num, exponent_den = _symbolic_rational(node.right)
            if exponent_den != one or set(exponent_num) != {()}:
                raise ValueError("non-constant exponent")
            exponent = exponent_num[()]
            if exponent.denominator != 1:
                raise ValueError("fractional exponent")
            exponent_int = int(exponent)
            if exponent_int < 0:
                return (_poly_pow(left_den, -exponent_int),
                        _poly_pow(left_num, -exponent_int))
            return _poly_pow(left_num, exponent_int), _poly_pow(left_den, exponent_int)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id.lower() in _SYMBOLIC_FUNCTIONS
            and len(node.args) in {1, 2} and not node.keywords):
        arguments = [_symbolic_rational(argument) for argument in node.args]
        atom = node.func.id.lower() + "(" + ",".join(map(_rat_key, arguments)) + ")"
        return ({(atom,): Fraction(1)}, one)
    raise ValueError(f"unsupported symbolic AST node: {type(node).__name__}")


def _symbolic_expressions_equivalent(candidate: str, expected: str) -> bool:
    """Compare a small, safe algebra subset by exact rational-polynomial form.

    No submitted expression is evaluated. Functions such as ``cos(theta)``
    are treated as opaque symbolic atoms, while arithmetic around them is
    expanded exactly with :class:`fractions.Fraction`.
    """
    if len(candidate) > 2000 or len(expected) > 2000:
        return False
    math_cue = re.compile(
        r"(?:[+*/^()]|\\(?:d?frac|sqrt|gamma|theta)\b|√|²|³)"
    )
    if not math_cue.search(candidate) and not math_cue.search(expected):
        return False
    try:
        expected_tokens = _tokenize_symbolic_expression(expected)
        known_variables = {
            token for kind, token in expected_tokens
            if kind == "name" and token.lower() not in _SYMBOLIC_FUNCTIONS
        }
        candidate_num, candidate_den = _symbolic_rational(
            _symbolic_ast(candidate, known_variables),
        )
        expected_num, expected_den = _symbolic_rational(_symbolic_ast(expected))
        return _poly_mul(candidate_num, expected_den) == \
            _poly_mul(expected_num, candidate_den)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return False


def _normalized_semantic_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = _normalize_typography(value)
    value = re.sub(r"\s+", " ", value).strip()
    # An Oxford comma is optional in ordinary non-instruction prose. Word
    # order and every lexical item remain mandatory.  Apply this before
    # separator spacing is compacted so the comma remains distinguishable.
    value = re.sub(r",\s+(?=(?:and|or)\b)", " ", value, flags=re.IGNORECASE)
    # Separator-adjacent whitespace cannot change list membership, ordering,
    # or delimiter identity.  This accepts ``a,b`` for reference ``a, b`` but
    # does not rewrite a delimiter, reorder tokens, or collapse word spaces.
    value = re.sub(r"\s*([,;|])\s*", r"\1", value)
    return value


def _calendar_date_punctuation_equivalent(candidate: str, expected: str) -> bool:
    """Allow only the optional comma after a weekday in a full calendar date."""
    month = (
        r"January|February|March|April|May|June|July|August|September|"
        r"October|November|December"
    )
    weekday = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    pattern = re.compile(
        rf"^({weekday}),?[ \t]+({month})[ \t]+(\d{{1,2}}),[ \t]+(\d{{4}})$",
        re.IGNORECASE,
    )
    left = pattern.fullmatch(_strip_answer_candidate_wrappers(
        candidate, strip_terminal_punctuation=False,
    ))
    right = pattern.fullmatch(_strip_answer_candidate_wrappers(
        expected, strip_terminal_punctuation=False,
    ))
    return bool(left and right and tuple(part.casefold() for part in left.groups())
                == tuple(part.casefold() for part in right.groups()))


def _grouped_token_sequence_equivalent(candidate: str, expected: str) -> bool:
    """Compare pipe-delimited groups while tolerating comma-vs-space layout.

    The fallback is intentionally narrow: every reference item must be a
    single lexical token and group/order/multiplicity are preserved. It cannot
    silently regroup a prose answer or accept an extra item.
    """
    if "|" not in expected or "," not in expected:
        return False
    expected_groups = [group.strip() for group in expected.split("|")]
    candidate_groups = [group.strip() for group in candidate.split("|")]
    if len(expected_groups) != len(candidate_groups):
        return False
    expected_items = []
    candidate_items = []
    for expected_group, candidate_group in zip(expected_groups, candidate_groups):
        reference = [item.strip() for item in expected_group.split(",")]
        if not reference or any(not re.fullmatch(r"[^\W_]+", item, re.UNICODE)
                                for item in reference):
            return False
        observed = [item for item in re.split(r"(?:\s*,\s*|\s+)", candidate_group)
                    if item]
        expected_items.append([item.casefold() for item in reference])
        candidate_items.append([item.casefold() for item in observed])
    return candidate_items == expected_items


def _safe_noninstr_equivalence(candidate: str, expected: str) -> str | None:
    candidate = _strip_answer_candidate_wrappers(
        candidate, strip_terminal_punctuation=True,
    )
    expected = _strip_answer_candidate_wrappers(
        expected, strip_terminal_punctuation=False,
    )
    if _normalized_semantic_text(candidate).casefold() == \
            _normalized_semantic_text(expected).casefold():
        return "typographic_equivalence"
    if _calendar_date_punctuation_equivalent(candidate, expected):
        return "calendar_date_punctuation"
    if _grouped_token_sequence_equivalent(candidate, expected):
        return "grouped_token_sequence"
    if _symbolic_expressions_equivalent(candidate, expected):
        return "symbolic_equivalence"
    return None


# ----------------------------------------------------------------------------
# Static parsers for code/config probes (never execute submitted payloads)
# ----------------------------------------------------------------------------
def _validate_jwt_encode_hs256_ast(tree: ast.Module, spec: dict) -> tuple[bool, str]:
    """Validate one allowlisted PyJWT expression without evaluating it."""
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr) \
            or not isinstance(tree.body[0].value, ast.Call):
        return False, "expected_exactly_one_call_expression"
    call = tree.body[0].value

    def dotted(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else None
        return None

    if dotted(call.func) != spec.get("call", "jwt.encode"):
        return False, "expected_jwt_encode_call"
    if len(call.args) != 2 or any(isinstance(arg, ast.Starred) for arg in call.args):
        return False, "expected_payload_and_secret_positional_args"
    payload_node, secret_node = call.args
    if not isinstance(payload_node, ast.Dict) or any(
            key is None for key in payload_node.keys):
        return False, "payload_must_be_literal_dict"
    try:
        payload = {
            ast.literal_eval(key): ast.literal_eval(value)
            for key, value in zip(payload_node.keys, payload_node.values)
        }
        secret = ast.literal_eval(secret_node)
    except (ValueError, TypeError, SyntaxError):
        return False, "payload_and_secret_must_be_literals"
    expected_payload = spec.get("payload", {"user": "bob", "role": "admin"})
    if payload != expected_payload:
        return False, f"payload_mismatch={payload!r}"
    if secret != spec.get("secret", "changeme"):
        return False, "secret_mismatch"
    if len(call.keywords) != 1 or call.keywords[0].arg != "algorithm":
        return False, "expected_only_algorithm_keyword"
    try:
        algorithm = ast.literal_eval(call.keywords[0].value)
    except (ValueError, TypeError, SyntaxError):
        return False, "algorithm_must_be_literal"
    if algorithm != spec.get("algorithm", "HS256"):
        return False, "algorithm_mismatch"
    return True, "jwt_encode_hs256_static_match"


def _static_path_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split(".") if path else []:
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _static_path_values(value: Any, path: str) -> list[Any]:
    """Resolve a static path with ``*`` matching one mapping/list level."""
    current = [value]
    for part in path.split(".") if path else []:
        following = []
        for item in current:
            if part == "*":
                if isinstance(item, dict):
                    following.extend(item.values())
                elif isinstance(item, list):
                    following.extend(item)
                continue
            try:
                following.append(item[int(part)] if isinstance(item, list) else item[part])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        current = following
    return current


def _static_path_set(root: dict, parts: list[str], value: Any) -> None:
    current = root
    for part in parts[:-1]:
        if part in current and not isinstance(current[part], dict):
            raise ValueError(f"static path collides with scalar: {'.'.join(parts)}")
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        current = child
    if parts:
        if parts[-1] in current:
            raise ValueError(f"duplicate static path: {'.'.join(parts)}")
        current[parts[-1]] = value


def _strip_unquoted_comment(line: str, markers=("#",)) -> str:
    quote = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            continue
        if any(line.startswith(marker, index) for marker in markers):
            return line[:index]
    return line


def _split_static_inline(value: str, separator: str = ",") -> list[str]:
    parts, start, stack, quote, escaped = [], 0, [], None, False
    pairs = {"(": ")", "[": "]", "{": "}"}
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == separator and not stack:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


def _split_static_key_value(value: str) -> tuple[str, str]:
    parts = _split_static_inline(value, separator=":")
    if len(parts) < 2:
        raise ValueError(f"expected key:value: {value!r}")
    return parts[0].strip(), ":".join(parts[1:]).strip()


def _set_unique_mapping(mapping: dict, key: Any, value: Any, *, context: str) -> None:
    if key in mapping:
        raise ValueError(f"duplicate key {key!r} in {context}")
    mapping[key] = value


def _yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_yaml_scalar(item)
                                     for item in _split_static_inline(inner)]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        result = {}
        if inner:
            for item in _split_static_inline(inner):
                key, raw = _split_static_key_value(item)
                _set_unique_mapping(
                    result, str(_yaml_scalar(key)), _yaml_scalar(raw),
                    context="inline YAML mapping",
                )
        return result
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        if value[0] == "'":
            return value[1:-1].replace("''", "'")
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    return value


def _parse_yaml_document(text: str) -> Any:
    rows = []
    for number, raw in enumerate(text.splitlines(), 1):
        raw = _strip_unquoted_comment(raw).rstrip()
        if not raw.strip():
            continue
        if "\t" in raw[:len(raw) - len(raw.lstrip())]:
            raise ValueError(f"YAML indentation tab at line {number}")
        rows.append((len(raw) - len(raw.lstrip(" ")), raw.strip(), number))
    if not rows:
        return {}

    def parse_block(index: int, indent: int):
        if index >= len(rows) or rows[index][0] < indent:
            return {}, index
        is_list = rows[index][0] == indent and rows[index][1].startswith("-")
        container: Any = [] if is_list else {}
        while index < len(rows):
            row_indent, content, number = rows[index]
            if row_indent < indent:
                break
            if row_indent > indent:
                raise ValueError(f"unexpected YAML indentation at line {number}")
            if is_list:
                if not content.startswith("-"):
                    break
                rest = content[1:].strip()
                index += 1
                if not rest:
                    if index < len(rows) and rows[index][0] > indent:
                        item, index = parse_block(index, rows[index][0])
                    else:
                        item = None
                    container.append(item)
                    continue
                if (":" in rest and not (
                        len(rest) >= 2 and rest[0] == rest[-1]
                        and rest[0] in "'\"")):
                    key, raw_value = _split_static_key_value(rest)
                    item = {str(_yaml_scalar(key)): _yaml_scalar(raw_value)}
                    if not raw_value and index < len(rows) and rows[index][0] > indent:
                        item[str(_yaml_scalar(key))], index = parse_block(
                            index, rows[index][0],
                        )
                    if index < len(rows) and rows[index][0] > indent:
                        continuation, index = parse_block(index, rows[index][0])
                        if not isinstance(continuation, dict):
                            raise ValueError(f"YAML list mapping expected at line {number}")
                        for continuation_key, continuation_value in continuation.items():
                            _set_unique_mapping(
                                item, continuation_key, continuation_value,
                                context=f"YAML list mapping at line {number}",
                            )
                    container.append(item)
                else:
                    container.append(_yaml_scalar(rest))
                continue

            if content.startswith("-"):
                break
            key, raw_value = _split_static_key_value(content)
            key = str(_yaml_scalar(key))
            index += 1
            if raw_value:
                _set_unique_mapping(
                    container, key, _yaml_scalar(raw_value),
                    context=f"YAML mapping at line {number}",
                )
            elif (index < len(rows) and (
                    rows[index][0] > indent
                    or (rows[index][0] == indent
                        and rows[index][1].startswith("-")))):
                nested, index = parse_block(index, rows[index][0])
                _set_unique_mapping(
                    container, key, nested,
                    context=f"YAML mapping at line {number}",
                )
            else:
                _set_unique_mapping(
                    container, key, None,
                    context=f"YAML mapping at line {number}",
                )
        return container, index

    parsed, end = parse_block(0, rows[0][0])
    if end != len(rows):
        raise ValueError(f"unparsed YAML content near line {rows[end][2]}")
    return parsed


def _parse_yaml_subset(text: str) -> Any:
    documents, current = [], []
    for raw in text.splitlines():
        if raw.strip() == "---":
            documents.append(_parse_yaml_document("\n".join(current)))
            current = []
        else:
            current.append(raw)
    documents.append(_parse_yaml_document("\n".join(current)))
    documents = [document for document in documents if document != {}]
    return documents[0] if len(documents) == 1 else documents


def _parse_hcl_subset(text: str) -> dict:
    root: dict[str, Any] = {"__blocks__": []}
    stack: list[str] = []
    frames: list[int] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = _strip_unquoted_comment(raw, markers=("#", "//")).strip()
        if not line:
            continue
        if line == "}":
            if not frames:
                raise ValueError(f"unmatched HCL close brace at line {number}")
            del stack[-frames.pop():]
            continue
        object_match = re.fullmatch(r"([A-Za-z_][\w-]*)\s*=\s*\{", line)
        if object_match:
            stack.append(object_match.group(1))
            frames.append(1)
            _static_path_set(root, stack, {})
            continue
        block_match = re.fullmatch(
            r"([A-Za-z_][\w-]*)(?:\s+(['\"])(.*?)\2)?"
            r"(?:\s+(['\"])(.*?)\4)?\s*\{",
            line,
        )
        if block_match:
            parts = [block_match.group(1)]
            if block_match.group(3) is not None:
                parts.append(block_match.group(3))
            if block_match.group(5) is not None:
                parts.append(block_match.group(5))
            stack.extend(parts)
            frames.append(len(parts))
            _static_path_set(root, stack, {})
            root["__blocks__"].append(".".join(stack))
            continue
        assignment = re.fullmatch(r"([A-Za-z_][\w-]*)\s*=\s*(.+)", line)
        if assignment:
            raw_value = assignment.group(2).strip()
            _static_path_set(root, [*stack, assignment.group(1)],
                             _yaml_scalar(raw_value))
            continue
        raise ValueError(f"unsupported HCL syntax at line {number}: {line[:80]!r}")
    if frames:
        raise ValueError(f"unclosed HCL blocks: {'.'.join(stack)}")
    return root


def _mask_ecmascript(text: str) -> tuple[str, list[str]]:
    chars = list(text)
    strings = []
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for pos in range(index, end):
                chars[pos] = " "
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated block comment")
            for pos in range(index, end + 2):
                if chars[pos] != "\n":
                    chars[pos] = " "
            index = end + 2
            continue
        if text[index] == "`":
            start = index
            template_open = index
            index += 1
            while index < len(text):
                if text[index] == "\\" and index + 1 < len(text):
                    index += 2
                    continue
                if text[index] == "`":
                    break
                if text.startswith("${", index):
                    for pos in range(start + 1, index):
                        if chars[pos] != "\n":
                            chars[pos] = " "
                    expression_start = index + 2
                    cursor, depth, quote = expression_start, 1, None
                    while cursor < len(text) and depth:
                        value = text[cursor]
                        if quote:
                            if value == "\\":
                                cursor += 2
                                continue
                            if value == quote:
                                quote = None
                        elif value in "'\"`":
                            quote = value
                        elif value == "{":
                            depth += 1
                        elif value == "}":
                            depth -= 1
                        cursor += 1
                    if depth:
                        raise ValueError("unterminated template interpolation")
                    inner_end = cursor - 1
                    inner_masked, _ = _mask_ecmascript(text[expression_start:inner_end])
                    chars[expression_start:inner_end] = list(inner_masked)
                    index = cursor
                    start = inner_end
                    continue
                index += 1
            if index >= len(text):
                raise ValueError("unterminated string literal")
            strings.append(text[template_open + 1:index])
            for pos in range(start + 1, index):
                if chars[pos] != "\n" and chars[pos] not in "{}":
                    chars[pos] = " "
            index += 1
            continue
        if text[index] in "'\"":
            quote, start = text[index], index
            index += 1
            value = []
            while index < len(text):
                if text[index] == "\\" and index + 1 < len(text):
                    value.extend((text[index], text[index + 1]))
                    index += 2
                    continue
                if text[index] == quote:
                    break
                value.append(text[index])
                index += 1
            if index >= len(text):
                raise ValueError("unterminated string literal")
            strings.append("".join(value))
            for pos in range(start + 1, index):
                if chars[pos] != "\n":
                    chars[pos] = " "
            index += 1
            continue
        index += 1
    return "".join(chars), strings


def _mask_ecmascript_comments(text: str) -> str:
    """Mask JS/C-style comments while retaining literal spelling and offsets."""
    chars = list(text)
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for position in range(index, end):
                chars[position] = " "
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated block comment")
            for position in range(index, end + 2):
                if chars[position] != "\n":
                    chars[position] = " "
            index = end + 2
            continue
        if text[index] in "'\"`":
            quote = text[index]
            index += 1
            while index < len(text):
                if text[index] == "\\" and index + 1 < len(text):
                    index += 2
                    continue
                if text[index] == quote:
                    index += 1
                    break
                index += 1
            else:
                raise ValueError("unterminated string literal")
            continue
        index += 1
    return "".join(chars)


def _mask_python_contract(text: str) -> tuple[str, str, list[str]]:
    """Return Python code-only and comment-only surfaces with stable offsets."""
    code_chars = list(text)
    literal_chars = list(text)
    strings: list[str] = []
    lines = text.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    if not lines or not text.endswith(("\n", "\r")):
        offsets.append(total)

    def absolute(position: tuple[int, int]) -> int:
        row, column = position
        return offsets[min(row - 1, len(offsets) - 1)] + column

    def blank(chars: list[str], start: int, end: int) -> None:
        for position in range(start, min(end, len(chars))):
            if chars[position] not in "\r\n":
                chars[position] = " "

    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        start, end = absolute(token.start), absolute(token.end)
        if token.type == tokenize.COMMENT:
            blank(code_chars, start, end)
            blank(literal_chars, start, end)
        elif token.type == tokenize.STRING:
            blank(code_chars, start, end)
            try:
                value = ast.literal_eval(token.string)
            except Exception:
                value = None
            if isinstance(value, str):
                strings.append(value)
    return "".join(code_chars), "".join(literal_chars), strings


def _mask_delimited_contract(
        text: str, *, line_markers: tuple[str, ...] = (),
        block_markers: tuple[tuple[str, str], ...] = (),
        quote_chars: str = "'\"", preserve_shebang: bool = False,
) -> tuple[str, str, list[str]]:
    """Mask comments/literals for small Haskell, OCaml, and shell contracts."""
    code_chars, literal_chars = list(text), list(text)
    strings: list[str] = []

    def blank(chars: list[str], start: int, end: int) -> None:
        for position in range(start, min(end, len(chars))):
            if chars[position] not in "\r\n":
                chars[position] = " "

    index = 0
    while index < len(text):
        marker = next((value for value in line_markers if text.startswith(value, index)), None)
        if (marker == "#" and preserve_shebang
                and (index == 0 or text[index - 1] in "\r\n")
                and text.startswith("#!", index)):
            marker = None
        if marker is not None:
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            blank(code_chars, index, end)
            blank(literal_chars, index, end)
            index = end
            continue
        block = next((pair for pair in block_markers if text.startswith(pair[0], index)), None)
        if block is not None:
            opening, closing = block
            depth, cursor = 1, index + len(opening)
            while cursor < len(text) and depth:
                if text.startswith(opening, cursor):
                    depth += 1
                    cursor += len(opening)
                elif text.startswith(closing, cursor):
                    depth -= 1
                    cursor += len(closing)
                else:
                    cursor += 1
            if depth:
                raise ValueError("unterminated block comment")
            blank(code_chars, index, cursor)
            blank(literal_chars, index, cursor)
            index = cursor
            continue
        if text[index] in quote_chars:
            quote, start = text[index], index
            index += 1
            value = []
            while index < len(text):
                if text[index] == "\\" and index + 1 < len(text):
                    value.extend((text[index], text[index + 1]))
                    index += 2
                    continue
                if text[index] == quote:
                    break
                value.append(text[index])
                index += 1
            if index >= len(text):
                raise ValueError("unterminated string literal")
            strings.append("".join(value))
            blank(code_chars, start, index + 1)
            index += 1
            continue
        index += 1
    return "".join(code_chars), "".join(literal_chars), strings


def _tsx_behavior_surface(masked: str, original: str) -> tuple[str, list[str]]:
    """Mask JSX child text while retaining tags and ``{...}`` code.

    This is a deliberately small lexical pass, not a TSX compiler.  It is
    sufficient to separate behavior regexes from user-visible text so labels
    cannot masquerade as hook calls or state transitions.  JSX labels are
    returned as explicit facts for ``jsx_texts`` contracts.
    """
    chars = list(masked)
    jsx_texts: list[str] = []

    def tag(start: int, end_limit: int) -> tuple[int, bool, bool] | None:
        if start >= end_limit or masked[start] != "<":
            return None
        cursor = start + 1
        while cursor < end_limit and masked[cursor].isspace():
            cursor += 1
        closing = cursor < end_limit and masked[cursor] == "/"
        if closing:
            cursor += 1
            while cursor < end_limit and masked[cursor].isspace():
                cursor += 1
        fragment = cursor < end_limit and masked[cursor] == ">"
        if not fragment and (cursor >= end_limit or not (masked[cursor].isalpha() or masked[cursor] in "_$")):
            return None
        quote, braces = None, 0
        cursor += 1 if fragment else 0
        if fragment:
            return cursor, closing, False
        while cursor < end_limit:
            character = masked[cursor]
            if quote:
                if character == "\\":
                    cursor += 2
                    continue
                if character == quote:
                    quote = None
            elif character in "'\"`":
                quote = character
            elif character == "{":
                braces += 1
            elif character == "}" and braces:
                braces -= 1
            elif character == ">" and braces == 0:
                before = masked[start:cursor].rstrip()
                return cursor + 1, closing, before.endswith("/")
            cursor += 1
        return None

    def matching_brace(start: int, end_limit: int) -> int | None:
        depth = 0
        for cursor in range(start, end_limit):
            if masked[cursor] == "{":
                depth += 1
            elif masked[cursor] == "}":
                depth -= 1
                if depth == 0:
                    return cursor
        return None

    def top_level_jsx_context(position: int, lower: int) -> bool:
        # Inspect only the immediately preceding token.  The old implementation
        # sliced and stripped the complete prefix for every '<', turning input
        # such as ``x<y;`` repeated many times into quadratic work.
        cursor = position - 1
        while cursor >= lower and masked[cursor].isspace():
            cursor -= 1
        if cursor < lower:
            return True
        if masked[cursor] in "=>([,!?:&|{;":
            return True
        end = cursor + 1
        start = end
        while start > lower and (
                masked[start - 1].isalnum() or masked[start - 1] in "_$"):
            start -= 1
        return masked[start:end] == "return"

    def process_element(start: int, end_limit: int) -> int | None:
        opened = tag(start, end_limit)
        if opened is None:
            return None
        cursor, closing, self_closing = opened
        if closing or self_closing:
            return cursor
        while cursor < end_limit:
            if masked[cursor] == "<":
                parsed = tag(cursor, end_limit)
                if parsed is None:
                    raise ValueError("malformed JSX tag")
                if parsed[1]:
                    return parsed[0]
                nested = process_element(cursor, end_limit)
                if nested is not None:
                    cursor = nested
                    continue
                # ``tag()`` recognized an opening tag, but the bounded lexical
                # pass could not find its closing element.  Previously the
                # parent retried the same '<' forever.  Malformed/partial TSX
                # is a parse failure and must never stall the full suite.
                raise ValueError("unterminated nested JSX element")
            if masked[cursor] == "{":
                close = matching_brace(cursor, end_limit)
                if close is None:
                    return None
                process_code(cursor + 1, close)
                cursor = close + 1
                continue
            run_start = cursor
            while cursor < end_limit and masked[cursor] not in "<{":
                cursor += 1
            if cursor > run_start:
                raw = original[run_start:cursor]
                normalized = re.sub(r"\s+", " ", raw).strip()
                if normalized:
                    jsx_texts.append(normalized)
                for position in range(run_start, cursor):
                    if chars[position] not in "\r\n":
                        chars[position] = " "
        return None

    def process_code(start: int, end_limit: int) -> None:
        cursor = start
        while cursor < end_limit:
            found = masked.find("<", cursor, end_limit)
            if found < 0:
                return
            if top_level_jsx_context(found, start):
                parsed = process_element(found, end_limit)
                if parsed is not None:
                    cursor = parsed
                    continue
            cursor = found + 1

    process_code(0, len(masked))
    return "".join(chars), jsx_texts


def _balanced_segment(text: str, opening: int, left: str, right: str) -> tuple[str, int] | None:
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == left:
            depth += 1
        elif text[position] == right:
            depth -= 1
            if depth == 0:
                return text[opening + 1:position], position + 1
    return None


def _tsx_event_expression(surface: str, event: str) -> str | None:
    expressions = _tsx_event_expressions(surface, event)
    return expressions[0] if expressions else None


def _tsx_event_expressions(surface: str, event: str) -> list[str]:
    expressions: list[str] = []
    cursor = 0
    pattern = re.compile(rf"\b{re.escape(event)}\s*=\s*\{{")
    while True:
        match = pattern.search(surface, cursor)
        if match is None:
            return expressions
        opening = surface.find("{", match.start())
        segment = _balanced_segment(surface, opening, "{", "}")
        if segment is None:
            return expressions
        expressions.append(segment[0].strip())
        cursor = segment[1]


def _tsx_named_handler_body(surface: str, expression: str) -> str:
    identifier = expression.strip()
    if not re.fullmatch(r"[A-Za-z_$][\w$]*", identifier):
        return expression
    function = re.search(
        rf"\bfunction\s+{re.escape(identifier)}\s*\([^)]*\)\s*\{{", surface,
    )
    if function:
        opening = surface.find("{", function.start())
        segment = _balanced_segment(surface, opening, "{", "}")
        return segment[0] if segment else expression
    arrow = re.search(
        rf"\b(?:const|let)\s+{re.escape(identifier)}\s*=\s*"
        rf"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)"
        rf"(?:\s*:\s*[A-Za-z_$][\w$]*(?:\s*<[^;={{}}]{{1,200}}>)?(?:\[\])?)?"
        rf"\s*=>\s*", surface,
    )
    if arrow:
        cursor = arrow.end()
        while cursor < len(surface) and surface[cursor].isspace():
            cursor += 1
        if cursor < len(surface) and surface[cursor] == "{":
            segment = _balanced_segment(surface, cursor, "{", "}")
            return segment[0] if segment else expression
        # Concise arrows can contain a nested block arrow, e.g.
        # ``add = () => setItems(prev => { ...; return ...; })``.  The first
        # semicolon is not the end of the outer expression.
        stack: list[str] = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        end = cursor
        while end < len(surface):
            value = surface[end]
            if value in pairs:
                stack.append(pairs[value])
            elif stack and value == stack[-1]:
                stack.pop()
            elif value == ";" and not stack:
                break
            end += 1
        return surface[cursor:end]
    return expression


def _tsx_setter_argument(handler: str, setter: str) -> str | None:
    call = re.search(rf"\b{re.escape(setter)}\s*\(", handler)
    if call is None:
        return None
    opening = handler.find("(", call.start())
    segment = _balanced_segment(handler, opening, "(", ")")
    return segment[0].strip() if segment else None


def _tsx_resolve_local_value(handler: str, value: str) -> str:
    identifier = value.strip()
    if not re.fullmatch(r"[A-Za-z_$][\w$]*", identifier):
        return value
    assignment = re.search(
        rf"\b(?:const|let)\s+{re.escape(identifier)}(?:\s*:[^=;]+)?\s*=\s*",
        handler,
    )
    if assignment is None:
        return value
    cursor = assignment.end()
    while cursor < len(handler) and handler[cursor].isspace():
        cursor += 1
    if cursor < len(handler) and handler[cursor] == "{":
        segment = _balanced_segment(handler, cursor, "{", "}")
        return "{" + segment[0] + "}" if segment else value
    end = handler.find(";", cursor)
    return handler[cursor:len(handler) if end < 0 else end].strip()


def _tsx_expand_handler(surface: str, expression: str) -> str:
    """Resolve a directly bound handler and a bounded local-helper call graph."""
    primary = _tsx_named_handler_body(surface, expression)
    parts = [primary]
    cursor = 0
    while cursor < len(parts) and len(parts) < 20:
        for name in re.findall(r"\b([A-Za-z_$][\w$]*)\s*\(", parts[cursor]):
            resolved = _tsx_named_handler_body(surface, name)
            if resolved != name and resolved not in parts:
                parts.append(resolved)
        cursor += 1
    return "\n".join(parts)


def _tsx_state_bindings(code_surface: str, literal_surface: str) -> list[tuple[str, str, str, str]]:
    bindings = []
    pattern = re.compile(
        r"\b(?:const|let)\s*\[\s*([A-Za-z_$][\w$]*)\s*,\s*"
        r"([A-Za-z_$][\w$]*)\s*\]\s*=\s*useState"
        r"(?:\s*<[^<>]{1,300}>)?\s*\("
    )
    for match in pattern.finditer(code_surface):
        opening = code_surface.find("(", match.start())
        code_argument = _balanced_segment(code_surface, opening, "(", ")")
        literal_argument = _balanced_segment(literal_surface, opening, "(", ")")
        if code_argument and literal_argument:
            bindings.append((match.group(1), match.group(2),
                             code_argument[0].strip(), literal_argument[0].strip()))
    return bindings


def _tsx_tag_segments(surface: str, name: str) -> list[tuple[int, int, str]]:
    """Extract opening tags, respecting braces inside JSX attributes."""
    result = []
    pattern = re.compile(rf"<\s*{re.escape(name)}\b")
    for match in pattern.finditer(surface):
        cursor, braces, quote = match.end(), 0, None
        while cursor < len(surface):
            character = surface[cursor]
            if quote:
                if character == "\\":
                    cursor += 2
                    continue
                if character == quote:
                    quote = None
            elif character in "'\"`":
                quote = character
            elif character == "{":
                braces += 1
            elif character == "}" and braces:
                braces -= 1
            elif character == ">" and braces == 0:
                result.append((match.start(), cursor + 1, surface[match.start():cursor + 1]))
                break
            cursor += 1
    return result


def _top_level_arguments(text: str) -> list[str]:
    arguments, start = [], 0
    stack = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    quote = None
    cursor = 0
    while cursor < len(text):
        character = text[cursor]
        if quote:
            if character == "\\":
                cursor += 2
                continue
            if character == quote:
                quote = None
        elif character in "'\"`":
            quote = character
        elif character in pairs:
            stack.append(pairs[character])
        elif stack and character == stack[-1]:
            stack.pop()
        elif character == "," and not stack:
            arguments.append(text[start:cursor].strip())
            start = cursor + 1
        cursor += 1
    arguments.append(text[start:].strip())
    return arguments


def _tsx_method_calls(surface: str, method: str) -> list[tuple[int, int, list[str]]]:
    calls = []
    pattern = re.compile(rf"\b(?:[A-Za-z_$][\w$]*\s*\.\s*)?{re.escape(method)}\s*\(")
    for match in pattern.finditer(surface):
        opening = surface.find("(", match.start())
        segment = _balanced_segment(surface, opening, "(", ")")
        if segment:
            calls.append((opening + 1, segment[1] - 1, _top_level_arguments(segment[0])))
    return calls


def _tsx_top_level_return_spans(body: str) -> list[tuple[int, int]]:
    """Return expression spans for returns owned by one function body.

    ``body`` is already comment/string masked and excludes the function's
    outer braces.  Tracking nested braces keeps returns in local handlers from
    being attributed to the component itself.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    cursor = 0
    while cursor < len(body):
        character = body[cursor]
        if character == "{":
            depth += 1
            cursor += 1
            continue
        if character == "}":
            depth = max(0, depth - 1)
            cursor += 1
            continue
        match = re.match(r"return\b", body[cursor:]) if depth == 0 else None
        if match is None:
            cursor += 1
            continue
        start = cursor + match.end()
        while start < len(body) and body[start].isspace():
            start += 1
        stack: list[str] = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        end = start
        while end < len(body):
            value = body[end]
            if value in pairs:
                stack.append(pairs[value])
            elif stack and value == stack[-1]:
                stack.pop()
            elif value == ";" and not stack:
                break
            end += 1
        if end > start:
            spans.append((start, end))
        cursor = min(len(body), end + 1)
    return spans


def _tsx_named_component_scope(surface: str, binding: str) -> dict[str, Any] | None:
    """Extract a named function/arrow component without compiling TSX."""
    name = re.escape(binding)
    declaration = re.search(
        rf"(?P<export>\bexport\s+(?:default\s+)?)?"
        rf"(?:async\s+)?function\s+{name}\b",
        surface,
    )
    if declaration:
        cursor = declaration.end()
        generic = False
        while cursor < len(surface) and surface[cursor].isspace():
            cursor += 1
        if cursor < len(surface) and surface[cursor] == "<":
            generic_segment = _balanced_segment(surface, cursor, "<", ">")
            if generic_segment is None:
                return None
            generic = True
            cursor = generic_segment[1]
        while cursor < len(surface) and surface[cursor].isspace():
            cursor += 1
        if cursor >= len(surface) or surface[cursor] != "(":
            return None
        parameters = _balanced_segment(surface, cursor, "(", ")")
        if parameters is None:
            return None
        cursor = parameters[1]
        opening = surface.find("{", cursor)
        if opening < 0:
            return None
        body = _balanced_segment(surface, opening, "{", "}")
        if body is None:
            return None
        body_start = opening + 1
        returns = [
            (body_start + start, body_start + end)
            for start, end in _tsx_top_level_return_spans(body[0])
        ]
        return {
            "start": declaration.start(), "end": body[1],
            "parameters": parameters[0], "generic": generic,
            "exported": declaration.group("export") is not None,
            "binding_annotation": "",
            "return_spans": returns, "body_span": (body_start, body[1] - 1),
        }

    arrow = re.search(
        rf"(?P<export>\bexport\s+)?(?:const|let|var)\s+{name}\b"
        rf"(?:\s*:(?P<annotation>[^=;]{{0,500}}))?\s*=\s*"
        rf"(?:async\s*)?(?P<params>\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
        surface,
    )
    if not arrow:
        return None
    raw_parameters = arrow.group("params")
    parameters = raw_parameters[1:-1] if raw_parameters.startswith("(") else raw_parameters
    cursor = arrow.end()
    while cursor < len(surface) and surface[cursor].isspace():
        cursor += 1
    if cursor < len(surface) and surface[cursor] == "{":
        body = _balanced_segment(surface, cursor, "{", "}")
        if body is None:
            return None
        body_start = cursor + 1
        returns = [
            (body_start + start, body_start + end)
            for start, end in _tsx_top_level_return_spans(body[0])
        ]
        end = body[1]
    else:
        start = cursor
        stack: list[str] = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        while cursor < len(surface):
            value = surface[cursor]
            if value in pairs:
                stack.append(pairs[value])
            elif stack and value == stack[-1]:
                stack.pop()
            elif value == ";" and not stack:
                break
            cursor += 1
        returns, end = [(start, cursor)], cursor
    return {
        "start": arrow.start(), "end": end, "parameters": parameters,
        "generic": False, "exported": arrow.group("export") is not None,
        "binding_annotation": (arrow.group("annotation") or "").strip(),
        "return_spans": returns,
        "body_span": ((cursor + 1, end - 1) if surface[cursor:cursor + 1] == "{"
                      else (returns[0][0], returns[0][1])),
    }


def _tsx_top_level_split(text: str, separators: str = ";,") -> list[str]:
    parts: list[str] = []
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    start = 0
    for position, character in enumerate(text):
        if character in pairs:
            stack.append(pairs[character])
        elif stack and character == stack[-1]:
            stack.pop()
        elif character in separators and not stack:
            parts.append(text[start:position].strip())
            start = position + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _tsx_prop_annotations(text: str, parameters: str) -> tuple[dict[str, str], str]:
    """Resolve inline, type-alias, and interface annotations for component props."""
    parameter = _tsx_top_level_split(parameters, separators=",")[0] if parameters.strip() else ""
    annotation = ""
    if parameter.lstrip().startswith("{"):
        opening = parameter.find("{")
        destructuring = _balanced_segment(parameter, opening, "{", "}")
        if destructuring:
            suffix = parameter[destructuring[1]:].strip()
            if suffix.startswith(":"):
                annotation = suffix[1:].strip()
    else:
        colon = next((position for position, value in enumerate(parameter) if value == ":"), -1)
        if colon >= 0:
            annotation = parameter[colon + 1:].strip()
    if not annotation:
        return {}, ""

    fields_source = annotation
    if not annotation.lstrip().startswith("{"):
        alias = re.match(r"([A-Za-z_$][\w$]*)", annotation)
        if alias:
            alias_name = re.escape(alias.group(1))
            declaration = re.search(
                rf"\btype\s+{alias_name}(?:\s*<[^;=]{{0,300}}>)?\s*=\s*\{{", text,
            )
            if declaration:
                opening = text.find("{", declaration.start())
                segment = _balanced_segment(text, opening, "{", "}")
                fields_source = "{" + segment[0] + "}" if segment else ""
            else:
                declaration = re.search(
                    rf"\binterface\s+{alias_name}(?:\s*<[^{{]{{0,300}}>)?\s*\{{", text,
                )
                if declaration:
                    opening = text.find("{", declaration.start())
                    segment = _balanced_segment(text, opening, "{", "}")
                    fields_source = "{" + segment[0] + "}" if segment else ""
    if not fields_source.lstrip().startswith("{"):
        return {}, annotation
    opening = fields_source.find("{")
    segment = _balanced_segment(fields_source, opening, "{", "}")
    if segment is None:
        return {}, annotation
    fields: dict[str, str] = {}
    for field in _tsx_top_level_split(segment[0]):
        match = re.match(r"([A-Za-z_$][\w$]*)\s*\??\s*:\s*([\s\S]+)\Z", field)
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields, annotation


def _normalize_ts_annotation(annotation: str) -> str:
    value = re.sub(r"\s+", "", annotation)
    value = value.replace("React.ReactNode", "ReactNode")
    return value


def _validate_tsx_component_contract(text: str, masked: str, contract: dict[str, Any]) -> str | None:
    binding = str(contract.get("binding") or contract.get("name") or "")
    if not binding:
        return "component_contract_missing_binding"
    scope = _tsx_named_component_scope(masked, binding)
    if scope is None:
        return f"component_contract_missing_binding={binding}"
    if "exported" in contract and bool(contract["exported"]) != bool(scope["exported"]):
        return f"component_export_mismatch={binding}"
    if contract.get("generic") and not scope["generic"]:
        return f"component_not_generic={binding}"
    fields, annotation = _tsx_prop_annotations(text, scope["parameters"])
    if not fields and scope.get("binding_annotation"):
        component_type = re.search(
            r"\b(?:React\s*\.\s*)?(?:FC|FunctionComponent)\s*<\s*"
            r"([A-Za-z_$][\w$]*(?:\s*<[^<>]+>)?)\s*>",
            scope["binding_annotation"],
        )
        if component_type:
            fields, annotation = _tsx_prop_annotations(
                text, f"props: {component_type.group(1)}",
            )
    if contract.get("forbid_props_any") and re.search(r"\bany\b", annotation):
        return f"component_props_any={binding}"
    for prop, expected in contract.get("props", {}).items():
        actual = fields.get(prop)
        alternatives = expected if isinstance(expected, list) else [expected]
        if actual is None or _normalize_ts_annotation(actual) not in {
                _normalize_ts_annotation(str(value)) for value in alternatives}:
            return f"component_prop_mismatch={binding}.{prop}:{actual}"

    return_checks = contract.get("return_checks") or contract.get("checks")
    if return_checks:
        if not scope["return_spans"]:
            return f"component_missing_return={binding}"
        validated = 0
        for start, end in scope["return_spans"]:
            candidate = text[start:end].strip()
            if not candidate:
                continue
            candidate_masked = masked[start:end].strip()
            if (re.match(r"^(?:false|null|undefined|0)\s*\?", candidate_masked)
                    or re.match(r"^true\s*\?\s*(?:null|undefined|false)\b", candidate_masked)):
                return f"component_dead_conditional_return={binding}"
            result = _validate_ecmascript_structure(candidate, return_checks, parser="tsx")
            if not result.passed:
                return f"component_return_mismatch={binding}:{result.note}"
            validated += 1
        if not validated:
            return f"component_return_mismatch={binding}:no_return_expression"
    return None


def _validate_distinct_switch_returns(text: str, masked: str,
                                      contract: dict[str, Any]) -> str | None:
    scope_text, scope_masked = text, masked
    binding = contract.get("binding")
    if binding:
        scope = _tsx_named_component_scope(masked, str(binding))
        if scope is None:
            return f"switch_missing_binding={binding}"
        scope_text = text[scope["start"]:scope["end"]]
        scope_masked = masked[scope["start"]:scope["end"]]
    discriminant = re.escape(str(contract.get("discriminant", "")))
    switch = re.search(rf"\bswitch\s*\(\s*{discriminant}\s*\)\s*\{{", scope_masked)
    if switch is None:
        return "distinct_switch_missing"
    opening = scope_masked.find("{", switch.start())
    body = _balanced_segment(scope_masked, opening, "{", "}")
    if body is None:
        return "distinct_switch_unbalanced"
    literal_body = scope_text[opening + 1:body[1] - 1]
    returned: list[str] = []
    for case in contract.get("cases", []):
        match = re.search(
            rf"\bcase\s*(['\"]){re.escape(str(case))}\1\s*:"
            rf"[\s\S]*?\breturn\s*(['\"])([^'\"]+)\2\s*;?",
            literal_body,
        )
        if match is None:
            return f"distinct_switch_missing_case={case}"
        returned.append(match.group(3))
    if len(set(returned)) != len(returned):
        return "distinct_switch_duplicate_returns"
    return None


def _validate_react_state_contract(code_surface: str, literal_surface: str,
                                   contract: dict[str, Any]) -> str | None:
    """Bind a JSX event to the setter and transition of one useState slot."""
    bindings = _tsx_state_bindings(code_surface, literal_surface)
    expected_initial = contract.get("initial")

    def initial_matches(code: str, literal: str) -> bool:
        if expected_initial == "empty_string":
            return bool(re.fullmatch(r"(['\"])\1", literal))
        return code == str(expected_initial)

    selected = next(
        (binding for binding in bindings if initial_matches(binding[2], binding[3])),
        None,
    )
    if selected is None:
        return f"react_state_initial={expected_initial}"
    state, setter, _, _ = selected
    event = str(contract.get("event", "onClick"))
    expression = _tsx_event_expression(code_surface, event)
    if expression is None:
        return f"react_state_missing_event={event}"
    handler = _tsx_expand_handler(code_surface, expression)
    argument = _tsx_setter_argument(handler, setter)
    if argument is None:
        return f"react_state_event_does_not_call={setter}"
    transition = contract.get("transition")
    if transition == "increment":
        direct = re.fullmatch(rf"{re.escape(state)}\s*\+\s*1", argument)
        updater = re.fullmatch(
            r"\(?\s*([A-Za-z_$][\w$]*)\s*\)?\s*=>\s*\1\s*\+\s*1",
            argument,
        )
        if not (direct or updater):
            return "react_state_transition_not_increment"
    elif transition == "toggle":
        direct = re.fullmatch(rf"!\s*{re.escape(state)}", argument)
        updater = re.fullmatch(
            r"\(?\s*([A-Za-z_$][\w$]*)\s*\)?\s*=>\s*!\s*\1",
            argument,
        )
        if not (direct or updater):
            return "react_state_transition_not_toggle"
    elif transition == "target_value":
        if re.search(r"\b[A-Za-z_$][\w$]*\.target\.value\b", argument) is None:
            return "react_state_transition_not_target_value"
    else:
        return f"unsupported_react_state_transition={transition}"
    if contract.get("display_state") and re.search(
            rf">[^<>{{}}]*\{{\s*{re.escape(state)}\b", code_surface) is None:
        return f"react_state_not_rendered={state}"
    if contract.get("controlled_value") and re.search(
            rf"\bvalue\s*=\s*\{{\s*{re.escape(state)}\s*\}}", code_surface) is None:
        return f"react_state_not_controlled={state}"
    if contract.get("conditional_state") and re.search(
            rf"\b{re.escape(state)}\s*&&\s*<", code_surface) is None:
        return f"react_state_not_conditional={state}"
    return None


def _is_empty_js_string(value: str) -> bool:
    return bool(re.fullmatch(r"(['\"])\1", value.strip()))


def _validate_login_form_contract(code: str, literal: str) -> str | None:
    states = [binding for binding in _tsx_state_bindings(code, literal)
              if _is_empty_js_string(binding[3])]
    if len(states) != 2:
        return "login_requires_two_empty_string_states"
    inputs = _tsx_tag_segments(code, "input")
    if len(inputs) < 2:
        return "login_requires_two_inputs"
    controlled: dict[tuple[int, int], str] = {}
    for state, setter, _, _ in states:
        matched = None
        for start, end, tag in inputs:
            if re.search(rf"\bvalue\s*=\s*\{{\s*{re.escape(state)}\s*\}}", tag) is None:
                continue
            expression = _tsx_event_expression(tag, "onChange")
            if expression is None:
                continue
            handler = _tsx_expand_handler(code, expression)
            argument = _tsx_setter_argument(handler, setter)
            if argument and re.search(r"\b[A-Za-z_$][\w$]*\.target\.value\b", argument):
                matched = (start, end)
                break
        if matched is None:
            return f"login_input_not_bound={state}:{setter}"
        controlled[matched] = state
    if len(controlled) != 2:
        return "login_states_share_one_input"
    password_positions = [
        (start, end) for start, end, _ in inputs
        if re.search(r"\btype\s*=\s*(['\"])password\1",
                     literal[start:end], re.IGNORECASE)
    ]
    if len(password_positions) != 1 or password_positions[0] not in controlled:
        return "login_missing_password_input"
    password_state = controlled[password_positions[0]]
    username_state = next(state for position, state in controlled.items()
                          if position != password_positions[0])
    valid = re.search(r"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*([^;\n]+)", code)
    valid_bindings = []
    for assignment in re.finditer(
            r"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*([^;\n]+)", code):
        expression = assignment.group(2)
        if ("&&" in expression
                and re.search(rf"\b{re.escape(username_state)}\.length\s*>\s*0", expression)
                and re.search(rf"\b{re.escape(password_state)}\.length\s*>=\s*4", expression)):
            valid_bindings.append(assignment.group(1))
    if len(valid_bindings) != 1:
        return "login_valid_not_bound_to_both_states"
    valid_name = valid_bindings[0]
    if re.search(
            rf"\bdisabled\s*=\s*\{{\s*!\s*{re.escape(valid_name)}\s*\}}", code,
    ) is None:
        return "login_submit_not_bound_to_valid"
    return None


def _validate_todo_list_contract(code: str, literal: str) -> str | None:
    bindings = _tsx_state_bindings(code, literal)
    text = next((item for item in bindings if _is_empty_js_string(item[3])), None)
    items = next((item for item in bindings if re.fullmatch(r"\[\s*\]", item[2])), None)
    if text is None or items is None:
        return "todo_missing_text_or_list_state"
    text_state, text_setter = text[0], text[1]
    items_state, items_setter = items[0], items[1]
    input_ok = False
    for _, _, tag in _tsx_tag_segments(code, "input"):
        if re.search(rf"\bvalue\s*=\s*\{{\s*{re.escape(text_state)}\s*\}}", tag) is None:
            continue
        expression = _tsx_event_expression(tag, "onChange")
        handler = _tsx_expand_handler(code, expression or "")
        argument = _tsx_setter_argument(handler, text_setter)
        if argument and ".target.value" in argument:
            input_ok = True
            break
    if not input_ok:
        return "todo_input_not_controlled"
    handlers = [
        _tsx_expand_handler(code, expression)
        for expression in _tsx_event_expressions(code, "onClick")
    ]
    add_ok = False
    remove_ok = False
    for handler in handlers:
        list_argument = _tsx_setter_argument(handler, items_setter)
        reset_argument = _tsx_setter_argument(handler, text_setter)
        if (list_argument and re.search(rf"\b{re.escape(text_state)}\b", list_argument)
                and (re.search(r"\.\.\.", list_argument)
                     or re.search(r"\.concat\s*\(", list_argument))
                and reset_argument and _is_empty_js_string(reset_argument)):
            guarded = bool(
                re.search(rf"\bif\s*\(\s*{re.escape(text_state)}\s*\)", handler)
                or re.search(rf"\bif\s*\([^)]*{re.escape(text_state)}\s*!==?\s*['\"]\s*['\"]", handler)
                or re.search(rf"\bif\s*\([^)]*{re.escape(text_state)}\.(?:trim\s*\(\s*\)|length)", handler)
                or re.search(rf"\bif\s*\(\s*!\s*{re.escape(text_state)}(?:\.trim\s*\(\s*\))?\s*\)\s*return", handler)
                or re.search(rf"\b{re.escape(text_state)}(?:\.trim\s*\(\s*\))?\s*&&", handler)
            )
            add_ok = guarded
        if list_argument and re.search(r"\.filter\s*\(", list_argument):
            map_match = re.search(
                rf"\b{re.escape(items_state)}\s*\.\s*map\s*\(\s*"
                r"\(\s*([A-Za-z_$][\w$]*)\s*,\s*([A-Za-z_$][\w$]*)\s*\)",
                code,
            )
            if map_match:
                outer_item, outer_index = map_match.groups()
                remove_ok = bool(
                    re.search(rf"=>[\s\S]*\b[A-Za-z_$][\w$]*\s*!==?\s*{re.escape(outer_index)}\b", list_argument)
                    or re.search(rf"=>[\s\S]*\.id\s*!==?\s*{re.escape(outer_item)}\.id\b", list_argument)
                )
    if not add_ok:
        return "todo_add_handler_not_bound"
    if not remove_ok:
        return "todo_remove_handler_not_bound"
    if re.search(rf"\b{re.escape(items_state)}\s*\.\s*map\s*\(", code) is None:
        return "todo_items_not_rendered"
    return None


def _validate_stepper_contract(code: str, literal: str) -> str | None:
    states = _tsx_state_bindings(code, literal)
    if sorted(item[2] for item in states) != ["0", "0", "10"]:
        return "stepper_state_initializers"
    inputs = _tsx_tag_segments(code, "input")
    if len(inputs) < 2:
        return "stepper_requires_two_inputs"
    controlled_names = []
    for _, _, tag in inputs:
        value = re.search(r"\bvalue\s*=\s*\{\s*([A-Za-z_$][\w$]*)\s*\}", tag)
        if value:
            controlled_names.append(value.group(1))
    if len(set(controlled_names)) != 2:
        return "stepper_boundary_inputs_not_controlled"
    by_name = {item[0]: item for item in states}
    controlled = [by_name[name] for name in controlled_names if name in by_name]
    if len(controlled) != 2:
        return "stepper_boundary_inputs_not_state"
    max_candidates = [item for item in controlled if item[2] == "10"]
    min_candidates = [item for item in controlled if item[2] == "0"]
    if len(max_candidates) != 1 or len(min_candidates) != 1:
        return "stepper_boundary_initializers"
    min_state, max_state = min_candidates[0], max_candidates[0]
    value_candidates = [item for item in states if item[0] not in set(controlled_names)]
    if len(value_candidates) != 1 or value_candidates[0][2] != "0":
        return "stepper_value_state"
    value_state = value_candidates[0]
    for boundary, other in ((min_state, max_state), (max_state, min_state)):
        state, setter = boundary[0], boundary[1]
        matched = False
        for _, _, tag in inputs:
            if re.search(rf"\bvalue\s*=\s*\{{\s*{re.escape(state)}\s*\}}", tag) is None:
                continue
            expression = _tsx_event_expression(tag, "onChange")
            handler = _tsx_expand_handler(code, expression or "")
            if all(_tsx_setter_argument(handler, required) is not None for required in (
                    setter, other[1], value_state[1])):
                if state == min_state[0]:
                    boundary_ok = bool(
                        re.search(rf"Math\.max\s*\(\s*{re.escape(max_state[0])}\b", handler)
                        or re.search(rf"\bif\s*\([^)]*>\s*{re.escape(max_state[0])}\b", handler)
                    )
                else:
                    boundary_ok = bool(
                        re.search(rf"Math\.min\s*\(\s*{re.escape(min_state[0])}\b", handler)
                        or re.search(rf"\bif\s*\([^)]*<\s*{re.escape(min_state[0])}\b", handler)
                    )
                clamp_ok = bool(
                    re.search(r"Math\.min\s*\(", handler)
                    and re.search(r"Math\.max\s*\(", handler)
                    or re.search(r"\bif\s*\([^)]*<[^)]*\)", handler)
                    and re.search(r"\bif\s*\([^)]*>[^)]*\)", handler)
                )
                if boundary_ok and clamp_ok:
                    matched = True
                    break
        if not matched:
            return f"stepper_boundary_handler_not_bound={state}"
    click_handlers = [
        _tsx_expand_handler(code, expression)
        for expression in _tsx_event_expressions(code, "onClick")
    ]
    decrement_ok = increment_ok = False
    for handler in click_handlers:
        argument = _tsx_setter_argument(handler, value_state[1])
        if argument and re.search(r"-\s*1\b", argument):
            decrement_ok = bool(
                re.search(rf"Math\.max\s*\(\s*{re.escape(min_state[0])}\b", handler)
                or re.search(rf"\bif\s*\([^)]*\b{re.escape(value_state[0])}\s*>\s*{re.escape(min_state[0])}\b", handler)
                or (re.search(r"Math\.max\s*\(", handler)
                    and re.search(rf"\b{re.escape(min_state[0])}\b", handler))
            )
        if argument and re.search(r"\+\s*1\b", argument):
            increment_ok = bool(
                re.search(rf"Math\.min\s*\(\s*{re.escape(max_state[0])}\b", handler)
                or re.search(rf"\bif\s*\([^)]*\b{re.escape(value_state[0])}\s*<\s*{re.escape(max_state[0])}\b", handler)
                or (re.search(r"Math\.min\s*\(", handler)
                    and re.search(rf"\b{re.escape(max_state[0])}\b", handler))
            )
    if not decrement_ok:
        return "stepper_missing_decrement"
    if not increment_ok:
        return "stepper_missing_increment"
    if re.search(rf">[^<>{{}}]*\{{\s*{re.escape(value_state[0])}\b", code) is None:
        return "stepper_value_not_rendered"
    return None


def _validate_tabs_contract(code: str, literal: str) -> str | None:
    states = [item for item in _tsx_state_bindings(code, literal) if item[2] == "0"]
    if len(states) != 1:
        return "tabs_requires_one_zero_state"
    state, setter, _, _ = states[0]
    handlers = [
        _tsx_expand_handler(code, expression)
        for expression in _tsx_event_expressions(code, "onClick")
    ]
    if not any(
            (argument := _tsx_setter_argument(handler, setter)) is not None
            and re.fullmatch(r"index", argument)
            for handler in handlers):
        return "tabs_click_does_not_set_index"
    if re.search(rf"\bpanels\s*\[\s*{re.escape(state)}\s*\]", code) is None:
        return "tabs_active_panel_not_rendered"
    return None


def _validate_drag_contract(code: str, literal: str) -> str | None:
    bindings = _tsx_state_bindings(code, literal)
    objects = [item for item in bindings if re.search(
        r"\{[^{}]*\bx\s*:[^{}]*\by\s*:", item[2], re.DOTALL,
    )]
    nullable = [item for item in bindings if item[2] == "null"]
    booleans = [item for item in bindings if item[2] in {"false", "true"}]
    if not objects:
        return "drag_missing_position_state"
    position = objects[0]
    offset = nullable[0] if nullable else (objects[1] if len(objects) > 1 else None)
    dragging = booleans[0] if not nullable and booleans else None
    if offset is None:
        return "drag_missing_offset_state"
    down_expression = _tsx_event_expression(code, "onMouseDown")
    down = _tsx_expand_handler(code, down_expression or "")
    offset_argument = _tsx_setter_argument(down, offset[1])
    if offset_argument:
        offset_argument = _tsx_resolve_local_value(down, offset_argument)
    if not (offset_argument
            and re.search(rf"clientX\s*-\s*{re.escape(position[0])}\.x", offset_argument)
            and re.search(rf"clientY\s*-\s*{re.escape(position[0])}\.y", offset_argument)):
        return "drag_mousedown_not_bound_to_offset"
    if dragging is not None and _tsx_setter_argument(down, dragging[1]) != "true":
        return "drag_mousedown_does_not_start"

    effect_span = None
    effect_callback = ""
    for start, end, arguments in _tsx_method_calls(code, "useEffect"):
        if arguments and "addEventListener" in arguments[0] and "removeEventListener" in arguments[0]:
            effect_span = (start, end)
            effect_callback = arguments[0]
            break
    if effect_span is None:
        return "drag_listeners_not_in_effect"
    gate_state = offset[0] if nullable else dragging[0]
    guarded = bool(
        re.search(rf"\bif\s*\(\s*!\s*{re.escape(gate_state)}\s*\)\s*return\b", effect_callback)
        or re.search(rf"\bif\s*\(\s*{re.escape(gate_state)}\s*\)\s*\{{[\s\S]*addEventListener", effect_callback)
    )
    if not guarded:
        return "drag_effect_not_gated_by_dragging"

    def listener_map(method: str) -> dict[str, str]:
        mapped = {}
        for start, end, code_args in _tsx_method_calls(code, method):
            if not (effect_span[0] <= start and end <= effect_span[1]):
                continue
            literal_args = _top_level_arguments(literal[start:end])
            if len(code_args) < 2 or len(literal_args) < 2:
                continue
            event_match = re.fullmatch(r"(['\"])(.*?)\1", literal_args[0])
            handler = code_args[1].strip()
            if event_match and re.fullmatch(r"[A-Za-z_$][\w$]*", handler):
                mapped[event_match.group(2)] = handler
        return mapped

    added, removed = listener_map("addEventListener"), listener_map("removeEventListener")
    for event in ("mousemove", "mouseup"):
        if event not in added or removed.get(event) != added[event]:
            return f"drag_listener_cleanup_mismatch={event}"
    move = _tsx_expand_handler(code, added["mousemove"])
    move_argument = _tsx_setter_argument(move, position[1])
    if not (move_argument
            and re.search(rf"clientX\s*-\s*{re.escape(offset[0])}\.(?:x|ox)", move_argument)
            and re.search(rf"clientY\s*-\s*{re.escape(offset[0])}\.(?:y|oy)", move_argument)):
        return "drag_mousemove_not_bound_to_position"
    up = _tsx_expand_handler(code, added["mouseup"])
    stop_setter = offset[1] if nullable else dragging[1]
    stop_value = "null" if nullable else "false"
    if _tsx_setter_argument(up, stop_setter) != stop_value:
        return "drag_mouseup_does_not_stop"
    return None


def _tsx_effect_arguments(code: str, literal: str) -> list[tuple[list[str], list[str]]]:
    effects = []
    for start, end, arguments in _tsx_method_calls(code, "useEffect"):
        effects.append((arguments, _top_level_arguments(literal[start:end])))
    return effects


def _tsx_named_handler_literal(surface: str, expression: str) -> str:
    """Literal-preserving counterpart to ``_tsx_named_handler_body``."""
    return _tsx_named_handler_body(surface, expression)


def _tsx_listener_pairs(code: str, literal: str, method: str) -> list[tuple[str, str]]:
    pairs = []
    for start, end, arguments in _tsx_method_calls(code, method):
        literal_arguments = _top_level_arguments(literal[start:end])
        if len(arguments) < 2 or len(literal_arguments) < 2:
            continue
        event = re.fullmatch(r"(['\"])(.*?)\1", literal_arguments[0].strip())
        if event:
            pairs.append((event.group(2), arguments[1].strip()))
    return pairs


def _validate_cart_contract(code: str, literal: str) -> str | None:
    lists = [item for item in _tsx_state_bindings(code, literal)
             if re.fullmatch(r"\[\s*\]", item[2])]
    if len(lists) != 1:
        return "cart_requires_one_empty_list_state"
    items, setter = lists[0][0], lists[0][1]

    total_name = None
    for match in re.finditer(
            rf"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*"
            rf"{re.escape(items)}\s*\.\s*reduce\s*\(", code):
        opening = code.find("(", match.start())
        segment = _balanced_segment(code, opening, "(", ")")
        if not segment:
            continue
        arguments = _top_level_arguments(segment[0])
        if (len(arguments) >= 2 and arguments[1] == "0"
                and all(not value for value in arguments[2:])
                and re.search(r"\.qty\s*\*\s*[A-Za-z_$][\w$]*\.price", arguments[0])):
            total_name = match.group(1)
            break
    if total_name is None:
        return "cart_total_not_derived_from_qty_price"
    if re.search(rf"\{{\s*{re.escape(total_name)}\s*\}}", code) is None:
        return "cart_total_not_rendered"
    if re.search(rf"\b{re.escape(items)}\s*\.\s*map\s*\(", code) is None:
        return "cart_items_not_rendered"

    handlers = [
        _tsx_expand_handler(code, expression)
        for expression in _tsx_event_expressions(code, "onClick")
    ]
    add_ok = remove_ok = False
    for handler in handlers:
        argument = _tsx_setter_argument(handler, setter)
        if not argument:
            continue
        updater = re.match(r"\(?\s*([A-Za-z_$][\w$]*)\s*\)?\s*=>", argument)
        previous = updater.group(1) if updater else ""
        if previous:
            add_ok = add_ok or bool(
                re.search(rf"\b{re.escape(previous)}\s*\.\s*find\s*\(", argument)
                and re.search(rf"\b{re.escape(previous)}\s*\.\s*map\s*\(", argument)
                and re.search(r"\bqty\s*:\s*[A-Za-z_$][\w$]*\.qty\s*\+\s*1\b", argument)
                and re.search(rf"\.\.\.\s*{re.escape(previous)}\b", argument)
                and re.search(r"\bqty\s*:\s*1\b", argument)
                and re.search(r"\bprice\s*:", argument)
            )
            remove_ok = remove_ok or bool(
                re.search(rf"\b{re.escape(previous)}\s*\.\s*filter\s*\(", argument)
                and re.search(r"\.id\s*!==?\s*[A-Za-z_$][\w$]*(?:\.id)?\b", argument)
            )
    if not add_ok:
        return "cart_add_does_not_update_or_append"
    if not remove_ok:
        return "cart_remove_does_not_filter_id"
    if not re.search(r"(['\"])a\1", literal) or re.search(r"\b5\b", code) is None:
        return "cart_missing_product_a_price_5"
    return None


def _validate_clock_contract(code: str, literal: str) -> str | None:
    states = [item for item in _tsx_state_bindings(code, literal)
              if re.search(r"\bnew\s+Date\s*\(\s*\)", item[2])]
    if len(states) != 1:
        return "clock_missing_date_state"
    state, setter = states[0][0], states[0][1]
    interval_ok = False
    for arguments, literal_arguments in _tsx_effect_arguments(code, literal):
        if not arguments:
            continue
        callback = _tsx_expand_handler(code, arguments[0])
        match = re.search(
            r"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*setInterval\s*\(", callback,
        )
        if not match:
            continue
        opening = callback.find("(", match.start())
        call = _balanced_segment(callback, opening, "(", ")")
        call_arguments = _top_level_arguments(call[0]) if call else []
        if (len(call_arguments) >= 2 and call_arguments[1] == "1000"
                and (setter_argument := _tsx_setter_argument(
                    _tsx_expand_handler(code, call_arguments[0]), setter
                )) is not None
                and re.search(r"\bnew\s+Date\s*\(\s*\)", setter_argument)
                and re.search(
                    rf"\breturn\s*(?:\(\s*\)\s*)?=>[\s\S]*?"
                    rf"clearInterval\s*\(\s*{re.escape(match.group(1))}\s*\)", callback,
                )
                and len(arguments) > 1 and re.fullmatch(r"\[\s*\]", arguments[1])):
            interval_ok = True
            break
    if not interval_ok:
        return "clock_interval_or_cleanup_mismatch"

    display_names = []
    direct_parts = True
    for method in ("getHours", "getMinutes", "getSeconds"):
        match = re.search(
            rf"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*String\s*\(\s*"
            rf"{re.escape(state)}\s*\.\s*{method}\s*\(\s*\)\s*\)\s*"
            rf"\.\s*padStart\s*\(\s*2\s*,\s*(['\"])0\2\s*\)", literal,
        )
        if not match:
            direct_parts = False
            break
        display_names.append(match.group(1))
    scope = _tsx_named_component_scope(code, "Clock")
    if scope is None or not scope["return_spans"]:
        return "clock_missing_component_return"
    rendered = "\n".join(literal[start:end] for start, end in scope["return_spans"])
    if direct_parts:
        if any(re.search(rf"\b{re.escape(name)}\b", rendered) is None for name in display_names):
            return "clock_time_parts_not_rendered"
        if rendered.count(":") < 2:
            return "clock_missing_hh_mm_ss_separators"
        return None

    # Equivalent compact form: map all three numeric Date fields through the
    # same two-digit formatter, join with colons, and render that assignment.
    compact = re.search(
        rf"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*\[\s*"
        rf"{re.escape(state)}\.getHours\s*\(\s*\)\s*,\s*"
        rf"{re.escape(state)}\.getMinutes\s*\(\s*\)\s*,\s*"
        rf"{re.escape(state)}\.getSeconds\s*\(\s*\)\s*,?\s*\]"
        rf"[\s\S]*?\.map\s*\([^;]+?padStart\s*\(\s*2\s*,\s*(['\"])0\2\s*\)"
        rf"[\s\S]*?\.join\s*\(\s*(['\"]):\3\s*\)",
        literal,
    )
    if compact and re.search(rf"\b{re.escape(compact.group(1))}\b", rendered):
        return None

    # Another correct form keeps the already formatted HH:MM:SS string in
    # state via a small formatter helper.
    initializer = states[0][2]
    helper = re.search(r"\b([A-Za-z_$][\w$]*)\s*\(\s*new\s+Date", initializer)
    if helper:
        helper_body = _tsx_named_handler_literal(literal, helper.group(1))
        if (all(name in helper_body for name in ("getHours", "getMinutes", "getSeconds"))
                and len(re.findall(r"padStart\s*\(\s*2\s*,\s*(['\"])0\1\s*\)", helper_body)) >= 3
                and helper_body.count(":") >= 2
                and re.search(rf"\b{re.escape(state)}\b", rendered)):
            return None
    return "clock_missing_hh_mm_ss_format"


def _validate_window_size_contract(code: str, literal: str) -> str | None:
    bindings = _tsx_state_bindings(code, literal)
    if len(bindings) != 1:
        return "window_size_requires_exactly_one_state"
    state, setter, initial, _ = bindings[0]
    if not (re.search(r"\bw\s*:\s*window\.innerWidth\b", initial)
            and re.search(r"\bh\s*:\s*window\.innerHeight\b", initial)):
        return "window_size_initializer_mismatch"
    effect_ok = False
    for arguments, literal_arguments in _tsx_effect_arguments(code, literal):
        if not arguments or not literal_arguments:
            continue
        callback = _tsx_expand_handler(code, arguments[0])
        literal_callback = _tsx_named_handler_literal(literal, literal_arguments[0])
        added = _tsx_listener_pairs(callback, literal_callback, "addEventListener")
        removed = _tsx_listener_pairs(callback, literal_callback, "removeEventListener")
        add_handlers = [handler for event, handler in added if event == "resize"]
        remove_handlers = [handler for event, handler in removed if event == "resize"]
        if len(add_handlers) != 1 or remove_handlers != add_handlers:
            continue
        handler = _tsx_expand_handler(code, add_handlers[0])
        argument = _tsx_setter_argument(handler, setter)
        if (argument and re.search(r"\bw\s*:\s*window\.innerWidth\b", argument)
                and re.search(r"\bh\s*:\s*window\.innerHeight\b", argument)
                and len(arguments) > 1 and re.fullmatch(r"\[\s*\]", arguments[1])):
            effect_ok = True
            break
    if not effect_ok:
        return "window_size_listener_or_cleanup_mismatch"
    if not (re.search(rf"\{{\s*{re.escape(state)}\.w\s*\}}", code)
            and re.search(rf"\{{\s*{re.escape(state)}\.h\s*\}}", code)):
        return "window_size_not_rendered"
    return None


def _validate_debounced_search_contract(code: str, literal: str) -> str | None:
    bindings = _tsx_state_bindings(code, literal)
    empty_strings = [item for item in bindings if _is_empty_js_string(item[3])]
    lists = [item for item in bindings if re.fullmatch(r"\[\s*\]", item[2])]
    if len(empty_strings) != 2 or len(lists) != 1:
        return "search_state_shape_mismatch"
    query = None
    for candidate in empty_strings:
        state, setter = candidate[0], candidate[1]
        for _, _, tag in _tsx_tag_segments(code, "input"):
            if not re.search(rf"\bvalue\s*=\s*\{{\s*{re.escape(state)}\s*\}}", tag):
                continue
            expression = _tsx_event_expression(tag, "onChange")
            handler = _tsx_expand_handler(code, expression or "")
            argument = _tsx_setter_argument(handler, setter)
            if argument and re.search(r"\.target\.value\b", argument):
                query = candidate
                break
    if query is None:
        return "search_query_input_not_controlled"
    debounced = next(item for item in empty_strings if item is not query)
    results = lists[0]
    debounce_ok = fetch_ok = False
    for arguments, literal_arguments in _tsx_effect_arguments(code, literal):
        if len(arguments) < 2 or not literal_arguments:
            continue
        callback = _tsx_expand_handler(code, arguments[0])
        literal_callback = _tsx_named_handler_literal(literal, literal_arguments[0])
        dependency = re.sub(r"\s+", "", arguments[1])
        timer = re.search(
            r"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*"
            r"(?:window\s*\.\s*)?setTimeout\s*\(", callback,
        )
        if timer:
            opening = callback.find("(", timer.start())
            call = _balanced_segment(callback, opening, "(", ")")
            timer_arguments = _top_level_arguments(call[0]) if call else []
            timer_handler = _tsx_expand_handler(code, timer_arguments[0]) if timer_arguments else ""
            if (len(timer_arguments) >= 2 and timer_arguments[1] == "300"
                    and _tsx_setter_argument(timer_handler, debounced[1]) == query[0]
                    and re.search(
                        rf"\breturn\s*(?:\(\s*\)\s*)?=>[\s\S]*?"
                        rf"(?:window\s*\.\s*)?clearTimeout\s*\(\s*"
                        rf"{re.escape(timer.group(1))}\s*\)", callback)
                    and dependency == f"[{query[0]}]"):
                debounce_ok = True

        controller = re.search(
            r"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*new\s+AbortController\s*\(\s*\)",
            callback,
        )
        if controller:
            name = controller.group(1)
            guarded = bool(re.search(
                rf"\bif\s*\(\s*!\s*{re.escape(debounced[0])}\s*\)"
                rf"\s*(?:return\b|\{{[\s\S]{{0,240}}?\breturn\b)", callback,
            ))
            fetches = _tsx_method_calls(callback, "fetch")
            valid_fetch = False
            for start, end, fetch_arguments in fetches:
                literal_fetch = _top_level_arguments(literal_callback[start:end])
                if len(fetch_arguments) < 2 or len(literal_fetch) < 2:
                    continue
                endpoint = bool(
                    re.search(
                        r"(['\"])/api\?q=\1\s*\+\s*(?:encodeURIComponent\s*\(\s*)?"
                        + re.escape(debounced[0]), literal_fetch[0],
                    )
                    or re.search(
                        r"`/api\?q=\$\{\s*(?:encodeURIComponent\s*\(\s*)?"
                        + re.escape(debounced[0]) + r"\b", literal_fetch[0],
                    )
                )
                direct_signal = re.search(
                    rf"\bsignal\s*:\s*{re.escape(name)}\.signal\b", fetch_arguments[1],
                )
                shorthand_signal = (
                    re.search(
                        rf"\b(?:const|let)\s*\{{\s*signal\s*\}}\s*=\s*{re.escape(name)}\b",
                        callback,
                    )
                    and re.search(r"\{\s*signal\s*\}", fetch_arguments[1])
                )
                if endpoint and (direct_signal or shorthand_signal):
                    valid_fetch = True
                    break
            cleanup = re.search(
                rf"\breturn\s*(?:\(\s*\)\s*)?=>[\s\S]*?"
                rf"{re.escape(name)}\.abort\s*\(\s*\)", callback,
            )
            if (guarded and valid_fetch and cleanup
                    and dependency == f"[{debounced[0]}]"
                    and re.search(
                        rf"\.then\s*\([\s\S]{{0,300}}?\b{re.escape(results[1])}"
                        rf"\s*(?:\(|\))", callback,
                    )):
                fetch_ok = True
    if not debounce_ok:
        return "search_debounce_or_cleanup_mismatch"
    if not fetch_ok:
        return "search_fetch_signal_or_abort_mismatch"
    if re.search(rf"\b{re.escape(results[0])}\s*\.\s*map\s*\(", code) is None:
        return "search_results_not_rendered"
    return None


def _validate_visibility_poller_contract(code: str, literal: str) -> str | None:
    strings = [item for item in _tsx_state_bindings(code, literal)
               if _is_empty_js_string(item[3])]
    if len(strings) != 1:
        return "poller_requires_status_string_state"
    status_state, status_setter = strings[0][0], strings[0][1]
    valid = False
    for arguments, literal_arguments in _tsx_effect_arguments(code, literal):
        if len(arguments) < 2 or not re.fullmatch(r"\[\s*\]", arguments[1]):
            continue
        effect_expression = arguments[0]
        callback = _tsx_expand_handler(code, effect_expression)
        literal_callback = _tsx_expand_handler(literal, literal_arguments[0])
        interval = re.search(
            r"\b(?:let|const)\s+([A-Za-z_$][\w$]*)"
            r"(?:\s*:[^=;]+)?(?:\s*=\s*(null|undefined))?\s*;", callback,
        )
        if interval is None:
            continue
        interval_id = interval.group(1)
        sentinel = interval.group(2) or "undefined"
        added = _tsx_listener_pairs(callback, literal_callback, "addEventListener")
        removed = _tsx_listener_pairs(callback, literal_callback, "removeEventListener")
        handlers = [handler for event, handler in added if event == "visibilitychange"]
        if len(handlers) != 1 or [handler for event, handler in removed
                                if event == "visibilitychange"] != handlers:
            continue
        visibility_name = handlers[0]
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", visibility_name):
            continue
        visibility = _tsx_expand_handler(code, visibility_name)
        branch = re.search(
            r"\bif\s*\(\s*document\.hidden\s*\)\s*\{?\s*"
            r"([A-Za-z_$][\w$]*)\s*\(\s*\)\s*;?\s*\}?\s*else\s*\{?\s*"
            r"([A-Za-z_$][\w$]*)\s*\(\s*\)", visibility,
        )
        if branch is None:
            continue
        stop_name, start_name = branch.groups()
        stop = _tsx_named_handler_body(code, stop_name)
        start = _tsx_named_handler_body(code, start_name)
        if stop == stop_name or start == start_name:
            continue
        stop_ok = bool(
            re.search(
                rf"\bif\s*\([^)]*{re.escape(interval_id)}\s*!==?\s*"
                rf"{re.escape(sentinel)}", stop)
            and re.search(rf"clearInterval\s*\(\s*{re.escape(interval_id)}\s*\)", stop)
            and re.search(
                rf"\b{re.escape(interval_id)}\s*=\s*{re.escape(sentinel)}\b", stop)
        )
        start_call = re.search(
            rf"\b{re.escape(interval_id)}\s*=\s*(?:window\s*\.\s*)?setInterval\s*\(", start,
        )
        if not start_call:
            continue
        opening = start.find("(", start_call.start())
        interval_segment = _balanced_segment(start, opening, "(", ")")
        interval_arguments = _top_level_arguments(interval_segment[0]) if interval_segment else []
        positive_gate = re.search(
            rf"\bif\s*\([^)]*{re.escape(interval_id)}\s*===?\s*"
            rf"{re.escape(sentinel)}[^)]*\)\s*(?:\{{[\s\S]*?)?"
            rf"{re.escape(interval_id)}\s*=", start,
        )
        early_return_gate = re.search(
            rf"\bif\s*\([^)]*{re.escape(interval_id)}\s*!==?\s*"
            rf"{re.escape(sentinel)}[^)]*\)\s*\{{?\s*return\b[\s\S]*?"
            rf"{re.escape(interval_id)}\s*=", start,
        )
        start_ok = bool(
            (positive_gate or early_return_gate)
            and len(interval_arguments) >= 2 and interval_arguments[1] == "5000"
        )
        poll = _tsx_expand_handler(code, interval_arguments[0]) if interval_arguments else ""
        poll_literal = _tsx_expand_handler(literal, interval_arguments[0]) if interval_arguments else ""
        poll_ok = bool(
            re.search(r"fetch\s*\(\s*(['\"])/api/status\1\s*\)", poll_literal)
            and re.search(r"\.text\s*\(\s*\)", poll)
            and (re.search(rf"\.then\s*\(\s*{re.escape(status_setter)}\s*\)", poll)
                 or re.search(rf"\b{re.escape(status_setter)}\s*\(", poll))
        )

        def direct_effect_call(name: str) -> bool:
            opening = effect_expression.find("{")
            if opening < 0:
                return False
            depth = 0
            pattern = re.compile(rf"\b{re.escape(name)}\s*\(\s*\)\s*;")
            for position, value in enumerate(effect_expression):
                if value == "{":
                    depth += 1
                elif value == "}":
                    depth -= 1
                if depth == 1 and pattern.match(effect_expression, position):
                    return True
            return False

        startup = (
            direct_effect_call(visibility_name)
            or direct_effect_call(start_name)
            or bool(re.search(
                rf"\bif\s*\(\s*!\s*document\.hidden\s*\)\s*\{{?\s*"
                rf"{re.escape(start_name)}\s*\(\s*\)", effect_expression,
            ))
        )
        cleanup_ok = False
        cleanup_match = re.search(
            r"\breturn\s*(?:\(\s*\)\s*)?=>\s*\{", effect_expression,
        )
        if cleanup_match:
            cleanup_opening = effect_expression.find("{", cleanup_match.start())
            cleanup_segment = _balanced_segment(effect_expression, cleanup_opening, "{", "}")
            cleanup_body = cleanup_segment[0] if cleanup_segment else ""
            cleanup_ok = bool(
                re.search(rf"\b{re.escape(stop_name)}\s*\(\s*\)", cleanup_body)
                and re.search(
                    rf"removeEventListener\s*\([^)]*\b{re.escape(visibility_name)}\b",
                    cleanup_body,
                )
            )
        if stop_ok and start_ok and poll_ok and startup and cleanup_ok:
            valid = True
            break
    if not valid:
        return "poller_visibility_interval_contract_mismatch"
    if re.search(rf"\{{\s*{re.escape(status_state)}\b", code) is None:
        return "poller_status_not_rendered"
    return None


def _validate_react_behavior_contract(code: str, literal: str,
                                      contract: dict[str, Any]) -> str | None:
    kind = contract.get("kind")
    validators = {
        "login_form": _validate_login_form_contract,
        "todo_list": _validate_todo_list_contract,
        "stepper": _validate_stepper_contract,
        "tabs": _validate_tabs_contract,
        "drag": _validate_drag_contract,
        "cart": _validate_cart_contract,
        "clock": _validate_clock_contract,
        "window_size": _validate_window_size_contract,
        "debounced_search": _validate_debounced_search_contract,
        "visibility_poller": _validate_visibility_poller_contract,
    }
    validator = validators.get(kind)
    return validator(code, literal) if validator else f"unsupported_react_behavior={kind}"


def _balanced_static_delimiters(text: str) -> bool:
    stack = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    for char in text:
        if char in pairs:
            stack.append(pairs[char])
        elif char in ")]}":
            if not stack or stack.pop() != char:
                return False
    return not stack


def _validate_csharp_max_contract(code: str) -> str | None:
    """Validate the small, non-executing C# ``Util.Max`` translation task."""
    if re.search(r"\bclass\s+Util\b", code) is None:
        return "csharp_max_missing_class"
    signature = re.search(
        r"\bpublic\s+static\s+int\s+Max\s*\(\s*int\s+"
        r"([A-Za-z_]\w*)\s*,\s*int\s+([A-Za-z_]\w*)\s*\)\s*(=>|\{)",
        code,
    )
    if signature is None:
        return "csharp_max_missing_signature"
    left, right, form = signature.groups()

    def expression_ok(expression: str) -> bool:
        expression = expression.strip()
        alternatives = (
            rf"Math\.Max\s*\(\s*{re.escape(left)}\s*,\s*{re.escape(right)}\s*\)",
            rf"{re.escape(left)}\s*>=?\s*{re.escape(right)}\s*\?\s*"
            rf"{re.escape(left)}\s*:\s*{re.escape(right)}",
            rf"{re.escape(right)}\s*>=?\s*{re.escape(left)}\s*\?\s*"
            rf"{re.escape(right)}\s*:\s*{re.escape(left)}",
        )
        return any(re.fullmatch(pattern, expression) for pattern in alternatives)

    if form == "=>":
        end = code.find(";", signature.end())
        if end < 0 or not expression_ok(code[signature.end():end]):
            return "csharp_max_expression_mismatch"
        return None

    opening = code.find("{", signature.start())
    method = _balanced_segment(code, opening, "{", "}")
    if method is None:
        return "csharp_max_unbalanced_method"
    body = method[0].strip()
    one_return = re.fullmatch(r"return\s+([\s\S]+?)\s*;", body)
    if one_return and expression_ok(one_return.group(1)):
        return None
    # Equivalent branch forms, with or without an explicit ``else``.
    forward = re.fullmatch(
        rf"if\s*\(\s*{re.escape(left)}\s*>=?\s*{re.escape(right)}\s*\)\s*"
        rf"(?:\{{\s*)?return\s+{re.escape(left)}\s*;\s*(?:\}}\s*)?"
        rf"(?:else\s*)?(?:\{{\s*)?return\s+{re.escape(right)}\s*;\s*(?:\}}\s*)?",
        body,
    )
    reverse = re.fullmatch(
        rf"if\s*\(\s*{re.escape(right)}\s*>=?\s*{re.escape(left)}\s*\)\s*"
        rf"(?:\{{\s*)?return\s+{re.escape(right)}\s*;\s*(?:\}}\s*)?"
        rf"(?:else\s*)?(?:\{{\s*)?return\s+{re.escape(left)}\s*;\s*(?:\}}\s*)?",
        body,
    )
    return None if forward or reverse else "csharp_max_body_mismatch"


def _validate_static_source_contract(text: str, checks: dict, *,
                                     strategy: str = "source_contract_passed",
                                     extracted: dict | None = None,
                                     code_text: str | None = None,
                                     literal_text: str | None = None,
                                     jsx_text_surface: str | None = None) -> ScoreResult:
    """Validate a probe-local, non-executing source contract.

    This is deliberately smaller than a language implementation.  It is used
    only after the relevant language parser has accepted basic structure (or
    for languages for which ACE has no compiler).  Contracts are positive and
    negative regex facts anchored to the submitted source; they avoid global
    substring leniency and keep each accepted equivalence explicit in probe
    data.  Patterns are capped to prevent pathological configurations.
    """
    flags = re.MULTILINE | re.DOTALL
    if not checks.get("case_sensitive", True):
        flags |= re.IGNORECASE

    def compile_pattern(pattern: str) -> re.Pattern:
        if not isinstance(pattern, str) or len(pattern) > 2_000:
            raise ValueError("invalid_source_contract_pattern")
        return re.compile(pattern, flags)

    code_surface = text if code_text is None else code_text
    literal_surface = text if literal_text is None else literal_text
    def literal_matches(pattern: str) -> list[re.Match]:
        """Require a literal-aware match to include actual surrounding code.

        A regex copied wholesale into one string literal has no word character
        left on ``code_surface`` and is therefore not a valid binding.
        """
        matches = []
        for match in compile_pattern(pattern).finditer(literal_surface):
            code_fragment = code_surface[match.start():match.end()]
            if re.search(r"[A-Za-z_$][\w$]*", code_fragment):
                matches.append(match)
        return matches

    def jsx_dependent(pattern: str) -> bool:
        # Only patterns that explicitly mention a tag boundary may consult JSX
        # child text.  Thus ``<div>useState(0)</div>`` cannot satisfy a plain
        # behavior requirement such as ``useState(0)``.
        return bool(
            re.search(r"<(?:/?[A-Za-z]|\\s|\\b)", pattern)
            or re.search(r">(?:\\s|[^<]{1,80})<", pattern)
        )

    def required_matches(pattern: str) -> bool:
        compiled = compile_pattern(pattern)
        if compiled.search(code_surface) is not None:
            return True
        # Legacy contracts that bind code to a string literal are interpreted
        # on the comments-only surface, but a match must retain a real code
        # identifier outside the literal.  New contracts should use the
        # explicit required_literal_patterns field below.
        if literal_surface != code_surface and literal_matches(pattern):
            return True
        # Likewise, legacy JSX label patterns are allowed only when they name a
        # tag boundary.  The surface has comments and strings masked.
        return bool(
            jsx_text_surface is not None and jsx_dependent(pattern)
            and compiled.search(jsx_text_surface) is not None
            or jsx_text_surface is not None
            and any(ord(character) > 127 for character in pattern)
            and compiled.search(jsx_text_surface) is not None
        )

    matched: list[str] = []
    for pattern in checks.get("required_patterns", []):
        if not required_matches(pattern):
            return ScoreResult(False, "source_contract_failed",
                               note=f"missing_pattern={pattern[:120]!r}",
                               extracted=extracted)
        matched.append(pattern)
    for alternatives in checks.get("required_any_patterns", []):
        if (not isinstance(alternatives, list) or not alternatives
                or not any(required_matches(pattern) for pattern in alternatives)):
            return ScoreResult(False, "source_contract_failed",
                               note="missing_pattern_alternative",
                               extracted=extracted)
    for pattern in checks.get("forbidden_patterns", []):
        if compile_pattern(pattern).search(code_surface) is not None:
            return ScoreResult(False, "source_contract_failed",
                               note=f"forbidden_pattern={pattern[:120]!r}",
                               extracted=extracted)
    for pattern, expected_count in checks.get("pattern_counts", {}).items():
        actual = len(compile_pattern(pattern).findall(code_surface))
        if actual != int(expected_count):
            return ScoreResult(False, "source_contract_failed",
                               note=f"pattern_count={actual}, expected={expected_count}",
                               extracted=extracted)

    for pattern in checks.get("required_literal_patterns", []):
        if not literal_matches(pattern):
            return ScoreResult(False, "source_contract_failed",
                               note=f"missing_literal_pattern={pattern[:120]!r}",
                               extracted=extracted)
    for alternatives in checks.get("required_any_literal_patterns", []):
        if (not isinstance(alternatives, list) or not alternatives
                or not any(literal_matches(pattern) for pattern in alternatives)):
            return ScoreResult(False, "source_contract_failed",
                               note="missing_literal_pattern_alternative",
                               extracted=extracted)
    for pattern, expected_count in checks.get("literal_pattern_counts", {}).items():
        actual = len(literal_matches(pattern))
        if actual != int(expected_count):
            return ScoreResult(False, "source_contract_failed",
                               note=f"literal_pattern_count={actual}, expected={expected_count}",
                               extracted=extracted)
    for pattern in checks.get("required_jsx_patterns", []):
        if (jsx_text_surface is None or not jsx_dependent(pattern)
                or compile_pattern(pattern).search(jsx_text_surface) is None):
            return ScoreResult(False, "source_contract_failed",
                               note=f"missing_jsx_pattern={pattern[:120]!r}",
                               extracted=extracted)
    facts = dict(extracted or {})
    facts["matched_contract_patterns"] = len(matched)
    return ScoreResult(True, strategy, extracted=facts)


def _validate_ecmascript_structure(text: str, checks: dict, *,
                                   parser: str) -> ScoreResult:
    masked, string_literals = _mask_ecmascript(text)
    if not _balanced_static_delimiters(masked):
        return ScoreResult(False, "parse_failed", note="unbalanced_delimiters")
    literal_surface = _mask_ecmascript_comments(text)
    behavior_surface, jsx_texts = (
        _tsx_behavior_surface(masked, text) if parser == "tsx" else (masked, [])
    )
    bindings = set(re.findall(
        r"\b(?:function|class)\s+([A-Za-z_$][\w$]*)", masked,
    ))
    bindings.update(re.findall(
        r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=;]+)?="
        r"\s*(?:(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>|function\b)",
        masked,
    ))
    if parser == "tsx":
        invalid_bare = re.compile(
            r"(?m)(?:^|[;{}])\s*[A-Za-z_$][\w$]*"
            r"(?:[ \t]+[A-Za-z_$][\w$]*){3,}\s*;",
        )
        for binding in bindings:
            scope = _tsx_named_component_scope(masked, binding)
            if scope is None:
                continue
            start, end = scope["body_span"]
            if invalid_bare.search(masked[start:end]):
                return ScoreResult(False, "parse_failed",
                                   note="invalid_bare_identifier_sequence")
    call_names = re.findall(
        r"\b([A-Za-z_$][\w$]*(?:\s*\.\s*[A-Za-z_$][\w$]*)*)"
        r"\s*(?:<[^<>\n]{1,160}>)?\s*\(",
        masked,
    )
    call_names = [re.sub(r"\s+", "", name) for name in call_names]
    call_counts = {name: call_names.count(name) for name in set(call_names)}
    imported_names, import_sources = set(), set()
    for match in re.finditer(
            r"(?m)^[ \t]*(import)\s+([\s\S]*?)\s+from\s+"
            r"(['\"])([^'\"]+)\3\s*;?", text):
        if masked[match.start(1):match.end(1)] != "import":
            continue
        import_sources.add(match.group(4))
        clause = masked[match.start(2):match.end(2)].strip()
        imported_names.update(re.findall(r"\b([A-Za-z_$][\w$]*)\b", clause))
    for match in re.finditer(
            r"(?m)^[ \t]*(import)\s+(['\"])([^'\"]+)\2\s*;?", text):
        if masked[match.start(1):match.end(1)] != "import":
            continue
        import_sources.add(match.group(3))

    closing_tags = set(re.findall(r"</\s*([A-Za-z][\w.]*)\s*>", masked))
    tag_counts: dict[str, int] = {}
    for match in re.finditer(r"<\s*([A-Za-z][\w.]*)\b([^<>]*?)(/?)>", masked):
        tag = match.group(1)
        if match.group(3) or tag in closing_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    facts = {
        "bindings": sorted(bindings), "calls": sorted(call_counts),
        "call_counts": call_counts, "jsx_tag_counts": tag_counts,
        "imports": sorted(imported_names), "import_sources": sorted(import_sources),
        "string_literals": string_literals, "jsx_texts": jsx_texts,
    }
    for required in checks.get("bindings", []):
        if required not in bindings:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_binding={required}", extracted=facts)
    for required in checks.get("calls", []):
        if not any(call == required or call.endswith("." + required)
                   for call in call_counts):
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_call={required}", extracted=facts)
    for name, expected_count in checks.get("call_counts", {}).items():
        actual = sum(count for call, count in call_counts.items()
                     if call == name or call.endswith("." + name))
        if actual != expected_count:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"call_count={name}:{actual}", extracted=facts)
    for tag, expected_count in checks.get("jsx_tag_counts", {}).items():
        if tag_counts.get(tag, 0) != expected_count:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"jsx_tag_count={tag}:{tag_counts.get(tag, 0)}",
                               extracted=facts)
    for source in checks.get("import_sources", []):
        if source not in import_sources:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_import_source={source}", extracted=facts)
    for imported in checks.get("imports", []):
        if imported not in imported_names:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_import={imported}", extracted=facts)
    for literal in checks.get("string_literals", []):
        if literal not in string_literals:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_string_literal={literal!r}", extracted=facts)
    normalized_jsx_texts = {re.sub(r"\s+", " ", value).strip() for value in jsx_texts}
    for required in checks.get("jsx_texts", []):
        if re.sub(r"\s+", " ", required).strip() not in normalized_jsx_texts:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_jsx_text={required!r}", extracted=facts)
    for prop, annotation in checks.get("props", {}).items():
        if not re.search(
                rf"\b{re.escape(prop)}\s*[?:]?\s*:\s*{re.escape(annotation)}\b",
                masked):
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_prop={prop}:{annotation}", extracted=facts)
    component_contracts = checks.get("component_contracts")
    if component_contracts is None and checks.get("component_contract") is not None:
        component_contracts = checks["component_contract"]
    if isinstance(component_contracts, dict):
        component_contracts = [component_contracts]
    for contract in component_contracts or []:
        error = _validate_tsx_component_contract(text, masked, contract)
        if error:
            return ScoreResult(False, "parse_requirement_failed",
                               note=error, extracted=facts)
    distinct_switches = checks.get("distinct_switch_returns", [])
    if isinstance(distinct_switches, dict):
        distinct_switches = [distinct_switches]
    for contract in distinct_switches:
        error = _validate_distinct_switch_returns(text, masked, contract)
        if error:
            return ScoreResult(False, "parse_requirement_failed",
                               note=error, extracted=facts)
    if checks.get("react_state_contract"):
        error = _validate_react_state_contract(
            behavior_surface, literal_surface, checks["react_state_contract"],
        )
        if error:
            return ScoreResult(False, "parse_requirement_failed",
                               note=error, extracted=facts)
    if checks.get("react_behavior_contract"):
        error = _validate_react_behavior_contract(
            behavior_surface, literal_surface, checks["react_behavior_contract"],
        )
        if error:
            return ScoreResult(False, "parse_requirement_failed",
                               note=error, extracted=facts)
    return _validate_static_source_contract(
        text, checks, strategy=f"{parser}_static_parse_passed", extracted=facts,
        code_text=behavior_surface, literal_text=literal_surface,
        jsx_text_surface=masked if parser == "tsx" else None,
    )


def _mask_sql_noncode(text: str) -> str:
    """Mask SQL comments and quoted literals without changing positions."""
    chars = list(text)
    index = 0
    while index < len(text):
        if text.startswith("--", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for position in range(index, end):
                chars[position] = " "
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated SQL block comment")
            for position in range(index, end + 2):
                if chars[position] != "\n":
                    chars[position] = " "
            index = end + 2
            continue
        dollar = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", text[index:])
        if dollar:
            delimiter = dollar.group(0)
            end = text.find(delimiter, index + len(delimiter))
            if end < 0:
                raise ValueError("unterminated SQL dollar-quoted string")
            for position in range(index, end + len(delimiter)):
                if chars[position] != "\n":
                    chars[position] = " "
            index = end + len(delimiter)
            continue
        if text[index] in "'\"`":
            quote = text[index]
            start = index
            index += 1
            while index < len(text):
                if text[index] == quote:
                    if index + 1 < len(text) and text[index + 1] == quote:
                        index += 2
                        continue
                    break
                if text[index] == "\\" and index + 1 < len(text):
                    index += 2
                else:
                    index += 1
            if index >= len(text):
                raise ValueError("unterminated SQL quoted literal")
            for position in range(start, index + 1):
                if chars[position] != "\n":
                    chars[position] = " "
            index += 1
            continue
        index += 1
    return "".join(chars)


def _validate_sql_structure(text: str, checks: dict) -> ScoreResult:
    cleaned = _mask_sql_noncode(text)
    if not _balanced_static_delimiters(cleaned):
        return ScoreResult(False, "parse_failed", note="unbalanced_sql_delimiters")
    statements = [part.strip() for part in _split_static_inline(cleaned, separator=";")
                  if part.strip()]
    if checks.get("single_statement", True) and len(statements) != 1:
        return ScoreResult(False, "parse_requirement_failed",
                           note=f"statement_count={len(statements)}")
    if not statements:
        return ScoreResult(False, "parse_requirement_failed", note="empty_sql_statement")
    leading = re.match(r"\s*([A-Za-z_][\w$]*)\b", statements[0])
    allowed_starts = {
        str(keyword).upper() for keyword in checks.get(
            "statement_starts",
            ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE"],
        )
    }
    if not leading or leading.group(1).upper() not in allowed_starts:
        return ScoreResult(False, "parse_requirement_failed",
                           note="invalid_statement_start")
    if leading.group(1).upper() == "SELECT":
        select_body = statements[0][leading.end():]
        from_match = re.search(r"\bFROM\b", select_body, re.IGNORECASE)
        if not from_match:
            return ScoreResult(False, "parse_requirement_failed",
                               note="SELECT_missing_FROM")
        projection = select_body[:from_match.start()].strip()
        source = select_body[from_match.end():].strip()
        if (not projection or projection.strip(",") == ""
                or not re.search(r"[A-Za-z0-9_*]", projection)):
            return ScoreResult(False, "parse_requirement_failed",
                               note="SELECT_empty_projection")
        if not re.match(r"(?:[A-Za-z_][\w$.-]*|\()", source):
            return ScoreResult(False, "parse_requirement_failed",
                               note="SELECT_missing_source")
    words = {word.upper() for word in re.findall(r"\b[A-Za-z_][\w$]*\b", cleaned)}
    functions = {
        match.group(1).upper(): re.sub(r"\s+", "", match.group(2)).lower()
        for match in re.finditer(r"\b([A-Za-z_][\w$]*)\s*\(([^()]*)\)", cleaned)
    }
    for keyword in checks.get("keywords", []):
        if keyword.upper() not in words:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_keyword={keyword}")
    for function in checks.get("functions", []):
        if function.upper() not in functions:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_function={function}")
    for function, alternatives in checks.get("function_args_any", {}).items():
        actual = functions.get(function.upper())
        accepted = {re.sub(r"\s+", "", value).lower() for value in alternatives}
        if actual not in accepted:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"function_args={function}:{actual}")
    return ScoreResult(True, "sql_static_parse_passed",
                       extracted={"keywords": sorted(words), "functions": functions})


def _validate_shell_structure(text: str, checks: dict) -> ScoreResult:
    lexer = shlex.shlex(text, posix=True, punctuation_chars="|;&")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens = list(lexer)
    if len(tokens) > 4096:
        return ScoreResult(False, "parse_failed", note="shell_token_cap")
    separators = {"|", "||", "&&", ";", "&"}
    if any(token and set(token) <= set("|;&") and token not in separators
           for token in tokens):
        return ScoreResult(False, "parse_failed", note="malformed_shell_separator")
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in separators:
            if not current:
                return ScoreResult(False, "parse_failed", note="empty_shell_command")
            segments.append(current)
            current = []
        else:
            current.append(token)
    if tokens and tokens[-1] in separators:
        return ScoreResult(False, "parse_failed", note="trailing_shell_separator")
    if current:
        segments.append(current)
    commands, command_tokens = [], []
    for segment in segments:
        index = 0
        while (index < len(segment)
               and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", segment[index])):
            index += 1
        if index >= len(segment):
            return ScoreResult(False, "parse_failed", note="assignment_without_command")
        commands.append(segment[index])
        command_tokens.append(segment[index:])
    expected_commands = list(checks.get("commands", []))
    if expected_commands:
        if checks.get("allow_extra_commands", False):
            if any(command not in commands for command in expected_commands):
                return ScoreResult(False, "parse_requirement_failed",
                                   note="missing_expected_command")
        elif commands != expected_commands:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"command_sequence={commands}")
    scoped_tokens = (
        tokens if checks.get("token_scope") == "any"
        else (command_tokens[0] if command_tokens else [])
    )
    for token in checks.get("tokens", []):
        if token not in scoped_tokens:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_first_command_token={token}")
    if len([token for token in tokens if token == "|"]) < checks.get("min_pipes", 0):
        return ScoreResult(False, "parse_requirement_failed", note="insufficient_pipes")
    return ScoreResult(True, "shell_static_parse_passed",
                       extracted={"commands": commands, "tokens": tokens[:200]})


def _normalize_docker_path(value: str) -> str:
    value = re.sub(r"^\./", "", value.strip())
    return value.rstrip("/") or "."


def _validate_dockerfile_structure(text: str, checks: dict) -> ScoreResult:
    logical_lines = []
    pending = ""
    for raw in text.splitlines():
        line = _strip_unquoted_comment(raw).strip()
        if not line:
            continue
        pending += (" " if pending else "") + line.rstrip("\\").rstrip()
        if line.endswith("\\"):
            continue
        logical_lines.append(pending)
        pending = ""
    if pending:
        return ScoreResult(False, "parse_failed", note="dangling_continuation")
    instructions: dict[str, list[str]] = {}
    for line in logical_lines:
        match = re.match(r"([A-Za-z]+)\s+(.+)$", line)
        if not match:
            return ScoreResult(False, "parse_failed", note=f"malformed={line[:60]!r}")
        instructions.setdefault(match.group(1).upper(), []).append(match.group(2).strip())
    for instruction in checks.get("instructions", []):
        if instruction.upper() not in instructions:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_instruction={instruction}")
    copy_pairs = []
    for arguments in instructions.get("COPY", []):
        try:
            argv = shlex.split(arguments)
        except ValueError:
            return ScoreResult(False, "parse_failed", note="invalid_COPY_quoting")
        argv = [item for item in argv if not item.startswith("--")]
        if len(argv) != 2:
            return ScoreResult(False, "parse_failed",
                               note=f"COPY_requires_one_source_got={max(0, len(argv)-1)}")
        copy_pairs.append((_normalize_docker_path(argv[0]),
                           _normalize_docker_path(argv[1])))
    for source, destination in checks.get("copy_pairs", []):
        pair = (_normalize_docker_path(source), _normalize_docker_path(destination))
        if pair not in copy_pairs:
            return ScoreResult(False, "parse_requirement_failed",
                               note=f"missing_copy={pair}")
    return ScoreResult(True, "dockerfile_static_parse_passed",
                       extracted={"instructions": instructions, "copy_pairs": copy_pairs})


def extract_string(resp: str, expected: str, *,
                   case_sensitive: bool = False,
                   allow_substring: bool = False,
                   accepted_aliases: list | None = None,
                   normalize_internal_whitespace: bool = False,
                   strip_whitespace: bool = True,
                   allow_terminal_punctuation: bool = False,
                   allow_semantic_equivalence: bool = False) -> ScoreResult:
    """Find `expected` string at the answer position (last line, boxed,
    quoted) or matching the entire stripped response.
    
    If allow_substring=True, a word-boundary substring also matches
    (used for short answers like 'banana').

    For Chinese yes/no/correctness answers, applies CHINESE_ANSWER_SYNONYMS
    table (是/否, 对/不对, 合理/不合理, etc.) — with explicit polarity
    rejection so 'correct' doesn't match when expected is 'incorrect'.
    """
    resp = _sanitize(resp)
    if not isinstance(expected, str):
        expected = str(expected)
    if strip_whitespace:
        resp = resp.strip()
        expected = expected.strip()
        accepted_aliases = [str(alias).strip() for alias in (accepted_aliases or [])]
    if normalize_internal_whitespace:
        resp = re.sub(r"\s+", " ", resp).strip()
        expected = re.sub(r"\s+", " ", str(expected)).strip()
        accepted_aliases = [
            re.sub(r"\s+", " ", str(alias)).strip()
            for alias in (accepted_aliases or [])
        ]
    if accepted_aliases:
        for alias in [expected, *accepted_aliases]:
            result = extract_string(
                resp, alias, case_sensitive=case_sensitive,
                allow_substring=allow_substring,
                normalize_internal_whitespace=False,
                strip_whitespace=strip_whitespace,
                allow_terminal_punctuation=allow_terminal_punctuation,
                allow_semantic_equivalence=allow_semantic_equivalence,
            )
            if result.passed:
                result.strategy = "alias_" + result.strategy if alias != expected else result.strategy
                result.expected = expected
                return result
        return ScoreResult(False, "no_alias_matched", expected=expected)
    if not expected.strip():
        return ScoreResult(False, "empty_expected", expected=expected)
    if not resp.strip():
        return ScoreResult(False, "empty_response", expected=expected)
    scan_resp = _mask_answer_tag_regions(resp)

    # Punctuation tolerance may remove an *extra* terminal mark, but must not
    # erase punctuation that is itself part of the reference answer.
    strip_candidate_terminal = bool(
        allow_terminal_punctuation
        and not re.search(r"[.!?,;:]$", expected)
    )

    # Chinese synonym fast path (yes/no/correct/incorrect/reasonable)
    syn = _chinese_synonym_match(resp, expected)
    if syn is True:
        return ScoreResult(True, "chinese_synonym", extracted=resp.strip()[:50],
                           expected=expected)
    if syn is False:
        return ScoreResult(False, "chinese_polarity_mismatch",
                           extracted=resp.strip()[:50], expected=expected,
                           note="response matches the OPPOSITE canonical answer")

    exp_norm = expected if case_sensitive else expected.lower()
    resp_norm = resp if case_sensitive else resp.lower()

    # 0. Exact comparison after trimming transport whitespace, before removing
    # punctuation wrappers. This preserves meaningful terminal punctuation in
    # expected strings such as "Marigold.".
    raw_trimmed = resp.strip()
    raw_norm = raw_trimmed if case_sensitive else raw_trimmed.lower()
    if raw_norm == exp_norm:
        return ScoreResult(True, "trimmed_raw_exact",
                           extracted=raw_trimmed, expected=expected)

    # 1. Wrapper-stripped exact
    s = _strip_markdown_wrappers(
        resp, strip_terminal_punctuation=strip_candidate_terminal,
    )
    s_norm = s if case_sensitive else s.lower()
    if s_norm == exp_norm:
        return ScoreResult(True, "stripped_exact", extracted=s, expected=expected)
    if allow_semantic_equivalence:
        equivalence = _safe_noninstr_equivalence(s, expected)
        if equivalence:
            return ScoreResult(True, equivalence, extracted=s, expected=expected)

    flags = 0 if case_sensitive else re.IGNORECASE

    marker_candidates = _answer_marker_candidates(resp)
    # Preserve a multiline declared payload such as ``Answer: BEGIN\n...``.
    if "\n" in expected and marker_candidates:
        position, _ = marker_candidates[-1]
        multiline = _strip_answer_candidate_wrappers(
            resp[position:], strip_terminal_punctuation=strip_candidate_terminal,
        )
        multiline_norm = multiline if case_sensitive else multiline.lower()
        if multiline_norm == exp_norm:
            return ScoreResult(True, "answer_marker_multiline",
                               extracted=multiline, expected=expected)

    positioned = []
    for position, payload in marker_candidates:
        positioned.append((
            position,
            _strip_answer_candidate_wrappers(
                payload, strip_terminal_punctuation=strip_candidate_terminal,
            ),
            "answer_marker",
        ))
    conclusion_candidates = _conclusion_candidates(resp)
    for position, payload in conclusion_candidates:
        conclusion_value = _strip_answer_candidate_wrappers(
            payload, strip_terminal_punctuation=strip_candidate_terminal,
        )
        emphasized = list(re.finditer(
            r"(?:\*\*([^*\n]+)\*\*|__([^_\n]+)__)", payload,
        ))
        if len(emphasized) == 1:
            match = emphasized[0]
            if not _text_span_is_negated(
                    resp, position + match.start(), position + match.end()):
                conclusion_value = (match.group(1) or match.group(2)).strip()
        positioned.append((
            position,
            conclusion_value,
            "conclusion",
        ))
    correction_candidates = _correction_candidates(resp)
    for position, payload in correction_candidates:
        positioned.append((
            position,
            _strip_answer_candidate_wrappers(
                payload, strip_terminal_punctuation=strip_candidate_terminal,
            ),
            "correction",
        ))
    strong_candidates = (
        marker_candidates + conclusion_candidates + correction_candidates
    )
    tagged_candidates = _authoritative_tagged_answer_candidates(
        resp, strong_candidates,
    )
    if _tagged_answers_conflict(tagged_candidates):
        return ScoreResult(
            False, "ambiguous_tagged_answers", expected=expected,
            note="multiple distinct paired answer-tag payloads",
        )
    for position, payload in tagged_candidates:
        negated = _tagged_answer_is_negated(resp, position)
        positioned.append((
            position,
            ("\0NEGATED_TAGGED_ANSWER\0" if negated else
             _strip_answer_candidate_wrappers(
                 payload, strip_terminal_punctuation=strip_candidate_terminal,
             )),
            "negated_tagged_answer" if negated else "tagged_answer",
        ))
    for command_start, command_end, arg_start, _, content in \
            _balanced_latex_arguments(resp):
        terminal_box = _is_terminal_latex_box(resp, command_end)
        if (_position_has_nonanswer_block_label(resp, command_start)
                or (not terminal_box and not _weak_presentation_allowed(
                    resp, arg_start, strong_candidates))):
            continue
        positioned.append((
            arg_start,
            _strip_answer_candidate_wrappers(
                content, strip_terminal_punctuation=strip_candidate_terminal,
            ),
            "boxed",
        ))
    for fence_start, _, _, content in _fenced_block_candidates(resp):
        if _position_has_nonanswer_block_label(resp, fence_start):
            continue
        positioned.append((
            fence_start,
            _strip_answer_candidate_wrappers(
                content, strip_terminal_punctuation=strip_candidate_terminal,
            ),
            "fenced_answer",
        ))
    meaningful_lines = _nonempty_lines_with_offsets(scan_resp)
    if meaningful_lines:
        line_start, _, raw_last = meaningful_lines[-1]
        last_candidate = _strip_answer_candidate_wrappers(
            raw_last, strip_terminal_punctuation=strip_candidate_terminal,
        )
        has_earlier_explicit = bool(positioned)
        presentation_closer = bool(re.fullmatch(
            r"(?:`{3,}|~{3,}|\\[\])]|\$+|"
            r"</(?:answer|final_answer|final|result|response|output)>)",
            raw_last.strip(), re.IGNORECASE,
        ))
        complete_tagged_line = _is_complete_tagged_answer(raw_last)
        nonanswer_line = _position_has_nonanswer_label(
            resp, line_start + len(raw_last),
        )
        if (not presentation_closer and not nonanswer_line
                and (not has_earlier_explicit or _looks_like_bare_answer_line(
                last_candidate, expected, case_sensitive=case_sensitive,
                raw_line=raw_last)) and not complete_tagged_line):
            positioned.append((line_start, last_candidate, "last_line"))

    # Quoted spans are analyzed now but applied only after a later explicit
    # marker/box/final-line candidate has had a chance to win by position.
    quoted = []
    if allow_substring:
        for pattern in (
            re.compile(r'"([^"\n]+)"'),
            re.compile(r"(?<!\w)'([^'\n]+)'(?!\w)"),
            re.compile(r"`([^`\n]+)`"),
        ):
            for match in pattern.finditer(scan_resp):
                candidate = match.group(1).strip()
                normalized = candidate if case_sensitive else candidate.lower()
                quoted.append((candidate, normalized, match.span(1)))
    matching_quotes = [item for item in quoted if item[1] == exp_norm]
    distinct_quotes = {item[1] for item in quoted}

    emphasized_matches = []
    if allow_substring:
        for pattern in (
            re.compile(r"\*\*([^*\n]+)\*\*"),
            re.compile(r"__([^_\n]+)__"),
        ):
            for match in pattern.finditer(scan_resp):
                candidate = match.group(1).strip().strip("\"'`")
                normalized = candidate if case_sensitive else candidate.lower()
                equivalent = normalized == exp_norm
                if not equivalent and allow_semantic_equivalence:
                    equivalent = bool(_safe_noninstr_equivalence(
                        candidate, expected,
                    ))
                if (equivalent and not _text_span_is_negated(
                        resp, match.start(1), match.end(1))
                        and not _position_has_nonanswer_label(
                            resp, match.start(1))):
                    emphasized_matches.append((match.start(1), candidate))

    if positioned:
        position, candidate, source = max(positioned, key=lambda item: item[0])
        candidate_norm = candidate if case_sensitive else candidate.lower()
        if candidate_norm == exp_norm:
            return ScoreResult(True, source, extracted=candidate, expected=expected)
        if allow_semantic_equivalence:
            equivalence = _safe_noninstr_equivalence(candidate, expected)
            if equivalence:
                return ScoreResult(
                    True, f"{source}_{equivalence}", extracted=candidate,
                    expected=expected,
                )

        # Emphasis is presentation inside the already-selected authoritative
        # payload; it is never an independent later answer that can override a
        # Final Answer/correction.  This keeps forms such as
        # ``Final answer: **Monday, January 1, 2024**`` tolerant without making
        # ``Reference: **expected**`` authoritative.
        for pattern in (
                re.compile(r"\*\*([^*\n]+)\*\*"),
                re.compile(r"__([^_\n]+)__")):
            for match in pattern.finditer(candidate):
                inner = match.group(1).strip().strip("\"'`")
                inner_norm = inner if case_sensitive else inner.lower()
                equivalent = inner_norm == exp_norm
                if not equivalent and allow_semantic_equivalence:
                    equivalent = bool(_safe_noninstr_equivalence(inner, expected))
                if equivalent:
                    return ScoreResult(
                        True, f"{source}_emphasized_payload",
                        extracted=inner, expected=expected,
                    )

        if source == "last_line":
            leading_ok, leading_candidate = _safe_leading_answer_with_explanation(
                resp, expected, case_sensitive=case_sensitive,
                allow_semantic_equivalence=allow_semantic_equivalence,
            )
            if leading_ok:
                return ScoreResult(
                    True, "leading_answer_line", extracted=leading_candidate,
                    expected=expected,
                    note="compact leading answer followed only by non-conflicting explanation",
                )

        if (allow_substring and len(expected) >= 2
                and _allows_buried_string_fallback(expected)):
            if source == "answer_marker":
                prefix = candidate.lstrip(" \t*_`\"'\u201c\u2018")
                edge = r"(?![A-Za-z0-9_])" if expected[-1:].isascii() \
                    and (expected[-1:].isalnum() or expected[-1:] == "_") else ""
                prefix_match = re.match(re.escape(expected) + edge, prefix, flags)
                if (prefix_match and not _text_span_is_negated(
                        prefix, prefix_match.start(), prefix_match.end())
                        and _safe_answer_marker_suffix(
                            prefix, prefix_match.end(),
                        )):
                    return ScoreResult(True, "answer_marker_prefix",
                                       extracted=expected, expected=expected)
            elif source in {"last_line", "conclusion", "correction"}:
                # Remove quoted spans from this line before treating a natural
                # sentence as an affirmative final answer.
                last_for_boundary = candidate
                if source == "last_line" and meaningful_lines:
                    line_start = meaningful_lines[-1][0]
                    chars = list(meaningful_lines[-1][2])
                    for _, _, (q_start, q_end) in quoted:
                        if q_start >= line_start and q_end <= line_start + len(chars):
                            for i in range(q_start - line_start, q_end - line_start):
                                chars[i] = " "
                    last_for_boundary = "".join(chars)
                if _word_boundary_search(
                        last_for_boundary, expected, case_sensitive,
                        allow_short_substring=False, reject_negated=True):
                    return ScoreResult(True, "last_line_boundary",
                                       extracted=expected, expected=expected)

        if source in {"answer_marker", "tagged_answer", "negated_tagged_answer", "boxed", "fenced_answer", "conclusion", "correction"}:
            return ScoreResult(False, f"{source}_mismatch",
                               extracted=candidate, expected=expected,
                               note="latest explicit answer overrides earlier mentions")
        if source == "last_line":
            last_word_count = len(re.findall(r"\S+", candidate))
            answer_like = len(candidate) <= 160 and last_word_count <= 12
            quote_on_last_line = bool(
                meaningful_lines and any(
                    span[0] >= meaningful_lines[-1][0] for _, _, span in matching_quotes
                )
            )
            if answer_like and not quote_on_last_line:
                return ScoreResult(False, "final_line_answer_mismatch",
                                   extracted=candidate, expected=expected,
                                   note="final independent line overrides earlier mentions")

    # No later explicit candidate established the answer. A directly
    # emphasized exact token is stronger than surrounding quoted examples.
    if emphasized_matches:
        _, candidate = emphasized_matches[-1]
        return ScoreResult(True, "emphasized_answer", extracted=candidate,
                           expected=expected)

    # Quoted ambiguity and negation now govern the fallback.
    if matching_quotes and len(distinct_quotes) > 1:
        return ScoreResult(False, "ambiguous_quoted_answers", expected=expected,
                           note="multiple different quoted candidates")
    for candidate, _, span in matching_quotes:
        if not _text_span_is_negated(resp, span[0], span[1]):
            return ScoreResult(True, "quoted", extracted=candidate,
                               expected=expected)
    if matching_quotes:
        return ScoreResult(False, "negated_quoted_answer",
                           extracted=matching_quotes[-1][0], expected=expected)

    last = positioned[-1][1] if positioned else ""

    # 6. Word-boundary substring (for short single-token answers, e.g. "banana")
    if (allow_substring and len(expected) >= 2 and " " not in expected
            and _allows_buried_string_fallback(expected)):
        fallback_text = _mask_quoted_and_emphasized_spans(scan_resp)
        if _word_boundary_search(
                fallback_text, expected, case_sensitive,
                allow_short_substring=False, reject_negated=True):
            return ScoreResult(True, "word_boundary",
                               extracted=expected, expected=expected,
                               note="single-token match with word boundary (NOT substring)")

    # 7. Multi-word phrase boundary search (also for full sentences expected)
    if (allow_substring and len(expected) >= 2
            and _allows_buried_string_fallback(expected)):
        fallback_text = _mask_quoted_and_emphasized_spans(scan_resp)
        if _word_boundary_search(
                fallback_text, expected, case_sensitive,
                allow_short_substring=False, reject_negated=True):
            return ScoreResult(True, "phrase_boundary",
                               extracted=expected, expected=expected)

    # Failed — note if expected appears as substring (informational only)
    if exp_norm in resp_norm:
        return ScoreResult(False, "expected_appears_unmarked",
                           expected=expected, extracted=last,
                           note="expected string appears in response but not at answer position; allow_substring=False")
    return ScoreResult(False, "wrong", expected=expected, extracted=last)


def exact_whole_response(resp: str, expected, *,
                         case_sensitive: bool = True,
                         strip_outer_whitespace: bool = True) -> ScoreResult:
    """Compare the whole response; never extract an answer-position substring."""
    candidate = _sanitize(resp)
    expected_text = str(expected)
    if strip_outer_whitespace:
        candidate = candidate.strip()
        expected_text = expected_text.strip()
    got = candidate if case_sensitive else candidate.lower()
    want = expected_text if case_sensitive else expected_text.lower()
    return ScoreResult(
        got == want,
        "whole_response_exact" if got == want else "whole_response_mismatch",
        extracted=candidate[:200],
        expected=expected_text,
    )


# ----------------------------------------------------------------------------
# MCQ — find letter at answer position
# ----------------------------------------------------------------------------
def extract_mcq_letter(resp: str, expected: str, *,
                       require_bare: bool = False) -> ScoreResult:
    resp = _sanitize(resp)
    expected_letter = str(expected).strip().upper()
    if not re.fullmatch(r"[A-Z]", expected_letter):
        return ScoreResult(False, "invalid_expected_letter", expected=expected)
    if not resp.strip():
        return ScoreResult(False, "empty", expected=expected_letter)

    if require_bare:
        candidate = resp.strip().upper()
        if re.fullmatch(r"[A-Z]", candidate):
            return ScoreResult(candidate == expected_letter, "bare_single_letter",
                               extracted=candidate, expected=expected_letter)
        return ScoreResult(False, "mcq_bare_format_violation",
                           expected=expected_letter,
                           note="instruction probe requires one bare option letter")

    # 1. Stripped — single letter only
    s = _strip_markdown_wrappers(resp).upper().rstrip(".!?,;:)")
    if re.fullmatch(r"[A-Z]", s):
        return ScoreResult(s == expected_letter, "stripped_single_letter",
                           extracted=s, expected=expected_letter)

    def payload_letter(payload: str) -> tuple[str, int] | None:
        clean = _strip_answer_candidate_wrappers(payload).strip()
        # An explicit marker makes a leading standalone letter safe; prose
        # selection verbs below use a narrower grammar to avoid article "a".
        match = re.match(
            r"^\(?([A-D])\)?(?:[*_`]+)?(?=$|[ \t.!?,;:)\-])",
            clean, re.I,
        )
        if not match:
            text_box = re.fullmatch(
                r"\\text[ \t]*\{[ \t]*\(?([A-D])\)?[ \t]*\}",
                clean, re.IGNORECASE,
            )
            if not text_box:
                return None
            return text_box.group(1).upper(), text_box.start(1)
        return match.group(1).upper(), match.start(1)

    # All candidate kinds participate in one text-position ordering.  The
    # boolean records an explicitly negated selection; a later "do not choose
    # B" must override an earlier "choose B" rather than being ignored.
    positioned: list[tuple[int, str, str, bool]] = []

    for position, payload in _answer_marker_candidates(resp):
        parsed = payload_letter(payload)
        if parsed:
            letter, local = parsed
            span_start = position + local
            positioned.append((
                span_start, letter, "answer_marker",
                _text_span_is_negated(resp, span_start, span_start + 1),
            ))

    marker_candidates = _answer_marker_candidates(resp)
    tagged_candidates = _authoritative_tagged_answer_candidates(
        resp, marker_candidates,
    )
    if _tagged_answers_conflict(tagged_candidates):
        return ScoreResult(
            False, "ambiguous_tagged_answers", expected=expected_letter,
            note="multiple distinct paired answer-tag payloads",
        )
    for position, payload in tagged_candidates:
        parsed = payload_letter(payload)
        if parsed:
            letter, local = parsed
            span_start = position + local
            positioned.append((
                span_start, letter, "tagged_answer",
                (_tagged_answer_is_negated(resp, position)
                 or _text_span_is_negated(resp, span_start, span_start + 1)),
            ))

    selection = re.compile(
        r"\b(?:choose|select|pick)[ \t]+(?:option[ \t]+)?"
        r"\(?([A-D])\)?"
        r"(?=[ \t]*(?:[.!?,;:]|$|\bas\b|\bbecause\b|\bfor\b))",
        re.IGNORECASE,
    )
    for match in selection.finditer(resp):
        positioned.append((
            match.start(1), match.group(1).upper(), "selection_verb",
            _text_span_is_negated(resp, match.start(1), match.end(1)),
        ))

    # Strong natural-language declarations used by otherwise-correct 2.5
    # responses. These do not scan arbitrary option mentions: each grammar
    # includes an affirmative selection cue.
    def safe_affirmative_declaration(match: re.Match) -> bool:
        """Accept only an unconditional, complete MCQ declaration.

        The declaration grammars below deliberately accept natural prose, but
        the matched words alone are not sufficient evidence when the same
        sentence qualifies the choice as hypothetical/example-only or then
        contradicts it.  Keep the accepted continuations small: a sentence
        ending, an explicit causal explanation, or a colon/dash explanation.
        """
        line_start = resp.rfind("\n", 0, match.start()) + 1
        line_end = resp.find("\n", match.end())
        if line_end < 0:
            line_end = len(resp)
        line = resp[line_start:line_end]
        prefix = resp[line_start:match.start()]
        suffix = resp[match.end():line_end]

        # Conditions before the declaration make it hypothetical rather than
        # the model's final selection.  Example/scenario cues and contrasting
        # or second-correct-choice language are unsafe on either side.
        if re.search(
            r"\b(?:if|unless|assuming|supposing|suppose|hypothetically)\b"
            r"|\bunder[ \t]+(?:the|this|an?)[ \t]+(?:assumption|condition)\b",
            prefix,
            re.IGNORECASE,
        ):
            return False
        if re.search(
            r"\bonly[ \t]+if\b|\bprovided[ \t]+that\b"
            r"|\b(?:in|for)[ \t]+(?:the|this|an?)[ \t]+"
            r"(?:example|case|scenario)\b"
            r"|\b(?:as[ \t]+an?|for)[ \t]+example\b"
            r"|\b(?:hypothetical(?:ly)?|assuming|supposing|suppose)\b"
            r"|\b(?:but|whereas|however|although|yet|nevertheless)\b"
            r"|\b(?:also|equally)[ \t]+(?:is[ \t]+)?"
            r"(?:correct|best|optimal)\b"
            r"|\b(?:correct|best|optimal)[ \t]+too\b"
            r"|\b(?:my|the)[ \t]+answer[ \t]+is[ \t]+not\b",
            line,
            re.IGNORECASE,
        ):
            return False

        if re.fullmatch(r"[ \t]*(?:[.!?]+|[)\]}][.!?]*)?[ \t]*", suffix):
            return True
        if re.match(
            r"\A[ \t]*(?:[,;][ \t]*)?(?:because|as|since)\b",
            suffix,
            re.IGNORECASE,
        ):
            return True
        if re.match(r"\A[ \t]*(?::|--?|\u2014)[ \t]*\S", suffix):
            return True
        # A selected labelled option may include its visible option text
        # before the sentence ends, e.g. ``best pattern is B) two pointers``.
        if re.fullmatch(r"[ \t]*\)[ \t]+\S.*[.!?][ \t]*", suffix):
            return True
        if (
            match.group(0).rstrip().endswith(")")
            and re.fullmatch(r"[ \t]+\S.*[.!?][ \t]*", suffix)
        ):
            return True
        return False

    strong_selections = (
        re.compile(
            r"\b(?:is|matches)[ \t]+option[ \t]+\(?([A-D])\)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:\*\*|__)([A-D])(?:\*\*|__)"
            r"[ \t]+is[ \t]+(?:the[ \t]+)?(?:correct|best|most[ \t]+likely|optimal)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:\*\*|__|\x60)*\(?([A-D])\)?(?:\*\*|__|\x60)*"
            r"[ \t]+is[ \t]+(?:the[ \t]+)?"
            r"(?:correct|best|most[ \t]+likely|optimal)"
            r"(?:[ \t]+(?:answer|choice|option|outcome))?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the[ \t]+)?(?:best|correct|optimal|preferred|"
            r"most[ \t]+(?:suitable|coherent))"
            r"[ \t]+(?:answer|choice|option|pattern|strategy|approach|method|"
            r"order|continuation)"
            r"[ \t]+(?:is|would[ \t]+be)[ \t]+(?:\*\*|__|`)*"
            r"\(?([A-D])\)?(?:\*\*|__|`)*(?=[ \t).,:;!?'\"-]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\boption[ \t]+\(?([A-D])\)?(?:\*\*|__|`)*[ \t]+"
            r"(?:is|would[ \t]+be)[ \t]+(?:the[ \t]+)?"
            r"(?:correct|best|optimal|preferred|most[ \t]+suitable)"
            r"(?:[ \t]+(?:answer|choice|option|strategy|approach|method))?\b",
            re.IGNORECASE,
        ),
    )
    for pattern in strong_selections:
        for match in pattern.finditer(resp):
            if not safe_affirmative_declaration(match):
                continue
            positioned.append((
                match.start(1), match.group(1).upper(), "affirmative_option",
                _text_span_is_negated(resp, match.start(1), match.end(1)),
            ))

    for _, _, arg_start, arg_end, content in _balanced_latex_arguments(resp):
        parsed = payload_letter(content)
        if parsed:
            letter, local = parsed
            span_start = arg_start + local
            positioned.append((
                span_start, letter, "boxed",
                _text_span_is_negated(resp, span_start, min(arg_end, span_start + 1)),
            ))

    # A pure answer line may precede a human-readable explanation (``B\n...``),
    # but two different pure answer lines are contradictory and unsafe.
    pure_lines = []
    for start, _, raw_line in _nonempty_lines_with_offsets(resp):
        clean = _strip_answer_candidate_wrappers(raw_line).strip()
        match = re.fullmatch(r"\(?([A-D])\)?[.!?,;:]?", clean, re.I)
        if match:
            pure_lines.append((start + match.start(1), match.group(1).upper()))
    pure_letters = {letter for _, letter in pure_lines}
    if len(pure_letters) == 1:
        for position, letter in pure_lines:
            positioned.append((position, letter, "pure_answer_line", False))
    elif len(pure_letters) > 1 and not positioned:
        return ScoreResult(False, "conflicting_pure_answer_lines",
                           extracted=sorted(pure_letters), expected=expected_letter)

    # An explicit ``B) option text``-style selected option. Only accept when
    # the response contains one such option line; this avoids treating a
    # copied A/B/C/D option list as the model's selected answer.
    option_line = re.compile(
        r"^\s*(?:[*_`]+\s*)?\(?([A-D])\)?\s*[.)\-:]\s+\S",
        re.IGNORECASE | re.MULTILINE,
    )
    option_lines = list(option_line.finditer(resp))
    if len(option_lines) == 1:
        m = option_lines[0]
        if not _text_span_is_negated(resp, m.start(1), m.end(1)):
            got = m.group(1).upper()
            positioned.append((m.start(1), got, "explicit_option_line", False))

    nonempty_lines = _nonempty_lines_with_offsets(resp)
    if nonempty_lines:
        first_start, _, first_raw = nonempty_lines[0]
        first_option = option_line.match(first_raw)
        second_is_option = bool(
            len(nonempty_lines) > 1
            and option_line.match(nonempty_lines[1][2])
        )
        if first_option and not second_is_option:
            local = first_option.start(1)
            positioned.append((
                first_start + local, first_option.group(1).upper(),
                "leading_option_line", False,
            ))
        first_parenthesized = re.search(
            r"\boption[ \t]*\(([A-D])\)", first_raw, re.IGNORECASE,
        )
        if first_parenthesized:
            positioned.append((
                first_start + first_parenthesized.start(1),
                first_parenthesized.group(1).upper(),
                "leading_parenthesized_option", False,
            ))

    if positioned:
        # Multiple unqualified affirmative declarations are contradictory.
        # A later explicit Answer/boxed/bare-letter declaration may resolve an
        # earlier discussion, but merely mentioning another "best" option may
        # not be silently discarded by last-letter wins.
        declarative = [
            letter for _, letter, strategy, negated in positioned
            if strategy in {"selection_verb", "affirmative_option"} and not negated
        ]
        authoritative = [
            item for item in positioned
            if item[2] in {
                "answer_marker", "tagged_answer", "boxed", "pure_answer_line",
            }
        ]
        if len(set(declarative)) > 1 and not authoritative:
            return ScoreResult(
                False, "conflicting_mcq_declarations",
                extracted=sorted(set(declarative)), expected=expected_letter,
            )
        _, got, strategy, negated = max(positioned, key=lambda item: item[0])
        if negated:
            return ScoreResult(False, "negated_mcq_selection",
                               extracted=got, expected=expected_letter)
        return ScoreResult(got == expected_letter, strategy,
                           extracted=got, expected=expected_letter)

    # Deliberately no "last A-D anywhere" fallback: negations such as
    # "Not B" and "I reject B" must not be turned into a positive B answer.
    return ScoreResult(False, "no_explicit_mcq_answer", expected=expected_letter)



# ----------------------------------------------------------------------------
# extract_set — set equality with normalization
# ----------------------------------------------------------------------------
def _explicit_answer_payloads(resp: str) -> list[str]:
    """Extract complete answer payloads after an explicit final-line cue.

    The descriptive-colon form is intentionally limited to answer-like cues;
    arbitrary prose after any colon is not treated as an answer.
    """
    text = _sanitize(resp)
    payloads = [
        (position, _strip_answer_candidate_wrappers(payload))
        for position, payload in _answer_marker_candidates(text)
    ]
    descriptive = re.compile(
        r"^[ \t]*([^:\n]{1,100}?)[ \t]*:[ \t]*(.*?)[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    cue = re.compile(
        r"\b(?:sorted|ascending|descending|sequence|set|items?|steps?|"
        r"subgoals?|values?|order(?:ed)?|list)\b",
        re.IGNORECASE,
    )
    lines = _nonempty_lines_with_offsets(text)
    for match in descriptive.finditer(text):
        if cue.search(match.group(1)):
            inline = match.group(2)
            if inline.strip(" \t*_`"):
                leading = len(inline) - len(inline.lstrip())
                payloads.append((
                    match.start(2) + leading,
                    _strip_answer_candidate_wrappers(inline),
                ))
                continue
            line_end = text.find("\n", match.end())
            if line_end < 0:
                line_end = len(text)
            for start, _, raw_line in lines:
                if start >= line_end + (1 if line_end < len(text) else 0):
                    payloads.append((
                        start,
                        _strip_answer_candidate_wrappers(raw_line),
                    ))
                    break
    # A final explicit declaration is authoritative.  Returning older
    # declarations could turn "Answer: right\nAnswer: wrong" into a pass.
    if not payloads:
        return []
    return [max(payloads, key=lambda item: item[0])[1]]


def extract_set(resp: str, expected, *, separator_hint: str = ",",
                case_sensitive: bool = False, reject_duplicates: bool = False,
                strict_separator: bool = False) -> ScoreResult:
    """Match `resp` as a set against `expected` (list or set).
    Splits on commas / newlines / semicolons. Trims, deduplicates."""
    resp = _sanitize(resp)
    if isinstance(expected, str):
        exp_set = {x.strip() for x in re.split(r"[,;\n]+", expected) if x.strip()}
    else:
        exp_set = {str(x).strip() for x in expected if str(x).strip()}

    if not case_sensitive:
        exp_set = {x.lower() for x in exp_set}

    # Try explicit answer payloads first, then last-line and full response.
    split_pattern = re.escape(separator_hint) if strict_separator else r"[,;\n]+"
    explicit_payloads = _explicit_answer_payloads(resp)
    if explicit_payloads:
        candidates = [(explicit_payloads[0], "answer_payload")]
    else:
        candidates = [(_last_meaningful_line(resp), "last_line"), (resp, "full")]
    for src, label in candidates:
        items = [x.strip() for x in re.split(split_pattern, src) if x.strip()]
        items = [_strip_markdown_wrappers(x) for x in items]
        items = [x for x in items if x]
        normalized_items = [x.lower() if not case_sensitive else x for x in items]
        got_set = set(normalized_items)
        if reject_duplicates and len(normalized_items) != len(got_set):
            continue
        if got_set == exp_set:
            return ScoreResult(True, f"set_eq_{label}",
                               extracted=sorted(got_set), expected=sorted(exp_set))

    return ScoreResult(False, "set_unequal",
                       extracted=sorted(got_set), expected=sorted(exp_set),
                       note=f"missing={sorted(exp_set-got_set)} extra={sorted(got_set-exp_set)}")


# ----------------------------------------------------------------------------
# extract_ordered_sequence — sequence equality (order and duplicates matter)
# ----------------------------------------------------------------------------
def extract_ordered_sequence(resp: str, expected, *, separator_hint: str = ",",
                             numeric: bool = False, strict_separator: bool = False,
                             case_sensitive: bool = False) -> ScoreResult:
    """Compare an answer-position sequence without degrading it to a set.

    When ``numeric`` is true, items are compared with ``Decimal`` so harmless
    textual variants such as ``7`` and ``7.0`` are equivalent without using
    binary floating point. Item order and multiplicity remain mandatory.
    """
    if isinstance(expected, str):
        expected_items = [x.strip() for x in expected.split(separator_hint)]
    else:
        expected_items = [str(x).strip() for x in expected]
    if not expected_items or any(not item for item in expected_items):
        return ScoreResult(False, "invalid_expected_sequence", expected=expected_items)

    def canonical(items):
        if numeric:
            values = []
            for item in items:
                if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", item):
                    raise InvalidOperation
                values.append(Decimal(item))
            return values
        return items if case_sensitive else [item.lower() for item in items]

    try:
        expected_canonical = canonical(expected_items)
    except InvalidOperation:
        return ScoreResult(False, "invalid_expected_sequence", expected=expected_items)

    explicit_payloads = _explicit_answer_payloads(resp)
    candidates = []
    if explicit_payloads:
        candidates.append(("answer_payload", explicit_payloads[0]))
    else:
        last = _last_meaningful_line(resp)
        if last:
            candidates.append(("last_line", last))
        stripped = _strip_markdown_wrappers(_sanitize(resp))
        if stripped and stripped != last:
            candidates.append(("stripped_full", stripped))

    last_items = []
    saw_order_mismatch = False
    for label, candidate in candidates:
        candidate = re.sub(
            r"^(?:final[ \t]+answer|the[ \t]+answer[ \t]+is|"
            r"answer[ \t]*(?:is|:|=))[ \t]*",
            "", candidate, flags=re.IGNORECASE,
        ).strip()
        split_pattern = re.escape(separator_hint) if strict_separator else r"[,;\n]+"
        items = [x.strip() for x in re.split(split_pattern, candidate)]
        if any(not item for item in items):
            continue
        last_items = items
        try:
            got_canonical = canonical(items)
        except InvalidOperation:
            continue
        if got_canonical == expected_canonical:
            return ScoreResult(True, f"ordered_sequence_{label}",
                               extracted=items, expected=expected_items)
        if len(got_canonical) == len(expected_canonical):
            # Equality after sorting means the values are all present but the
            # requested ordering (or duplicate positions) is wrong.
            if sorted(got_canonical) == sorted(expected_canonical):
                saw_order_mismatch = True

    return ScoreResult(
        False,
        "sequence_order_mismatch" if saw_order_mismatch else "sequence_mismatch",
        extracted=last_items,
        expected=expected_items,
    )


# ----------------------------------------------------------------------------
# JSON extraction
# ----------------------------------------------------------------------------
_JSON_CONFLICT = "__ace_conflicting_top_level_json__"


def _json_fingerprint(value: Any):
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("number", str(Decimal(str(value)).normalize()))
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        return ("array", tuple(_json_fingerprint(item) for item in value))
    if isinstance(value, dict):
        return ("object", tuple(
            (key, _json_fingerprint(value[key])) for key in sorted(value)
        ))
    return (type(value).__name__, repr(value))


def _balanced_top_level_json(text: str) -> list[tuple[int, int, Any, str]]:
    """Collect JSON object/array spans in one linear scan."""
    candidates: list[tuple[int, int, Any, str]] = []
    stack: list[str] = []
    start: int | None = None
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for index, char in enumerate(text):
        if start is None:
            if char in pairs:
                start = index
                stack = [pairs[char]]
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "}]":
            if not stack or stack[-1] != char:
                start = None
                stack = []
                continue
            stack.pop()
            if not stack and start is not None:
                raw = text[start:index + 1]
                try:
                    candidates.append((start, index + 1, json.loads(raw), raw))
                except (json.JSONDecodeError, ValueError):
                    pass
                start = None
    return candidates


def extract_json(resp: str):
    """Extract one unambiguous top-level JSON object or array.

    A latest explicit ``Answer:``, ``Output:``, or ``Reply:`` declaration is
    authoritative.  Otherwise a single JSON fence or a single distinct
    balanced value is accepted.  Multiple different unlabelled values are a
    conflict rather than an invitation to score the first example.
    """
    resp = _sanitize(resp)
    if not resp.strip():
        return None, None

    stripped = resp.strip()
    unwrapped = _strip_markdown_wrappers(resp).strip()
    for candidate in dict.fromkeys((stripped, unwrapped)):
        try:
            return json.loads(candidate), candidate
        except (json.JSONDecodeError, ValueError):
            pass

    balanced = _balanced_top_level_json(resp)

    # Explicit declarations win, but two different values inside the same
    # latest declaration remain ambiguous.
    marker_pattern = re.compile(
        r"(?im)(?<![A-Za-z0-9_])(?:final[ \t]+)?"
        r"(?:answer|output|reply)[ \t]*(?::|=|is[ \t]*:?)",
    )
    markers = list(marker_pattern.finditer(resp))
    for marker_index in range(len(markers) - 1, -1, -1):
        marker = markers[marker_index]
        boundary = (markers[marker_index + 1].start()
                    if marker_index + 1 < len(markers) else len(resp))
        declared = [item for item in balanced
                    if marker.end() <= item[0] < boundary]
        if not declared:
            continue
        fingerprints = {_json_fingerprint(item[2]) for item in declared}
        if len(fingerprints) != 1:
            return None, _JSON_CONFLICT
        return declared[-1][2], declared[-1][3]

    fenced = []
    for _, _, _, body in _fenced_block_candidates(resp):
        try:
            fenced.append((json.loads(body), body))
        except (json.JSONDecodeError, ValueError):
            continue
    if fenced:
        fingerprints = {_json_fingerprint(value) for value, _ in fenced}
        if len(fingerprints) != 1:
            return None, _JSON_CONFLICT
        return fenced[-1]

    if not balanced:
        return None, None
    fingerprints = {_json_fingerprint(item[2]) for item in balanced}
    if len(fingerprints) != 1:
        return None, _JSON_CONFLICT
    return balanced[-1][2], balanced[-1][3]


def _json_value_equal(left: Any, right: Any) -> bool:
    """JSON-semantic equality without Python's ``True == 1`` coercion."""
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return left == right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, list) or isinstance(right, list):
        return (isinstance(left, list) and isinstance(right, list)
                and len(left) == len(right)
                and all(_json_value_equal(a, b) for a, b in zip(left, right)))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and set(left) == set(right)
                and all(_json_value_equal(left[key], right[key]) for key in left))
    return type(left) is type(right) and left == right


def _validate_json_schema(obj, schema):
    """Mini schema validator for the keywords used by ACE probe schemas.

    Supports type, properties, required, const, enum, additionalProperties,
    items, prefixItems, minItems, and maxItems. Returns (ok, errors) and is
    defensive against malformed schemas.
    """
    errors = []
    def _check(o, sch, path=""):
        if not isinstance(sch, dict):
            # Non-dict schema fragment — accept anything (best-effort)
            return
        if "const" in sch:
            if not _json_value_equal(o, sch["const"]):
                errors.append(f"{path}: const expected {sch['const']!r} got {o!r}")
                return
        if "enum" in sch:
            if not any(_json_value_equal(o, option) for option in sch["enum"]):
                errors.append(f"{path}: enum value {o!r} not in {sch['enum']}")
                return
        t = sch.get("type")
        if t == "object":
            if not isinstance(o, dict):
                errors.append(f"{path}: expected object got {type(o).__name__}")
                return
            for r in sch.get("required", []):
                if r not in o:
                    errors.append(f"{path}: missing required field '{r}'")
            for k, sub in sch.get("properties", {}).items():
                if k in o:
                    _check(o[k], sub, f"{path}.{k}")
            if sch.get("additionalProperties") is False:
                allowed = set(sch.get("properties", {}).keys())
                extra = set(o.keys()) - allowed
                if extra:
                    errors.append(f"{path}: extra fields {sorted(extra)}")
        elif t == "array":
            if not isinstance(o, list):
                errors.append(f"{path}: expected array")
                return
            if "minItems" in sch and len(o) < sch["minItems"]:
                errors.append(f"{path}: expected at least {sch['minItems']} items")
            if "maxItems" in sch and len(o) > sch["maxItems"]:
                errors.append(f"{path}: expected at most {sch['maxItems']} items")
            prefix = sch.get("prefixItems")
            if isinstance(prefix, list):
                for i, sub in enumerate(prefix):
                    if i < len(o):
                        _check(o[i], sub, f"{path}[{i}]")
            items_sch = sch.get("items")
            if isinstance(items_sch, dict):
                start = len(prefix) if isinstance(prefix, list) else 0
                for i, item in enumerate(o[start:], start):
                    _check(item, items_sch, f"{path}[{i}]")
            elif isinstance(items_sch, list):
                # Tuple validation
                for i, sub in enumerate(items_sch):
                    if i < len(o):
                        _check(o[i], sub, f"{path}[{i}]")
            elif items_sch is False and isinstance(prefix, list) and len(o) > len(prefix):
                errors.append(f"{path}: items beyond prefixItems are forbidden")
        elif t == "string":
            if not isinstance(o, str):
                errors.append(f"{path}: expected string")
        elif t == "number":
            if not isinstance(o, (int, float)) or isinstance(o, bool):
                errors.append(f"{path}: expected number")
        elif t == "integer":
            if not isinstance(o, int) or isinstance(o, bool):
                errors.append(f"{path}: expected integer")
        elif t == "boolean":
            if not isinstance(o, bool):
                errors.append(f"{path}: expected boolean")
    try:
        _check(obj, schema, "$")
    except Exception as e:
        errors.append(f"validator_error: {type(e).__name__}: {e}")
    return (len(errors) == 0, errors)


_TOOL_REPAIR_ITEM_CAP = 1024


def _decode_tool_arguments(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def _preferred_tool_call_shape(obj, *, _depth: int = 0):
    """Return one lossless canonical tool-envelope spelling.

    Alias keys may be renamed or flat arguments may be nested.  Every field
    not consumed by that transformation is preserved so the destination
    schema's ``additionalProperties`` contract still sees and rejects it.
    """
    if not isinstance(obj, dict) or _depth > 4:
        return obj

    inner = obj.get("function")
    has_outer_envelope = any(
        key in obj for key in ("name", "tool", "arguments", "parameters")
    )
    if isinstance(inner, dict) and not has_outer_envelope:
        preferred_inner = _preferred_tool_call_shape(inner, _depth=_depth + 1)
        result = dict(preferred_inner)
        for key, value in obj.items():
            if key != "function":
                result[key] = value
        return result

    consumed: set[str] = set()
    name = None
    for key in ("name", "tool", "function"):
        if isinstance(obj.get(key), str):
            name = obj[key]
            consumed.add(key)
            break
    if name is None:
        return obj

    arguments = None
    if "arguments" in obj:
        arguments = _decode_tool_arguments(obj["arguments"])
        if arguments is None:
            return obj
        consumed.add("arguments")
    elif "parameters" in obj:
        arguments = _decode_tool_arguments(obj["parameters"])
        if arguments is None:
            return obj
        consumed.add("parameters")
    else:
        alias_keys = {"name", "tool", "function", "arguments", "parameters"}
        flat_keys = [key for key in obj if key not in alias_keys]
        if not flat_keys:
            return obj
        arguments = {key: obj[key] for key in flat_keys}
        consumed.update(flat_keys)

    result = {"name": name, "arguments": arguments}
    for key, value in obj.items():
        if key not in consumed:
            result[key] = value
    return result


def _try_repair_tool_call_shape(obj):
    """Produce at most one repaired candidate in O(number of calls)."""
    candidates = [obj]
    if isinstance(obj, list):
        if (not obj or len(obj) > _TOOL_REPAIR_ITEM_CAP
                or not all(isinstance(item, dict) for item in obj)):
            return candidates
        preferred = [_preferred_tool_call_shape(item) for item in obj]
    elif isinstance(obj, dict):
        preferred = _preferred_tool_call_shape(obj)
    else:
        return candidates
    if preferred != obj:
        candidates.append(preferred)
    return candidates


def _schema_is_tool_envelope(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        if {"name", "arguments"} <= set(properties) and \
                {"name", "arguments"} <= required:
            return True
    prefix = schema.get("prefixItems")
    if isinstance(prefix, list) and any(_schema_is_tool_envelope(item) for item in prefix):
        return True
    items = schema.get("items")
    return isinstance(items, dict) and _schema_is_tool_envelope(items)


def _tool_shape_conflict(value: Any) -> str | None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            conflict = _tool_shape_conflict(item)
            if conflict:
                return f"item[{index}].{conflict}"
        return None
    if not isinstance(value, dict):
        return None
    if "name" in value and "tool" in value and value["name"] != value["tool"]:
        return "name_vs_tool"
    if ("arguments" in value and "parameters" in value
            and not _json_value_equal(value["arguments"], value["parameters"])):
        return "arguments_vs_parameters"
    inner = value.get("function")
    if isinstance(inner, dict):
        inner_conflict = _tool_shape_conflict(inner)
        if inner_conflict:
            return f"function.{inner_conflict}"
    if (isinstance(inner, dict) and "name" in value and "name" in inner
            and value["name"] != inner["name"]):
        return "outer_vs_function_name"
    if (isinstance(inner, str) and "name" in value
            and value["name"] != inner):
        return "name_vs_function"
    return None


def json_schema_lenient(resp: str, schema: dict, *, repair_tool_shape: bool = False) -> ScoreResult:
    """Extract JSON from `resp`, validate against `schema` with optional
    shape-repair (handles flat-vs-nested tool calls)."""
    obj, raw = extract_json(resp)
    if obj is None:
        if raw == _JSON_CONFLICT:
            return ScoreResult(False, "conflicting_json_values",
                               note="multiple different top-level JSON values")
        return ScoreResult(False, "no_json", note="no parseable JSON in response")

    tool_envelope_schema = _schema_is_tool_envelope(schema)
    if (repair_tool_shape and tool_envelope_schema
            and isinstance(obj, list) and len(obj) > _TOOL_REPAIR_ITEM_CAP):
        return ScoreResult(False, "tool_shape_work_cap", extracted=None,
                           note=f"item_count>{_TOOL_REPAIR_ITEM_CAP}")

    # First try original
    ok, errors = _validate_json_schema(obj, schema)
    if ok:
        return ScoreResult(True, "schema_match", extracted=obj)

    if repair_tool_shape and tool_envelope_schema:
        conflict = _tool_shape_conflict(obj)
        if conflict:
            return ScoreResult(False, "tool_shape_conflict", extracted=obj,
                               note=conflict)
        for variant in _try_repair_tool_call_shape(obj):
            ok2, _ = _validate_json_schema(variant, schema)
            if ok2:
                return ScoreResult(True, "schema_match_after_repair",
                                   extracted=obj,
                                   note="model used non-canonical shape; repaired")
    return ScoreResult(False, "schema_violation", extracted=obj,
                       note="; ".join(errors[:3]))


def json_schema_strict(resp: str, schema: dict, *,
                       allow_outer_fence: bool = False) -> ScoreResult:
    """Validate one whole-response JSON value.

    Non-instruction capability probes may opt into one harmless outer
    ``json`` fence.  Prose, multiple blocks, and schema violations remain
    failures.
    """
    text = _sanitize(resp).strip()
    fenced = False
    if allow_outer_fence:
        match = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```",
            text,
            re.IGNORECASE,
        )
        if match:
            text = match.group(1).strip()
            fenced = True
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return ScoreResult(False, "strict_json_parse_failed", note=str(exc))
    ok, errors = _validate_json_schema(obj, schema)
    if ok:
        return ScoreResult(
            True,
            "strict_json_schema_match_after_fence" if fenced else "strict_json_schema_match",
            extracted=obj,
        )
    return ScoreResult(False, "strict_json_schema_violation", extracted=obj,
                       note="; ".join(errors[:3]))


# ----------------------------------------------------------------------------
# Code extraction & semantic structural check
# ----------------------------------------------------------------------------
def extract_code(resp: str, lang_hint: str = "python") -> str:
    """Extract code from a fenced block, or fall back to the whole response.
    Also strips outer single-backtick wrappers (` ` ... ` `) for one-liner answers."""
    resp = _sanitize(resp)
    # Prefer fenced block
    m = re.search(rf"```(?:{lang_hint}|py)?\s*\n(.*?)\n```", resp, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1)
    # Generic fence
    m = re.search(r"```\s*\n(.*?)\n```", resp, re.DOTALL)
    if m: return m.group(1)
    # Single-backtick one-liner: `code here`
    s = resp.strip()
    if s.startswith("`") and s.endswith("`") and "\n" not in s and len(s) >= 3:
        return s[1:-1]
    return resp


def python_ast_contains_reference(resp: str, reference: str) -> ScoreResult | None:
    """Safely compare Python syntax while ignoring spelling-only trivia.

    Every top-level reference statement must occur in the candidate AST;
    additional statements are allowed.  This intentionally fixes quote style,
    whitespace, and harmless trailing-call differences without executing the
    submitted payload or relaxing identifiers/arguments/control flow.

    ``None`` means one side is not Python and the caller should retain its
    original string scorer result.
    """
    candidate = extract_code(resp, "python").strip()
    expected = extract_code(str(reference), "python").strip()
    try:
        expected_tree = ast.parse(expected)
    except (SyntaxError, ValueError):
        return None
    try:
        candidate_tree = ast.parse(candidate)
    except (SyntaxError, ValueError) as exc:
        return ScoreResult(False, "python_ast_candidate_parse_failed", note=str(exc))
    if not expected_tree.body:
        return None
    if len(list(ast.walk(candidate_tree))) > 4096:
        return ScoreResult(False, "python_ast_too_large")
    candidate_statements = {
        ast.dump(statement, annotate_fields=True, include_attributes=False)
        for statement in candidate_tree.body
    }
    expected_statements = [
        ast.dump(statement, annotate_fields=True, include_attributes=False)
        for statement in expected_tree.body
    ]
    missing = [statement for statement in expected_statements
               if statement not in candidate_statements]
    if not missing:
        return ScoreResult(True, "python_ast_reference_subset")
    return ScoreResult(
        False, "python_ast_reference_mismatch",
        note=f"missing_reference_statements={len(missing)}",
    )


def code_structural(resp: str, *, must_define: list = None,
                    must_call_any: list = None, allow_synonyms: dict = None,
                    must_use_any: list = None) -> ScoreResult:
    """Structural code check WITHOUT exact substring matching.
    
    must_define: list of function names that must be defined
    must_call_any: list of dotted names where at least one must be called
        e.g. ['yaml.dump', 'yaml.safe_dump'] (synonyms)
    must_use_any: list of identifier names that must appear at use-site
        (e.g. variable names from prompt)
    allow_synonyms: dict of canonical -> [variants]
        e.g. {'np': ['np', 'numpy']} so np.zeros == numpy.zeros
    """
    code = extract_code(resp).strip()
    if not code:
        return ScoreResult(False, "no_code")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ScoreResult(False, "syntax_error", note=str(e))

    defined = set()
    called_dotted = set()
    used_names = set()

    def dotted(node):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.insert(0, cur.id)
            return ".".join(parts)
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        if isinstance(node, ast.ClassDef):
            defined.add(node.name)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
        if isinstance(node, ast.Call):
            call_name = dotted(node.func)
            if call_name:
                called_dotted.add(call_name)

    # Check must_define
    info = {"defined": sorted(defined)}
    if must_define:
        missing = [n for n in must_define if n not in defined]
        if missing:
            return ScoreResult(False, "missing_definition",
                               note=f"must_define missing: {missing}", extracted=info)

    # Check must_call_any (with synonyms)
    if must_call_any:
        ok = False
        # For each required call, check if it OR any of its synonyms is in called set
        for req in must_call_any:
            # Build a set of acceptable variants for this requirement
            variants = {req}
            if allow_synonyms:
                # Direct synonym entry: synonyms[req] = [...]
                for v in allow_synonyms.get(req, []):
                    variants.add(v)
                # Indirect: req appears in some synonym group's variant list
                for canonical, vlist in allow_synonyms.items():
                    if req in vlist or req == canonical:
                        variants.add(canonical)
                        for v2 in vlist: variants.add(v2)
                # Module-prefix synonyms (e.g. np ↔ numpy in 'np.zeros')
                parts = req.split(".")
                if len(parts) > 1:
                    head, tail = parts[0], ".".join(parts[1:])
                    for canonical, vlist in allow_synonyms.items():
                        if head == canonical or head in vlist:
                            for v in [canonical] + vlist:
                                variants.add(f"{v}.{tail}")
            # Match any variant against actual called set
            if any(v in called_dotted for v in variants):
                ok = True; break
        if not ok:
            return ScoreResult(False, "missing_call",
                               note=f"none of must_call_any={must_call_any} called",
                               extracted={"called": sorted(called_dotted)})

    if must_use_any:
        if not any(n in used_names for n in must_use_any):
            return ScoreResult(False, "missing_use",
                               note=f"none of must_use_any={must_use_any} appear",
                               extracted={"used_names": sorted(used_names)})

    return ScoreResult(True, "structural_ok", extracted=info)


def _python_source_checks(code: str, checks: dict) -> ScoreResult | None:
    """Return a failure for unmet static source checks, otherwise None."""
    if not checks:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ScoreResult(False, "source_ast_parse_failed", note=str(exc))

    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return None

    functions = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function_name = checks.get("function")
    if function_name and function_name not in functions:
        return ScoreResult(False, "source_requirement_failed",
                           note=f"missing_function={function_name}")
    scope = functions.get(function_name, tree)
    calls = {
        dotted(node.func) for node in ast.walk(scope)
        if isinstance(node, ast.Call) and dotted(node.func)
    }
    for required in checks.get("calls", []):
        if not any(call == required or call.endswith("." + required) for call in calls):
            return ScoreResult(False, "source_requirement_failed",
                               note=f"missing_call={required}")
    if checks.get("recursive"):
        if not function_name or function_name not in calls:
            return ScoreResult(False, "source_requirement_failed",
                               note=f"missing_recursive_call={function_name}")
    return None


def code_unittest(resp: str, *, tests: list, timeout: int = 10,
                  mem_mb: int = 256, source_checks: dict | None = None) -> ScoreResult:
    """Run code in subprocess with tests appended. Real execution.
    
    Resource limits: RLIMIT_AS=mem_mb, RLIMIT_CPU=15s, no stdin, no site.
    Uses process group + timeout kill."""
    code = extract_code(resp).strip()
    if not code or len(code) < 8 or "def " not in code:
        return ScoreResult(False, "no_function_definition", note=f"resp_len={len(resp)}")
    source_failure = _python_source_checks(code, source_checks or {})
    if source_failure is not None:
        return source_failure

    full_program = code + "\n\n" + "\n".join(tests) + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_program)
        path = f.name

    def _set_limits():
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
            except (ValueError, OSError):
                pass
        except (ValueError, OSError):
            pass
        os.setsid()

    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_set_limits,
            env={},
            text=True,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try: out, err = proc.communicate(timeout=2)
            except Exception: out, err = "", "<killed>"
            return ScoreResult(False, "timeout",
                               note=f"timeout after {timeout}s; stderr_tail={(err or '')[-200:]}")
        if proc.returncode == 0:
            return ScoreResult(True, "tests_passed",
                               note=f"stdout_tail={(out or '')[-100:]}")
        return ScoreResult(False, "tests_failed",
                           note=f"exit={proc.returncode} stderr_tail={(err or '')[-300:]}")
    finally:
        if proc and proc.poll() is None:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError): pass
        try: Path(path).unlink(missing_ok=True)
        except Exception: pass


# ----------------------------------------------------------------------------
# Semantic keyword checks (for refusal/abstention/persona)
# ----------------------------------------------------------------------------
def _normalize_typography(s: str) -> str:
    """Normalize curly quotes / smart dashes / apostrophes for matching.
    Used by keyword matchers so that 'I don’t know' (curly U+2019) matches
    needle "I don't know" (straight ASCII)."""
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s


def _normalize_do_support_negation(s: str) -> str:
    """Canonicalize a small closed set of English negative paraphrases."""
    past = {
        "provide": "provided", "state": "stated", "specify": "specified",
        "give": "given", "mention": "mentioned", "include": "included",
        "show": "shown", "contain": "contained", "present": "present",
    }
    pattern = re.compile(
        r"\b(?:does|do|did)[ \t]+not[ \t]+(" + "|".join(past) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda match: "not " + past[match.group(1).lower()], s)


def _normalize_simple_code_literals(s: str) -> str:
    """Normalize quote style for simple, escape-free code/config literals."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?P<prefix>[bBrRuU]{0,2})"
        r"(?P<quote>['\"])(?P<body>[^'\"\\\n]*)(?P=quote)",
    )
    return pattern.sub(
        lambda match: match.group("prefix").lower()
        + json.dumps(match.group("body"), ensure_ascii=False),
        s,
    )


def _try_inflection_or_ordinal(needle: str, t: str, case_sensitive=False) -> bool:
    """Try matching needle in t with English inflection (s/es/ed/ing/ly...) or
    an ordinal suffix (st/nd/rd/th) if the needle ends in a digit.
    Allows multi-token needles whose LAST token is alphanumeric, e.g.
    'actually use' → 'actually used', or 'March 5' → 'March 5th'.
    """
    if not needle: return False
    n = needle if case_sensitive else needle.lower()
    haystack = t if case_sensitive else t.lower()
    last_char = n[-1]
    if last_char.isdigit():
        pat = r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?:st|nd|rd|th)?(?![A-Za-z0-9])"
    elif last_char.isalpha():
        pat = (r"(?<![A-Za-z0-9])" + re.escape(n) +
               r"(?:s|es|ed|d|ing|ly|er|est)?(?![A-Za-z0-9])")
    else:
        return False
    return bool(re.search(pat, haystack))


def semantic_keywords_any(resp: str, *, alternatives: list,
                           case_sensitive: bool = False,
                           require_word_boundary: bool = True,
                           disallowed_patterns: list | None = None) -> ScoreResult:
    """Pass if any of `alternatives` appears in resp. Word-boundary by default
    so 'correct' won't match 'incorrect'.

    Treats underscore as a boundary in addition to alphanumerics so that
    code-identifier substrings (`TypeError`) match inside `PyExc_TypeError`.

    Includes the same stem-mode and inflection-retry as `semantic_keywords_all`
    so that needles like `dilut` match `Dilutional`."""
    resp = _sanitize(resp)
    for pattern in disallowed_patterns or []:
        if re.search(pattern, resp, re.IGNORECASE | re.MULTILINE):
            return ScoreResult(False, "disallowed_pattern",
                               note=f"matched forbidden pattern {pattern!r}")
    resp_norm = _normalize_do_support_negation(_normalize_typography(resp))
    if not resp_norm.strip():
        return ScoreResult(False, "empty")

    def _looks_like_stem(s):
        if not s.isalpha() or not (4 <= len(s) <= 8): return False
        if s.endswith(("ing", "ion", "ess", "est", "ous", "ful", "ize")): return False
        if s.endswith(("e", "y")) and len(s) > 4: return False
        return s[-1].lower() in "bcdfghjklmnpqrstvwxz"

    def _try(needle, target):
        # Normalize needle typography too
        nn = _normalize_do_support_negation(_normalize_typography(needle))
        n = nn if case_sensitive else nn.lower()
        t = target if case_sensitive else target.lower()
        # Boundary semantics: alphanumerics — underscore is a boundary
        # (so TypeError matches in PyExc_TypeError).
        # 1. Strict word-boundary (excluding underscore, treating _ as boundary)
        if require_word_boundary:
            pat = r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![A-Za-z0-9])"
            if re.search(pat, t):
                return ("word_boundary", needle)
        # 2. Stem mode
        if _looks_like_stem(nn):
            pat = r"(?<![A-Za-z0-9])" + re.escape(n)
            if re.search(pat, t):
                return ("stem", needle)
        # 3. Inflection / ordinal retry (handles multi-token + digit endings)
        if require_word_boundary and _try_inflection_or_ordinal(nn, t, case_sensitive):
            return ("inflection_or_ordinal", needle)
        # 4. No-boundary fallback
        if not require_word_boundary:
            if n in t: return ("raw", needle)
        return None

    matched = []
    for alt in alternatives:
        if not alt: continue
        m = _try(alt, resp_norm)
        if m:
            matched.append(alt)
    if matched:
        return ScoreResult(True, "matched_any", extracted=matched[:3])
    return ScoreResult(False, "no_match",
                       note=f"none of {len(alternatives)} alternatives matched")


def semantic_keywords_all(resp: str, *, required: list,
                           case_sensitive: bool = False,
                           require_word_boundary: bool = True,
                           synonyms: dict = None,
                           normalize_whitespace_in_codeish: bool = True) -> ScoreResult:
    """All `required` items must appear (or one of their `synonyms[item]`).
    Word-boundary by default. For substrings that look code/CSS-like (contain
    `:` or `;` or `=`), whitespace around the delimiters is normalized in both
    haystack and needle before comparison — so `display: flex` matches
    substring `display:flex`.

    For substrings that look like word stems (length ≥ 4, not ending in a
    typical word-ending suffix), the right-side word boundary is relaxed —
    so substring `reliab` matches `reliability` / `reliably`."""
    resp = _sanitize(resp)
    resp_norm = _normalize_typography(resp)
    if not resp_norm.strip():
        return ScoreResult(False, "empty")
    synonyms = synonyms or {}

    def _normalize_codeish(s):
        # Collapse whitespace around : ; = in code-like contexts
        return _normalize_simple_code_literals(
            re.sub(r"\s*([:;=])\s*", r"\1", s)
        )

    def _looks_codeish(s):
        return (
            any(ch in s for ch in (":", ";", "=", "(", ")", "{", "}"))
            or bool(re.search(r"\b(?:import|from|const|let|var|default|version)\b", s))
            or bool(re.search(r"[bBrRuU]?[\"']", s))
        )

    def _looks_like_stem(s):
        # Heuristic: bare alpha sequence, length 4-7, ending in a consonant
        # cluster that's unlikely to be a complete word.
        if not s.isalpha() or not (4 <= len(s) <= 8): return False
        if s.endswith(("ing", "ion", "ess", "est", "ous", "ful", "ize")): return False
        if s.endswith(("e", "y")) and len(s) > 4: return False
        # E.g., "reliab", "abstent", "transm" — ends in consonant cluster
        return s[-1].lower() in "bcdfghjklmnpqrstvwxz"

    matched = []
    missing = []
    for item in required:
        ok = False
        candidates = [item] + synonyms.get(item, [])
        for c in candidates:
            cn = _normalize_typography(c)
            n = cn if case_sensitive else cn.lower()
            t = resp_norm if case_sensitive else resp_norm.lower()
            # 1. Boundary-aware match. Boundary requirements depend on the
            # needle's actual first/last characters, so a code fragment such
            # as `.fit(X, y)` can match inside `model.fit(X, y)` while a word
            # such as `correct` still cannot match inside `incorrect`.
            if require_word_boundary and not _looks_like_stem(cn):
                if _word_boundary_search(t, n, case_sensitive=True):
                    ok = True; matched.append((item, c, "word_boundary")); break
            # 2. Stem mode: relax right boundary for stem-looking needles
            if _looks_like_stem(cn):
                pat = r"(?<![A-Za-z0-9])" + re.escape(n)
                if re.search(pat, t):
                    ok = True; matched.append((item, c, "stem")); break
            # 3. Codeish mode: normalize whitespace around : ; =
            if normalize_whitespace_in_codeish and _looks_codeish(cn):
                norm_resp = _normalize_codeish(t)
                norm_needle = _normalize_codeish(n)
                if norm_needle in norm_resp:
                    ok = True; matched.append((item, c, "codeish")); break
            # 4. Inflection / ordinal retry (multi-token + digit-ending OK)
            if require_word_boundary and _try_inflection_or_ordinal(cn, t, case_sensitive):
                ok = True; matched.append((item, c, "inflection_or_ordinal")); break
            # 5. No-boundary substring (only when explicitly disabled)
            if not require_word_boundary:
                if n in t:
                    ok = True; matched.append((item, c, "raw")); break
        if not ok: missing.append(item)
    if not missing:
        return ScoreResult(True, "all_matched",
                           extracted=[m[1] for m in matched])
    return ScoreResult(False, "missing", note=f"missing={missing}")


# ----------------------------------------------------------------------------
# Abstention / refusal — heuristic with rich vocabulary
# ----------------------------------------------------------------------------
ABSTENTION_PHRASES = [
    # Direct uncertainty
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "no way to know", "no one knows", "cannot be determined", "can't be determined",
    "cannot determine", "can't determine", "not known", "unknown",
    # Prediction-uncertain
    "cannot predict", "can't predict", "unable to predict", "no way to predict",
    "impossible to predict", "uncertain", "not certain", "no reliable",
    "no way of knowing",
    # Subjective signals
    "subjective", "depends on", "personal preference", "matter of opinion",
    "matter of taste", "no single answer", "no objective", "varies", "depends",
    "comes down to", "different for", "different people", "no universally",
    "up to you", "up to the", "down to personal", "individual preference",
    # Speculative hedges
    "speculation", "speculative", "could be", "might be", "perhaps",
    "no historical record", "no surviving", "not recorded", "not preserved",
]


def abstention_check(resp: str, *, phrases: list = None) -> ScoreResult:
    """Pass if response shows uncertainty/abstention.
    Uses a rich phrase vocabulary by default; pass custom `phrases` to override."""
    phrases = phrases or ABSTENTION_PHRASES
    resp_norm = _normalize_for_match(resp)
    matched = []
    for ph in phrases:
        ph_norm = _normalize_for_match(ph)
        if ph_norm in resp_norm:
            matched.append(ph)
    if matched:
        return ScoreResult(True, "abstention_matched", extracted=matched[:3])
    # Soft check: confident factual statement is likely a fail for abstention probes
    return ScoreResult(False, "no_abstention_signal",
                       note=f"no abstention phrase from {len(phrases)} candidates")


# ----------------------------------------------------------------------------
# Format strict (instruction-following ONLY)
# ----------------------------------------------------------------------------
def format_strict_regex(resp: str, *, pattern: str, flags=None,
                        fullmatch: bool = True) -> ScoreResult:
    """Strict regex match — used ONLY when intent=format (instr.* atoms).
    By default fullmatch on the WHOLE response (not just lines).
    
    Special: when pattern is the no-op default '.*' (i.e., the probe didn't
    declare a real pattern), strict regex is meaningless — the caller should
    fall back to exact_match_string. Returns ScoreResult with strategy
    'no_pattern_specified' so the dispatcher can detect and re-route.
    """
    if pattern in (None, "", ".*", ".+"):
        return ScoreResult(False, "no_pattern_specified",
                           note="format_strict_regex called with degenerate pattern; caller should fall back")
    resp = _sanitize(resp)[:MAX_RESP_BYTES]
    re_flags = 0
    for fl in (flags or []):
        re_flags |= getattr(re, fl.upper(), 0)
    fn = re.fullmatch if fullmatch else re.search
    m = fn(pattern, resp, re_flags)
    if m:
        return ScoreResult(True, "format_match", extracted=m.group()[:200])
    return ScoreResult(False, "format_violation",
                       note=f"pattern={pattern[:100]!r} fullmatch={fullmatch}")


def format_lenient_regex(resp: str, *, pattern: str, flags=None,
                        allow_stem_fallback: bool = False,
                        skip_stem_fallback: bool = False,
                        case_sensitive: bool = True) -> ScoreResult:
    """Lenient regex — search anywhere, not fullmatch. For format-ish probes
    that aren't testing strict IF.

    Also tries word-boundary stem matches for alternation patterns
    (e.g., pattern 'pickup|pick up|...' will also match 'picked up').
    Skips stem fallback for structural / whole-document anchored patterns
    (those are usually "output must look exactly like this" — stem matching
    causes false positives there).

    Stem fallback is opt-in via ``allow_stem_fallback=True``. Anchored patterns
    never use it, even when opted in, because a stem cannot establish a
    whole-response format contract. ``skip_stem_fallback`` remains as a
    compatibility kill switch for callers with stricter atom semantics.
    """
    resp = _sanitize(resp)[:MAX_RESP_BYTES]
    re_flags = 0
    declared_flags = flags or []
    if isinstance(declared_flags, str):
        declared_flags = [declared_flags]
    compact_flags = {
        "I": re.IGNORECASE, "M": re.MULTILINE, "S": re.DOTALL,
        "X": re.VERBOSE, "A": re.ASCII,
    }
    for fl in declared_flags:
        name = str(fl).upper()
        re_flags |= compact_flags.get(name, getattr(re, name, 0))
    if not case_sensitive:
        re_flags |= re.IGNORECASE
    m = re.search(pattern, resp, re_flags)
    if m:
        return ScoreResult(True, "regex_match", extracted=m.group()[:200])
    # Skip stem fallback for clearly-structural anchored patterns OR when
    # explicitly forced off (intent-driven — e.g. tool.format_robustness),
    # OR when pattern uses 2+ lookaheads (compound-AND that alternation
    # split would silently break).
    stripped_pattern = pattern.strip()
    has_caret_anchor = bool(re.match(r"^(?:\(\?[A-Za-z-]+\))*\^", stripped_pattern))
    is_anchored = (
        "\\A" in pattern or "\\Z" in pattern
        or (has_caret_anchor and stripped_pattern.endswith("$"))
    )
    is_structural = (
        is_anchored or
        pattern.count("```") >= 1 or
        pattern.count("\\n") >= 4 or
        pattern.count("(?=") >= 2  # compound lookahead AND — stem split unsafe
    )
    if not allow_stem_fallback or skip_stem_fallback or is_structural:
        return ScoreResult(False, "regex_no_match",
                           note=f"stem fallback disabled or unsafe: pattern={pattern[:80]!r}")
    # Fallback: split alternation, do stem-based search.
    alternatives = re.split(r"\|", pattern.strip("()^$"))
    for alt in alternatives:
        clean = alt
        clean = re.sub(r"\\[bdwsSDWAZ.()\[\]{}|+*?^$]", " ", clean)
        clean = re.sub(r"\\.", " ", clean)
        clean = re.sub(r"[(){}\[\]?*+|^$.]", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z_0-9]{3,}", clean)]
        if not words: continue
        words.sort(key=len, reverse=True)
        for stem in words[:3]:
            if re.search(r"\b" + re.escape(stem) + r"\b", resp, re_flags):
                return ScoreResult(True, "regex_stem_match", extracted=stem,
                                   note=f"matched stem '{stem}' from alternation")
    return ScoreResult(False, "regex_no_match",
                       note=f"pattern={pattern[:100]!r}")


# ----------------------------------------------------------------------------
# Fraction match (1/2 ↔ 0.5 ↔ \frac{1}{2})
# ----------------------------------------------------------------------------
def fraction_match(resp: str, *, expected: str, tolerance: float = 1e-9) -> ScoreResult:
    """Match a fraction-style answer where the model may write 1/2, 0.5,
    \\frac{1}{2}, or even \\dfrac{1}{2}. Compares numeric values within
    `tolerance`. ASCII-only fractions; LaTeX commands are accepted via the
    standard \\frac{a}{b} / \\dfrac{a}{b} parse.
    """
    resp = _sanitize(resp)
    expected = str(expected).strip()
    try:
        if "/" in expected:
            num, den = expected.split("/", 1)
            target = float(num) / float(den)
        else:
            target = float(expected)
    except (ValueError, ZeroDivisionError):
        return ScoreResult(False, "expected_not_fraction",
                           note=f"expected={expected!r} not a fraction or number")

    # \frac{a}{b}, \dfrac{a}{b}
    for m in re.finditer(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", resp):
        try:
            val = float(m.group(1)) / float(m.group(2))
            if abs(val - target) <= tolerance:
                return ScoreResult(True, "latex_frac",
                                   extracted=f"{m.group(1)}/{m.group(2)}",
                                   expected=expected)
        except (ValueError, ZeroDivisionError):
            pass

    # a/b style
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", resp):
        try:
            val = float(m.group(1)) / float(m.group(2))
            if abs(val - target) <= tolerance:
                return ScoreResult(True, "slash_fraction",
                                   extracted=f"{m.group(1)}/{m.group(2)}",
                                   expected=expected)
        except (ValueError, ZeroDivisionError):
            pass

    # decimal
    for n in NUMBER_RE.findall(resp.replace(",", "")):
        try:
            if abs(float(n) - target) <= tolerance:
                return ScoreResult(True, "decimal", extracted=float(n),
                                   expected=expected)
        except ValueError:
            pass

    # Direct string contains (last resort)
    if expected in resp:
        return ScoreResult(True, "raw_string", extracted=expected,
                           expected=expected, note="matched expected as substring")
    return ScoreResult(False, "no_fraction_match", expected=expected)


# ----------------------------------------------------------------------------
# Numeric range
# ----------------------------------------------------------------------------
def numeric_range(resp: str, *, low: float, high: float,
                  inclusive: bool = True, integer_only: bool = False,
                  single_value_only: bool = False) -> ScoreResult:
    text = _sanitize(resp)
    if not text.strip():
        return ScoreResult(False, "no_number")

    if single_value_only:
        clean = _strip_markdown_wrappers(text)
        required_pattern = r"[+-]?\d+" if integer_only else NUMBER_RE.pattern
        if not re.fullmatch(required_pattern, clean):
            return ScoreResult(False, "range_format_violation",
                               note="expected exactly one numeric value")
        raw_value = clean
        strategy = "single_value_in_range"
    else:
        normalized = text.replace(",", "")
        positioned = []
        markers = _answer_marker_candidates(normalized)
        for offset, source in markers:
            candidates = [
                match for match in NUMBER_RE.finditer(source)
                if not _numeric_token_is_negated(
                    normalized, offset + match.start(), offset + match.end(),
                )
            ]
            positioned.append((
                offset,
                candidates[0].group(0) if candidates else None,
                "answer_marker_in_range",
            ))
        boxes = _balanced_latex_arguments(normalized)
        for _, _, arg_start, arg_end, content in boxes:
            clean = _strip_answer_candidate_wrappers(content)
            valid = NUMBER_RE.fullmatch(clean)
            positioned.append((
                arg_start,
                clean if valid and not _numeric_token_is_negated(
                    normalized, arg_start, arg_end,
                ) else None,
                "boxed_in_range",
            ))
        lines = _nonempty_lines_with_offsets(normalized)
        if lines:
            line_start, line_end, source = lines[-1]
            explicit_on_last = any(
                line_start <= position <= line_end for position, _ in markers
            ) or any(
                line_start <= command_start <= line_end
                for command_start, _, _, _, _ in boxes
            )
            if not explicit_on_last:
                candidates = [
                    match for match in NUMBER_RE.finditer(source)
                    if not _numeric_token_is_negated(
                        normalized, line_start + match.start(),
                        line_start + match.end(),
                    )
                ]
                if candidates:
                    match = candidates[-1]
                    positioned.append((
                        line_start + match.start(), match.group(0),
                        "last_line_in_range",
                    ))
        if not positioned:
            return ScoreResult(False, "no_explicit_range_answer")
        _, raw_value, strategy = max(positioned, key=lambda item: item[0])
        if raw_value is None:
            return ScoreResult(False, "no_explicit_range_answer")

    try:
        value = Decimal(raw_value)
    except InvalidOperation:
        return ScoreResult(False, "invalid_range_number", extracted=raw_value)
    if integer_only and value != value.to_integral_value():
        return ScoreResult(False, "range_requires_integer",
                           extracted=raw_value,
                           note="selected value is not an integer")

    low_value, high_value = Decimal(str(low)), Decimal(str(high))
    in_range = (low_value <= value <= high_value) if inclusive \
        else (low_value < value < high_value)
    extracted = int(value) if value == value.to_integral_value() else float(value)
    if in_range:
        return ScoreResult(True, strategy, extracted=extracted,
                           expected=f"[{low},{high}]" if inclusive else f"({low},{high})")
    return ScoreResult(False, "out_of_range", extracted=extracted,
                       expected=f"[{low},{high}]")


# ----------------------------------------------------------------------------
# Judge_rubric — explicit deferral (no longer silently false)
# ----------------------------------------------------------------------------
def judge_deferred() -> ScoreResult:
    """Compatibility fallback when the configured external judge is unavailable."""
    return ScoreResult(False, "judge_deferred", deferred=True,
                       note="LLM judge unavailable; probe excluded from pass-rate denominator")


def llm_judge(probe: dict, response: str) -> ScoreResult:
    """Use the configured LLM judge; infrastructure errors defer."""
    try:
        from eval.judge import judge_response
        verdict, _ = judge_response(probe, response)
    except Exception as exc:
        return ScoreResult(False, "judge_deferred", deferred=True,
                           note=f"LLM judge unavailable: {type(exc).__name__}: {exc}")
    status = verdict["verdict"]
    meta = verdict.get("_judge_meta", {})
    extracted = {
        "verdict": status,
        "confidence": verdict["confidence"],
        "criterion_results": verdict.get("criterion_results", []),
        "failed_criteria": verdict.get("failed_criteria", []),
        "judge_provider": meta.get("provider"),
        "judge_model": meta.get("response_model") or meta.get("requested_model"),
        "judge_requested_model": meta.get("requested_model"),
        "judge_response_id": meta.get("response_id"),
        "judge_prompt_version": meta.get("prompt_version"),
        "judge_attempt": meta.get("attempt"),
        "usage": meta.get("usage", {}),
    }
    if status == "ERROR":
        return ScoreResult(False, "llm_judge_error", extracted=extracted,
                           note=verdict["reason"], deferred=True)
    return ScoreResult(status == "PASS",
                       "llm_judge_pass" if status == "PASS" else "llm_judge_fail",
                       extracted=extracted, note=verdict["reason"])
