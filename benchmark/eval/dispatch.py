"""Route each probe using its atom policy and scorer type.

USAGE
-----
    from eval.dispatch import score_probe, load_policy
    
    policy = load_policy("policy/atom_intents.yaml")
    result, denom_count = score_probe(probe, response, policy)
    # result.passed: bool
    # result.strategy: which fallback succeeded
    # denom_count: 1 if probe counts toward main score, 0 if excluded (e.g. judge stub)
"""
from __future__ import annotations
import re, json, fnmatch, signal, time
from pathlib import Path
from typing import Any
from eval.scorer import (
    ScoreResult, extract_numeric, extract_string, extract_mcq_letter,
    extract_set, extract_ordered_sequence, exact_integer_output, extract_integer,
    exact_whole_response, extract_json, json_schema_lenient,
    json_schema_strict, code_structural,
    code_unittest, semantic_keywords_any, semantic_keywords_all,
    abstention_check, format_strict_regex, format_lenient_regex,
    numeric_range, judge_deferred, llm_judge, _word_boundary_search, fraction_match,
    code_parse_ast, _answer_marker_candidates, _balanced_latex_arguments,
    _nonempty_lines_with_offsets, _strip_answer_candidate_wrappers,
    _strip_markdown_wrappers, _fenced_block_candidates,
    python_ast_contains_reference, _symbolic_expressions_equivalent,
    _safe_noninstr_equivalence, _correction_candidates,
    _tagged_answer_candidates, _authoritative_tagged_answer_candidates,
    _tagged_answer_is_negated, _tagged_answers_conflict,
    _is_complete_tagged_answer, _position_has_nonanswer_label,
    _position_has_nonanswer_block_label,
    _answer_payload_is_presentation_opener,
    _is_terminal_latex_box,
    normalize_noninstr_transport,
)


class _RegexTimeout(Exception): pass


class _MainThreadAlarm:
    """SIGALRM-based wall clock timeout. Only works in the MAIN thread.
    On Windows or non-main threads, degrades to no-op (caller must enforce
    timeout via subprocess or other means)."""
    def __init__(self, seconds=2):
        self.seconds = seconds
        self._installed = False
        self._previous_handler = None
        self._previous_timer = (0.0, 0.0)
        self._started = 0.0
    def __enter__(self):
        try:
            self._previous_handler = signal.getsignal(signal.SIGALRM)
            self._previous_timer = signal.getitimer(signal.ITIMER_REAL)
            self._started = time.monotonic()
            delay = float(self.seconds)
            previous_delay = self._previous_timer[0]
            if previous_delay > 0:
                delay = min(delay, previous_delay)
            signal.signal(signal.SIGALRM, self._handler)
            signal.setitimer(signal.ITIMER_REAL, delay)
            self._installed = True
        except (ValueError, AttributeError, OSError):
            self._installed = False
        return self
    def __exit__(self, *exc):
        if self._installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, self._previous_handler)
                previous_delay, previous_interval = self._previous_timer
                if previous_delay > 0:
                    elapsed = time.monotonic() - self._started
                    previous_delay = max(previous_delay - elapsed, 1e-6)
                signal.setitimer(
                    signal.ITIMER_REAL, previous_delay, previous_interval,
                )
            except (ValueError, AttributeError, OSError):
                pass
        return False
    @staticmethod
    def _handler(*a): raise _RegexTimeout()


def load_policy(path: str | Path) -> dict:
    """Tiny YAML reader (no PyYAML dependency for trivial cases)."""
    try:
        import yaml
        return yaml.safe_load(open(path))
    except ImportError:
        # Fallback: extremely simple parser — only handles the structure we use.
        return _simple_yaml_load(open(path).read())


def _strip_unquoted_yaml_comment(line: str) -> str:
    """Remove a YAML line comment while preserving ``#`` inside quotes."""
    quote = None
    escaped = False
    i = 0
    while i < len(line):
        ch = line[i]
        if quote == '"':
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quote = None
        elif quote == "'":
            if ch == "'" and i + 1 < len(line) and line[i + 1] == "'":
                i += 1  # YAML single-quoted escape: ''
            elif ch == "'":
                quote = None
        elif ch in {'"', "'"}:
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
        i += 1
    return line.rstrip()


def _simple_yaml_scalar(value: str):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1]
        items, start, quote, escaped = [], 0, None, False
        for i, ch in enumerate(body):
            if quote == '"':
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    quote = None
            elif quote == "'":
                if ch == "'":
                    quote = None
            elif ch in {'"', "'"}:
                quote = ch
            elif ch == ",":
                item = body[start:i].strip()
                if item:
                    items.append(_simple_yaml_scalar(item))
                start = i + 1
        item = body[start:].strip()
        if item:
            items.append(_simple_yaml_scalar(item))
        return items
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~"}:
        return None
    return value


def _simple_yaml_load(text):
    """Minimal, safe YAML parser for ``policy/atom_intents.yaml``.

    It supports nested mappings, scalar block lists, and inline scalar lists.
    The fallback deliberately implements only this policy's data shape; unlike
    the previous parser it handles block lists and does not retain unquoted
    line-end comments as part of intent names.
    """
    lines = []
    for raw in text.splitlines():
        cleaned = _strip_unquoted_yaml_comment(raw)
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append((indent, cleaned.strip()))

    def parse_node(index: int, indent: int):
        is_list = lines[index][1].startswith("- ")
        node = [] if is_list else {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"unexpected YAML indentation near {content!r}")
            if is_list:
                if not content.startswith("- "):
                    break
                value = content[2:].strip()
                if not value:
                    if index + 1 >= len(lines) or lines[index + 1][0] <= indent:
                        node.append(None)
                        index += 1
                    else:
                        child, index = parse_node(index + 1, lines[index + 1][0])
                        node.append(child)
                else:
                    node.append(_simple_yaml_scalar(value))
                    index += 1
                continue

            if content.startswith("- ") or ":" not in content:
                break
            key, value = content.split(":", 1)
            key = str(_simple_yaml_scalar(key.strip()))
            value = value.strip()
            if value:
                node[key] = _simple_yaml_scalar(value)
                index += 1
            elif index + 1 < len(lines) and lines[index + 1][0] > indent:
                child, index = parse_node(index + 1, lines[index + 1][0])
                node[key] = child
            else:
                node[key] = {}
                index += 1
        return node, index

    if not lines:
        return {}
    parsed, final_index = parse_node(0, lines[0][0])
    if final_index != len(lines):
        raise ValueError(f"could not parse YAML near {lines[final_index][1]!r}")
    return parsed


# ----------------------------------------------------------------------------
# Atom-aware intent resolution
# ----------------------------------------------------------------------------
def resolve_intent(probe: dict, policy: dict) -> str | None:
    """Given a probe and the policy, return the intent name or None."""
    atom = probe.get("canonical_id", "")
    prefixes = (policy or {}).get("prefixes", {})
    # Exact atom match wins
    if atom in prefixes:
        return prefixes[atom].get("intent")
    # Glob-style prefix match (e.g. arith.*)
    for pattern, conf in prefixes.items():
        if fnmatch.fnmatch(atom, pattern):
            return conf.get("intent")
    return None


def is_excluded(probe: dict, policy: dict) -> tuple[bool, str]:
    """Return (excluded, reason). Excluded probes don't count toward main score."""
    if probe.get("enabled") is False:
        return (True, "probe explicitly disabled")
    excl = (policy or {}).get("excluded_from_main_score", {})
    sc_types = excl.get("scorer_types", []) or []
    if probe.get("scorer", {}).get("type") in sc_types:
        return (True, f"scorer_type {probe['scorer']['type']} excluded")
    atom = probe.get("canonical_id", "")
    for prefix in (excl.get("atom_prefixes", []) or []):
        if atom.startswith(prefix):
            return (True, f"atom prefix {prefix} excluded")
    return (False, "")


# ----------------------------------------------------------------------------
# Default routing: Map a declared scorer type to its implementation
# ----------------------------------------------------------------------------
def _strict_instruction_following(probe: dict) -> bool:
    """Only the dedicated instruction-following category is format-strict.

    Other capability categories are scored for answer correctness and retain
    the safe extraction scaffolding used by ACE 2.5.  Functional contracts
    such as parseable JSON/code/AST remain handled by their own scorers.
    """
    return str(probe.get("category", "")).strip().lower() == "instr"


def _code_payload_candidate(response: str) -> tuple[str | None, str | None]:
    """Select one functional code/config answer without accepting buried prose."""
    text = str(response).strip()
    # One outer code fence whose closing delimiter alone was truncated is a
    # code-answer transport defect.  Repair it locally here, never in global
    # normalization: output artifacts and byte-exact formats must still fail
    # when a required closing fence is absent.
    unclosed = re.fullmatch(
        r"(?P<fence>```|~~~)[A-Za-z0-9_+.-]*[ \t]*\r?\n"
        r"(?P<body>[\s\S]+)",
        text,
    )
    if (unclosed and not re.search(
            r"(?m)^[ \t]*(?:```|~~~)", unclosed.group("body"))):
        newline = "\r\n" if "\r\n" in text else "\n"
        text = text + newline + unclosed.group("fence")
    fences = _fenced_block_candidates(text)
    if fences:
        bodies = {body.strip() for _, _, _, body in fences}
        if len(bodies) > 1:
            # Worked examples may precede a separately labelled final block.
            # Select the final fence only when an explicit answer marker lies
            # after every earlier fence and immediately introduces that final
            # block.  Without this positional proof, multiple distinct code
            # payloads remain ambiguous.
            last_start, _, _, last_body = fences[-1]
            previous_end = fences[-2][1]
            interstitial = text[previous_end:last_start]
            affirmative_final_heading = re.search(
                r"(?is)(?:^|\r?\n)[ \t]*(?:#{1,6}[ \t]+)?"
                r"(?:[*_`]+[ \t]*)?"
                r"final[ \t]+(?:answer|response|output|result)"
                r"(?:[ \t]*[*_`]+)?[ \t]*:?[ \t]*\s*"
                r"(?:```|~~~)[A-Za-z0-9_+.-]*[ \t]*\r?\n\Z",
                interstitial,
            )
            if not affirmative_final_heading:
                return None, "conflicting_code_fences"
            body = last_body.strip()
        else:
            body = fences[-1][3].strip()
        _, fence_end, _, body = fences[-1]
        trailing = text[fence_end:].strip()
        if trailing and re.match(
                r"(?i)^(?:actually|correction|corrected|rather|instead)\b",
                trailing):
            return None, "code_answer_corrected_after_fence"
        return body.strip(), None
    if (len(text) >= 2 and text.startswith("`") and text.endswith("`")
            and not text.startswith("```") and "\n" not in text):
        return text[1:-1].strip(), None
    return text, None


def _unresolved_code_refusal(text: str) -> bool:
    """Reject refusal prose that merely repeats all required code tokens."""
    prefix = text.lstrip()[:500]
    return bool(re.search(
        r"(?i)^(?:i[ \t]+(?:cannot|can't|won't|will[ \t]+not|must[ \t]+not|"
        r"am[ \t]+unable[ \t]+to)|sorry\b[^\n]{0,80}\b(?:cannot|can't)|"
        r"(?:cannot|can't|unable[ \t]+to)[ \t]+(?:provide|produce|help|comply)|"
        r"无法|不能|拒绝)",
        prefix,
    ))


def _narrative_tail_after_explicit_answer(
        text: str, *, allow_descriptive_math_context: bool = False) -> bool:
    """Identify closing prose that should not hide an explicit answer.

    Corrections, negations, and anything containing a value remain
    authoritative.  This is used only after a marker, box, fence, or math
    presentation has already exposed an answer candidate.
    """
    value = text.strip()
    if not value or not re.search(r"[.!?。！？]\s*$", value):
        return False
    if re.search(
            r"\b(?:but|however|actually|instead|wrong|incorrect|not|correction)\b",
            value, re.IGNORECASE):
        return False
    if (_answer_marker_candidates(value)
            or _correction_candidates(value)
            or _tagged_answer_candidates(value)
            or _balanced_latex_arguments(value)
            or _fenced_block_candidates(value)):
        return False
    # A sentence that declares a new final/actual value is a competing answer,
    # even when it begins with the otherwise-narrative-looking ``This is``.
    # Check this before the descriptive-sentence shortcut below.
    if re.search(
            r"(?i)\b(?:actual(?:ly)?[ \t]+)?final[ \t]+"
            r"(?:answer|result|value)\b|"
            r"^this[ \t]+is[ \t]+(?:the[ \t]+)?(?:actual[ \t]+)?"
            r"(?:final[ \t]+)?(?:answer|result|value)\b"
            r"(?![ \t]+of\b)",
            value):
        return False
    declaration_text = re.sub(r"(?:\*\*|__)", "", value)
    if re.search(
            r"(?i)^this[ \t]+is[ \t]+(?:the|an?)[ \t]+"
            r"(?:[A-Za-z]+[ \t]+){0,3}"
            r"(?:answer|result|value|number|fraction|expression|solution)\b"
            r"(?![ \t]+(?:of|after|for|from|in|under|when)\b)",
            declaration_text):
        return False
    words = re.findall(r"[^\W\d_]+", value, re.UNICODE)
    descriptive_this_is = bool(
        len(words) >= 5
        and re.match(
            r"(?i)^this[ \t]+is[ \t]+(?:the|an?|already)\b", value,
        )
    )
    if allow_descriptive_math_context and descriptive_this_is:
        return True
    # Any visible number/math expression makes the line answer-bearing.  This
    # must also precede the ``This is the ...`` shortcut so a later conflicting
    # scalar cannot be dismissed as prose.
    cue_text = re.sub(r"(?:\*\*|__)(.+?)(?:\*\*|__)", r"\1", value)
    if re.search(r"\d|[=+*/√$]|\\(?:frac|sqrt|boxed)\b", cue_text):
        return False
    # A long closing sentence beginning "This is the ..." describes the
    # already selected answer; it is not a second answer.  Explicit answer /
    # result declarations remain authoritative and are excluded here.
    if descriptive_this_is:
        return True
    return len(words) >= 5 and bool(re.match(
        r"(?i)^(?:that(?:'|’)?s|that[ \t]+is|this[ \t]+is|these[ \t]+are|"
        r"those[ \t]+are|this[ \t]+gives|which[ \t]+is)\b",
        value,
    ))


def _final_answer_heading_directly_introduces_box(
        text: str, box_command_start: int) -> bool:
    """Prove that a line-only Final Answer marker directly owns a box."""
    prefix = str(text)[:max(0, box_command_start)]
    return bool(re.search(
        r"(?is)(?:^|\r?\n)[ \t]*(?:#{1,6}[ \t]+)?"
        r"(?:[^\w\s*_`#:\r\n]{0,8}[ \t]*)?"
        r"(?:[*_`]+[ \t]*)?final[ \t]+answer"
        r"(?:[ \t]*[*_`]+)?[ \t]*:?[ \t]*"
        r"(?:(?:\r?\n[ \t]*)+)?"
        r"(?:(?:\$\$|\\\[)[ \t]*(?:\r?\n[ \t]*)?)?\Z",
        prefix,
    ))


def _boxed_final_has_only_one_trailing_sentence(
        text: str, box_command_end: int, final_line_start: int) -> bool:
    """No hidden answer-bearing line may occur between the box and tail."""
    if final_line_start <= box_command_end:
        return False
    between = str(text)[box_command_end:final_line_start]
    return bool(re.fullmatch(
        r"(?:[ \t]*(?:(?:\$\$|\\\])[ \t]*)?\r?\n)+[ \t]*",
        between,
    ))


def _regex_semantic_candidates(candidate: str) -> list[str]:
    """Expose safe answer-level spellings inside a selected regex candidate."""
    variants = [candidate]
    terminal_stripped = _strip_answer_candidate_wrappers(
        candidate, strip_terminal_punctuation=True,
    )
    if terminal_stripped not in variants:
        variants.append(terminal_stripped)

    # A derivation ending in ``... = answer`` transports the final answer on
    # its right-hand side.  Comparisons such as <=, !=, and == are untouched.
    for value in list(variants):
        equals = list(re.finditer(r"(?<![<>=!])=(?!=)", value))
        if equals:
            rhs = value[equals[-1].end():].strip()
            rhs = _strip_answer_candidate_wrappers(
                rhs, strip_terminal_punctuation=True,
            )
            if rhs and rhs not in variants:
                variants.append(rhs)

    # ``X (or Y)`` is one answer only when both displayed forms are proven
    # equivalent by the exact, non-evaluating symbolic normalizer.
    for value in list(variants):
        alternative = re.fullmatch(
            r"([\s\S]*?)\s*\((?:or|equivalently)\s+([\s\S]+)\)\s*",
            value, re.IGNORECASE,
        )
        if alternative:
            first = alternative.group(1).strip()
            second = alternative.group(2).strip()
            if _symbolic_expressions_equivalent(first, second):
                for form in (first, second):
                    if form not in variants:
                        variants.append(form)
    return variants


def _tool_format_transport_candidate(
        response: str, pattern: str) -> tuple[str | None, str | None]:
    """Unwrap one whole-response tool-call transport convention.

    ``tool.format_robustness`` measures the call payload, not whether a model
    used one Markdown wrapper.  Keep this deliberately narrower than ordinary
    answer extraction: exactly one inline-code wrapper, exactly one fenced
    block, or one terminal sentence mark on an otherwise complete payload.
    Prose, multiple fences, and wrapper-plus-punctuation combinations remain
    invalid.  The returned payload must still satisfy the probe's original
    anchored regex.
    """
    stripped_pattern = str(pattern).strip()
    anchored = bool(
        ("\\A" in stripped_pattern and "\\Z" in stripped_pattern)
        or (re.match(r"^(?:\(\?[A-Za-z-]+\))*\^", stripped_pattern)
            and stripped_pattern.endswith("$"))
    )
    if not anchored:
        return None, None

    text = str(response).strip()
    inline = re.fullmatch(r"`([^`\r\n]+)`", text)
    if inline:
        return inline.group(1), "tool_format_single_inline_wrapper"

    fences = _fenced_block_candidates(text)
    if (len(fences) == 1 and fences[0][0] >= 0
            and not text[:fences[0][0]].strip("`~ \t\r\nABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_+.-")
            and not text[fences[0][1]:].strip()):
        # Re-check the complete outer shape so a second/interstitial delimiter
        # cannot be swallowed as fence body by the bounded block extractor.
        outer = re.fullmatch(
            r"(?P<fence>```|~~~)[A-Za-z0-9_+.-]*[ \t]*\r?\n"
            r"(?P<body>[\s\S]*?)\r?\n(?P=fence)",
            text,
        )
        if (outer and not re.search(
                r"(?m)^[ \t]*(?:```|~~~)[A-Za-z0-9_+.-]*[ \t]*$",
                outer.group("body"))):
            return outer.group("body"), "tool_format_single_fence_wrapper"

    if (len(text) >= 2 and text[-1] in ".?!"
            and text[-2] not in ".?!"):
        return text[:-1], "tool_format_single_terminal_punctuation"
    return None, None


def default_route(probe: dict, response: str, policy: dict) -> ScoreResult:
    """Dispatch using the scorer type declared in the probe."""
    sc = probe.get("scorer", {})
    t = sc.get("type")
    gt = probe.get("ground_truth")
    synonyms = (policy or {}).get("synonyms", {})
    strict_if = _strict_instruction_following(probe)

    if t == "exact_match_numeric":
        return extract_numeric(response, sc.get("expected", gt),
                               tolerance=float(sc.get("tolerance", 0)),
                               require_only_numeric=(
                                   strict_if and (
                                       sc.get("require_only_numeric_output", False)
                                       or sc.get("accepted_output_format") == "numeric_only"
                                   )
                               ))

    if t == "exact_match_string":
        strict_no_extra_text = str(
            probe.get("canonical_id", "")
        ).startswith("output.no_extra_text")
        if strict_if or strict_no_extra_text:
            return exact_whole_response(
                response, sc.get("expected", gt),
                case_sensitive=sc.get("case_sensitive", False),
                strip_outer_whitespace=sc.get("strip_whitespace", True),
            )
        expected_text = str(sc.get("expected", gt)).strip()
        prompt_text = str(probe.get("prompt", ""))
        if (re.fullmatch(r"\d+", expected_text)
                and re.search(
                    r"\b(?:how[ \t]+many|number[ \t]+of|count[ \t]+of)\b",
                    prompt_text, re.IGNORECASE,
                )):
            return extract_numeric(
                response, expected_text, tolerance=0,
                allow_leading_count=True,
            )
        return extract_string(
            response, sc.get("expected", gt),
            case_sensitive=sc.get("case_sensitive", False),
            # ACE 2.5-style answer extraction, hardened with token boundaries
            # and local negation guards in extract_string.
            allow_substring=True,
            accepted_aliases=sc.get("accepted_aliases"),
            normalize_internal_whitespace=sc.get("normalize_internal_whitespace", False),
            strip_whitespace=True,
            allow_terminal_punctuation=sc.get("allow_terminal_punctuation", True),
            allow_semantic_equivalence=True,
        )

    if t == "exact_integer":
        if strict_if:
            return exact_integer_output(response, sc.get("expected", gt))
        return extract_integer(response, sc.get("expected", gt))

    if t == "mcq_letter":
        return extract_mcq_letter(
            response, sc.get("expected", gt), require_bare=strict_if,
        )

    if t == "set_equal":
        items = sc.get("items") or sc.get("expected") or gt
        return extract_set(response, items,
                           separator_hint=sc.get("delimiter", ","),
                           case_sensitive=sc.get("case_sensitive", False),
                           reject_duplicates=(strict_if and sc.get(
                               "reject_duplicates",
                               sc.get("reject_duplicate_items", False),
                           )),
                           strict_separator=(strict_if and bool(sc.get("delimiter"))))

    if t == "sequence_equal":
        items = sc.get("items") or sc.get("expected") or gt
        return extract_ordered_sequence(
            response, items,
            separator_hint=sc.get("delimiter", ","),
            numeric=sc.get("numeric", False),
            strict_separator=(strict_if and bool(sc.get("delimiter"))),
            case_sensitive=sc.get("case_sensitive", False),
        )

    if t == "regex":
        # The regex remains the source of truth.  Outside ``instr`` we may
        # remove harmless answer wrappers, then apply that same regex again.
        atom = probe.get("canonical_id", "")
        # Atoms that explicitly test format compliance — disable stem fallback
        # to prevent matching a literal stem (e.g. 'sphereVolume') from a
        # short anchored format pattern.
        is_format_atom = (
            atom.startswith((
                "tool.format_", "output.format_", "output.artifact_",
                "output.no_extra_text",
            ))
            or probe.get("category") == "instr"
            or str((probe.get("axis_values") or {}).get(
                "format_strictness", "",
            )) in {"byte_exact", "exact_single_artifact"}
        )
        is_tool_format_robustness = (
            atom == "tool.format_robustness"
            or atom.startswith("tool.format_robustness.")
        )
        scoring_response = response
        if not strict_if and not is_format_atom:
            # Markdown hard-break spaces are presentation-only for a
            # non-executing regex match.  Keep this variant local to the regex
            # route: code_unittest/static parser/heredoc inputs must receive
            # the original line bytes.
            scoring_response = "\n".join(
                line.rstrip() for line in scoring_response.split("\n")
            )
        if not strict_if and probe.get("category") == "code":
            scoring_response, payload_error = _code_payload_candidate(response)
            if payload_error:
                return ScoreResult(False, payload_error)
            if scoring_response is None or _unresolved_code_refusal(scoring_response):
                return ScoreResult(False, "code_refusal_without_answer")
        authoritative_override = False
        if not strict_if:
            authority_text = scoring_response.strip()
            authority_strong = (
                _answer_marker_candidates(authority_text)
                + _correction_candidates(authority_text)
            )
            authoritative_override = bool(
                authority_strong
                or _authoritative_tagged_answer_candidates(
                    authority_text, authority_strong,
                )
                or _balanced_latex_arguments(authority_text)
                or _fenced_block_candidates(authority_text)
            )
        try:
            with _MainThreadAlarm(seconds=3):
                result = format_lenient_regex(
                    scoring_response, pattern=sc["pattern"], flags=sc.get("flags"),
                    allow_stem_fallback=sc.get("allow_stem_fallback", False),
                    skip_stem_fallback=is_format_atom,
                    case_sensitive=sc.get("case_sensitive", True),
                )
                if result.passed:
                    declared = sc.get("flags") or []
                    if isinstance(declared, str):
                        declared = [declared]
                    ignore_case = (
                        sc.get("case_sensitive") is False
                        or any(str(flag).upper() in {"I", "IGNORECASE"}
                               for flag in declared)
                    )
                    if (probe.get("category") == "code" and ignore_case
                            and re.search(r"\b(?:JavaScript|TypeScript|TSX)\b",
                                          str(probe.get("prompt", "")), re.I)):
                        case_flags = 0
                        for flag in declared:
                            name = str(flag).upper()
                            if name not in {"I", "IGNORECASE"}:
                                case_flags |= getattr(re, name, 0)
                        if not re.search(sc["pattern"], scoring_response, case_flags):
                            return ScoreResult(
                                False, "code_regex_case_mismatch",
                                note="identifier-sensitive language cannot use IGNORECASE fallback",
                            )
                    # Output-artifact probes may intentionally require the
                    # complete fenced document.  Once their anchored contract
                    # has matched the whole response, inner fence extraction
                    # must not discard that PASS and retry the fence body
                    # against a regex which itself requires the fence.
                    output_whole_response_contract = (
                        probe.get("category") == "output"
                        and (
                            ("\\A" in sc["pattern"] and "\\Z" in sc["pattern"])
                            or (
                                sc["pattern"].lstrip().startswith("^")
                                and sc["pattern"].rstrip().endswith("$")
                            )
                        )
                    )
                    if not authoritative_override or output_whole_response_contract:
                        return result
                if is_tool_format_robustness and not strict_if:
                    tool_candidate, tool_strategy = (
                        _tool_format_transport_candidate(
                            scoring_response, sc["pattern"],
                        )
                    )
                    if tool_candidate is not None:
                        tool_result = format_lenient_regex(
                            tool_candidate,
                            pattern=sc["pattern"], flags=sc.get("flags"),
                            allow_stem_fallback=False,
                            skip_stem_fallback=True,
                            case_sensitive=sc.get("case_sensitive", True),
                        )
                        if tool_result.passed:
                            tool_result.strategy = tool_strategy or tool_result.strategy
                            tool_result.note = (
                                "whole-response presentation wrapper removed "
                                "before original anchored regex"
                            )
                            return tool_result
                if is_format_atom:
                    # Exact artifact/format atoms score the submitted bytes;
                    # answer extraction must not synthesize or unwrap tokens.
                    return result
                # ACE 2.5-style scaffolding for every non-instruction
                # capability.  We intentionally do not restore arbitrary stem
                # or suffix matching: an extracted candidate must still pass
                # the probe's original regex.
                if not strict_if:
                    re_flags = 0
                    declared_flags = sc.get("flags") or []
                    if isinstance(declared_flags, str):
                        declared_flags = [declared_flags]
                    compact_flags = {
                        "I": re.IGNORECASE, "M": re.MULTILINE,
                        "S": re.DOTALL, "X": re.VERBOSE,
                        "A": re.ASCII,
                    }
                    for fl in declared_flags:
                        name = str(fl).upper()
                        re_flags |= compact_flags.get(name, getattr(re, name, 0))
                    if not sc.get("case_sensitive", True):
                        re_flags |= re.IGNORECASE

                    text = scoring_response.strip()
                    # A full outer fence is one presentation wrapper.  In all
                    # other cases, marker/box/final-line candidates compete by
                    # their payload positions; fixed marker priority would get
                    # ``Answer: old\nnew`` and ``boxed{old}\nAnswer:new`` wrong.
                    outer_fence = re.fullmatch(
                        r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n"
                        r"([\s\S]*?)\r?\n```",
                        text,
                    )
                    explicit_trailing_conflict = False
                    if outer_fence:
                        candidates = [outer_fence.group(1).strip()]
                    else:
                        positioned = []
                        explicit_positions = []
                        answer_markers = _answer_marker_candidates(text)
                        correction_markers = _correction_candidates(text)
                        strong_markers = answer_markers + correction_markers
                        expects_multiline = (
                            r"\n" in sc["pattern"]
                            or "\n" in sc["pattern"]
                            or r"[\s\S]" in sc["pattern"]
                        )
                        # For an explicitly multiline regex, everything after
                        # the latest marker is the declared structured answer.
                        # Otherwise physical answer candidates use the normal
                        # latest-position precedence.
                        if answer_markers and expects_multiline:
                            position, _ = answer_markers[-1]
                            declared_section = re.split(
                                r"\n[ \t]*\n", text[position:], maxsplit=1,
                            )[0]
                            explicit_positions.append(position)
                            positioned.append((
                                position,
                                _strip_answer_candidate_wrappers(
                                    declared_section,
                                ),
                            ))
                        for position, payload in (
                                [] if expects_multiline else answer_markers):
                            explicit_positions.append(position)
                            positioned.append((
                                position,
                                _strip_answer_candidate_wrappers(payload),
                            ))
                        for position, payload in correction_markers:
                            explicit_positions.append(position)
                            positioned.append((
                                position,
                                _strip_answer_candidate_wrappers(payload),
                            ))
                        tagged_answers = _authoritative_tagged_answer_candidates(
                            text, strong_markers,
                        )
                        if _tagged_answers_conflict(tagged_answers):
                            return ScoreResult(
                                False, "ambiguous_tagged_answers",
                                note="multiple distinct paired answer-tag payloads",
                            )
                        for position, payload in tagged_answers:
                            explicit_positions.append(position)
                            positioned.append((
                                position,
                                ("\0NEGATED_TAGGED_ANSWER\0"
                                 if _tagged_answer_is_negated(text, position)
                                 else _strip_answer_candidate_wrappers(payload)),
                            ))
                        latest_strong = max(
                            strong_markers, key=lambda item: item[0],
                            default=None,
                        )
                        strong_payload_is_opener = bool(
                            latest_strong
                            and _answer_payload_is_presentation_opener(
                                latest_strong[1],
                            )
                        )
                        strong_line_end = -1
                        if latest_strong:
                            strong_line_end = text.find("\n", latest_strong[0])
                            if strong_line_end < 0:
                                strong_line_end = len(text)
                        boxed_args = _balanced_latex_arguments(text)
                        for box_start, box_end, arg_start, _, content in boxed_args:
                            terminal_box = _is_terminal_latex_box(text, box_end)
                            if (latest_strong
                                    and not strong_payload_is_opener
                                    and not terminal_box
                                    and not (latest_strong[0] <= arg_start
                                             <= strong_line_end)):
                                continue
                            if _position_has_nonanswer_block_label(
                                    text, box_start):
                                continue
                            explicit_positions.append(arg_start)
                            positioned.append((
                                arg_start,
                                _strip_answer_candidate_wrappers(content),
                            ))
                        for fence_start, _, _, content in _fenced_block_candidates(text):
                            if (latest_strong
                                    and not strong_payload_is_opener
                                    and not (latest_strong[0] <= fence_start
                                             <= strong_line_end)):
                                continue
                            if _position_has_nonanswer_block_label(
                                    text, fence_start):
                                continue
                            explicit_positions.append(fence_start)
                            positioned.append((
                                fence_start,
                                _strip_answer_candidate_wrappers(content),
                            ))
                        # Inline/display math, code, and emphasis are
                        # presentation scaffolding commonly used by 2.5
                        # responses.  When an answer marker exists, only spans
                        # after that marker compete; derivation examples above
                        # it cannot override the declared final section.  The
                        # selected payload must still satisfy the original
                        # anchored regex or a proven-safe equivalence.
                        presentation_floor = (
                            latest_strong[0] if latest_strong else -1
                        )
                        presentation_wrappers = () if (
                            boxed_args
                            or (latest_strong and not strong_payload_is_opener)
                        ) else (
                            re.compile(r"\\\(([\s\S]*?)\\\)"),
                            re.compile(r"\\\[([\s\S]*?)\\\]"),
                            re.compile(r"\$\$([\s\S]*?)\$\$"),
                            re.compile(
                                r"(?<!\$)\$(?!\$)([^$\n]+)(?<!\$)\$(?!\$)"
                            ),
                            re.compile(r"(?<!`)`([^`\n]+)`(?!`)"),
                            re.compile(r"\*\*([^*\n]+)\*\*"),
                            re.compile(r"__([^_\n]+)__"),
                        )
                        for wrapper in presentation_wrappers:
                            for wrapped in wrapper.finditer(text):
                                if wrapped.start(1) < presentation_floor:
                                    continue
                                if _position_has_nonanswer_label(
                                        text, wrapped.start(1)):
                                    continue
                                explicit_positions.append(wrapped.end())
                                positioned.append((
                                    wrapped.end(),
                                    _strip_answer_candidate_wrappers(
                                        wrapped.group(1),
                                    ),
                                ))
                        raw_lines = _nonempty_lines_with_offsets(text)
                        if raw_lines and not (answer_markers and expects_multiline):
                            line_start, _, final_line = raw_lines[-1]
                            final_clean = _strip_answer_candidate_wrappers(final_line)
                            later_than_explicit = bool(
                                explicit_positions
                                and line_start > max(explicit_positions)
                            )
                            final_without_terminal = (
                                _strip_answer_candidate_wrappers(
                                    final_line,
                                    strip_terminal_punctuation=True,
                                )
                            )
                            bare_matches_original = bool(
                                re.fullmatch(
                                    sc["pattern"], final_clean, re_flags,
                                )
                                or re.fullmatch(
                                    sc["pattern"], final_without_terminal,
                                    re_flags,
                                )
                            )
                            boxed_tail_opt_in = False
                            if (later_than_explicit
                                    and sc.get(
                                        "allow_boxed_final_descriptive_math_tail"
                                    ) is True
                                    and boxed_args):
                                box_start, box_end, box_arg_start, _, _ = (
                                    boxed_args[-1]
                                )
                                boxed_tail_opt_in = bool(
                                    max(explicit_positions, default=-1)
                                    == box_arg_start
                                    and _final_answer_heading_directly_introduces_box(
                                        text, box_start,
                                    )
                                    and _boxed_final_has_only_one_trailing_sentence(
                                        text, box_end, line_start,
                                    )
                                )
                            narrative_tail = (
                                later_than_explicit
                                and _narrative_tail_after_explicit_answer(
                                    final_line,
                                    allow_descriptive_math_context=(
                                        boxed_tail_opt_in
                                    ),
                                )
                            )
                            # A worked derivation may end with the same answer
                            # in a displayed equation.  Treat that as
                            # consistent narration, but only after the inner
                            # math payload (or its final RHS) independently
                            # satisfies the original regex/GT.  A different
                            # trailing equation remains an authoritative
                            # conflict.
                            if later_than_explicit and not narrative_tail:
                                consistent_math_tail = False
                                for wrapper in (
                                    re.compile(r"\\\(([\s\S]*?)\\\)"),
                                    re.compile(r"\\\[([\s\S]*?)\\\]"),
                                    re.compile(r"(?<!\$)\$([^$]+)\$(?!\$)"),
                                    re.compile(r"\*\*([^*\n]+)\*\*"),
                                    re.compile(r"__([^_\n]+)__"),
                                    re.compile(r"(?<!`)`([^`\n]+)`(?!`)"),
                                ):
                                    for wrapped in wrapper.finditer(final_line):
                                        for variant in _regex_semantic_candidates(
                                                wrapped.group(1)):
                                            cleaned = _strip_answer_candidate_wrappers(
                                                variant,
                                                strip_terminal_punctuation=True,
                                            )
                                            if re.fullmatch(
                                                    sc["pattern"], cleaned,
                                                    re_flags):
                                                consistent_math_tail = True
                                                break
                                            expected_values = sc.get("expected", gt)
                                            if isinstance(expected_values, str):
                                                expected_values = [expected_values]
                                            for value in (expected_values or []):
                                                if not isinstance(value, str):
                                                    continue
                                                equivalence = (
                                                    _safe_noninstr_equivalence(
                                                        cleaned, value,
                                                    )
                                                )
                                                if (equivalence
                                                        and not (
                                                            equivalence
                                                            == "symbolic_equivalence"
                                                            and sc.get(
                                                                "allow_symbolic_equivalence",
                                                                True,
                                                            ) is False
                                                        )):
                                                    consistent_math_tail = True
                                                    break
                                            if consistent_math_tail:
                                                break
                                        if consistent_math_tail:
                                            break
                                    if consistent_math_tail:
                                        break
                                narrative_tail = consistent_math_tail
                            presentation_closer = bool(re.fullmatch(
                                r"(?:\\[\])]|\$+|`{3,}|~{3,}|"
                                r"</(?:answer|final_answer|final|result|response|output)>)",
                                final_line.strip(),
                                re.IGNORECASE,
                            ))
                            complete_tagged_line = bool(re.fullmatch(
                                r"<(?P<tag>answer|final_answer|final|result|response|output)"
                                r"[ \t]*>[\s\S]*?</(?P=tag)[ \t]*>",
                                final_line.strip(), re.IGNORECASE,
                            ))
                            if (later_than_explicit and not presentation_closer
                                    and not narrative_tail
                                    and not bare_matches_original):
                                explicit_trailing_conflict = True
                            if (not presentation_closer and not narrative_tail
                                    and not complete_tagged_line):
                                positioned.append((line_start, final_clean))
                        candidates = [
                            max(positioned, key=lambda item: item[0])[1]
                        ] if positioned else []

                        # A single math wrapper on the selected final answer is
                        # another harmless presentation layer.
                        if candidates:
                            selected = candidates[0]
                            for wrapper in (
                                r"\\\((.+)\\\)", r"\\\[(.+)\\\]",
                                r"\$+([^$]+)\$+",
                            ):
                                wrapped = re.fullmatch(wrapper, selected)
                                if wrapped:
                                    candidates.append(wrapped.group(1).strip())
                    if explicit_trailing_conflict:
                        return ScoreResult(
                            False, "conflicting_trailing_answer",
                            note="text after explicit answer is not harmless narration",
                        )
                    # Unicode superscripts → ASCII (so '²/6' matches pattern '2/6')
                    SUPER_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
                    # LaTeX command canonicalization: \sqrt{X}→sqrt(X), etc.
                    def _latex_canonicalize(s):
                        s = re.sub(r"\\(?:left|right)\b", "", s)
                        s = re.sub(r"\\(?:\(|\)|\[|\])", "", s)
                        s = s.replace(r"\$", "$")
                        s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)
                        s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
                        s = re.sub(r"\^[ \t]*\{([^{}]+)\}", r"^\1", s)
                        s = re.sub(r"\\sqrt\s*\{([^}]+)\}", r"sqrt(\1)", s)
                        fraction_replacement = (
                            r"\1/(\2)"
                            if sc.get("latex_fraction_preserve_grouping")
                            else r"\1/\2"
                        )
                        s = re.sub(
                            r"\\d?frac\s*\{([^}]+)\}\s*\{([^}]+)\}",
                            fraction_replacement, s,
                        )
                        s = re.sub(r"\\(?:cdot|times)\b", "*", s)
                        s = re.sub(r"\\pi\b", "pi", s)
                        s = re.sub(r"\\[ \t]+", " ", s)
                        s = re.sub(r"\\(cot|tan|sin|cos|ln|log)\b", r"\1", s)
                        s = re.sub(
                            r"\b(cot|tan|sin|cos|ln|log)[ \t]+([A-Za-z0-9]+)\b",
                            r"\1(\2)", s,
                        )
                        s = re.sub(r"\\[,;!]", " ", s)
                        return s
                    semantic_candidates = []
                    for candidate in candidates:
                        for variant in _regex_semantic_candidates(candidate):
                            if variant not in semantic_candidates:
                                semantic_candidates.append(variant)
                    expanded = []
                    for c in semantic_candidates:
                        for variant in (c, c.translate(SUPER_MAP), _latex_canonicalize(c)):
                            if variant not in expanded:
                                expanded.append(variant)
                    for cand in expanded:
                        cand_clean = _strip_markdown_wrappers(
                            cand, strip_terminal_punctuation=False,
                        )
                        cand_without_terminal = _strip_markdown_wrappers(
                            cand, strip_terminal_punctuation=True,
                        )
                        for c in (cand, cand_clean, cand_without_terminal):
                            if re.search(sc["pattern"], c, re_flags):
                                return ScoreResult(
                                    True, "regex_after_answer_extract",
                                    extracted=c[:200],
                                    note="matched against extracted answer position",
                                )
                        expected_values = sc.get("expected", gt)
                        if isinstance(expected_values, str):
                            expected_values = [expected_values]
                        if isinstance(expected_values, list):
                            for expected_value in expected_values:
                                if not isinstance(expected_value, str):
                                    continue
                                equivalence = _safe_noninstr_equivalence(
                                    cand_without_terminal, expected_value,
                                )
                                if (equivalence == "symbolic_equivalence"
                                        and sc.get("allow_symbolic_equivalence", True)
                                        is False):
                                    equivalence = None
                                equivalence_guard = sc.get(
                                    "symbolic_equivalence_guard_pattern"
                                )
                                if (equivalence == "symbolic_equivalence"
                                        and equivalence_guard
                                        and not re.search(
                                            equivalence_guard,
                                            cand_without_terminal,
                                            re_flags,
                                        )):
                                    equivalence = None
                                if (probe.get("category") == "code"
                                        and equivalence not in {
                                            None, "symbolic_equivalence",
                                        }):
                                    # Identifiers and string literals in code
                                    # remain case-sensitive; prose-level
                                    # typography folding must not bypass that.
                                    equivalence = None
                                if equivalence:
                                    return ScoreResult(
                                        True, f"regex_{equivalence}",
                                        extracted=cand_without_terminal[:200],
                                        expected=expected_value,
                                    )
                    # For semantic capability categories, an anchored legacy
                    # regex often encodes only a bare reference answer (for
                    # example ``^Berlin$``).  Reuse the ambiguity/negation-
                    # aware string extractor only when that original regex
                    # independently accepts the declared expected value.  The
                    # fallback is never enabled for code/tool/agent/output or
                    # instruction-format contracts.
                    semantic_categories = {
                        "arith", "causal", "dialog", "know", "lang", "logic",
                        "longctx", "math", "meta", "plan", "read", "truth",
                        "world",
                    }
                    expected_value = sc.get("expected", gt)
                    if (probe.get("category") in semantic_categories
                            and isinstance(expected_value, str)
                            and 0 < len(expected_value) <= 256
                            and not re.search(r"[/\\]", expected_value)
                            and not sc.get("symbolic_equivalence_guard_pattern")
                            and sc.get("allow_symbolic_equivalence", True)
                            is not False
                            and re.fullmatch(sc["pattern"], expected_value, re_flags)):
                        semantic_result = extract_string(
                            scoring_response, expected_value,
                            case_sensitive=not bool(re_flags & re.IGNORECASE),
                            allow_substring=True,
                            strip_whitespace=True,
                            allow_terminal_punctuation=True,
                            allow_semantic_equivalence=True,
                        )
                        if semantic_result.passed:
                            return ScoreResult(
                                True,
                                f"regex_semantic_{semantic_result.strategy}",
                                extracted=semantic_result.extracted,
                                expected=expected_value,
                                note="declared reference extracted before applying legacy bare-answer contract",
                            )
                if authoritative_override:
                    return ScoreResult(
                        False, "explicit_answer_mismatch",
                        note="later explicit answer/correction did not match regex",
                    )
                return result
        except _RegexTimeout:
            return ScoreResult(False, "regex_timeout")

    if t == "format_check":
        # Scorer schema: {must_satisfy: [{name, regex}]}
        # In v2: lenient by default; strict (fullmatch) only when category=instr.
        must_satisfy = sc.get("must_satisfy")
        if must_satisfy:
            failed = []
            from eval.scorer import _sanitize, MAX_RESP_BYTES
            r = _sanitize(response)[:MAX_RESP_BYTES]
            try:
                with _MainThreadAlarm(seconds=3):
                    for chk in must_satisfy:
                        pat = chk.get("regex", ".*")
                        # A must_satisfy regex controls its own scope. Adding
                        # MULTILINE/DOTALL here lets anchored ^...$ patterns
                        # match one valid middle line while ignoring extra
                        # output before or after it.
                        check_flags = 0
                        declared_flags = chk.get("flags") or []
                        if isinstance(declared_flags, str):
                            declared_flags = [declared_flags]
                        compact_flags = {
                            "I": re.IGNORECASE, "M": re.MULTILINE,
                            "S": re.DOTALL, "X": re.VERBOSE,
                            "A": re.ASCII,
                        }
                        for flag in declared_flags:
                            # Accept compact flags ("i", "m", "s") and the
                            # standard Python names ("IGNORECASE", etc.).
                            name = str(flag).upper()
                            check_flags |= compact_flags.get(name, getattr(re, name, 0))
                        if not re.search(pat, r, check_flags):
                            failed.append(chk.get("name", pat[:30]))
            except _RegexTimeout:
                return ScoreResult(False, "regex_timeout",
                                   note="regex timeout in format_check; possible catastrophic backtrack")
            if not failed:
                return ScoreResult(True, "all_must_satisfy_passed")
            return ScoreResult(False, "format_violation",
                               note=f"failed_checks={failed}")
        # Fall-through: legacy 'pattern' field
        pat = sc.get("pattern", ".*")
        try:
            with _MainThreadAlarm(seconds=3):
                if probe.get("category") == "instr":
                    return format_strict_regex(response, pattern=pat,
                                               flags=sc.get("flags"))
                return format_lenient_regex(response, pattern=pat,
                                            flags=sc.get("flags"),
                                            allow_stem_fallback=sc.get("allow_stem_fallback", False),
                                            case_sensitive=sc.get("case_sensitive", True))
        except _RegexTimeout:
            return ScoreResult(False, "regex_timeout")

    if t == "substring_any":
        return semantic_keywords_any(response, alternatives=sc.get("substrings", []),
                                     case_sensitive=sc.get("case_sensitive", False),
                                     require_word_boundary=True,
                                     disallowed_patterns=sc.get("disallowed_patterns"))

    if t == "substring_all":
        payload = response
        if not strict_if and probe.get("category") == "code":
            payload, payload_error = _code_payload_candidate(response)
            if payload_error:
                return ScoreResult(False, payload_error)
            if payload is None or _unresolved_code_refusal(payload):
                return ScoreResult(False, "code_refusal_without_answer")
        result = semantic_keywords_all(
            payload, required=sc.get("substrings", []),
            case_sensitive=sc.get("case_sensitive", False),
            require_word_boundary=True, synonyms=synonyms,
        )
        if strict_if or probe.get("category") != "code":
            return result
        reference = probe.get("ground_truth")
        if isinstance(reference, str):
            ast_result = python_ast_contains_reference(payload, reference)
            if ast_result is not None and ast_result.passed:
                return ast_result
            if (result.passed and ast_result is not None
                    and ast_result.strategy == "python_ast_candidate_parse_failed"):
                return ast_result
        if result.passed:
            return result
        return result

    if t == "structured_json":
        return json_schema_lenient(response, sc.get("schema", {}),
                                   repair_tool_shape=bool(
                                       sc.get("repair_tool_shape", False)
                                   ))

    if t == "strict_json":
        return json_schema_strict(
            response, sc.get("schema", {}), allow_outer_fence=not strict_if,
        )

    if t == "code_unittest":
        return code_unittest(response, tests=sc.get("tests", []),
                             timeout=int(sc.get("timeout", 10)),
                             mem_mb=int(sc.get("memory_mb", 256)),
                             source_checks=sc.get("source_checks"))

    if t == "code_parse_ast":
        # Deterministic scorers run on the evaluator's main thread.  Keep a
        # final wall-clock guard around every static parser so an adversarial
        # or merely truncated model completion cannot stall the whole suite.
        try:
            with _MainThreadAlarm(seconds=5):
                return code_parse_ast(response, sc)
        except _RegexTimeout:
            return ScoreResult(
                False,
                "ast_timeout",
                note="static parser exceeded 5s safety limit",
            )

    if t == "numeric_range":
        # Accept both 'min'/'max' and 'low'/'high'.
        low = sc.get("low", sc.get("min"))
        high = sc.get("high", sc.get("max"))
        if low is None or high is None:
            return ScoreResult(False, "missing_range_bounds",
                               note=f"scorer={sc}")
        return numeric_range(response, low=float(low), high=float(high),
                             inclusive=sc.get("inclusive", True),
                             integer_only=sc.get("integer_only", False),
                             single_value_only=(strict_if and sc.get("single_integer_only", False)))

    if t == "fraction_match":
        return fraction_match(response,
                              expected=str(sc.get("expected", gt)),
                              tolerance=float(sc.get("tolerance", 1e-9)))

    if t == "refusal_check":
        return abstention_check(response, phrases=sc.get("phrases"))

    if t == "judge_rubric":
        return judge_deferred()

    return ScoreResult(False, "unknown_scorer_type", note=f"scorer.type={t!r}")


# ----------------------------------------------------------------------------
# Intent-driven routing
# ----------------------------------------------------------------------------
def by_intent(intent: str, probe: dict, response: str, policy: dict) -> ScoreResult:
    sc = probe.get("scorer", {})
    gt = probe.get("ground_truth")
    synonyms = (policy or {}).get("synonyms", {})
    strict_if = _strict_instruction_following(probe)

    if intent == "extract_numeric":
        # A broad arithmetic/math policy must not erase a probe's declared
        # structural contract. Coerce only actual numeric scorers (or probes
        # that explicitly opt in); regex/string/fraction/range scorers retain
        # their own routing semantics.
        if (sc.get("type") != "exact_match_numeric"
                and not sc.get("coerce_numeric_by_intent", False)):
            return default_route(probe, response, policy)
        exp = sc.get("expected", gt)
        try: float(exp)
        except (TypeError, ValueError):
            # fall back to default routing
            return default_route(probe, response, policy)
        return extract_numeric(
            response, exp, tolerance=float(sc.get("tolerance", 0)),
            require_only_numeric=(
                strict_if and (
                    sc.get("require_only_numeric_output", False)
                    or sc.get("accepted_output_format") == "numeric_only"
                )
            ),
        )

    if intent == "exact_integer":
        if strict_if:
            return exact_integer_output(response, sc.get("expected", gt))
        return extract_integer(response, sc.get("expected", gt))

    if intent == "strict_output":
        if not strict_if:
            return default_route(probe, response, policy)
        if sc.get("type") == "regex" and sc.get("pattern"):
            return format_strict_regex(
                response, pattern=sc["pattern"], flags=sc.get("flags"),
                fullmatch=True,
            )
        return exact_whole_response(
            response, gt,
            case_sensitive=sc.get("case_sensitive", True),
            strip_outer_whitespace=sc.get("strip_whitespace", True),
        )

    if intent == "extract_string":
        return default_route(probe, response, policy)

    if intent == "extract_string_loose":
        return default_route(probe, response, policy)

    if intent == "mcq_letter":
        return extract_mcq_letter(
            response, sc.get("expected", gt), require_bare=strict_if,
        )

    if intent == "set_match":
        items = sc.get("items") or sc.get("expected") or gt
        return extract_set(
            response, items, separator_hint=sc.get("delimiter", ","),
            case_sensitive=sc.get("case_sensitive", False),
            reject_duplicates=(strict_if and sc.get(
                "reject_duplicates", sc.get("reject_duplicate_items", False),
            )),
            strict_separator=(strict_if and bool(sc.get("delimiter"))),
        )

    if intent == "sequence_match":
        items = sc.get("items") or sc.get("expected") or gt
        return extract_ordered_sequence(
            response, items,
            separator_hint=sc.get("delimiter", ","),
            numeric=True,
            strict_separator=(strict_if and bool(sc.get("delimiter"))),
            case_sensitive=sc.get("case_sensitive", False),
        )

    if intent == "abstention":
        # Use phrases from scorer if provided; else default rich vocab
        phrases = sc.get("substrings") or sc.get("phrases")
        return abstention_check(response, phrases=phrases)

    if intent == "format_strict":
        if not strict_if:
            return default_route(probe, response, policy)
        # If the probe declares must_satisfy (format_check schema), defer
        # to default_route which handles that schema. We must NOT silently
        # fall through to extract_string against a placeholder ground_truth
        # like 'format_only' (R1 regression).
        if sc.get("must_satisfy"):
            return default_route(probe, response, policy)
        pat = sc.get("pattern", ".*")
        result = format_strict_regex(response, pattern=pat, flags=sc.get("flags"))
        # If the probe didn't declare a real pattern, fall back to
        # exact_match_string against the ground_truth (still strict by intent),
        # but ONLY if ground_truth looks like a real expected string — not a
        # sentinel like 'format_only' / 'judge_rubric' / '<...>' / placeholder.
        if result.strategy == "no_pattern_specified":
            gt = probe.get("ground_truth")
            sentinel = isinstance(gt, str) and (
                gt in ("format_only", "judge_rubric", "refusal",
                       "deferred", "no_record_exists")
                or (gt.startswith("<") and gt.endswith(">"))
            )
            if gt and not sentinel:
                return exact_whole_response(
                    response, str(gt),
                    case_sensitive=sc.get("case_sensitive", False),
                    strip_outer_whitespace=sc.get("strip_whitespace", True),
                )
            # No usable spec — conservatively defer to default routing
            return default_route(probe, response, policy)
        return result

    if intent == "format_lenient":
        # If probe has no real pattern (e.g. exact_match_string scorer routed
        # here by atom policy), defer to default_route to avoid trivial '.*'
        # match.
        if not strict_if:
            return default_route(probe, response, policy)
        if (not sc.get("pattern")
                or sc.get("type") in ("exact_match_string", "exact_match_numeric")):
            return default_route(probe, response, policy)
        return format_strict_regex(
            response, pattern=sc.get("pattern", ".*"),
            flags=sc.get("flags"), fullmatch=True,
        )

    if intent == "keywords_any":
        return semantic_keywords_any(response, alternatives=sc.get("substrings", []),
                                     case_sensitive=sc.get("case_sensitive", False),
                                     require_word_boundary=True,
                                     disallowed_patterns=sc.get("disallowed_patterns"))

    if intent == "keywords_all":
        if not sc.get("substrings"):
            return default_route(probe, response, policy)
        return semantic_keywords_all(response, required=sc.get("substrings", []),
                                     case_sensitive=sc.get("case_sensitive", False),
                                     require_word_boundary=True,
                                     synonyms=synonyms)

    if intent == "keywords_all_smart":
        if not sc.get("substrings"):
            return default_route(probe, response, policy)
        # "all" is a hard contract. Synonyms may satisfy an individual
        # requirement, but deleting any required semantic element must fail.
        return semantic_keywords_all(
            response, required=sc.get("substrings", []),
            case_sensitive=sc.get("case_sensitive", False),
            require_word_boundary=True, synonyms=synonyms,
        )

    if intent == "code_structural":
        # Need scorer config — read from probe.scorer.structural if present
        sc_struct = sc.get("structural", {})
        if not sc_struct:
            return default_route(probe, response, policy)
        return code_structural(
            response,
            must_define=sc_struct.get("must_define"),
            must_call_any=sc_struct.get("must_call_any"),
            must_use_any=sc_struct.get("must_use_any"),
            allow_synonyms=synonyms,
        )

    if intent == "code_unittest":
        return code_unittest(response, tests=sc.get("tests", []),
                             timeout=int(sc.get("timeout", 10)),
                             mem_mb=int(sc.get("memory_mb", 256)),
                             source_checks=sc.get("source_checks"))

    if intent == "json_schema":
        return json_schema_lenient(response, sc.get("schema", {}),
                                   repair_tool_shape=bool(
                                       sc.get("repair_tool_shape", False)
                                   ))

    if intent == "strict_json":
        return json_schema_strict(
            response, sc.get("schema", {}), allow_outer_fence=not strict_if,
        )

    if intent == "numeric_range":
        low = sc.get("low", sc.get("min"))
        high = sc.get("high", sc.get("max"))
        if low is None or high is None:
            return default_route(probe, response, policy)
        return numeric_range(response, low=float(low), high=float(high),
                             inclusive=sc.get("inclusive", True),
                             integer_only=sc.get("integer_only", False),
                             single_value_only=(strict_if and sc.get("single_integer_only", False)))

    if intent == "fraction_match":
        return fraction_match(response,
                              expected=str(sc.get("expected", gt)),
                              tolerance=float(sc.get("tolerance", 1e-9)))

    if intent == "judge_deferred":
        return judge_deferred()

    if intent == "multimodal_defer":
        # For multimodal probes, try the underlying scorer but mark deferred
        result = default_route(probe, response, policy)
        result.deferred = True
        result.note = (result.note or "") + " | multimodal: prompt contains [ATTACHED_*] placeholder"
        return result

    # Fallback
    return default_route(probe, response, policy)


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def score_probe(probe: dict, response: str, policy: dict) -> tuple[ScoreResult, int]:
    """Returns (result, denom_count).
    denom_count = 1 if this probe counts toward overall pass-rate denominator,
    0 if excluded (judge stub, multimodal placeholder, etc.).
    """
    # Disabled probes are stable dataset exclusions (for example, an absent
    # media asset). Return before scorer dispatch so they can never trigger an
    # LLM judge call or accidentally appear as a candidate failure.
    if probe.get("enabled") is False:
        return (
            ScoreResult(
                False,
                "probe_disabled",
                deferred=True,
                note="excluded: probe explicitly disabled",
            ),
            0,
        )

    # 1. Check exclusion
    excluded, reason = is_excluded(probe, policy)

    # Static parser/AST scoring is an explicit per-probe contract and must not
    # be weakened by a broad atom intent such as code_structural.
    probe_scorer = probe.get("scorer", {})
    explicit_static = probe_scorer.get("type") == "code_parse_ast"
    explicit_judge = probe_scorer.get("type") == "judge_rubric"
    # A small number of probes intentionally specialize a broad canonical
    # policy (for example, one exact empty-set answer inside an abstention
    # family).  This opt-in flag keeps the exception local to that probe.
    explicit_default_route = probe_scorer.get("override_atom_intent") is True

    # Resolve before transport normalization because a legacy ``regex`` probe
    # may be routed by policy to an executing/structured scorer.
    intent = resolve_intent(probe, policy)

    # Only scalar/text non-instruction scorers receive transport cleanup.
    # ``instr`` preserves the exact model bytes/line endings, while LLM judge
    # probes retain the original response for semantic adjudication.
    scoring_response = response
    scalar_transport_types = {
        "exact_match_numeric", "exact_match_string", "exact_integer",
        "mcq_letter", "set_equal", "sequence_equal", "numeric_range",
        "fraction_match",
    }
    atom = str(probe.get("canonical_id", ""))
    exact_artifact_contract = (
        atom.startswith((
            "output.artifact_", "output.format_", "output.no_extra_text",
            "tool.format_",
        ))
        or str((probe.get("axis_values") or {}).get("format_strictness", ""))
        in {"byte_exact", "exact_single_artifact"}
    )
    ordinary_regex = (
        probe_scorer.get("type") == "regex"
        and str(probe.get("category", "")).lower() != "code"
        and intent not in {"code_unittest", "code_structural", "json_shape"}
    )
    if (not explicit_judge and not _strict_instruction_following(probe)
            and not exact_artifact_contract
            and (probe_scorer.get("type") in scalar_transport_types
                 or ordinary_regex)):
        scoring_response = normalize_noninstr_transport(
            response, repair_unclosed_outer_fence=False,
        )

    # 3. Score (defensive — never crash the runner)
    try:
        if explicit_judge:
            result = llm_judge(probe, response)
        elif explicit_static or explicit_default_route:
            result = default_route(probe, scoring_response, policy)
        elif intent:
            result = by_intent(intent, probe, scoring_response, policy)
        else:
            result = default_route(probe, scoring_response, policy)
    except Exception as e:
        result = ScoreResult(False, "scorer_error",
                             note=f"{type(e).__name__}: {e}")

    # 4. Apply exclusion to denominator
    denom = 0 if (excluded or result.deferred) else 1
    if excluded and not result.deferred:
        result.deferred = True
        result.note = (result.note or "") + f" | excluded: {reason}"

    return (result, denom)
