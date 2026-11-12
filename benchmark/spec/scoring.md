# ACE 2.6.4 Scoring Specification

## 1. Scope and dataset invariants

- The benchmark release is `2.6.4`, as recorded in [`VERSION`](../VERSION).
- The scorer version is `2.6.4`; the evaluation contract version is
  `ace-eval-contract-v2.6.4-public`; and the required judge prompt version is
  `ace-binary-judge-v1-deterministic-observations`.
- The canonical judge response budget is **4,096 tokens** and is included in
  the evaluation signature.

- **2,565 probes** across **187 canonical atoms** and **19 categories**.
- **348** probes declare `scorer.type = judge_rubric`.
- **76** probes declare `enabled: false`. Sixty lack required binary media,
  15 are excluded security probes, and one is an excluded external quotation.
  The runner skips these probes before making candidate or judge requests.
- The evaluator loads the materialized dataset from `probes/*.jsonl`.

Recalculate these counts from the JSONL files whenever the probe set changes.

## 2. Evaluation flow

For each selected probe, the runner applies the following order:

1. If `enabled` is false, emit a deferred `probe_disabled` record with
   denominator weight 0. Do not call the candidate or judge endpoint.
2. Call the candidate endpoint and retain final content, reasoning content,
   finish reason, latency, and token usage.
3. If candidate transport fails after retries, emit `model_call_error` with
   denominator weight 0. Do not turn provider failure into a model failure.
4. For `judge_rubric`, submit the candidate response and canonical rubric to
   the configured `deepseek-v4-flash` judge. For all other enabled probes, resolve the scoring
   intent and run the deterministic scorer. Scoring uses the complete returned
   content; the persistence caps described below are applied only afterward.
5. Convert the scorer result to `(ScoreResult, denominator_weight)`, persist it
   immediately, and aggregate only records whose weight is 1.

Explicit `code_parse_ast` and `judge_rubric` scorer contracts take precedence
over broad atom-level intents. Otherwise, `policy/atom_intents.yaml` may
override routing; probes with no matching intent use their declared
`scorer.type` through the default route.

## 3. Result and denominator contract

Every per-probe record includes at least:

| Field | Meaning |
|---|---|
| `passed` | Boolean scorer decision; meaningful for effective rows |
| `score_strategy` | The extraction/check/judge path used |
| `score_extracted` | Auditable extracted value or judge metadata, including per-criterion results |
| `score_note` | Failure, deferral, or decision explanation |
| `deferred` | The row could not produce a benchmark verdict |
| `excluded_from_score` | The row has denominator weight 0 |

Every newly produced per-probe record also identifies the evaluation release
and contract:

| Field | Meaning |
|---|---|
| `benchmark_version` | Authoritative ACE release from `VERSION` |
| `scorer_version` | Deterministic scorer semantics release |
| `evaluation_contract_version` | Result/denominator/protocol contract |
| `judge_prompt_version` | LLM-judge protocol version, even for deterministic rows |
| `evaluation_contract_sha256` | Digest of version/data/policy/runtime contract identity |
| `evaluator_runtime_sha256` | Digest of the evaluator source bundle |
| `evaluation_signature` | Digest binding code/data/config identity for resume |

The summary repeats those fields under `meta` and retains the signature
payload plus probe-suite, selected-probe, and policy SHA-256 digests. The
contract digest covers benchmark/scorer/contract/prompt versions, those data
and policy digests, and evaluator runtime digest. The full
evaluation signature additionally covers candidate endpoint/model/generation
configuration, persisted response/reasoning caps, and judge configuration.
Result consumers use these fields to verify evaluation identity.

`--resp-cap` and `--reasoning-cap` truncate only the response/reasoning copies
stored in the detail JSONL. They do not truncate the candidate content before
deterministic or LLM-judge scoring. Regex scorer routes independently enforce
their own bounded input and timeout. Static parser artifacts reject inputs
longer than 200,000 characters, and the main-thread `code_parse_ast` route has
a five-second termination guard.

Aggregation uses:

```text
n_eff = count(rows where excluded_from_score is false)
n_pass = count(rows where excluded_from_score is false and passed is true)
overall_pass_rate = n_pass / n_eff
```

The following outcomes have denominator weight 0:

- one of the 76 explicitly disabled probes;
- a candidate transport/provider error;
- a judge `ERROR` verdict;
- a judge network, API, protocol, or model-identity failure;
- a `judge_rubric` probe when LLM judging is disabled or unconfigured;
- any probe explicitly excluded by policy.

Successful judge `PASS` and `FAIL` outcomes both have denominator weight 1.
`FAIL` therefore lowers the score; the judge is not merely called for logging.

A local deterministic scorer mismatch has denominator weight 1. An unexpected
local scorer exception is recorded as `scorer_error` rather than silently
passing.

Denominator exclusion and publication readiness are separate. Stable disabled
probe exclusions are part of the declared benchmark scope and do not
make a run partial. In contrast, any candidate/provider error,
`judge_deferred`, `llm_judge_error`, or `scorer_error` causes the summary to
contain:

```json
{
  "meta": {
    "score_status": "partial",
    "publishable": false,
    "n_transient_deferred": 1,
    "deferred_by_reason": {"judge_deferred": 1},
    "deferred_probe_ids": ["probe-id"]
  }
}
```

The exact reason counts and complete affected probe-ID list are retained even
when the caller disables strict process exits. Only a summary with
`meta.official_full_suite = true`, an empty `meta.official_scope_issues`,
`meta.score_status = complete`, and `meta.publishable = true` is publishable as
a formal ACE 2.6.4 score.

The full-suite gate compares the loaded/selected probe manifest and policy to
the canonical release. It requires the endpoint configured in `ACE_JUDGE_BASE_URL` with
`deepseek-v4-flash`, the 4,096-token judge budget, canonical persistence
caps, and `n_total=2565`, `n_eff=2489`, `n_excluded=76`. Filtered, limited,
skipped-category, modified-data/policy, or custom-judge runs remain valid
diagnostics, but are marked `partial` / `publishable:false` even when they have
no transient provider error.

## 4. Deterministic scorer semantics

### 4.1 Exact strings and whole-response output

Scoring first derives `strict_instruction_following` from the probe category:
it is true only when `category == "instr"`. `exact_match_string` then routes
to whole-response equality for `instr`, and to guarded answer extraction for
all other categories. Unless a probe explicitly changes the fields:

- outer transport whitespace is stripped;
- comparison is case-insensitive;
- non-`instr` extraction may use boundary-aware substring/phrase recovery;
- internal whitespace is not collapsed;
- non-`instr` extraction may discard one extra terminal punctuation mark
  only when punctuation is not itself part of the reference answer.

The non-`instr` answer positions include the whole/stripped response, a guarded
answer-like final line, a balanced `\boxed{...}`, and an explicit answer
marker. Quoted, emphasized, or boundary matches are accepted only when they are
unambiguous and affirmative. A later explicit/final answer overrides an
earlier mention; conflicting lists, negation, token-boundary violations,
punctuation-only containment, and sign changes such as `$1,200` inside
`-$1,200` are rejected. `accepted_aliases` remains explicit probe data, not a
global fuzzy-match rule.

For `instr`, `strip_whitespace: false` preserves leading/trailing whitespace
in whole-response equality. `strict_output` and format-strict intents enforce
whole-response presentation only for `instr`; outside that category they route
back to the probe's capability scorer. Functional validity requirements such
as JSON schema, parser, code, or AST checks remain binding independently of
this presentation policy.

### 4.2 Regex and format checks

Ordinary `regex` routing uses `re.search` and is case-sensitive by default.
No implicit `IGNORECASE`, `MULTILINE`, or `DOTALL` flags are added; a probe must
declare `i`, `m`, `s`, or their long names. Numeric/command probes may
explicitly re-test the declared regex against extracted answer-position
candidates. In non-`instr` categories, a full outer fence or a guarded latest
answer marker, balanced box, math/code/emphasis span, or final answer line may
be unwrapped and the same regex retried; the regex itself is not widened.

Stem fallback is off by default. A probe must set
`allow_stem_fallback: true`, and even then it is disabled for anchored,
multi-line structural, fenced, or compound-lookahead patterns.

`format_check.must_satisfy` requires every listed regex to match and honors
only that check's declared flags. For `instr`, `format_strict` and
`strict_output` use whole-response fullmatch semantics where the intent says
formatting itself is the capability. Outside `instr`, presentation-only
strict intents route back to the capability scorer.

Regex input is length-capped. The default regex and `format_check` dispatch
paths additionally apply the runtime timeout.

### 4.3 Numeric values, integers, ranges, and sequences

`exact_match_numeric` compares an expected numeric value at an answer
position. Strategies include a numeric-only stripped response, `\boxed{...}`,
the last number on the final meaningful line, the last non-negated number, and
an explicit answer marker. A correct number that appears only in unrelated
working is reported as buried and does not pass.

Occurrences explicitly negated by constructions such as `not 5`, `!= 5`, or
their supported equivalents are not accepted as the answer. A minus sign in a
negative number is not treated as linguistic negation.

When an `instr` probe declares `require_only_numeric_output: true` or
`accepted_output_format: numeric_only`, any prose/extra formatting fails. In
other categories those presentation flags do not suppress safe extraction of
the substantive answer.

For `instr`, `exact_integer` requires exactly one canonical base-10 integer
after outer transport trimming. In other categories it extracts a canonical
integer only from the whole answer, a balanced box, an explicit marker, or the
end of the final meaningful line. Both paths reject decimal and exponent
spellings and compare text without converting to a binary float.

`sequence_equal` preserves item order and multiplicity. Numeric sequence
elements use decimal comparison, so `7` and `7.0` may be numerically equivalent
without losing ordering or duplicate positions. `set_equal` ignores order.
Non-`instr` probes may unwrap an answer label/Markdown and accept common
separators; `instr` additionally enforces declared separator and duplicate
policies.

### 4.4 Multiple choice

For `instr`, `mcq_letter` accepts only one bare option letter. For other
categories it accepts only an unambiguous answer position:

1. a standalone response letter;
2. a pure letter line, including one before an explanation;
3. an explicit `Answer: X`/`Choice: X` marker;
4. a balanced `\boxed{X}` or `\boxed{\text{X}}`;
5. a supported affirmative `choose`/`select`/`pick` phrase; or
6. one explicitly selected `X) option text` line (not a copied option list).

Candidates are ordered by their position in the response, so a later answer or
negation overrides an earlier choice. There is no last-letter-anywhere
fallback. Text such as `Not B` or `I reject B` does not become a positive B
answer, and the article in `select a pairwise comparison` is not option A.

### 4.5 Keywords

`substring_any` and `substring_all` use edge-aware boundaries and optional,
declared synonym groups. `substring_all` requires every required concept; it
does not use a partial 2-of-3 threshold. Code-like punctuation edges such as
`.fit(X, y)` and `<h1>` are handled without allowing `correct` to match inside
`incorrect`.

### 4.6 JSON and tool calls

`structured_json` extracts one unambiguous JSON value from an otherwise
presented answer, then validates its schema. Multiple different unlabelled
top-level values are a conflict; a latest explicit `Answer`/`Output`/`Reply`
declaration is authoritative. Shape repair is off unless the probe explicitly
sets `repair_tool_shape: true`. On that opt-in path, documented
`name`/`tool`/`function` and `arguments`/`parameters` aliases may be renamed or
unwrapped, but unknown fields are preserved for `additionalProperties`
validation, conflicts are rejected, and array repair has a fixed work cap.

`strict_json` parses the stripped **entire response** with `json.loads` for
`instr`. In other categories it may remove exactly one full outer JSON code
fence first. It never accepts surrounding prose, multiple blocks, JSON repair,
or tool-call shape repair. The supported schema subset includes `type`,
`properties`, `required`, `const`, `enum`,
`additionalProperties`, `items`, `prefixItems`, `minItems`, and `maxItems`.

### 4.7 Code, parser, and AST scoring

`code_parse_ast` is static-only: it does not compile or execute the submitted
program. Current parser contracts cover:

- Python AST structure and selected declared semantics;
- regex compilation plus positive/negative behavior examples;
- JSON, TOML, and constrained YAML paths;
- constrained HCL blocks and paths;
- static TSX/JavaScript/TypeScript, SQL, shell, and Dockerfile structure;
- a constrained CSS rule parser;
- symbolic OOXML construction for supported spreadsheet, document, and slide
  tasks.

The OOXML path imports neither `openpyxl`, `python-docx`, nor `python-pptx`.
It interprets a bounded subset of Python AST into symbolic artifacts and
checks the declared save path, content, layout, and formatting properties.
It never executes candidate code.

`code_structural` records actual AST definitions, `ast.Call` targets, and
loaded identifier uses. Probes that declare `code_unittest` can execute model
output in a Python subprocess with timeout and resource limits. Run these
probes in an isolated environment as described in the
[repository README](../../README.md#run-the-full-benchmark).

No full TSX, SQL, Cython, or general-purpose compiler was introduced. The
TSX/SQL checks above validate only explicitly declared static subsets. When a
property cannot be validated soundly with a small parser/AST contract, the
probe retains explicit string/regex scoring. Such rows must not be described
as compiled behavioral tests.

## 5. LLM judge

### 5.1 Canonical rubric

Every current `judge_rubric` scorer is an object with exactly the following
outer rubric keys:

```json
{
  "type": "judge_rubric",
  "rubric": {
    "version": "1",
    "pass_criteria": ["one or more criteria"],
    "fail_criteria": ["one or more criteria"],
    "pass_rule": "all pass criteria and no fail criteria"
  }
}
```

`eval/validate_judge_rubrics.py` is the authoritative static schema check.
At request time, every stored pass criterion is normalized to ordered IDs
`PASS-1`, `PASS-2`, and so on, and every fail criterion to `FAIL-1`, `FAIL-2`,
and so on. Plain string criteria remain `text`; structured criteria remain a
compact JSON `spec`, without dropping or prose-inflating their fields.

### 5.2 Request and prompt protocol

The judge uses an OpenAI-compatible `/v1/chat/completions` endpoint and
requests model `deepseek-v4-flash` at temperature 0. The current prompt version
is `ace-binary-judge-v1-deterministic-observations`. The task prompt, optional
system prompt, real reference answer, canonical rubric, candidate response,
authoritative `evaluation_mode`, and bounded locally computed mechanical
observations are serialized into a JSON evaluation packet.
Opaque internal ground-truth sentinels are omitted from `reference_answer`.

The runtime selects one of two category-aware modes:

- `STRICT_INSTRUCTION` is used only for `instr`. Every stated semantic and
  presentation constraint binds, including exact/verbatim output, wrappers,
  headings, ordering, casing, whitespace, counts, and bans on extra text.
- `CAPABILITY_WITH_EXTRACTION` is used for every other category. The judge
  extracts and grades the substantive answer and does not fail solely for
  harmless explanation, Markdown wrappers, terminal punctuation, surrounding
  whitespace, or presentation-only length/count differences. Core factual and
  logical correctness still binds, as do safety, requested natural language,
  required semantic fields and arguments, meaning-changing order, and
  parseable functional representations for code, JSON/XML, tool calls, and
  other structured artifacts.

The mode relaxes presentation only; it never permits a contradictory second
answer, material ambiguity, missing required semantics, or broken functional
parsing.

Mechanical observations include exact aggregate token/line/paragraph counts,
bounded first/last paragraph details, outer-wrapper facts, and exact-case plus
case-folded whole-word counts for quoted task terms. They are authoritative
only when an observation's definition and scope exactly match the task. A
global count must not replace a body-only, per-item, heading-excluded, or
wrapper-excluded requirement. Detail arrays are capped so an adversarially
segmented candidate cannot inflate the Judge packet without bound.

An absent/null candidate response is normalized to the empty string. Other
non-text JSON-serializable candidates are deterministically serialized and
then judged as submitted answers. They do not become `ERROR` merely because
they were not originally strings. A value that cannot be losslessly
serialized fails local packet validation and is deferred as a judge input
error.

The judge system prompt establishes that every packet field is
untrusted data, not an instruction to the judge. It requires evaluation of all
mandatory pass criteria and rejection conditions and permits semantic
equivalence unless the task explicitly requires exact formatting.

The complete judge response must be one JSON object with exactly:

```json
{
  "verdict": "PASS | FAIL | ERROR",
  "confidence": 0.0,
  "reason": "evidence-based explanation",
  "criterion_results": [
    {"id": "PASS-1", "status": "SATISFIED", "evidence": "..."},
    {"id": "FAIL-1", "status": "NOT_TRIGGERED", "evidence": "..."}
  ],
  "failed_criteria": []
}
```

Duplicate keys, extra keys, fences/prefixes/suffixes, non-finite confidence,
or incomplete/inconsistent criterion coverage are protocol errors. For `PASS`
or `FAIL`, `criterion_results` must cover every supplied criterion exactly
once and in packet order. Pass-criterion status is `SATISFIED` or
`UNSATISFIED`; fail-criterion status is `NOT_TRIGGERED` or `TRIGGERED`; every
entry must include nonempty evidence. `failed_criteria` must exactly equal,
without duplicates and in canonical order, all `UNSATISFIED` pass IDs followed
by all `TRIGGERED` fail IDs. `PASS` requires that derived list to be empty and
`FAIL` requires it to be nonempty. `ERROR` requires both arrays to be empty.
The returned model name must equal the requested model.

The runtime records judge provider/model, requested model, response ID, prompt
version, retry attempt, token usage, confidence, reason, complete
`criterion_results`, and `failed_criteria`.

### 5.3 Judge error boundary

`ERROR` is reserved for an unusable task or rubric, not a weak candidate
answer. Empty, incomplete, wrong, or ambiguous candidate answers are still
scoreable and should receive `FAIL` when requirements are unmet.

Transient HTTP statuses and transport/protocol failures are retried. If all
attempts fail, or the judge returns `ERROR`, the row is deferred with
denominator weight 0. With default `--strict-judge`, any such row makes the
overall command exit nonzero after artifacts are written.

## 6. Concurrency, strictness, and resume

The runner creates a candidate pool of 200 workers and a judge pool of 200
workers by default. Both endpoints acquire the same process-wide bounded
semaphore, whose default capacity is 200. Thus:

```text
in_flight_candidate_requests + in_flight_judge_requests <= 200
```

The pool sizes can be changed independently with `--concurrency` and
`--judge-concurrency`; the combined provider pressure is controlled with
`--total-api-concurrency`.

`--strict-judge` and `--strict-errors` are both enabled by default. They do not
rewrite deferred rows as failures; they make an incomplete run visibly fail at
the process level. The append-only detail JSONL is flushed per probe. Resume
keeps stable completed rows and retries candidate/judge infrastructure
deferrals **only when the stored `evaluation_signature` exactly matches the
new run**. This includes the persisted `--resp-cap` and `--reasoning-cap`
values. A missing or mismatched signature makes the old JSONL ineligible
for resume, even when its filename has the same model tag. The old artifact is
not rewritten; the runner starts a fresh detail file.

Disabling a strict flag changes only the final process exit code. It never
upgrades a transiently incomplete aggregate: `meta.score_status` remains
`partial`, `meta.publishable` remains false, and the summary still lists
`meta.deferred_by_reason` and every ID in `meta.deferred_probe_ids`. Such a
result must not be reported as an official benchmark score.

## 7. Authoring and validation rules

- Store reviewed probe changes directly in `probes/`.
- Use deterministic exact, schema, parser, AST, or static checks whenever the
  property can be expressed reliably.
- Use `judge_rubric` for genuinely semantic/open-ended evaluation and provide
  complete pass and fail criteria in canonical form.
- Do not invent a compiler scorer. Keep an explicit string/regex contract when
  a simple parser is insufficient and full compilation is outside scope.
- Use `strict_json` or `strict_output` for `instr` when extra prose is itself a
  tested instruction-following violation. In other categories, express any
  independently necessary functional contract with a schema/parser/AST check;
  do not depend on presentation strictness or incidental extractor behavior.
- Declare regex flags, aliases, punctuation tolerance, whitespace
  normalization, and substring tolerance explicitly.

Before publishing probe changes, at minimum run:

```bash
python3 eval/run_eval.py --version
python3 eval/validate_judge_rubrics.py
python3 eval/test_scorer.py
python3 eval/test_judge.py
python3 eval/test_run_eval.py
```
