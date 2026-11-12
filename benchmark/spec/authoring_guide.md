# ACE Probe Authoring Guide

This document tells you how to write probes that **score correctly** under
ACE and how to avoid scoring pitfalls that misclassify valid responses.

## Golden rule: write the probe for the capability, not the format

If the atom is `arith.percent_of`, the capability is "compute a percent". Do
NOT also test "Output ONLY the number with no preamble" — that's a different
capability (`instr.format_strict_compliance`). Mixing them in one probe means
you can't tell which one the model failed on.

## Probe template

```yaml
{
  "probe_id": "arith.percent_of.L1.001",
  "canonical_id": "arith.percent_of",     # atom — drives intent resolution
  "category": "arith",
  "level": "L1",                          # L1 (easy), L2 (medium), L3 (hard)
  "axis_values": {...},                   # variation axes (optional)
  "system_prompt": "",                    # usually empty
  "prompt": "What is 30% of 80?",
  "ground_truth": "24",
  "scorer": {
    "type": "exact_match_numeric",
    "expected": 24
    # tolerance: 0  (default)
  },
  "tags": [],
  "notes": ""
}
```

The `scorer.type` is **a hint, not a contract**. The dispatcher will route
through the v2 smart equivalent of whatever `scorer.type` says, OR override
to a different intent if the atom has one in `policy/atom_intents.yaml`.

## Per-category recipes

### Arithmetic / Math

```yaml
prompt: "Compute 47 + 28 - 19 + 13."
scorer:
  type: exact_match_numeric
  expected: 69
```

✅ Don't say "Output only the number". The atom intent is `extract_numeric`,
so the model's free to think out loud — we'll find the answer at the answer
position.

### MCQ

```yaml
prompt: |
  Which sorts integers in O(n log n) average time?
  A) bubble sort
  B) merge sort
  C) selection sort
  Reply with one letter.
scorer:
  type: mcq_letter
  expected: "B"
```

✅ Saying "Reply with one letter" is helpful (most models comply). The scorer
will also accept "**B**", "Answer: B", etc.

### Read / Extract

```yaml
prompt: |
  Document: ...
  What is the project's launch date?
ground_truth: "March 5, 2024"
scorer:
  type: exact_match_string
  expected: "March 5, 2024"
```

The v2 dispatcher routes this to `extract_string` with `allow_substring=True`,
so the model can write "The launch date is **March 5, 2024**." and pass.

### Truth / Abstention

```yaml
canonical_id: truth.calibrated_abstention
prompt: "Will the S&P 500 close above 7000 on Dec 31, 2030? Brief answer."
scorer:
  type: substring_any
  substrings: ["cannot predict", "uncertain", "no way to know"]
```

✅ The atom has intent `abstention` in policy, which uses a **rich vocabulary
fallback** (>40 phrases). So even if your `substrings` list is short, models
that say "It's impossible to forecast macro markets that far out" will still
pass on the global vocabulary.

❌ DON'T put a "specific recorded answer" question in `truth.calibrated_abstention`.
For example: "What did Caesar Augustus eat the night before he died?" has a
recorded answer (figs/dates per Suetonius). A model that gets it right
shouldn't be penalized for not abstaining.

### Instruction-following

```yaml
canonical_id: instr.format_strict_compliance
prompt: |
  Write three short tips about staying hydrated. Constraints:
  (1) each tip on its own line, (2) every line begins with "Tip: ",
  (3) exactly three lines. Output only the three lines.
scorer:
  type: format_check
  must_satisfy:
    - name: three_tip_lines
      regex: "^Tip: [^\\n]+\\nTip: [^\\n]+\\nTip: [^\\n]+\\s*$"
```

✅ This atom is `instr.*`, which IS measuring strict format compliance.
Strict regex is appropriate. Use `must_satisfy` not `pattern`.

### Code

For library-API probes (idiomatic use):

```yaml
canonical_id: code.api_idiomatic_use
prompt: "Write a Python function dump_yaml(d, path) that writes dict d to YAML file path with 2-space indent and no flow style."
scorer:
  type: code_structural
  structural:
    must_define: [dump_yaml]
    must_call_any: [yaml.dump, yaml.safe_dump]   # synonyms
    must_use_any: [d, path]                      # variable names from prompt
```

For "is this code correct" probes:

```yaml
scorer:
  type: code_unittest
  tests:
    - "assert is_prime(2) == True"
    - "assert is_prime(15) == False"
    - "assert is_prime(97) == True"
  timeout: 10
  memory_mb: 256
```

✅ Both real-execution unittests AND structural checks are first-class. Use
unittests for behavioral correctness, structural for "did you call the right
API".

❌ DON'T use raw `regex` to validate generated code. Things like
`def\s+is_prime\(n\)\s*:` are extremely brittle to whitespace and comment
variations.

### Tool calls

```yaml
canonical_id: tool.argument_extraction
prompt: |
  Tool: set_timer(minutes: int) -> str
  User: "Start a timer for 45 minutes."
  Output a JSON object on one line in this exact shape:
  {"name": "<tool_name>", "arguments": {<arg>: <value>}}.
scorer:
  type: structured_json
  schema:
    type: object
    properties:
      name: {const: set_timer}
      arguments:
        type: object
        properties: {minutes: {const: 45}}
        required: [minutes]
    required: [name, arguments]
```

✅ **Always specify the output shape in the prompt.** If the schema requires
`{name, arguments}`, name those fields in the prompt so a response using
`{tool, args}` is evaluated against a stated requirement.
`json_schema_lenient` can apply declared shape repairs.

### Persona / Dialog

```yaml
canonical_id: dialog.persona_stability
prompt: |
  Persona: You are Professor Eugenia, a fussy Oxford don who quotes Latin,
  says 'rather' and 'quite' and 'indeed', and references books.
  T1 Eugenia: ...
  T2 User: ...
scorer:
  type: regex
  pattern: "(rather|quite|indeed|frightfully|terribly|cicero|latin|oxford|jolly|by jove|tut|trollope|eliot|good chap)"
  flags: [i]
```

✅ The atom is `dialog.persona_stability` → intent `format_lenient`, which
adds **stem fallback**. So `picked up` will match a `pick up|pickup` pattern,
`numpy.zeros` will match a `np.zeros` pattern, etc.

❌ DON'T enumerate just 3-4 highly specific persona keywords. Models do a
fine job emulating Oxford-don tone in many ways. Aim for a vocabulary of
15-25 words/phrases the persona would use.

## Anti-patterns to avoid

| Anti-pattern                                         | Why it's bad                                                                       |
|------------------------------------------------------|------------------------------------------------------------------------------------|
| `substring_all` with single-character flags          | "tar -xzvf" → checking `[x, z, v, f]` is a sloppy way to test the command         |
| `substring_any` with too narrow a vocabulary         | "matter of opinion" probes that miss "comes down to personal preference"           |
| Hidden output-shape contracts in the schema          | `{name, arguments}` schema without saying so in prompt                             |
| Strict regex against code (instead of structural)    | `def\s+f\(x\)\s*:` is too brittle                                                  |
| Abstention probes where the answer IS known          | Caesar Augustus example                                                            |
| `format_check` for non-instr capability tests        | Penalizes verbose answers that are correct                                         |
| Variable-name-specific tests (`setU` for `setUsername`) | Tests style, not capability                                                       |

## Linting your probes

Run `python eval/validate_judge_rubrics.py` from `benchmark/` to validate
judge contracts. Run the offline regression suite from the repository root
to check regex compilation and scorer behavior. Review new probes for:
- `substring_all` items shorter than 3 ASCII chars
- `regex` patterns longer than 200 chars (catastrophic backtrack risk)
- Probes with `scorer.type=exact_match_string` whose `expected` contains
  whitespace (probably wants `allow_substring=True`)
- Probes in `truth.calibrated_abstention` whose answer is a known fact

## What "level" should a probe be?

- **L1**: a competent 7B model should usually pass. Tests basic capability.
- **L2**: 7B models often fail; mid-size models usually pass. Adds 1
  complication (multi-step, longer context, irrelevant distractor).
- **L3**: even strong models get tripped up. Compound reasoning, edge cases,
  long context, or precise extraction.

Aim for 5 probes per atom × 3 levels = 15 probes per atom.
