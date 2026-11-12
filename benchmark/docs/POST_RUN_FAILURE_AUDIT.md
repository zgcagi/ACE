# ACE post-run failure audit

`eval/audit_failures.py` performs an offline, read-only audit of completed
ACE result files. It does not rescore responses and never calls a model or an
LLM judge. Judge verdicts, failed criteria, confidence, and evidence are copied
only from the supplied result JSONL.

## Audit two completed runs

From the `benchmark` directory:

```bash
python3 eval/audit_failures.py \
  --results nonthinking=/path/to/results_MODEL_TIMESTAMP.jsonl \
  --results thinking=/path/to/results_MODEL-thinking_TIMESTAMP.jsonl \
  --summary nonthinking=/path/to/summary_MODEL_TIMESTAMP.json \
  --summary thinking=/path/to/summary_MODEL-thinking_TIMESTAMP.json \
  --probes-dir probes \
  --out-dir /path/to/post_run_audit \
  --name ace_failure_audit
```

The command writes:

- `ace_failure_audit.json`: full machine-readable review packets,
  including prompts, scorer contracts, full responses, answer-position hints,
  and result-record Judge evidence.
- `ace_failure_audit.md`: compact review report with aggregate
  tables and response excerpts.

`--excerpt-chars N` controls only Markdown excerpts. JSON retains full response
and reasoning text. `--summary` is optional, but supplying it enables consistency
checks against the deduplicated JSONL.

## Deduplication and failure scope

1. Within one run, the last JSONL row for each `probe_id` wins. This matches the
   append-only retry behavior of `run_eval.py`.
2. Across runs, one review item is emitted per `probe_id`.
3. An item is included if its final row failed in at least one run. Passes from
   another run remain attached and add `cross_run_disagreement`.
4. Excluded, deferred, and infrastructure-error rows remain visible, but are
   counted separately from effective failures.

## Automatic review flags

Flags are triage hints, not revised scores:

- `explicit_answer_matches_expected`: an Answer/Output/Reply/boxed/first-or-last
  candidate appears to match the probe reference despite the recorded failure.
- `response_contains_expected`: the expected value appears in the response.
- `cross_run_disagreement`: the same probe passes one run and fails another.
- `judge_fail_review` / `judge_failed_criteria`: a supplied result record contains
  an actual Judge `FAIL` and/or unsatisfied/triggered criteria.
- `regex_boundary_review`, `mcq_boundary_review`, `numeric_boundary_review`:
  deterministic scorer families where extraction boundaries deserve review.
- `scorer_error`, `deferred`, `excluded_from_score`: runtime or denominator state.

Review order is `critical`, `high`, `medium`, then `low`. A flagged response is
not automatically a scorer bug: for example, reasoning may mention the expected
answer and later explicitly reject it. The report preserves the final strategy,
score note, extracted value, Judge evidence, and all run outcomes for that manual
decision.

## Suggested parallel review workflow

1. Check `critical` items first and separate infrastructure/Judge availability
   failures from stable model failures.
2. Split `high` items by category or scorer type among review agents.
3. For each item, compare prompt, expected answer, response, final answer cue,
   scorer strategy, and (only when present) the recorded Judge criteria.
4. Record one disposition: `correct_fail`, `scorer_false_negative`,
   `probe_or_reference_issue`, `judge_issue`, or `infrastructure_issue`.
5. Deduplicate proposed scorer fixes by root cause before changing runtime code.

## Offline test

```bash
python3 eval/test_audit_failures.py
```

