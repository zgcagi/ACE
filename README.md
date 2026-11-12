# ACE 2.6.4

[简体中文](README.zh-CN.md) · [Visualization tutorial](examples/tutorial.md)

**ACE** stands for **Atom Capability Evaluation**. It measures individual
language-model capabilities, or **atoms**, and reports scores by atom,
category, and difficulty alongside per-probe outcomes.

## Design principles

- **Limit interference from other capabilities.** A model may solve a math
  problem and add an explanation despite an answer-only instruction. When the
  scoring rule allows extraction, ACE grades the recoverable answer. Probes
  that test instruction following or depend on a specific output structure
  check those requirements directly.
- **Vary inputs and difficulty.** Multiple probes test each atom across
  different inputs and three difficulty levels, L1–L3, to show where
  performance changes as the task gets harder.
- **Make scoring auditable.** ACE uses deterministic checks where possible
  and explicit pass and fail criteria for probes that need a semantic judge.

See also the [ZGCM-1 model project](https://github.com/zgcagi/ZGCM-1).

## Benchmark at a glance

| Item | ACE |
| --- | ---: |
| Canonical atoms | 187 |
| Capability categories | 19 |
| Total probes | 2,565 |
| Probes in the public full-suite score | 2,489 |
| Enabled probes scored with an LLM judge | 348 |
| Excluded probes | 76 |

The probes cover arithmetic, math, logic, causal reasoning, language,
reading, knowledge, world understanding, long context, dialogue,
truthfulness, metacognition, instruction following, output formatting, code,
planning, tool use, agent behavior, and multimodal tasks. Multimodal records
are included for traceability. Their 60 media-dependent probes are disabled
because the media files are unavailable. The public dataset also retains
metadata for 15 excluded security probes and one externally sourced quotation.
The runner skips all 76 exclusions before sending requests to the candidate or judge.

ACE 2.6.4 is the first public release. The release identifier is recorded in
[`benchmark/VERSION`](benchmark/VERSION).

## Install and verify

Run these commands from the repository root. The evaluator uses POSIX process
and resource controls. Use Linux for full evaluation. The offline suite was
verified with Python 3.12 on macOS.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python benchmark/eval/run_eval.py --version
python benchmark/eval/run_eval.py --help
python benchmark/eval/validate_judge_rubrics.py
```

The version command prints `ACE 2.6.4`. Rubric validation is offline and
reports 348 rubrics and zero errors. To run the included offline regression
suite, install the test dependencies and run pytest from this directory:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Try a small evaluation

Set `OPENAI_API_KEY` to the candidate endpoint's key. Replace the endpoint
and model ID in the example. This sends five arithmetic prompts to the
candidate and writes a diagnostic result without calling the judge.

```bash
export OPENAI_API_KEY='your-candidate-key'

python benchmark/eval/run_eval.py \
  --base-url https://candidate.example.com/v1 \
  --model-name candidate-model \
  --filter arith --limit 5 \
  --concurrency 2 --total-api-concurrency 2 \
  --no-llm-judge \
  --out-dir results/smoke \
  --tag candidate-smoke
```

## Run the full benchmark

Use a candidate endpoint that implements the OpenAI Chat Completions API.
Set `ACE_JUDGE_BASE_URL` to your reference judge's OpenAI-compatible endpoint
and `ACE_JUDGE_API_KEY` to its key. The full-suite scoring contract requires
`deepseek-v4-flash` with a 4,096-token response budget. For the 348 judge
probes, the judge receives the task, rubric, candidate answer, and relevant
reference information.

**The full suite executes model-generated Python for some code probes.** Run
full evaluations in an isolated, disposable Linux VM with a dedicated
unprivileged user, no mounted private data, restricted network access, and
scoped evaluation credentials. The runner's time and resource limits are
process controls and do not provide a security sandbox.

```bash
export OPENAI_API_KEY='your-candidate-key'
export ACE_JUDGE_BASE_URL='https://judge.example.com/v1'
export ACE_JUDGE_API_KEY='your-judge-key'

python benchmark/eval/run_eval.py \
  --base-url https://candidate.example.com/v1 \
  --model-name candidate-model \
  --judge-model deepseek-v4-flash \
  --judge-max-tokens 4096 \
  --concurrency 16 --judge-concurrency 16 --total-api-concurrency 16 \
  --out-dir results \
  --tag candidate-model-v1
```

For a single-model diagnostic run, `--judge-base-url` can override the
endpoint and `--judge-model` can select another model. To qualify for a
full-suite score, the endpoint must match `ACE_JUDGE_BASE_URL`. Results
from different judge endpoints carry different evaluation signatures.

The example uses 16 workers per pool and caps combined in-flight requests at
16. Without those flags, each default is 200. Set concurrency to match your
provider quota. The default candidate output budget is 16,384 tokens and the
default temperature is 0. Use `--omit-temperature` for endpoints that reject
the temperature field. `--no-system-prompt` moves system content into the user
message for endpoints that reject the system role.

The runner retries infrastructure failures in a second pass by default.
Each phase allows one candidate request and one judge attempt per probe.
A judge-only retry reuses the original candidate response. If the judge
endpoint rejects JSON response mode, the judge client may make a compatibility
request without that option within the same attempt. Use `--no-retry-failed`
to disable the second phase. Remaining candidate or judge transport failures
can cause a nonzero exit after results are written. Inspect the summary for
deferred probes before rerunning.

## Understand the score and output files

The scorer dispatches each probe using its declared contract and the
[atom policy](benchmark/policy/atom_intents.yaml). Deterministic checks cover
numbers, text, choice labels, keywords, structured data, and code. Selected
semantic probes use a binary LLM judge with explicit pass and fail criteria.
Instruction-following probes enforce their declared presentation constraints.
Other probes can accept harmless answer wrappers when the scoring contract
permits extraction. See the [scoring specification](benchmark/spec/scoring.md).

The overall score is `n_pass / n_eff`. Both scored passes and scored failures
count in `n_eff`. Stable exclusions and infrastructure deferrals do not count
in that denominator. A full-suite result must cover all 2,565 probes, with
`n_eff = 2489` and `n_excluded = 76`.

| File | Contents |
| --- | --- |
| `results_<tag>_<timestamp>.jsonl` | Per-probe verdict, score strategy, response excerpt, and judge evidence where applicable. Retries can append a later record for the same probe ID. Use the last record for each probe. |
| `summary_<tag>_<timestamp>.json` | Aggregate counts, overall pass rate, category and atom breakdowns, run identity, and any deferred probe IDs. |

Check these fields before reporting a full-suite score:

```text
meta.benchmark_version     = "2.6.4"
meta.official_full_suite   = true
meta.official_scope_issues = []
meta.score_status          = "complete"
meta.publishable           = true
```

The last field describes score eligibility. Review detailed outputs before
sharing them because they can include candidate responses and reasoning.
A subset, altered probes or policy, changed response caps, a different judge
configuration, or unresolved deferrals yields a diagnostic result. Use a
separate tag and output directory for diagnostics.

### Resume and reproduce runs

Resume is enabled by default. Rerun with the same tag, output directory,
benchmark files, model settings, and judge configuration to resume compatible
partial results. The runner compares `evaluation_signature`, which binds the
probe and policy content, the evaluator source digest, and the candidate
and judge settings. Old results with a different signature are left intact.
Use `--no-resume` to start a new run. Retry concurrency and timeout can be
adjusted without changing the signature. Enabling or disabling the second
retry phase changes the signed request limits.

Record the candidate model ID and snapshot, both provider endpoints, relevant
generation settings, run date, and the summary's `evaluation_signature`,
`evaluation_contract_sha256`, and `evaluator_runtime_sha256`. The signature
identifies the evaluation inputs and settings. Hosted models can still return
different outputs across runs.

## Batch evaluation and offline analysis

To evaluate several models, copy [`models.example.yaml`](models.example.yaml)
to `models.yaml` and replace the candidate endpoint, API key, and model ID.
Keep the filled configuration local. Set `ACE_JUDGE_BASE_URL` and
`ACE_JUDGE_API_KEY` as in the full-suite example. The batch runner uses
that judge endpoint with the reference model and token budget. Use the same
`ACE_JUDGE_BASE_URL` when checking status or exporting results.

```bash
cp models.example.yaml models.yaml
# Edit models.yaml before running these commands.
python run_all.py --status
python run_all.py --eval-only --model candidate-model
python run_all.py --build-only
```

Batch output goes to `results/`. The last command creates `dashboard/summary.json`
and `dashboard/all_results.json` for downstream analysis. It selects the newest
compatible full-suite result for each configured model ID and tag. For direct
CLI runs, use `--out-dir results` and match those IDs and tags to include the
runs in the export. See the
[offline failure audit guide](benchmark/docs/POST_RUN_FAILURE_AUDIT.md) for
triaging your own result files without making model calls.

## Frontend and model comparison

The ACE frontend provides a leaderboard, model filtering, a category heatmap,
a radar chart, and a side-by-side comparison table. Serve the repository root
with `python3 -m http.server 8000 --bind 127.0.0.1` and open
`http://127.0.0.1:8000/frontend/` to browse three archived ACE 2.6.4
results without API credentials. These aggregate samples use the historical
2,503-probe scoring scope. Current public runs score 2,489 probes.
[Visualization tutorial](examples/tutorial.md) ·
[Sample results](examples/results/)

## Read or extend the dataset

The [probe directory](benchmark/probes/) contains UTF-8 JSONL, with `//` comment
lines in some files. Skip blank and comment lines before decoding records.
`probe_id` is the unique question ID, `canonical_id` identifies its atom,
`category` and `level` support grouping, and `axis_values` records how the
probe varies. `prompt` and optional `system_prompt` form the model input.
`ground_truth` and `scorer` define the reference and scoring contract.
`enabled: false` preserves a probe in the dataset while excluding it from
the score. The [authoring guide](benchmark/spec/authoring_guide.md) explains
the probe format and the three difficulty levels.

| Location | Purpose |
| --- | --- |
| [`benchmark/probes/`](benchmark/probes/) | Materialized prompts, references, and per-probe scoring contracts |
| [`benchmark/policy/`](benchmark/policy/) and [`merged/`](benchmark/merged/) | Atom routing policy and supplemental capability definitions |
| [`benchmark/eval/`](benchmark/eval/) | Runner, scoring and judge implementations, validation, and regression tests |
| [`benchmark/spec/`](benchmark/spec/) and [`docs/`](benchmark/docs/) | Scoring, probe authoring, capability coverage, and offline auditing |
| [`run_all.py`](run_all.py) and [`build_dashboard.py`](build_dashboard.py) | Batch orchestration and local result export |

Static code scoring checks the specified syntax or behavior of a supported
subset. It does not establish general program equivalence. Since the
benchmark includes prompts and answers, keep them out of model training when
using ACE to measure held-out capability.

## License

ACE contributors created all probes and questions published in this repository.
The evaluator code and the complete probe dataset, including prompts, reference
answers, and per-probe scoring rules, are available under the MIT License.
[Code license](LICENSE) · [Probe dataset license](benchmark/probes/LICENSE)
