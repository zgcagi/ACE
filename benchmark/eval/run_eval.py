"""Run ACE against an OpenAI-compatible candidate endpoint.

Evaluation behavior:
  - Uses dispatch.score_probe (atom-aware, multi-strategy fallback)
  - Captures both `content` and `reasoning_content` (DeepSeek-style models)
  - judge_rubric probes use a configured judge; infrastructure failures
    and multimodal placeholders are removed from the pass-rate denominator
  - Probes are evaluated directly from the materialized dataset
  - Candidate + judge HTTP requests share one process-wide concurrency limit
  - Explicitly disabled probes never consume candidate-model API calls
  - Defensive against scorer crashes; per-call regex timeout
  - Concurrency-safe scoring on main thread (worker threads only do API calls)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.dispatch import score_probe, load_policy
from eval.judge import (
    configure_judge,
    judge_endpoint,
    DEFAULT_MODEL as DEFAULT_JUDGE_MODEL,
    PROMPT_VERSION as JUDGE_PROMPT_VERSION,
)
from eval.version import (
    BENCHMARK_VERSION, EVALUATION_CONTRACT_VERSION,
    OFFICIAL_DISABLED_PROBES, OFFICIAL_EFFECTIVE_PROBES,
    OFFICIAL_TOTAL_PROBES, SCORER_VERSION,
)

DEFAULT_CANDIDATE_CONCURRENCY = 200
DEFAULT_TOTAL_API_CONCURRENCY = 200
DEFAULT_JUDGE_CONCURRENCY = 200
DEFAULT_SKIP_CATEGORIES = ""
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.0
DEFAULT_JUDGE_MAX_TOKENS = 4096
DEFAULT_RESPONSE_CAP = 100000
DEFAULT_REASONING_CAP = 100000
DEFAULT_LLM_JUDGE = True
DEFAULT_STRICT_JUDGE = True
DEFAULT_STRICT_ERRORS = True
DEFAULT_RESUME = True
DEFAULT_RETRY_FAILED = True
CANDIDATE_SDK_MAX_RETRIES = 0
CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE = 1
JUDGE_REQUEST_ATTEMPTS_PER_PHASE = 1


def max_candidate_requests_per_probe(retry_failed: bool) -> int:
    """Return the per-invocation candidate request ceiling for one probe."""
    return CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE + int(bool(retry_failed))


def max_judge_requests_per_probe(retry_failed: bool) -> int:
    """Return the per-invocation judge request ceiling for one probe."""
    return JUDGE_REQUEST_ATTEMPTS_PER_PHASE + int(bool(retry_failed))


def configure_runtime_judge(**kwargs) -> None:
    """Configure one judge request per phase; the runner owns phase retries."""
    configure_judge(retries=JUDGE_REQUEST_ATTEMPTS_PER_PHASE, **kwargs)


def load_probes(probes_dir: Path):
    by_id = {}
    for f in sorted(probes_dir.glob("*.jsonl")):
        with f.open() as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("//"): continue
                try:
                    p = json.loads(ln)
                    by_id[p["probe_id"]] = p
                except json.JSONDecodeError as e:
                    print(f"WARN: bad JSONL in {f.name}: {e}", file=sys.stderr)
    return by_id


def probe_manifest_sha256(probes) -> str:
    """Hash the exact materialized probe objects in stable probe-ID order."""
    values = probes.values() if isinstance(probes, dict) else probes
    by_id = {probe["probe_id"]: probe for probe in values}
    digest = hashlib.sha256()
    for probe_id in sorted(by_id):
        digest.update(json.dumps(
            by_id[probe_id], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256(paths) -> str:
    """Hash evaluator sources with stable, repository-relative names."""
    digest = hashlib.sha256()
    for path in sorted((Path(p).resolve() for p in paths), key=lambda p: str(p)):
        try:
            label = path.relative_to(ROOT).as_posix()
        except ValueError:
            label = path.name
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def evaluator_source_paths() -> list[Path]:
    """Source files whose contents define the local scoring contract."""
    return [
        Path(__file__),
        ROOT / "eval" / "dispatch.py",
        ROOT / "eval" / "scorer.py",
        ROOT / "eval" / "static_ooxml.py",
        ROOT / "eval" / "static_python_semantics.py",
        ROOT / "eval" / "judge.py",
        ROOT / "eval" / "version.py",
    ]


def evaluation_signature(payload: dict) -> str:
    """Hash all non-secret inputs that determine generation and scoring."""
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def build_evaluation_contract(
    *,
    probe_suite_sha256: str,
    selected_probe_sha256: str,
    policy_sha256: str,
    evaluator_runtime_sha256: str,
) -> tuple[dict, str]:
    """Build the canonical contract payload and its SHA-256 identity."""
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "scorer_version": SCORER_VERSION,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "probe_suite_sha256": probe_suite_sha256,
        "selected_probe_sha256": selected_probe_sha256,
        "policy_sha256": policy_sha256,
        "evaluator_runtime_sha256": evaluator_runtime_sha256,
    }
    return payload, evaluation_signature(payload)


def aggregate(results):
    by_atom = defaultdict(lambda: {"L1": [], "L2": [], "L3": []})
    by_category = defaultdict(list)
    by_cat_level = defaultdict(lambda: defaultdict(list))
    n_excluded = 0
    n_pass = 0
    n_eff = 0  # effective denominator
    for r in results:
        if r.get("excluded_from_score"):
            n_excluded += 1
            continue
        n_eff += 1
        if r["passed"]: n_pass += 1
        by_atom[r["canonical_id"]][r["level"]].append(r["passed"])
        by_category[r["category"]].append(r["passed"])
        by_cat_level[r["category"]][r["level"]].append(r["passed"])

    def _r(bs):
        return round(sum(1 for b in bs if b)/len(bs), 4) if bs else None

    atom_summary = {}
    for aid, lv in by_atom.items():
        atom_summary[aid] = {
            "L1_pass_rate": _r(lv["L1"]),
            "L2_pass_rate": _r(lv["L2"]),
            "L3_pass_rate": _r(lv["L3"]),
            "n_total": sum(len(v) for v in lv.values()),
            "category": aid.split(".")[0],
        }
    cat_summary = {}
    for cat, passes in by_category.items():
        cat_summary[cat] = {
            "pass_rate": _r(passes), "n": len(passes),
            "L1": _r(by_cat_level[cat]["L1"]),
            "L2": _r(by_cat_level[cat]["L2"]),
            "L3": _r(by_cat_level[cat]["L3"]),
        }
    return {
        "n_total": len(results),
        "n_eff": n_eff,
        "n_excluded": n_excluded,
        "n_pass": n_pass,
        "overall_pass_rate": round(n_pass/n_eff, 4) if n_eff else None,
        "by_category": cat_summary,
        "by_atom": atom_summary,
    }


def transient_deferred_diagnostics(results):
    """Describe non-stable exclusions that make an aggregate non-publishable.

    Explicitly disabled probes are part of the declared effective benchmark
    scope and are not transient. Provider, judge-protocol, and scorer failures
    can be answer/task-correlated, so any such record makes the run partial.
    """
    transient_strategies = {
        "model_call_error", "judge_deferred", "llm_judge_error", "scorer_error",
    }
    affected = []
    by_reason = defaultdict(int)
    for record in results:
        strategy = record.get("score_strategy")
        if record.get("error") or strategy in transient_strategies:
            affected.append(record.get("probe_id"))
            by_reason[strategy or "unknown_error"] += 1
    return {
        "n_transient_deferred": len(affected),
        "deferred_by_reason": dict(sorted(by_reason.items())),
        "deferred_probe_ids": sorted(pid for pid in affected if pid),
        "score_status": "partial" if affected else "complete",
        "publishable": not affected,
    }


def official_scope_issues(
    *,
    probe_suite_sha256: str,
    selected_probe_sha256: str,
    policy_sha256: str,
    canonical_probe_sha256: str,
    canonical_policy_sha256: str,
    llm_judge: bool,
    judge_model: str,
    judge_base_url: str,
    reference_judge_base_url: str,
    judge_max_tokens: int,
    response_cap: int,
    reasoning_cap: int,
    aggregate_result: dict,
) -> list[str]:
    """Explain why a result is not an official full-suite ACE score."""
    issues = []
    if probe_suite_sha256 != canonical_probe_sha256:
        issues.append("noncanonical_probe_suite")
    if selected_probe_sha256 != canonical_probe_sha256:
        issues.append("selected_probe_subset_or_modified_scope")
    if policy_sha256 != canonical_policy_sha256:
        issues.append("noncanonical_scoring_policy")
    if not llm_judge:
        issues.append("llm_judge_disabled")
    if judge_model != DEFAULT_JUDGE_MODEL:
        issues.append("noncanonical_judge_model")
    if llm_judge and (not reference_judge_base_url
                      or judge_endpoint(judge_base_url) != judge_endpoint(reference_judge_base_url)):
        issues.append("noncanonical_judge_endpoint")
    if judge_max_tokens != DEFAULT_JUDGE_MAX_TOKENS:
        issues.append("noncanonical_judge_max_tokens")
    if response_cap != DEFAULT_RESPONSE_CAP:
        issues.append("noncanonical_response_cap")
    if reasoning_cap != DEFAULT_REASONING_CAP:
        issues.append("noncanonical_reasoning_cap")
    expected_counts = {
        "n_total": OFFICIAL_TOTAL_PROBES,
        "n_eff": OFFICIAL_EFFECTIVE_PROBES,
        "n_excluded": OFFICIAL_DISABLED_PROBES,
    }
    for key, expected in expected_counts.items():
        if aggregate_result.get(key) != expected:
            issues.append(
                f"{key}_mismatch:{aggregate_result.get(key)}!=expected_{expected}"
            )
    return issues


class Runner:
    def __init__(self, base_url, api_key, model_name,
                 max_tokens=16384, temperature=0.0, request_timeout=300,
                 request_limiter=None, thinking_mode=False,
                 omit_temperature=False):
        from openai import OpenAI
        # The runner owns retry policy explicitly.  Disabling SDK retries
        # prevents one logical call from silently expanding into three HTTP
        # requests before the runner's second-stage retry is considered.
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=CANDIDATE_SDK_MAX_RETRIES,
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = None if omit_temperature else temperature
        self.think_prefix = bool(thinking_mode)
        self.request_limiter = request_limiter

    def _with_request_limit(self, fn, **request_args):
        if self.request_limiter is None:
            return fn(**request_args)
        with self.request_limiter:
            return fn(**request_args)

    def call(self, prompt, system_prompt=""):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        # For thinking models: add assistant prefix to trigger <think> mode
        if self.think_prefix:
            messages.append({"role": "assistant", "content": "<think>\n", "prefix": True})
        request_args = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            request_args["temperature"] = self.temperature
        rsp = self._with_request_limit(
            self.client.chat.completions.create, **request_args
        )
        msg = rsp.choices[0].message
        content = msg.content or ""
        # Support multiple reasoning field names across providers:
        # - DeepSeek native: reasoning_content
        # - OpenRouter: reasoning
        reasoning = (getattr(msg, "reasoning_content", None)
                     or getattr(msg, "reasoning", None) or "")
        # For thinking models that put everything in reasoning with empty content,
        # use reasoning as content fallback for scoring
        if not content and reasoning:
            content = reasoning
        # Handle </think> tag: split thinking from final answer
        if "</think>" in content:
            parts = content.split("</think>", 1)
            reasoning = parts[0].strip()
            content = parts[1].strip()
        return {
            "content": content,
            "reasoning": reasoning,
            "finish_reason": rsp.choices[0].finish_reason,
            "completion_tokens": rsp.usage.completion_tokens if rsp.usage else None,
        }


def call_probe(runner, probe, no_system_prompt=False):
    if probe.get("enabled") is False:
        # Disabled probes (currently missing-media placeholders) are stable
        # benchmark exclusions. They must not incur a provider request.
        return {
            "probe": probe,
            "out": {
                "content": "", "reasoning": "", "finish_reason": None,
                "completion_tokens": None,
            },
            "err": None,
            "latency_s": 0.0,
            "candidate_request_attempts": 0,
        }
    prompt = probe.get("prompt", "")
    system_prompt = probe.get("system_prompt", "") or ""
    if no_system_prompt and system_prompt:
        prompt = system_prompt + "\n\n" + prompt
        system_prompt = ""
    err = None; out = None
    t0 = time.time()
    try:
        # Exactly one candidate request belongs to each phase.  When enabled,
        # retry_failed_results supplies the only second candidate request.
        out = runner.call(prompt, system_prompt=system_prompt)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        out = {"content": "", "reasoning": "", "finish_reason": None, "completion_tokens": None}
    return {
        "probe": probe,
        "out": out,
        "err": err,
        "latency_s": round(time.time()-t0, 3),
        "candidate_request_attempts": CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE,
    }


def score_one(call_result, policy, resp_cap=8000, reasoning_cap=8000,
              run_context=None):
    probe = call_result["probe"]
    out = call_result["out"]
    err = call_result["err"]
    response = out["content"]

    t0 = time.time()
    if err:
        from eval.scorer import ScoreResult
        # A transport/provider failure is not evidence that the candidate
        # answered incorrectly. Keep it visible and retryable, but exclude it
        # from the benchmark denominator.
        result, denom = ScoreResult(
            False, "model_call_error", note=err, deferred=True
        ), 0
    else:
        result, denom = score_probe(probe, response, policy)
    score_dt = time.time() - t0

    rec = {
        "benchmark_version": BENCHMARK_VERSION,
        "scorer_version": SCORER_VERSION,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "probe_id": probe["probe_id"],
        "canonical_id": probe["canonical_id"],
        "category": probe["category"],
        "level": probe["level"],
        "passed": result.passed if not err else False,
        "excluded_from_score": denom == 0,
        "score_strategy": result.strategy,
        "score_extracted": result.extracted,
        "score_note": result.note,
        "deferred": result.deferred,
        "response": response[:resp_cap],
        "reasoning": out["reasoning"][:reasoning_cap] if out["reasoning"] else "",
        "finish_reason": out["finish_reason"],
        "completion_tokens": out["completion_tokens"],
        "candidate_request_attempts": int(
            call_result.get("candidate_request_attempts", 0) or 0
        ),
        "latency_s": call_result["latency_s"],
        "score_s": round(score_dt, 3),
    }
    if run_context:
        rec.update({
            "evaluation_signature": run_context["evaluation_signature"],
            "evaluation_contract_sha256": run_context["evaluation_contract_sha256"],
            "evaluator_runtime_sha256": run_context["evaluator_runtime_sha256"],
            "probe_suite_sha256": run_context["probe_suite_sha256"],
            "selected_probe_sha256": run_context["selected_probe_sha256"],
            "policy_sha256": run_context["policy_sha256"],
        })
    if err: rec["error"] = err
    return rec


RETRYABLE_STRATEGIES = {
    "model_call_error", "judge_deferred", "llm_judge_error", "scorer_error",
}


def is_retryable_result(record):
    return bool(record.get("error")) or record.get("score_strategy") in RETRYABLE_STRATEGIES


def _counter_from_results(results):
    return {
        "done": len(results),
        "passed": sum(bool(r.get("passed")) for r in results),
        "excluded": sum(bool(r.get("excluded_from_score")) for r in results),
        "errors": sum(bool(r.get("error")) for r in results),
        "tokens": sum(r.get("completion_tokens", 0) or 0 for r in results),
        "request_attempts": sum(
            int(r.get("candidate_request_attempts", 0) or 0) for r in results
        ),
        "judge_deferred": sum(
            r.get("score_strategy") in {"judge_deferred", "llm_judge_error"}
            for r in results
        ),
    }


def retry_failed_results(
    results, probes_by_id, args, policy, run_context, request_limiter,
    detail_path,
):
    """Retry infrastructure-deferred records without retrying wrong answers.

    Candidate transport failures make a second candidate request. Judge-only
    deferrals reuse the already captured candidate response, avoiding an
    unnecessary and potentially different model generation.
    """
    pending = [
        (index, record, probes_by_id.get(record.get("probe_id")))
        for index, record in enumerate(results)
        if is_retryable_result(record)
    ]
    pending = [item for item in pending if item[2] is not None]
    retry_concurrency = args.retry_concurrency or max(
        1, min(8, max(1, args.concurrency // 4))
    )
    retry_timeout = args.retry_request_timeout or args.request_timeout
    stats = {
        "enabled": bool(args.retry_failed),
        "first_pass_deferred": len(pending),
        "attempted": 0,
        "recovered": 0,
        "still_deferred": len(pending),
        "concurrency": retry_concurrency,
        "request_timeout": retry_timeout,
        "candidate_sdk_max_retries": CANDIDATE_SDK_MAX_RETRIES,
        "candidate_request_attempts_per_phase": CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE,
        "max_candidate_requests_per_probe": max_candidate_requests_per_probe(
            args.retry_failed
        ),
        "judge_request_attempts_per_phase": JUDGE_REQUEST_ATTEMPTS_PER_PHASE,
        "max_judge_requests_per_probe": max_judge_requests_per_probe(
            args.retry_failed
        ),
    }
    if not args.retry_failed or not pending:
        return stats

    retry_runner = Runner(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_timeout=retry_timeout,
        request_limiter=request_limiter,
        thinking_mode=args.thinking_mode,
        omit_temperature=getattr(args, "omit_temperature", False),
    )

    def retry_one(record, probe):
        if (record.get("score_strategy") in {"judge_deferred", "llm_judge_error"}
                and not record.get("error")):
            call_result = {
                "probe": probe,
                "out": {
                    "content": record.get("response") or "",
                    "reasoning": record.get("reasoning") or "",
                    "finish_reason": record.get("finish_reason"),
                    "completion_tokens": record.get("completion_tokens"),
                },
                "err": None,
                "latency_s": record.get("latency_s", 0),
                "candidate_request_attempts": 0,
            }
        else:
            call_result = call_probe(
                retry_runner, probe, no_system_prompt=args.no_system_prompt
            )
        # Judge retries contain their own endpoint call and may remain in the
        # retry pool.  Deterministic scoring must return to the main thread so
        # process-wide parser/regex alarms remain effective; only candidate
        # network calls are parallelized for those probes.
        if (probe.get("scorer", {}).get("type") == "judge_rubric"
                and args.llm_judge and not call_result["err"]):
            return "scored", score_one(
                call_result, policy, args.resp_cap, args.reasoning_cap,
                run_context,
            )
        return "call_result", call_result

    print(
        f"\n=== Retry deferred probes ===\n"
        f"  retrying:     {len(pending)} infrastructure-deferred probes\n"
        f"  concurrency:  {retry_concurrency}\n"
        f"  timeout:      {retry_timeout}s",
        file=sys.stderr,
    )
    with (ThreadPoolExecutor(max_workers=retry_concurrency) as retry_ex,
          detail_path.open("a", encoding="utf-8") as detail_fh):
        futures = {
            retry_ex.submit(retry_one, record, probe): (index, record)
            for index, record, probe in pending
        }
        for retry_index, future in enumerate(as_completed(futures), start=1):
            index, first_record = futures[future]
            retry_kind, retry_value = future.result()
            if retry_kind == "scored":
                retry_record = retry_value
            else:
                retry_record = score_one(
                    retry_value, policy, args.resp_cap, args.reasoning_cap,
                    run_context,
                )
            retry_record.update({
                "retry_attempted": True,
                "retry_round": 2,
                "retry_index": retry_index,
                "retry_concurrency": retry_concurrency,
                "retry_request_timeout": retry_timeout,
                "retry_recovered": not is_retryable_result(retry_record),
                "first_round_error": first_record.get("error", ""),
                "first_round_score_strategy": first_record.get("score_strategy", ""),
                "candidate_request_attempts": (
                    int(first_record.get("candidate_request_attempts", 1) or 0)
                    + int(retry_record.get("candidate_request_attempts", 0) or 0)
                ),
            })
            detail_fh.write(json.dumps(
                retry_record, ensure_ascii=False, default=str
            ) + "\n")
            detail_fh.flush()
            results[index] = retry_record
            stats["attempted"] += 1
            if retry_record["retry_recovered"]:
                stats["recovered"] += 1
    stats["still_deferred"] = sum(is_retryable_result(r) for r in results)
    return stats


def find_partial_results(out_dir: Path, tag: str, expected_signature: str | None = None):
    """Find the most recent resumable results file.

    A file is resumable when it has no matching summary *or* contains a
    retryable model/judge infrastructure record.  This lets a later run repair
    transient judge deferrals even if the earlier run emitted a summary.
    Returns (jsonl_path, completed_probe_ids) or (None, set())."""
    import glob as _glob
    pattern = str(out_dir / f"results_{tag}_*.jsonl")
    jsonl_files = sorted(_glob.glob(pattern))
    if not jsonl_files:
        return None, set()
    # Check from most recent - find one without a summary (= partial)
    for jf in reversed(jsonl_files):
        # Extract timestamp from filename
        ts_part = jf.rsplit("_", 1)[-1].replace(".jsonl", "")
        summary_file = out_dir / f"summary_{tag}_{ts_part}.json"
        # Infrastructure failures remain retryable; stable exclusions (for
        # example a deliberately disabled media probe) count as completed so
        # resume does not loop forever.
        completed = set()
        retryable_ids = set()
        incompatible = False
        saw_record = False
        try:
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    saw_record = True
                    if (expected_signature is not None
                            and rec.get("evaluation_signature") != expected_signature):
                        incompatible = True
                        break
                    retryable = is_retryable_result(rec)
                    if retryable:
                        retryable_ids.add(rec["probe_id"])
                        completed.discard(rec["probe_id"])
                    else:
                        completed.add(rec["probe_id"])
                        retryable_ids.discard(rec["probe_id"])
        except (json.JSONDecodeError, KeyError):
            # A truncated/corrupt append-only log is not a safe resume base for
            # a signed release run.  Keep legacy best-effort behavior only for
            # callers that explicitly omit an expected signature.
            incompatible = True
        if expected_signature is not None and (incompatible or not saw_record):
            # Never append a new release/configuration to an old detail file.
            continue
        if not summary_file.exists() or retryable_ids:
            return Path(jf), completed
    return None, set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version", action="version", version=f"ACE {BENCHMARK_VERSION}"
    )
    ap.add_argument("--probes-dir", default=str(ROOT / "probes"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--policy", default=str(ROOT / "policy/atom_intents.yaml"))
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CANDIDATE_CONCURRENCY)
    ap.add_argument(
        "--total-api-concurrency", type=int, default=DEFAULT_TOTAL_API_CONCURRENCY,
        help=(
            "Process-wide cap shared by candidate and judge HTTP calls "
            "(default: 200)"
        ),
    )
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument(
        "--omit-temperature", action="store_true", default=False,
        help="Do not send a temperature field to the candidate endpoint",
    )
    ap.add_argument("--filter", default=None)
    ap.add_argument("--skip-categories", default=DEFAULT_SKIP_CATEGORIES)
    ap.add_argument("--probe-ids-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--request-timeout", type=int, default=300)
    ap.add_argument("--llm-judge", action=argparse.BooleanOptionalAction, default=DEFAULT_LLM_JUDGE,
                    help="Use the configured LLM judge for judge_rubric probes")
    reference_judge_base_url = os.environ.get("ACE_JUDGE_BASE_URL", "")
    ap.add_argument("--judge-base-url", default=reference_judge_base_url)
    ap.add_argument("--judge-api-key", default=os.environ.get("ACE_JUDGE_API_KEY", ""))
    ap.add_argument("--judge-model",
                    default=os.environ.get("ACE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    ap.add_argument("--judge-timeout", type=int, default=120)
    ap.add_argument("--judge-concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY)
    ap.add_argument("--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    ap.add_argument("--strict-judge", action=argparse.BooleanOptionalAction, default=DEFAULT_STRICT_JUDGE,
                    help="Exit nonzero if any enabled judge call remains deferred")
    ap.add_argument("--strict-errors", action=argparse.BooleanOptionalAction, default=DEFAULT_STRICT_ERRORS,
                    help="Exit nonzero if any candidate-model API call remains deferred")
    ap.add_argument("--resp-cap", type=int, default=DEFAULT_RESPONSE_CAP)
    ap.add_argument("--reasoning-cap", type=int, default=DEFAULT_REASONING_CAP)
    ap.add_argument("--resume", action="store_true", default=DEFAULT_RESUME,
                    help="Resume from partial results if available (default: on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Don't resume; start fresh")
    ap.add_argument(
        "--retry-failed", action=argparse.BooleanOptionalAction,
        default=DEFAULT_RETRY_FAILED,
        help=(
            "Make one additional request for infrastructure-deferred probes "
            "in a lower-concurrency second pass (default: on; at most two "
            "candidate requests per probe including the first pass)"
        ),
    )
    ap.add_argument(
        "--retry-concurrency", type=int, default=None,
        help="Second-pass concurrency (default: min(8, concurrency//4), at least 1)",
    )
    ap.add_argument(
        "--retry-request-timeout", type=int, default=None,
        help="Second-pass request timeout (default: --request-timeout)",
    )
    ap.add_argument(
        "--thinking-mode", action=argparse.BooleanOptionalAction, default=False,
        help="Add the model-specific thinking prefix and separate reasoning from the final answer",
    )
    ap.add_argument(
        "--no-system-prompt", action="store_true", default=False,
        help="Avoid the system role by prepending system content to the user message",
    )
    args = ap.parse_args()

    for name in ("concurrency", "judge_concurrency", "total_api_concurrency"):
        if getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("retry_concurrency", "retry_request_timeout"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be at least 1")

    if not args.api_key:
        print("ERROR: --api-key not provided", file=sys.stderr); sys.exit(2)

    if args.llm_judge and not args.judge_base_url:
        ap.error("set ACE_JUDGE_BASE_URL or pass --judge-base-url to enable the judge")

    judge_key = args.judge_api_key
    candidate_root = args.base_url.rstrip("/").removesuffix("/v1")
    judge_root = args.judge_base_url.rstrip("/").removesuffix("/v1")
    if args.llm_judge and not judge_key and candidate_root == judge_root:
        judge_key = args.api_key
    if args.llm_judge and not judge_key:
        print(
            "ERROR: Judge requires ACE_JUDGE_API_KEY or --judge-api-key",
            file=sys.stderr,
        )
        sys.exit(2)

    api_request_limiter = threading.BoundedSemaphore(args.total_api_concurrency)

    configure_runtime_judge(
        enabled=args.llm_judge,
        base_url=args.judge_base_url,
        api_key=judge_key,
        model=args.judge_model,
        timeout=args.judge_timeout,
        max_tokens=args.judge_max_tokens,
        request_limiter=api_request_limiter,
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    skip = {c.strip() for c in args.skip_categories.split(",") if c.strip()}

    only_ids = None
    if args.probe_ids_file:
        only_ids = set()
        with open(args.probe_ids_file) as f:
            for ln in f:
                ln = ln.strip()
                if ln: only_ids.add(ln)

    probes_by_id = load_probes(Path(args.probes_dir))
    probe_suite_hash = probe_manifest_sha256(probes_by_id)
    canonical_probe_hash = probe_manifest_sha256(load_probes(ROOT / "probes"))

    # Filter
    probes = []
    for pid, p in probes_by_id.items():
        if args.filter and p.get("category") != args.filter: continue
        if p.get("category") in skip: continue
        if only_ids is not None and pid not in only_ids: continue
        probes.append(p)
        if args.limit and len(probes) >= args.limit: break
    print(f"Selected {len(probes)} probes", file=sys.stderr)
    selected_probe_hash = probe_manifest_sha256(probes)

    # Load policy
    policy = load_policy(args.policy)
    policy_hash = file_sha256(Path(args.policy))
    canonical_policy_hash = file_sha256(ROOT / "policy" / "atom_intents.yaml")

    evaluator_runtime_hash = source_bundle_sha256(evaluator_source_paths())

    contract_payload, evaluation_contract_hash = build_evaluation_contract(
        probe_suite_sha256=probe_suite_hash,
        selected_probe_sha256=selected_probe_hash,
        policy_sha256=policy_hash,
        evaluator_runtime_sha256=evaluator_runtime_hash,
    )

    effective_temperature = None if args.omit_temperature else args.temperature
    signature_payload = {
        **contract_payload,
        "evaluation_contract_sha256": evaluation_contract_hash,
        "candidate": {
            "base_url": args.base_url.rstrip("/"),
            "model": args.model_name,
            "temperature": effective_temperature,
            "max_tokens": args.max_tokens,
            "response_cap": args.resp_cap,
            "reasoning_cap": args.reasoning_cap,
            "thinking_mode": args.thinking_mode,
            "no_system_prompt": args.no_system_prompt,
            "candidate_sdk_max_retries": CANDIDATE_SDK_MAX_RETRIES,
            "candidate_request_attempts_per_phase": CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE,
            "max_candidate_requests_per_probe": max_candidate_requests_per_probe(
                args.retry_failed
            ),
        },
        "judge": {
            "enabled": args.llm_judge,
            "base_url": args.judge_base_url.rstrip("/"),
            "model": args.judge_model,
            "max_tokens": args.judge_max_tokens,
            "request_attempts_per_phase": JUDGE_REQUEST_ATTEMPTS_PER_PHASE,
            "max_requests_per_probe": max_judge_requests_per_probe(
                args.retry_failed
            ),
        },
    }
    run_context = {
        **signature_payload,
        "evaluation_signature": evaluation_signature(signature_payload),
    }
    print(
        f"ACE {BENCHMARK_VERSION} contract={evaluation_contract_hash[:12]} "
        f"signature={run_context['evaluation_signature'][:12]}",
        file=sys.stderr,
    )

    # Run
    runner = Runner(
        base_url=args.base_url, api_key=args.api_key,
        model_name=args.model_name, max_tokens=args.max_tokens,
        temperature=args.temperature, request_timeout=args.request_timeout,
        request_limiter=api_request_limiter,
        thinking_mode=args.thinking_mode,
        omit_temperature=args.omit_temperature,
    )

    tag = args.tag or args.model_name.replace("/", "_")

    # --- Checkpoint / Resume logic ---
    resumed_results = []
    resumed_ids = set()
    partial_path = None

    if args.resume:
        partial_path, resumed_ids = find_partial_results(
            out_dir, tag, run_context["evaluation_signature"]
        )
        if resumed_ids:
            print(f"  ♻️  Resuming: found {len(resumed_ids)} completed probes in {partial_path.name}",
                  file=sys.stderr)
            # Load only the latest non-retryable record per completed probe.
            # Older deferred/error lines remain in the append-only detail log
            # for auditability but must not be double-counted.
            resumed_by_id = {}
            with partial_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        retryable = is_retryable_result(rec)
                        if rec.get("probe_id") in resumed_ids and not retryable:
                            resumed_by_id[rec["probe_id"]] = rec
                    except json.JSONDecodeError:
                        pass
            resumed_results.extend(resumed_by_id.values())
            # Filter out already-completed probes
            probes = [p for p in probes if p["probe_id"] not in resumed_ids]
            print(f"  ♻️  Remaining probes to evaluate: {len(probes)}", file=sys.stderr)

    if partial_path and args.resume:
        # Append to existing partial file
        detail_path = partial_path
        detail_fh = detail_path.open("a")
        ts = int(partial_path.stem.rsplit("_", 1)[-1])
    else:
        ts = int(time.time())
        detail_path = out_dir / f"results_{tag}_{ts}.jsonl"
        detail_fh = detail_path.open("w")

    results = list(resumed_results)  # Start with already-completed results
    counter = {
        "done": len(resumed_ids),
        "passed": sum(1 for r in resumed_results if r.get("passed")),
        "excluded": sum(1 for r in resumed_results if r.get("excluded_from_score")),
        "errors": sum(1 for r in resumed_results if r.get("error")),
        "tokens": sum(r.get("completion_tokens", 0) or 0 for r in resumed_results),
        "judge_deferred": sum(
            r.get("score_strategy") in {"judge_deferred", "llm_judge_error"}
            for r in resumed_results
        ),
    }
    n_remaining = len(probes)
    n_total = n_remaining + len(resumed_ids)
    t_start = time.time()

    if n_remaining == 0:
        print(f"  ✅ All probes already completed (resumed {len(resumed_ids)} results)",
              file=sys.stderr)
        detail_fh.close()
    else:
        def record_result(rec):
            detail_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            detail_fh.flush()
            results.append(rec)
            counter["done"] += 1
            if rec["passed"]: counter["passed"] += 1
            if rec["excluded_from_score"]: counter["excluded"] += 1
            if rec.get("error"): counter["errors"] += 1
            if rec.get("completion_tokens"): counter["tokens"] += rec["completion_tokens"]
            if rec.get("score_strategy") in {"judge_deferred", "llm_judge_error"}:
                counter["judge_deferred"] += 1
            new_done = counter["done"] - len(resumed_ids)
            if new_done % 50 == 0 or new_done == n_remaining:
                dt = time.time() - t_start
                rate = new_done/dt if dt > 0 else 0
                eta = (n_remaining - new_done)/rate if rate > 0 else 0
                eff = counter["done"] - counter["excluded"]
                rate_pct = counter["passed"]/eff*100 if eff else 0
                print(f"  [{counter['done']:>4}/{n_total}] pass={counter['passed']} "
                      f"({rate_pct:.1f}% of effective) "
                      f"excl={counter['excluded']} err={counter['errors']} "
                      f"toks={counter['tokens']/1000:.1f}k "
                      f"({rate:.1f}/s, eta={eta:.0f}s)", file=sys.stderr)

        try:
            with (ThreadPoolExecutor(max_workers=args.concurrency) as ex,
                  ThreadPoolExecutor(max_workers=args.judge_concurrency) as judge_ex):
                futures = {
                    ex.submit(
                        call_probe, runner, p,
                        no_system_prompt=args.no_system_prompt,
                    ): p
                    for p in probes
                }
                judge_futures = []
                for fut in as_completed(futures):
                    cr = fut.result()
                    is_judge = cr["probe"].get("scorer", {}).get("type") == "judge_rubric"
                    if is_judge and args.llm_judge and not cr["err"]:
                        judge_futures.append(judge_ex.submit(
                            score_one, cr, policy, args.resp_cap, args.reasoning_cap,
                            run_context))
                    else:
                        record_result(score_one(
                            cr, policy, args.resp_cap, args.reasoning_cap,
                            run_context))
                for judge_future in as_completed(judge_futures):
                    record_result(judge_future.result())
        finally:
            detail_fh.close()

    retry_stats = retry_failed_results(
        results, probes_by_id, args, policy, run_context,
        api_request_limiter, detail_path,
    )
    counter = _counter_from_results(results)

    # Aggregate
    agg = aggregate(results)
    deferred_diag = transient_deferred_diagnostics(results)
    scope_issues = official_scope_issues(
        probe_suite_sha256=probe_suite_hash,
        selected_probe_sha256=selected_probe_hash,
        policy_sha256=policy_hash,
        canonical_probe_sha256=canonical_probe_hash,
        canonical_policy_sha256=canonical_policy_hash,
        llm_judge=args.llm_judge,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        reference_judge_base_url=reference_judge_base_url,
        judge_max_tokens=args.judge_max_tokens,
        response_cap=args.resp_cap,
        reasoning_cap=args.reasoning_cap,
        aggregate_result=agg,
    )
    deferred_diag["official_full_suite"] = not scope_issues
    deferred_diag["official_scope_issues"] = scope_issues
    if scope_issues:
        deferred_diag["score_status"] = "partial"
        deferred_diag["publishable"] = False
    agg["meta"] = {
        "benchmark_name": "ACE",
        "benchmark_version": BENCHMARK_VERSION,
        "scorer_version": SCORER_VERSION,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "evaluation_contract_sha256": evaluation_contract_hash,
        "evaluator_runtime_sha256": evaluator_runtime_hash,
        "evaluation_signature": run_context["evaluation_signature"],
        "evaluation_signature_payload": signature_payload,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "probes_dir": str(Path(args.probes_dir).resolve()),
        "probe_suite_sha256": probe_suite_hash,
        "selected_probe_sha256": selected_probe_hash,
        "policy_path": str(Path(args.policy).resolve()),
        "policy_sha256": policy_hash,
        "model_name": args.model_name, "base_url": args.base_url,
        "tag": tag, "temperature": effective_temperature,
        "max_tokens": args.max_tokens, "concurrency": args.concurrency,
        "request_timeout": args.request_timeout,
        "thinking_mode": args.thinking_mode,
        "no_system_prompt": args.no_system_prompt,
        "retry_failed": args.retry_failed,
        "retry_concurrency": retry_stats["concurrency"],
        "retry_request_timeout": retry_stats["request_timeout"],
        "retry": retry_stats,
        "candidate_sdk_max_retries": CANDIDATE_SDK_MAX_RETRIES,
        "candidate_request_attempts_per_phase": CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE,
        "max_candidate_requests_per_probe": max_candidate_requests_per_probe(
            args.retry_failed
        ),
        "judge_request_attempts_per_phase": JUDGE_REQUEST_ATTEMPTS_PER_PHASE,
        "max_judge_requests_per_probe": max_judge_requests_per_probe(
            args.retry_failed
        ),
        "total_candidate_request_attempts": counter["request_attempts"],
        "total_api_concurrency": args.total_api_concurrency,
        "judge_model": args.judge_model,
        "judge_concurrency": args.judge_concurrency,
        "judge_max_tokens": args.judge_max_tokens,
        "strict_judge": args.strict_judge,
        "strict_errors": args.strict_errors,
        "wall_seconds": round(time.time() - t_start, 1),
        "timestamp": ts, "n_errors": counter["errors"],
        "n_judge_deferred": counter["judge_deferred"],
        "total_completion_tokens": counter["tokens"],
        **deferred_diag,
    }
    agg_path = out_dir / f"summary_{tag}_{ts}.json"
    agg_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2,
                                   default=lambda x: round(x, 4) if isinstance(x, float) else x))

    print("\n=== ACE evaluation complete ===")
    print(f"  model:        {args.model_name}")
    print(f"  benchmark:    ACE {BENCHMARK_VERSION}")
    print(f"  total:        {agg['n_total']}  effective: {agg['n_eff']}  excluded: {agg['n_excluded']}")
    print(f"  passed:       {agg['n_pass']}  rate (eff): {agg['overall_pass_rate']}")
    print(f"  wall time:    {agg['meta']['wall_seconds']}s")
    print(f"  detail:       {detail_path}")
    print(f"  summary:      {agg_path}")
    if not deferred_diag["publishable"]:
        print(
            "  ⚠️  PARTIAL / NON-PUBLISHABLE: "
            f"{deferred_diag['n_transient_deferred']} transient result(s); "
            f"scope issues={scope_issues or 'none'}. Inspect "
            "meta.deferred_by_reason, meta.deferred_probe_ids, and "
            "meta.official_scope_issues.",
            file=sys.stderr,
        )
    if args.llm_judge and args.strict_judge and counter["judge_deferred"]:
        print(
            f"ERROR: {counter['judge_deferred']} LLM judge result(s) remain deferred; "
            "rerun with --resume after restoring the judge service, or use "
            "--no-strict-judge for an explicitly partial score.",
            file=sys.stderr,
        )
        sys.exit(3)
    if args.strict_errors and counter["errors"]:
        print(
            f"ERROR: {counter['errors']} candidate-model request(s) remain deferred; "
            "rerun with --resume after resolving the model endpoint, or use "
            "--no-strict-errors for an explicitly partial score.",
            file=sys.stderr,
        )
        sys.exit(4)


if __name__ == "__main__":
    main()
