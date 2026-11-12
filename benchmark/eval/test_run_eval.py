"""Offline regression tests for runner exclusion and resume semantics."""

import json
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace


ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))

from eval.run_eval import (
    CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE, CANDIDATE_SDK_MAX_RETRIES,
    JUDGE_REQUEST_ATTEMPTS_PER_PHASE,
    call_probe, evaluation_signature, find_partial_results,
    is_retryable_result, official_scope_issues, probe_manifest_sha256,
    max_candidate_requests_per_probe, max_judge_requests_per_probe,
    Runner, score_one, source_bundle_sha256,
    transient_deferred_diagnostics,
)
from eval.version import BENCHMARK_VERSION
import eval.run_eval as run_module

assert BENCHMARK_VERSION == "2.6.4"


probe = {
    "probe_id": "test.runner.L1.001",
    "canonical_id": "test.runner",
    "category": "test",
    "level": "L1",
    "prompt": "test",
    "ground_truth": "ok",
    "scorer": {"type": "exact_match_string", "expected": "ok"},
}
record = score_one(
    {
        "probe": probe,
        "out": {
            "content": "",
            "reasoning": "",
            "finish_reason": None,
            "completion_tokens": None,
        },
        "err": "HTTP 503",
        "latency_s": 0.1,
    },
    {},
)
assert record["score_strategy"] == "model_call_error"
assert record["deferred"] and record["excluded_from_score"]
assert record["benchmark_version"] == "2.6.4"

signed_context = {
    "evaluation_signature": "1" * 64,
    "evaluation_contract_sha256": "2" * 64,
    "evaluator_runtime_sha256": "3" * 64,
    "probe_suite_sha256": "4" * 64,
    "selected_probe_sha256": "5" * 64,
    "policy_sha256": "6" * 64,
}
signed_record = score_one(
    {
        "probe": probe,
        "out": {
            "content": "ok", "reasoning": "", "finish_reason": "stop",
            "completion_tokens": 1,
        },
        "err": None,
        "latency_s": 0.1,
    },
    {},
    run_context=signed_context,
)
for key, expected in signed_context.items():
    assert signed_record[key] == expected

assert probe_manifest_sha256([probe]) == probe_manifest_sha256({probe["probe_id"]: probe})
changed_probe = {**probe, "prompt": "changed"}
assert probe_manifest_sha256([probe]) != probe_manifest_sha256([changed_probe])
assert evaluation_signature({"a": 1, "b": 2}) == evaluation_signature({"b": 2, "a": 1})
assert source_bundle_sha256([Path(__file__)]) != source_bundle_sha256([
    ACE / "eval" / "version.py"
])
assert CANDIDATE_SDK_MAX_RETRIES == 0
assert CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE == 1
assert JUDGE_REQUEST_ATTEMPTS_PER_PHASE == 1
assert max_candidate_requests_per_probe(True) == 2
assert max_candidate_requests_per_probe(False) == 1
assert max_judge_requests_per_probe(True) == 2
assert max_judge_requests_per_probe(False) == 1


# The first-pass judge must also make only one request. A judge-deferred record
# may receive exactly one more request in retry_failed_results.
judge_config_kwargs = {}
original_configure_judge = run_module.configure_judge
run_module.configure_judge = lambda **kwargs: judge_config_kwargs.update(kwargs)
try:
    run_module.configure_runtime_judge(
        enabled=True,
        base_url="https://judge.example/v1",
        api_key="key",
        model="judge",
        timeout=30,
        max_tokens=100,
        request_limiter=None,
    )
finally:
    run_module.configure_judge = original_configure_judge
assert judge_config_kwargs["retries"] == 1


# The OpenAI SDK must never add hidden retries beneath the runner policy.
openai_kwargs = {}
fake_openai_module = ModuleType("openai")


class FakeOpenAI:
    def __init__(self, **kwargs):
        openai_kwargs.update(kwargs)


fake_openai_module.OpenAI = FakeOpenAI
saved_openai_module = sys.modules.get("openai")
sys.modules["openai"] = fake_openai_module
try:
    sdk_runner = Runner(
        base_url="https://candidate.example/v1",
        api_key="key",
        model_name="candidate",
    )
finally:
    if saved_openai_module is None:
        del sys.modules["openai"]
    else:
        sys.modules["openai"] = saved_openai_module
assert openai_kwargs["max_retries"] == 0
assert sdk_runner.client is not None


class MustNotBeCalled:
    def call(self, *args, **kwargs):
        raise AssertionError("disabled probe made a candidate API call")


disabled_probe = {**probe, "enabled": False}
disabled_call = call_probe(MustNotBeCalled(), disabled_probe)
disabled_record = score_one(disabled_call, {})
assert disabled_record["score_strategy"] == "probe_disabled"
assert disabled_record["deferred"] and disabled_record["excluded_from_score"]
assert disabled_record["latency_s"] == 0.0
assert disabled_record["candidate_request_attempts"] == 0


class AlwaysFailRunner:
    def __init__(self):
        self.calls = 0

    def call(self, *_args, **_kwargs):
        self.calls += 1
        raise RuntimeError("upstream unavailable")


always_fail = AlwaysFailRunner()
failed_call = call_probe(always_fail, probe)
assert always_fail.calls == 1
assert failed_call["candidate_request_attempts"] == 1
assert failed_call["err"] == "RuntimeError: upstream unavailable"


class CaptureRunner:
    def __init__(self):
        self.calls = []

    def call(self, prompt, system_prompt=""):
        self.calls.append((prompt, system_prompt))
        return {
            "content": "ok", "reasoning": "", "finish_reason": "stop",
            "completion_tokens": 1,
        }


system_probe = {**probe, "system_prompt": "system rules"}
capture = CaptureRunner()
call_probe(capture, system_probe, no_system_prompt=True)
assert capture.calls == [("system rules\n\ntest", "")]
capture = CaptureRunner()
call_probe(capture, system_probe, no_system_prompt=False)
assert capture.calls == [("test", "system rules")]


class FakeEndpoint:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


chat_response = Obj(
    choices=[Obj(message=Obj(content="final", reasoning_content="reason"), finish_reason="stop")],
    usage=Obj(completion_tokens=2),
)
chat_endpoint = FakeEndpoint(chat_response)
fake_runner = Runner.__new__(Runner)
fake_runner.client = Obj(
    chat=Obj(completions=chat_endpoint),
)
fake_runner.model_name = "demo"
fake_runner.max_tokens = 100
fake_runner.temperature = 0
fake_runner.request_limiter = None
fake_runner.think_prefix = True
standard = fake_runner.call("question", "rules")
assert standard["content"] == "final" and standard["reasoning"] == "reason"
assert chat_endpoint.kwargs["temperature"] == 0
assert chat_endpoint.kwargs["messages"][-1] == {
    "role": "assistant", "content": "<think>\n", "prefix": True,
}
fake_runner.temperature = None
fake_runner.think_prefix = False
fake_runner.call("question", "rules")
assert "temperature" not in chat_endpoint.kwargs
fake_runner.temperature = 1.0
fake_runner.call("question", "rules")
assert chat_endpoint.kwargs["temperature"] == 1.0
assert is_retryable_result({"score_strategy": "model_call_error"})
assert is_retryable_result({"score_strategy": "judge_deferred"})
assert not is_retryable_result({"score_strategy": "exact_match", "passed": False})

diag = transient_deferred_diagnostics([
    disabled_record,
    {"probe_id": "judge-bad", "score_strategy": "judge_deferred"},
    {"probe_id": "stable", "score_strategy": "exact_match"},
])
assert diag["score_status"] == "partial" and not diag["publishable"]
assert diag["n_transient_deferred"] == 1
assert diag["deferred_probe_ids"] == ["judge-bad"]
assert diag["deferred_by_reason"] == {"judge_deferred": 1}

official_counts = {"n_total": 2565, "n_eff": 2489, "n_excluded": 76}
scope_args = {
    "probe_suite_sha256": "a" * 64,
    "selected_probe_sha256": "a" * 64,
    "policy_sha256": "b" * 64,
    "canonical_probe_sha256": "a" * 64,
    "canonical_policy_sha256": "b" * 64,
    "llm_judge": True,
    "judge_model": "deepseek-v4-flash",
    "judge_base_url": "https://judge.example/v1",
    "reference_judge_base_url": "https://judge.example",
    "judge_max_tokens": 4096,
    "response_cap": 100000,
    "reasoning_cap": 100000,
    "aggregate_result": official_counts,
}
assert official_scope_issues(**scope_args) == []
for reference in ("", "https://another-judge.example"):
    issues = official_scope_issues(**{**scope_args, "reference_judge_base_url": reference})
    assert "noncanonical_judge_endpoint" in issues
subset_issues = official_scope_issues(
    **{**scope_args, "selected_probe_sha256": "c" * 64,
       "aggregate_result": {**official_counts, "n_total": 100}}
)
assert "selected_probe_subset_or_modified_scope" in subset_issues
assert any(issue.startswith("n_total_mismatch:") for issue in subset_issues)

with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    detail = out / "results_demo_123.jsonl"
    rows = [
        {"probe_id": "stable", "score_strategy": "exact_match", "deferred": False},
        {"probe_id": "judge", "score_strategy": "judge_deferred", "deferred": True},
        {"probe_id": "model", "score_strategy": "model_call_error", "deferred": True,
         "error": "timeout"},
        {"probe_id": "media", "score_strategy": "multimodal_defer", "deferred": True},
    ]
    detail.write_text("".join(json.dumps(x) + "\n" for x in rows))
    path, completed = find_partial_results(out, "demo")
    assert path == detail
    assert completed == {"stable", "media"}, completed

    # A summary does not make transient failures permanent; it remains
    # resumable until a later successful record for the same probe appears.
    (out / "summary_demo_123.json").write_text("{}")
    path, completed = find_partial_results(out, "demo")
    assert path == detail and completed == {"stable", "media"}

    with detail.open("a") as fh:
        fh.write(json.dumps({
            "probe_id": "judge", "score_strategy": "llm_judge_pass",
            "deferred": False,
        }) + "\n")
        fh.write(json.dumps({
            "probe_id": "model", "score_strategy": "exact_match",
            "deferred": False,
        }) + "\n")
    path, completed = find_partial_results(out, "demo")
    assert path is None and completed == set()

with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    # Old rows without a signature, and rows from a different contract, must
    # never be appended to a 2.6.4 run.
    (out / "results_old_100.jsonl").write_text(
        json.dumps({"probe_id": "legacy", "score_strategy": "exact_match"}) + "\n"
    )
    path, completed = find_partial_results(out, "old", expected_signature="new")
    assert path is None and completed == set()

    (out / "results_wrong_101.jsonl").write_text(
        json.dumps({
            "probe_id": "wrong", "score_strategy": "exact_match",
            "evaluation_signature": "old",
        }) + "\n"
    )
    path, completed = find_partial_results(out, "wrong", expected_signature="new")
    assert path is None and completed == set()

    matching = out / "results_matching_102.jsonl"
    matching.write_text(json.dumps({
        "probe_id": "same", "score_strategy": "exact_match",
        "evaluation_signature": "new",
    }) + "\n")
    path, completed = find_partial_results(out, "matching", expected_signature="new")
    assert path == matching and completed == {"same"}


# Retry workers retain concurrent endpoint calls, but deterministic scoring
# must happen on the main thread where SIGALRM parser guards are effective.
with tempfile.TemporaryDirectory() as td:
    original_runner = run_module.Runner
    original_call_probe = run_module.call_probe
    original_score_one = run_module.score_one
    main_thread = threading.current_thread()
    call_threads = []
    score_threads = {}
    score_calls = []
    retry_runner_kwargs = []

    class FakeRetryRunner:
        def __init__(self, **kwargs):
            retry_runner_kwargs.append(kwargs)

    def fake_call_probe(_runner, retry_probe, *, no_system_prompt=False):
        assert not no_system_prompt
        call_threads.append(threading.current_thread())
        return {
            "probe": retry_probe,
            "out": {
                "content": "ok", "reasoning": "", "finish_reason": "stop",
                "completion_tokens": 1,
            },
            "err": None,
            "latency_s": 0.01,
            "candidate_request_attempts": 1,
        }

    def fake_score_one(call_result, *_args, **_kwargs):
        retry_probe = call_result["probe"]
        score_calls.append(retry_probe["probe_id"])
        score_threads[retry_probe["probe_id"]] = threading.current_thread()
        judge = retry_probe["scorer"]["type"] == "judge_rubric"
        return {
            "probe_id": retry_probe["probe_id"],
            "passed": True,
            "excluded_from_score": False,
            "score_strategy": "llm_judge_pass" if judge else "exact_match",
            "deferred": False,
            "candidate_request_attempts": call_result.get(
                "candidate_request_attempts", 0
            ),
        }

    run_module.Runner = FakeRetryRunner
    run_module.call_probe = fake_call_probe
    run_module.score_one = fake_score_one
    try:
        deterministic = {**probe, "probe_id": "retry-deterministic"}
        judge = {
            **probe,
            "probe_id": "retry-judge",
            "scorer": {"type": "judge_rubric", "rubric": {}},
        }
        retry_results = [
            {
                "probe_id": deterministic["probe_id"],
                "score_strategy": "model_call_error",
                "error": "timeout",
                "candidate_request_attempts": 1,
            },
            {
                "probe_id": judge["probe_id"],
                "score_strategy": "judge_deferred",
                "error": "",
                "response": "ok",
                "candidate_request_attempts": 1,
            },
        ]
        retry_args = SimpleNamespace(
            retry_concurrency=2,
            retry_request_timeout=30,
            request_timeout=30,
            retry_failed=True,
            base_url="https://candidate.example/v1",
            api_key="key",
            model_name="candidate",
            max_tokens=100,
            temperature=0.0,
            omit_temperature=True,
            thinking_mode=False,
            no_system_prompt=False,
            resp_cap=1000,
            reasoning_cap=1000,
            llm_judge=True,
        )
        retry_stats = run_module.retry_failed_results(
            retry_results,
            {deterministic["probe_id"]: deterministic, judge["probe_id"]: judge},
            retry_args,
            {},
            {},
            None,
            Path(td) / "retry.jsonl",
        )
        assert retry_stats["recovered"] == 2
        assert retry_stats["candidate_sdk_max_retries"] == 0
        assert retry_stats["candidate_request_attempts_per_phase"] == 1
        assert retry_stats["max_candidate_requests_per_probe"] == 2
        assert retry_stats["judge_request_attempts_per_phase"] == 1
        assert retry_stats["max_judge_requests_per_probe"] == 2
        assert retry_results[0]["candidate_request_attempts"] == 2
        # Judge-only retry reuses the first candidate response.
        assert retry_results[1]["candidate_request_attempts"] == 1
        assert retry_runner_kwargs[0]["omit_temperature"] is True
        assert call_threads and all(thread is not main_thread for thread in call_threads)
        assert score_threads[deterministic["probe_id"]] is main_thread
        assert score_threads[judge["probe_id"]] is not main_thread
        assert score_calls.count(judge["probe_id"]) == 1

        # Existing --no-retry-failed remains the compatible way to select a
        # strict one-request ceiling.
        call_threads.clear()
        retry_args.retry_failed = False
        no_retry_results = [{
            "probe_id": deterministic["probe_id"],
            "score_strategy": "model_call_error",
            "error": "timeout",
            "candidate_request_attempts": 1,
        }]
        no_retry_stats = run_module.retry_failed_results(
            no_retry_results,
            {deterministic["probe_id"]: deterministic},
            retry_args,
            {},
            {},
            None,
            Path(td) / "no-retry.jsonl",
        )
        assert no_retry_stats["attempted"] == 0
        assert no_retry_stats["max_candidate_requests_per_probe"] == 1
        assert no_retry_stats["max_judge_requests_per_probe"] == 1
        assert not call_threads
        assert no_retry_results[0]["candidate_request_attempts"] == 1
    finally:
        run_module.Runner = original_runner
        run_module.call_probe = original_call_probe
        run_module.score_one = original_score_one

print("run_eval offline tests: PASS")
