"""ACE release-result compatibility helpers used by batch/report entry points."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "benchmark"

import sys

sys.path.insert(0, str(BENCHMARK_DIR))
from eval.judge import judge_endpoint, PROMPT_VERSION as JUDGE_PROMPT_VERSION  # noqa: E402
from eval.version import (  # noqa: E402
    BENCHMARK_VERSION,
    EVALUATION_CONTRACT_VERSION,
    OFFICIAL_DISABLED_PROBES,
    OFFICIAL_EFFECTIVE_PROBES,
    OFFICIAL_TOTAL_PROBES,
    SCORER_VERSION,
)
from eval.run_eval import (  # noqa: E402
    DEFAULT_CANDIDATE_CONCURRENCY,
    DEFAULT_JUDGE_CONCURRENCY,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_LLM_JUDGE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_CAP,
    DEFAULT_RESPONSE_CAP,
    DEFAULT_SKIP_CATEGORIES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOTAL_API_CONCURRENCY,
    build_evaluation_contract,
    evaluation_signature,
    evaluator_source_paths,
    source_bundle_sha256,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RETRYABLE_STRATEGIES = {
    "model_call_error",
    "judge_deferred",
    "llm_judge_error",
    "scorer_error",
}


@dataclass(frozen=True)
class DetailValidation:
    compatible: bool
    reason: str
    completed_probe_ids: frozenset[str] = frozenset()
    retryable_probe_ids: frozenset[str] = frozenset()
    excluded_probe_ids: frozenset[str] = frozenset()


def probe_manifest_sha256(probes: Any) -> str:
    """Hash materialized probes exactly as the evaluation runner does."""
    values = probes.values() if isinstance(probes, dict) else probes
    by_id = {probe["probe_id"]: probe for probe in values}
    digest = hashlib.sha256()
    for probe_id in sorted(by_id):
        digest.update(json.dumps(
            by_id[probe_id],
            ensure_ascii=False,
            sort_keys=True,
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


@lru_cache(maxsize=1)
def current_release_hashes() -> dict[str, str]:
    """Return current full-suite data, policy, and evaluator source hashes."""
    probes: dict[str, Any] = {}
    for path in sorted((BENCHMARK_DIR / "probes").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                probe = json.loads(line)
                probes[probe["probe_id"]] = probe
    return {
        "probe_suite_sha256": probe_manifest_sha256(probes),
        "policy_sha256": file_sha256(
            BENCHMARK_DIR / "policy" / "atom_intents.yaml"
        ),
        "evaluator_runtime_sha256": source_bundle_sha256(
            evaluator_source_paths()
        ),
    }


def validate_summary_data(
    summary: Any,
    *,
    expected_signature: str | None = None,
    require_publishable: bool = True,
) -> tuple[bool, str]:
    """Validate that a summary belongs to the current audited release."""
    if not isinstance(summary, dict):
        return False, "summary is not an object"
    meta = summary.get("meta")
    if not isinstance(meta, dict):
        return False, "missing meta"

    required_meta = {
        "benchmark_version": BENCHMARK_VERSION,
        "scorer_version": SCORER_VERSION,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
    }
    for key, expected in required_meta.items():
        if meta.get(key) != expected:
            return False, f"{key}={meta.get(key)!r}, expected {expected!r}"

    signature = meta.get("evaluation_signature")
    if not isinstance(signature, str) or not _SHA256_RE.fullmatch(signature):
        return False, "missing or invalid evaluation_signature"
    if expected_signature is not None and signature != expected_signature:
        return False, "evaluation_signature does not match this invocation"

    payload = meta.get("evaluation_signature_payload")
    if not isinstance(payload, dict):
        return False, "missing evaluation_signature_payload"
    if evaluation_signature(payload) != signature:
        return False, "evaluation_signature does not authenticate its payload"
    for key, expected in required_meta.items():
        if payload.get(key) != expected:
            return False, f"signature payload has incompatible {key}"

    current_hashes = current_release_hashes()
    for key, expected in current_hashes.items():
        if payload.get(key) != expected:
            return False, f"signature payload does not match current {key}"
    if payload.get("selected_probe_sha256") != current_hashes["probe_suite_sha256"]:
        return False, "signature payload is not the current unfiltered full suite"
    judge = payload.get("judge")
    if not isinstance(judge, dict):
        return False, "signature payload is missing judge settings"
    official_judge = {
        "enabled": DEFAULT_LLM_JUDGE,
        "model": DEFAULT_JUDGE_MODEL,
        "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
    }
    for key, expected in official_judge.items():
        if judge.get(key) != expected:
            return False, f"signature payload has non-official judge.{key}"
    judge_base_url = judge.get("base_url")
    reference_judge_base_url = os.environ.get("ACE_JUDGE_BASE_URL", "")
    if (not reference_judge_base_url or not isinstance(judge_base_url, str)
            or judge_endpoint(judge_base_url) != judge_endpoint(reference_judge_base_url)):
        return False, "signature payload has non-official judge.base_url"
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        return False, "signature payload is missing candidate settings"
    if candidate.get("response_cap") != DEFAULT_RESPONSE_CAP:
        return False, "signature payload has non-official candidate.response_cap"
    if candidate.get("reasoning_cap") != DEFAULT_REASONING_CAP:
        return False, "signature payload has non-official candidate.reasoning_cap"

    contract_hash = meta.get("evaluation_contract_sha256")
    runtime_hash = meta.get("evaluator_runtime_sha256")
    if not isinstance(contract_hash, str) or not _SHA256_RE.fullmatch(contract_hash):
        return False, "missing or invalid evaluation_contract_sha256"
    if not isinstance(runtime_hash, str) or not _SHA256_RE.fullmatch(runtime_hash):
        return False, "missing or invalid evaluator_runtime_sha256"
    if payload.get("evaluation_contract_sha256") != contract_hash:
        return False, "contract hash differs between summary and signature payload"
    if payload.get("evaluator_runtime_sha256") != runtime_hash:
        return False, "runtime hash differs between summary and signature payload"
    contract_payload, canonical_contract_hash = build_evaluation_contract(
        probe_suite_sha256=payload.get("probe_suite_sha256"),
        selected_probe_sha256=payload.get("selected_probe_sha256"),
        policy_sha256=payload.get("policy_sha256"),
        evaluator_runtime_sha256=payload.get("evaluator_runtime_sha256"),
    )
    signed_contract_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"evaluation_contract_sha256", "candidate", "judge"}
    }
    if signed_contract_payload != contract_payload:
        return False, "evaluation contract payload is non-canonical"
    if canonical_contract_hash != contract_hash:
        return False, "evaluation_contract_sha256 does not authenticate its payload"

    if (summary.get("n_total") != OFFICIAL_TOTAL_PROBES
            or summary.get("n_eff") != OFFICIAL_EFFECTIVE_PROBES
            or summary.get("n_excluded") != OFFICIAL_DISABLED_PROBES):
        return False, (
            f"not an official {OFFICIAL_TOTAL_PROBES}-total/"
            f"{OFFICIAL_EFFECTIVE_PROBES}-effective full-suite result"
        )
    if meta.get("official_full_suite") is not True:
        return False, "summary is not marked official_full_suite"
    if meta.get("official_scope_issues") != []:
        return False, "summary has official_scope_issues"

    if require_publishable:
        if meta.get("publishable") is not True:
            return False, "result is partial/non-publishable"
        if meta.get("score_status") != "complete":
            return False, "score_status is not complete"
    return True, "compatible"


def load_compatible_summary(
    path: str | Path,
    *,
    expected_signature: str | None = None,
    require_publishable: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    try:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"cannot read summary: {exc}"
    compatible, reason = validate_summary_data(
        summary,
        expected_signature=expected_signature,
        require_publishable=require_publishable,
    )
    return (summary if compatible else None), reason


def validate_detail_file(
    path: str | Path,
    *,
    expected_signature: str,
    expected_contract_sha256: str | None = None,
    expected_runtime_sha256: str | None = None,
) -> DetailValidation:
    """Require every detail row to belong to one current-release run."""
    completed: set[str] = set()
    retryable: set[str] = set()
    excluded: set[str] = set()
    saw_record = False
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                saw_record = True
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    return DetailValidation(False, f"line {line_no}: {exc}")
                expected_fields = {
                    "benchmark_version": BENCHMARK_VERSION,
                    "scorer_version": SCORER_VERSION,
                    "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "evaluation_signature": expected_signature,
                }
                for key, expected in expected_fields.items():
                    if record.get(key) != expected:
                        return DetailValidation(
                            False,
                            f"line {line_no}: incompatible {key}",
                        )
                contract_hash = record.get("evaluation_contract_sha256")
                runtime_hash = record.get("evaluator_runtime_sha256")
                if (not isinstance(contract_hash, str)
                        or not _SHA256_RE.fullmatch(contract_hash)):
                    return DetailValidation(
                        False, f"line {line_no}: invalid evaluation_contract_sha256"
                    )
                if (not isinstance(runtime_hash, str)
                        or not _SHA256_RE.fullmatch(runtime_hash)):
                    return DetailValidation(
                        False, f"line {line_no}: invalid evaluator_runtime_sha256"
                    )
                if (expected_contract_sha256 is not None
                        and contract_hash != expected_contract_sha256):
                    return DetailValidation(
                        False, f"line {line_no}: contract hash mismatch"
                    )
                if (expected_runtime_sha256 is not None
                        and runtime_hash != expected_runtime_sha256):
                    return DetailValidation(
                        False, f"line {line_no}: runtime hash mismatch"
                    )
                probe_id = record.get("probe_id")
                if not isinstance(probe_id, str) or not probe_id:
                    return DetailValidation(False, f"line {line_no}: missing probe_id")
                is_retryable = bool(record.get("error")) or (
                    record.get("score_strategy") in _RETRYABLE_STRATEGIES
                )
                if is_retryable:
                    retryable.add(probe_id)
                    completed.discard(probe_id)
                else:
                    completed.add(probe_id)
                    retryable.discard(probe_id)
                if record.get("excluded_from_score"):
                    excluded.add(probe_id)
                else:
                    excluded.discard(probe_id)
    except (OSError, UnicodeError) as exc:
        return DetailValidation(False, f"cannot read detail: {exc}")

    if not saw_record:
        return DetailValidation(False, "detail file is empty")
    return DetailValidation(
        True,
        "compatible",
        frozenset(completed),
        frozenset(retryable),
        frozenset(excluded),
    )
