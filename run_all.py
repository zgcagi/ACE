#!/usr/bin/env python3
"""
ACE 自动化评测入口
读取 models.yaml，对尚未评测的模型运行评测，然后生成 dashboard 数据。
支持多 provider 配置和断点续传。

用法:
  python3 run_all.py                    # 评测所有未完成的模型 + 生成 dashboard
  python3 run_all.py --eval-only        # 仅评测，不生成 dashboard
  python3 run_all.py --build-only       # 仅根据已有结果生成 dashboard
  python3 run_all.py --model xxx        # 仅评测指定模型（可用逗号分隔多个）
  python3 run_all.py --force            # 强制重新评测（即使已有结果）
  python3 run_all.py --resume           # 从断点继续（默认行为）
  python3 run_all.py --provider xxx     # 仅评测指定 provider 的模型
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

from release_results import (
    BENCHMARK_VERSION,
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
    file_sha256,
    load_compatible_summary,
    probe_manifest_sha256,
    evaluator_source_paths,
    source_bundle_sha256,
    validate_detail_file,
)
from eval.run_eval import (
    CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE,
    CANDIDATE_SDK_MAX_RETRIES,
    JUDGE_REQUEST_ATTEMPTS_PER_PHASE,
    max_candidate_requests_per_probe,
    max_judge_requests_per_probe,
)

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "benchmark"
RESULTS_DIR = ROOT / "results"
DASHBOARD_DIR = ROOT / "dashboard"
POLICY_PATH = EVAL_DIR / "policy" / "atom_intents.yaml"

def load_config():
    import yaml

    with open(ROOT / "models.yaml") as f:
        return yaml.safe_load(f)


def resolve_provider(model_cfg, config):
    """Resolve base_url and api_key for a model, supporting both
    direct override and provider reference."""
    providers = config.get("providers", {})
    default_provider = config.get("default_provider")

    # Direct override takes priority
    base_url = model_cfg.get("base_url")
    api_key = model_cfg.get("api_key")

    if base_url and api_key:
        return base_url, api_key

    # Try provider reference
    provider_name = model_cfg.get("provider", default_provider)
    if provider_name and provider_name in providers:
        prov = providers[provider_name]
        if not base_url:
            base_url = prov["base_url"]
        if not api_key:
            api_key = prov["api_key"]
        return base_url, api_key

    # Legacy fallback: top-level api section
    if "api" in config:
        api = config["api"]
        if not base_url:
            base_url = api.get("base_url", "")
        if not api_key:
            api_key = api.get("api_key", "")

    return base_url or "", api_key or ""


def _load_probes():
    probes = {}
    for path in sorted((EVAL_DIR / "probes").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                probe = json.loads(line)
                probes[probe["probe_id"]] = probe
    return probes


def resolve_eval_settings(model_cfg, global_cfg, base_url):
    """Resolve the exact non-secret settings passed to run_eval.py."""
    eval_cfg = global_cfg.get("eval") or {}
    judge_cfg = global_cfg.get("judge") or {}
    # The runner records unavailable media probes as exclusions.
    official = {
        "concurrency": DEFAULT_CANDIDATE_CONCURRENCY,
        "total_api_concurrency": DEFAULT_TOTAL_API_CONCURRENCY,
        "judge_concurrency": DEFAULT_JUDGE_CONCURRENCY,
        "skip_categories": DEFAULT_SKIP_CATEGORIES,
        "llm_judge": DEFAULT_LLM_JUDGE,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "judge_base_url": os.environ.get("ACE_JUDGE_BASE_URL", ""),
        "judge_max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
        "response_cap": DEFAULT_RESPONSE_CAP,
        "reasoning_cap": DEFAULT_REASONING_CAP,
    }
    ignored_overrides = []
    for key, expected in official.items():
        if key == "judge_model":
            configured = judge_cfg.get("model", eval_cfg.get("judge_model"))
        elif key == "judge_base_url":
            configured = judge_cfg.get("base_url", eval_cfg.get("judge_base_url"))
        else:
            configured = eval_cfg.get(key)
        if configured is not None and configured != expected:
            ignored_overrides.append(f"{key}={configured!r}")

    omit_temperature = model_cfg.get(
        "omit_temperature", eval_cfg.get("omit_temperature", False)
    )
    return {
        "base_url": base_url,
        "model_name": model_cfg["id"],
        "max_tokens": int(model_cfg.get(
            "max_tokens", eval_cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
        )),
        "temperature": (
            None if omit_temperature
            else float(eval_cfg.get("temperature", DEFAULT_TEMPERATURE))
        ),
        "concurrency": official["concurrency"],
        "total_api_concurrency": official["total_api_concurrency"],
        "request_timeout": int(eval_cfg.get("request_timeout", 300)),
        "retry_failed": bool(eval_cfg.get("retry_failed", True)),
        "retry_concurrency": int(eval_cfg.get("retry_concurrency", 8)),
        "retry_request_timeout": int(eval_cfg.get("retry_request_timeout", 300)),
        "thinking_mode": bool(model_cfg.get(
            "thinking_mode", eval_cfg.get("thinking_mode", False)
        )),
        "no_system_prompt": bool(model_cfg.get(
            "no_system_prompt", eval_cfg.get("no_system_prompt", False)
        )),
        "skip_categories": official["skip_categories"],
        "llm_judge": official["llm_judge"],
        "judge_base_url": official["judge_base_url"],
        "judge_api_key": (
            judge_cfg.get("api_key")
            or eval_cfg.get("judge_api_key")
            or os.environ.get("ACE_JUDGE_API_KEY", "")
        ),
        "judge_model": official["judge_model"],
        "judge_concurrency": official["judge_concurrency"],
        "judge_max_tokens": official["judge_max_tokens"],
        "response_cap": official["response_cap"],
        "reasoning_cap": official["reasoning_cap"],
        "ignored_legacy_overrides": ignored_overrides,
    }


def build_signature_payload(settings):
    """Build the same release/configuration payload as run_eval.py."""
    probes_by_id = _load_probes()
    skipped = {
        category.strip()
        for category in settings["skip_categories"].split(",")
        if category.strip()
    }
    selected = [
        probe for probe in probes_by_id.values()
        if probe.get("category") not in skipped
    ]
    runtime_hash = source_bundle_sha256(evaluator_source_paths())
    contract_payload, contract_hash = build_evaluation_contract(
        probe_suite_sha256=probe_manifest_sha256(probes_by_id),
        selected_probe_sha256=probe_manifest_sha256(selected),
        policy_sha256=file_sha256(POLICY_PATH),
        evaluator_runtime_sha256=runtime_hash,
    )
    return {
        **contract_payload,
        "evaluation_contract_sha256": contract_hash,
        "candidate": {
            "base_url": settings["base_url"].rstrip("/"),
            "model": settings["model_name"],
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
            "response_cap": settings["response_cap"],
            "reasoning_cap": settings["reasoning_cap"],
            "thinking_mode": settings["thinking_mode"],
            "no_system_prompt": settings["no_system_prompt"],
            "candidate_sdk_max_retries": CANDIDATE_SDK_MAX_RETRIES,
            "candidate_request_attempts_per_phase": CANDIDATE_REQUEST_ATTEMPTS_PER_PHASE,
            "max_candidate_requests_per_probe": max_candidate_requests_per_probe(
                settings["retry_failed"]
            ),
        },
        "judge": {
            "enabled": settings["llm_judge"],
            "base_url": settings["judge_base_url"].rstrip("/"),
            "model": settings["judge_model"],
            "max_tokens": settings["judge_max_tokens"],
            "request_attempts_per_phase": JUDGE_REQUEST_ATTEMPTS_PER_PHASE,
            "max_requests_per_probe": max_judge_requests_per_probe(
                settings["retry_failed"]
            ),
        },
    }


def find_existing_results(tag, expected_signature=None):
    """Find a complete, publishable result from this exact ACE release/run."""
    pattern = str(RESULTS_DIR / f"summary_{tag}_*.json")
    for summary_file in reversed(sorted(glob.glob(pattern))):
        summary, _ = load_compatible_summary(
            summary_file, expected_signature=expected_signature
        )
        if summary is None:
            continue
        timestamp = Path(summary_file).stem.rsplit("_", 1)[-1]
        detail_file = RESULTS_DIR / f"results_{tag}_{timestamp}.jsonl"
        detail = validate_detail_file(
            detail_file,
            expected_signature=summary["meta"]["evaluation_signature"],
            expected_contract_sha256=summary["meta"]["evaluation_contract_sha256"],
            expected_runtime_sha256=summary["meta"]["evaluator_runtime_sha256"],
        )
        if not detail.compatible or detail.retryable_probe_ids:
            continue
        if len(detail.completed_probe_ids) != summary.get("n_total"):
            continue
        if len(detail.excluded_probe_ids) != summary.get("n_excluded"):
            continue
        return summary_file
    return None


def _summary_fingerprints(tag):
    """Snapshot summary content so post-run success cannot reuse an old file."""
    pattern = str(RESULTS_DIR / f"summary_{tag}_*.json")
    fingerprints = {}
    for path_str in glob.glob(pattern):
        path = Path(path_str)
        try:
            fingerprints[str(path)] = file_sha256(path)
        except OSError:
            continue
    return fingerprints


def _new_summary_diagnostic(tag, before, expected_signature):
    """Explain the newest summary written or changed by the child runner."""
    pattern = str(RESULTS_DIR / f"summary_{tag}_*.json")
    for path_str in reversed(sorted(glob.glob(pattern))):
        path = Path(path_str)
        try:
            digest = file_sha256(path)
        except OSError as exc:
            return f"{path.name}: cannot read ({exc})"
        if before.get(str(path)) == digest:
            continue
        _, reason = load_compatible_summary(
            path, expected_signature=expected_signature
        )
        return f"{path.name}: {reason}"
    return "runner wrote no new or changed summary"


def find_partial_results(
    tag, expected_signature, expected_contract_sha256=None,
    expected_runtime_sha256=None,
):
    """Find a resumable detail file for this exact invocation."""
    pattern = str(RESULTS_DIR / f"results_{tag}_*.jsonl")
    jsonl_files = sorted(glob.glob(pattern))
    for jf in reversed(jsonl_files):
        detail = validate_detail_file(
            jf,
            expected_signature=expected_signature,
            expected_contract_sha256=expected_contract_sha256,
            expected_runtime_sha256=expected_runtime_sha256,
        )
        if not detail.compatible:
            continue
        ts_part = jf.rsplit("_", 1)[-1].replace(".jsonl", "")
        summary_file = RESULTS_DIR / f"summary_{tag}_{ts_part}.json"
        summary, _ = load_compatible_summary(
            summary_file,
            expected_signature=expected_signature,
            require_publishable=True,
        )
        if summary is None or detail.retryable_probe_ids:
            return jf, len(detail.completed_probe_ids)
    return None, 0


def build_eval_command(model_cfg, global_cfg, *, force=False):
    """Return command/settings/signature without launching provider calls."""
    base_url, api_key = resolve_provider(model_cfg, global_cfg)
    settings = resolve_eval_settings(model_cfg, global_cfg, base_url)
    settings["candidate_api_key"] = api_key
    signature_payload = build_signature_payload(settings)
    signature = evaluation_signature(signature_payload)
    cmd = [
        sys.executable, str(EVAL_DIR / "eval" / "run_eval.py"),
        "--base-url", settings["base_url"],
        "--model-name", settings["model_name"],
        "--max-tokens", str(settings["max_tokens"]),
        "--concurrency", str(settings["concurrency"]),
        "--total-api-concurrency", str(settings["total_api_concurrency"]),
        "--judge-concurrency", str(settings["judge_concurrency"]),
        "--judge-base-url", settings["judge_base_url"],
        "--judge-model", settings["judge_model"],
        "--judge-max-tokens", str(settings["judge_max_tokens"]),
        "--resp-cap", str(settings["response_cap"]),
        "--reasoning-cap", str(settings["reasoning_cap"]),
        "--llm-judge" if settings["llm_judge"] else "--no-llm-judge",
        "--skip-categories", settings["skip_categories"],
        "--request-timeout", str(settings["request_timeout"]),
        "--out-dir", str(RESULTS_DIR),
        "--tag", model_cfg["tag"],
    ]
    if settings["temperature"] is None:
        cmd.append("--omit-temperature")
    else:
        cmd.extend(["--temperature", str(settings["temperature"])])
    cmd.extend([
        "--retry-failed" if settings["retry_failed"] else "--no-retry-failed",
        "--retry-concurrency", str(settings["retry_concurrency"]),
        "--retry-request-timeout", str(settings["retry_request_timeout"]),
        "--thinking-mode" if settings["thinking_mode"] else "--no-thinking-mode",
    ])
    if settings["no_system_prompt"]:
        cmd.append("--no-system-prompt")
    if force:
        cmd.append("--no-resume")
    return cmd, settings, signature, signature_payload


def run_eval(model_cfg, global_cfg, force=False):
    """Run ACE evaluation for a single model."""
    tag = model_cfg["tag"]
    cmd, settings, expected_signature, signature_payload = build_eval_command(
        model_cfg, global_cfg, force=force
    )

    if not force and find_existing_results(tag, expected_signature):
        print(
            f"  ⏭  {tag}: ACE {BENCHMARK_VERSION} result with matching "
            "evaluation signature already exists; skipping (use --force to re-run)"
        )
        return "skipped"

    # Diagnostic only: the runner independently enforces the same signature.
    partial_file, n_done = find_partial_results(
        tag,
        expected_signature,
        signature_payload["evaluation_contract_sha256"],
        signature_payload["evaluator_runtime_sha256"],
    )
    if partial_file and not force:
        print(f"  ♻️  {tag}: found partial results ({n_done} probes done), resuming...")

    print(f"  🚀 {tag}: starting evaluation...")
    print(f"     Provider: {model_cfg.get('provider', global_cfg.get('default_provider', 'default'))}")
    print(f"     Model ID: {model_cfg['id']}")
    print(f"     Base URL: {settings['base_url']}")
    print(f"     ACE release: {BENCHMARK_VERSION} ({expected_signature[:12]}…)")
    print(
        "     API concurrency: "
        f"candidate={settings['concurrency']}, judge={settings['judge_concurrency']}, "
        f"shared_total={settings['total_api_concurrency']}"
    )
    print(f"     Skipped categories: {settings['skip_categories'] or '(none)'}")
    if settings["ignored_legacy_overrides"]:
        print(
            "     Ignored legacy non-official settings: "
            + ", ".join(settings["ignored_legacy_overrides"])
        )
    before_summaries = _summary_fingerprints(tag)
    t0 = time.time()
    child_env = {
        **os.environ,
        "OPENAI_API_KEY": settings["candidate_api_key"],
        "ACE_JUDGE_API_KEY": settings["judge_api_key"],
    }
    result = subprocess.run(cmd, env=child_env, capture_output=False)
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"  ❌ {tag}: evaluation FAILED (exit code {result.returncode})")
        return "failed"

    published_summary = find_existing_results(tag, expected_signature)
    if published_summary is not None:
        published_path = Path(published_summary)
        try:
            was_changed = (
                before_summaries.get(str(published_path))
                != file_sha256(published_path)
            )
        except OSError:
            was_changed = False
        if was_changed:
            print(f"  ✅ {tag}: completed in {dt:.0f}s")
            return "done"

    diagnostic = _new_summary_diagnostic(
        tag, before_summaries, expected_signature
    )
    print(
        f"  ❌ {tag}: runner exited 0 but did not produce a new publishable "
        f"ACE {BENCHMARK_VERSION} full-suite result ({diagnostic})"
    )
    return "failed"


def build_dashboard(config):
    """Generate dashboard data from results."""
    print("\n📊 Building dashboard...")
    cmd = [sys.executable, str(ROOT / "build_dashboard.py")]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("  ❌ Dashboard build failed!")
        return False
    print("  ✅ Dashboard built successfully!")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=f"ACE {BENCHMARK_VERSION} Automated Benchmark Runner"
    )
    ap.add_argument("--version", action="version", version=f"ACE {BENCHMARK_VERSION}")
    ap.add_argument("--eval-only", action="store_true", help="Only run evaluations")
    ap.add_argument("--build-only", action="store_true", help="Only build dashboard from existing results")
    ap.add_argument("--model", type=str, default=None,
                    help="Run only specific model(s) (by tag, comma-separated)")
    ap.add_argument("--provider", type=str, default=None,
                    help="Run only models from a specific provider")
    ap.add_argument("--force", action="store_true", help="Force re-evaluation even if results exist")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Compatibility flag; exact-signature resume is already enabled by default",
    )
    ap.add_argument("--status", action="store_true", help="Show evaluation status for all models")
    args = ap.parse_args()

    if not os.environ.get("ACE_JUDGE_BASE_URL"):
        ap.error("set ACE_JUDGE_BASE_URL for batch evaluation and result selection")

    config = load_config()
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    DASHBOARD_DIR.mkdir(exist_ok=True, parents=True)

    # Status mode: show what's done and what's pending
    if args.status:
        print("=" * 60)
        print(f"📋 ACE {BENCHMARK_VERSION} Evaluation Status")
        print("=" * 60)
        for m in config["models"]:
            tag = m["tag"]
            provider = m.get("provider", config.get("default_provider", "?"))
            _, _, expected_signature, signature_payload = build_eval_command(m, config)
            if find_existing_results(tag, expected_signature):
                print(f"  ✅ {tag:30s} [{provider}] — done")
            else:
                partial, n = find_partial_results(
                    tag,
                    expected_signature,
                    signature_payload["evaluation_contract_sha256"],
                    signature_payload["evaluator_runtime_sha256"],
                )
                if partial:
                    print(f"  ♻️  {tag:30s} [{provider}] — partial ({n} probes)")
                else:
                    print(f"  ⏳ {tag:30s} [{provider}] — pending")
        return

    if not args.build_only:
        # Filter models by tag/provider
        models_to_eval = config["models"]
        if args.model:
            target_tags = {t.strip() for t in args.model.split(",")}
            models_to_eval = [m for m in models_to_eval if m["tag"] in target_tags]
        if args.provider:
            models_to_eval = [m for m in models_to_eval
                              if m.get("provider", config.get("default_provider")) == args.provider]

        # Show providers info
        providers = config.get("providers", {})
        provider_names = set(m.get("provider", config.get("default_provider", "default"))
                            for m in models_to_eval)

        print("=" * 60)
        print(f"🧪 ACE {BENCHMARK_VERSION} Automated Benchmark")
        print("=" * 60)
        print(f"  Models to evaluate: {len(models_to_eval)}")
        print(f"  Providers: {', '.join(sorted(provider_names))}")
        for pname in sorted(provider_names):
            if pname in providers:
                print(f"    • {pname}: {providers[pname].get('description', providers[pname]['base_url'])}")
        print(
            f"  Max tokens: "
            f"{(config.get('eval') or {}).get('max_tokens', DEFAULT_MAX_TOKENS)}"
        )
        print(
            "  Official API concurrency: "
            f"candidate={DEFAULT_CANDIDATE_CONCURRENCY}, "
            f"judge={DEFAULT_JUDGE_CONCURRENCY}, "
            f"shared_total={DEFAULT_TOTAL_API_CONCURRENCY}"
        )
        print(
            "  Official skipped categories: "
            f"{DEFAULT_SKIP_CATEGORIES or '(none)'}"
        )
        print(f"  Resume: enabled (use --force to override)")
        print()

        done = 0
        skipped = 0
        failed = 0

        for m in models_to_eval:
            status = run_eval(m, config, force=args.force)
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

        print(f"\n  Summary: {done} evaluated, {skipped} skipped, {failed} failed")

    if not args.eval_only:
        build_dashboard(config)

    print("\n🎉 All done!")
    print(f"   Results: {RESULTS_DIR}/")
    print(f"   Dashboard data: {DASHBOARD_DIR}/")


if __name__ == "__main__":
    main()
