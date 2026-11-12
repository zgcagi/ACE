#!/usr/bin/env python3
"""Offline regressions for ACE batch and dashboard result selection."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import build_dashboard
import run_all
from release_results import (
    BENCHMARK_VERSION,
    EVALUATION_CONTRACT_VERSION,
    JUDGE_PROMPT_VERSION,
    SCORER_VERSION,
    evaluation_signature,
    validate_summary_data,
)


class ReleaseEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.judge_env = mock.patch.dict(os.environ, {
            "ACE_JUDGE_BASE_URL": "https://judge.example",
            "ACE_JUDGE_API_KEY": "test-judge-key",
        })
        self.judge_env.start()
        self.addCleanup(self.judge_env.stop)
        self.tempdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tempdir.name)
        self.old_run_all_results = run_all.RESULTS_DIR
        self.old_dashboard_results = build_dashboard.RESULTS_DIR
        run_all.RESULTS_DIR = self.results_dir
        build_dashboard.RESULTS_DIR = self.results_dir

        config = {
            "api": {
                "base_url": "https://candidate.example/v1",
                "api_key": "test-key",
            },
            "eval": {},
        }
        model = {"id": "candidate-model", "tag": "demo"}
        self.command, self.settings, self.signature, self.payload = (
            run_all.build_eval_command(model, config)
        )

    def tearDown(self):
        run_all.RESULTS_DIR = self.old_run_all_results
        build_dashboard.RESULTS_DIR = self.old_dashboard_results
        self.tempdir.cleanup()

    def _summary(self, **overrides):
        meta = {
            "benchmark_version": BENCHMARK_VERSION,
            "scorer_version": SCORER_VERSION,
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "evaluation_signature": self.signature,
            "evaluation_signature_payload": self.payload,
            "evaluation_contract_sha256": self.payload[
                "evaluation_contract_sha256"
            ],
            "evaluator_runtime_sha256": self.payload["evaluator_runtime_sha256"],
            "official_full_suite": True,
            "official_scope_issues": [],
            "publishable": True,
            "score_status": "complete",
        }
        summary = {
            "n_total": 2565,
            "n_eff": 2489,
            "n_excluded": 76,
            "n_pass": 0,
            "overall_pass_rate": 0.0,
            "by_category": {},
            "meta": meta,
        }
        for key, value in overrides.items():
            if key.startswith("meta_"):
                meta[key.removeprefix("meta_")] = value
            else:
                summary[key] = value
        return summary

    def _detail_record(self, probe_id, **overrides):
        record = {
            "benchmark_version": BENCHMARK_VERSION,
            "scorer_version": SCORER_VERSION,
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "evaluation_signature": self.signature,
            "evaluation_contract_sha256": self.payload[
                "evaluation_contract_sha256"
            ],
            "evaluator_runtime_sha256": self.payload["evaluator_runtime_sha256"],
            "probe_id": probe_id,
            "score_strategy": "exact_match_string",
            "passed": False,
        }
        record.update(overrides)
        return record

    def _write_pair(self, tag, timestamp, *, summary=None, record_overrides=None):
        summary_path = self.results_dir / f"summary_{tag}_{timestamp}.json"
        detail_path = self.results_dir / f"results_{tag}_{timestamp}.jsonl"
        summary_path.write_text(
            json.dumps(summary or self._summary()), encoding="utf-8"
        )
        record_overrides = record_overrides or {}
        with detail_path.open("w", encoding="utf-8") as handle:
            for index in range(2565):
                row_overrides = {
                    "excluded_from_score": index >= 2489,
                    **record_overrides,
                }
                handle.write(json.dumps(self._detail_record(
                    f"probe-{index}", **row_overrides
                )) + "\n")
        return summary_path, detail_path

    def test_batch_defaults_match_official_full_suite(self):
        self.assertEqual(self.settings["concurrency"], 200)
        self.assertEqual(self.settings["judge_concurrency"], 200)
        self.assertEqual(self.settings["total_api_concurrency"], 200)
        self.assertEqual(self.settings["skip_categories"], "")
        self.assertEqual(
            self.command[self.command.index("--total-api-concurrency") + 1], "200"
        )
        self.assertEqual(self.signature, evaluation_signature(self.payload))

        legacy_config = {
            "api": {"base_url": "https://candidate.example/v1", "api_key": "x"},
            "judge": {
                "model": "legacy-judge",
                "base_url": "https://alternate-judge.example/v1",
            },
            "eval": {
                "concurrency": 8,
                "judge_concurrency": 8,
                "total_api_concurrency": 8,
                "skip_categories": "multimodal",
                "llm_judge": False,
                "judge_max_tokens": 512,
            },
        }
        legacy_settings = run_all.resolve_eval_settings(
            {"id": "candidate-model", "tag": "demo"},
            legacy_config,
            "https://candidate.example/v1",
        )
        self.assertEqual(legacy_settings["concurrency"], 200)
        self.assertEqual(legacy_settings["total_api_concurrency"], 200)
        self.assertEqual(legacy_settings["skip_categories"], "")
        self.assertTrue(legacy_settings["llm_judge"])
        self.assertEqual(legacy_settings["judge_model"], "deepseek-v4-flash")
        self.assertEqual(
            legacy_settings["judge_base_url"],
            "https://judge.example",
        )
        self.assertTrue(legacy_settings["ignored_legacy_overrides"])

        canonical_v1_settings = run_all.resolve_eval_settings(
            {"id": "candidate-model", "tag": "demo"},
            {"judge": {"base_url": "https://judge.example"}},
            "https://candidate.example/v1",
        )
        self.assertEqual(
            canonical_v1_settings["judge_base_url"],
            "https://judge.example",
        )
        self.assertFalse(any(
            value.startswith("judge_base_url=")
            for value in canonical_v1_settings["ignored_legacy_overrides"]
        ))

    def test_exit_zero_without_new_publishable_summary_is_failure(self):
        config = {
            "api": {
                "base_url": "https://candidate.example/v1",
                "api_key": "test-key",
            },
            "eval": {},
        }
        model = {"id": "candidate-model", "tag": "demo"}
        with mock.patch.object(
            run_all.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ):
            self.assertEqual(run_all.run_eval(model, config), "failed")

    def test_candidate_runtime_options_are_forwarded_and_signed(self):
        config = {
            "api": {
                "base_url": "https://candidate.example/v1",
                "api_key": "test-key",
            },
            "eval": {
                "thinking_mode": True,
                "no_system_prompt": True,
                "omit_temperature": True,
                "retry_failed": False,
                "retry_concurrency": 7,
                "retry_request_timeout": 456,
            },
        }
        model = {"id": "candidate-model", "tag": "options"}
        command, settings, signature, payload = run_all.build_eval_command(model, config)
        self.assertIn("--thinking-mode", command)
        self.assertIn("--no-system-prompt", command)
        self.assertIn("--no-retry-failed", command)
        self.assertIn("--omit-temperature", command)
        self.assertNotIn("--temperature", command)
        self.assertEqual(command[command.index("--retry-concurrency") + 1], "7")
        self.assertTrue(payload["candidate"]["thinking_mode"])
        self.assertTrue(payload["candidate"]["no_system_prompt"])
        self.assertIsNone(payload["candidate"]["temperature"])
        self.assertEqual(payload["candidate"]["max_candidate_requests_per_probe"], 1)
        self.assertEqual(payload["judge"]["max_requests_per_probe"], 1)
        self.assertEqual(signature, evaluation_signature(payload))
        baseline_payload = run_all.build_signature_payload({
            **settings,
            "thinking_mode": False,
        })
        self.assertNotEqual(signature, evaluation_signature(baseline_payload))

    def test_exit_zero_with_new_exact_publishable_pair_is_done(self):
        config = {
            "api": {
                "base_url": "https://candidate.example/v1",
                "api_key": "test-key",
            },
            "eval": {},
        }
        model = {"id": "candidate-model", "tag": "demo"}

        def fake_run(command, *, env, capture_output):
            self.assertNotIn("--api-key", command)
            self.assertNotIn("--judge-api-key", command)
            self.assertNotIn("test-key", command)
            self.assertNotIn("test-judge-key", command)
            self.assertEqual(env["OPENAI_API_KEY"], "test-key")
            self.assertEqual(env["ACE_JUDGE_API_KEY"], "test-judge-key")
            self._write_pair("demo", "123")
            return SimpleNamespace(returncode=0)

        with mock.patch.object(run_all.subprocess, "run", side_effect=fake_run):
            self.assertEqual(run_all.run_eval(model, config), "done")

    def test_old_same_tag_does_not_hide_matching_release_result(self):
        current_summary, current_detail = self._write_pair("demo", "100")
        # A newer legacy pair used to win solely because its filename sorted last.
        (self.results_dir / "summary_demo_999.json").write_text(
            json.dumps({"n_total": 2565, "meta": {"benchmark_version": "2.5.0"}}),
            encoding="utf-8",
        )
        (self.results_dir / "results_demo_999.jsonl").write_text(
            json.dumps({"probe_id": "legacy"}) + "\n", encoding="utf-8"
        )

        self.assertEqual(
            run_all.find_existing_results("demo", self.signature),
            str(current_summary),
        )
        self.assertEqual(
            build_dashboard.find_latest_result("demo"),
            (str(current_summary), str(current_detail)),
        )
        self.assertIsNone(run_all.find_existing_results("demo", "0" * 64))

    def test_summary_and_detail_must_share_timestamp_and_hashes(self):
        good_summary, good_detail = self._write_pair("demo", "100")
        self._write_pair(
            "demo",
            "999",
            record_overrides={"evaluator_runtime_sha256": "0" * 64},
        )
        self.assertEqual(
            build_dashboard.find_latest_result("demo"),
            (str(good_summary), str(good_detail)),
        )

    def test_partial_resume_ignores_legacy_and_wrong_signature(self):
        current = self.results_dir / "results_demo_100.jsonl"
        current.write_text(
            "\n".join(json.dumps(self._detail_record(f"probe-{i}")) for i in range(2))
            + "\n",
            encoding="utf-8",
        )
        (self.results_dir / "results_demo_999.jsonl").write_text(
            json.dumps({"probe_id": "legacy"}) + "\n", encoding="utf-8"
        )
        partial, done = run_all.find_partial_results(
            "demo",
            self.signature,
            self.payload["evaluation_contract_sha256"],
            self.payload["evaluator_runtime_sha256"],
        )
        self.assertEqual(partial, str(current))
        self.assertEqual(done, 2)

    def test_partial_or_non_full_summary_is_not_publishable(self):
        valid, reason = validate_summary_data(self._summary(n_total=2489))
        self.assertFalse(valid)
        self.assertIn("full-suite", reason)
        valid, reason = validate_summary_data(
            self._summary(meta_evaluator_runtime_sha256="not-a-hash")
        )
        self.assertFalse(valid)
        self.assertIn("evaluator_runtime_sha256", reason)


if __name__ == "__main__":
    unittest.main()
