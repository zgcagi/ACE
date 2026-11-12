"""Offline tests for the post-run failure audit utility."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.audit_failures import (
    build_audit,
    load_probes,
    load_run,
    main,
    render_markdown,
)


PROBES = [
    {
        "probe_id": "p.exact", "canonical_id": "p.exact", "category": "know",
        "level": "L1", "prompt": "Fruit?", "ground_truth": "banana",
        "scorer": {"type": "exact_match_string", "expected": "banana"},
    },
    {
        "probe_id": "p.mcq", "canonical_id": "p.mcq", "category": "logic",
        "level": "L2", "prompt": "Choose.", "ground_truth": "B",
        "scorer": {"type": "mcq_letter", "expected": "B"},
    },
    {
        "probe_id": "p.regex", "canonical_id": "p.regex", "category": "math",
        "level": "L3", "prompt": "Expression?", "ground_truth": "e^2",
        "scorer": {"type": "regex", "pattern": r"^e\^2$"},
    },
    {
        "probe_id": "p.judge", "canonical_id": "p.judge", "category": "instr",
        "level": "L3", "prompt": "Write exactly three words.",
        "ground_truth": None, "scorer": {"type": "judge_rubric"},
    },
    {
        "probe_id": "p.numeric", "canonical_id": "p.numeric", "category": "arith",
        "level": "L1", "prompt": "Number?", "ground_truth": 42,
        "scorer": {"type": "exact_match_numeric", "expected": 42},
    },
    {
        "probe_id": "p.error", "canonical_id": "p.error", "category": "tool",
        "level": "L2", "prompt": "Call?", "ground_truth": "ok",
        "scorer": {"type": "exact_match_string", "expected": "ok"},
    },
]


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def run_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        probes_dir = root / "probes"
        probes_dir.mkdir()
        write_jsonl(probes_dir / "synthetic.jsonl", PROBES)

        run_a_path = root / "run_a.jsonl"
        run_b_path = root / "run_b.jsonl"
        summary_a_path = root / "summary_a.json"
        write_jsonl(run_a_path, [
            # The second record supersedes this retryable/old failure.
            {"probe_id": "p.exact", "passed": False, "response": "banana",
             "score_strategy": "wrong"},
            {"probe_id": "p.exact", "passed": True, "response": "banana",
             "score_strategy": "stripped_exact"},
            {"probe_id": "p.mcq", "passed": False, "response": "Answer: B",
             "score_strategy": "no_explicit_mcq_answer"},
            {"probe_id": "p.judge", "passed": False,
             "response": "only two", "score_strategy": "llm_judge_fail",
             "score_extracted": {
                 "verdict": "FAIL", "confidence": 1.0,
                 "failed_criteria": ["PASS-1"],
                 "criterion_results": [{
                     "id": "PASS-1", "status": "UNSATISFIED",
                     "evidence": "Only two words.",
                 }],
             }},
            {"probe_id": "p.numeric", "passed": False,
             "response": "Final Answer: 42", "score_strategy": "wrong_number"},
            {"probe_id": "p.error", "passed": False, "response": "",
             "score_strategy": "scorer_error", "deferred": True,
             "excluded_from_score": True, "error": "synthetic scorer failure"},
        ])
        write_jsonl(run_b_path, [
            {"probe_id": "p.exact", "passed": False,
             "response": "Reply: banana", "score_strategy": "wrong"},
            {"probe_id": "p.regex", "passed": False,
             "response": r"\boxed{e^2}", "score_strategy": "regex_no_match"},
        ])
        summary_a_path.write_text(json.dumps({
            "n_total": 999, "n_pass": 1, "n_excluded": 1,
        }))

        probes = load_probes(probes_dir)
        run_a = load_run("plain", run_a_path, summary_a_path)
        run_b = load_run("thinking", run_b_path)
        assert run_a["duplicate_record_count"] == 1
        assert run_a["records"]["p.exact"]["passed"] is True
        assert run_a["summary_warnings"] == [
            "n_total: summary=999, deduped_results=5"
        ]

        audit = build_audit([run_a, run_b], probes)
        assert audit["aggregation"]["unique_failed_probes"] == 6
        assert audit["aggregation"]["fail_occurrences"] == 6
        by_id = {item["probe_id"]: item for item in audit["items"]}
        assert by_id["p.exact"]["passed_runs"] == ["plain"]
        assert "cross_run_disagreement" in by_id["p.exact"]["risk_flags"]
        assert "explicit_answer_matches_expected" in by_id["p.mcq"]["risk_flags"]
        assert "regex_boundary_review" in by_id["p.regex"]["risk_flags"]
        assert "judge_fail_review" in by_id["p.judge"]["risk_flags"]
        assert by_id["p.judge"]["occurrences"][0]["judge"]["failed_criteria"] \
            == ["PASS-1"]
        assert "numeric_boundary_review" in by_id["p.numeric"]["risk_flags"]
        assert by_id["p.error"]["risk_level"] == "critical"

        markdown = render_markdown(audit, excerpt_chars=100)
        assert "Offline report: no scorer or LLM judge was called" in markdown
        assert "`p.judge`" in markdown
        assert "PASS-1" in markdown

        out_dir = root / "out"
        assert main([
            "--results", f"plain={run_a_path}",
            "--results", f"thinking={run_b_path}",
            "--summary", f"plain={summary_a_path}",
            "--probes-dir", str(probes_dir),
            "--out-dir", str(out_dir),
            "--name", "review",
        ]) == 0
        payload = json.loads((out_dir / "review.json").read_text())
        assert payload["aggregation"]["unique_failed_probes"] == 6
        assert (out_dir / "review.md").is_file()


if __name__ == "__main__":
    run_test()
    print("audit_failures offline tests: PASS")
