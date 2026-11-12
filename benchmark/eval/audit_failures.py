#!/usr/bin/env python3
"""Offline post-run failure audit for ACE result JSONL files.

This tool never calls a scorer, model endpoint, or LLM judge.  It reads the
judge verdict and criterion evidence only from the supplied result records.
Within each run the last JSONL record for a probe wins (matching the append-only
retry behavior in ``run_eval.py``); across runs failures are deduplicated by
``probe_id`` and retain one occurrence per run.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


RISK_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
MARKER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[*_`]*[ \t]*)?"
    r"(?:final[ \t]+answer|the[ \t]+answer|final[ \t]+result|"
    r"answer|choice|option|response|reply|output|result)"
    r"(?:[ \t]*[*_`]*)[ \t]*(?:is[ \t]*:|is|:|=)[ \t]*",
    re.IGNORECASE,
)


def load_probes(probes_dir: Path) -> dict[str, dict]:
    probes: dict[str, dict] = {}
    for path in sorted(probes_dir.glob("*.jsonl")):
        for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
            if not raw.lstrip().startswith("{"):
                continue
            try:
                probe = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid probe JSON: {path}:{line_no}: {exc}") from exc
            probe_id = probe.get("probe_id")
            if not probe_id:
                raise ValueError(f"probe missing probe_id: {path}:{line_no}")
            probes[probe_id] = probe
    if not probes:
        raise ValueError(f"no probes found under {probes_dir}")
    return probes


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_run(label: str, results_path: Path,
             summary_path: Path | None = None) -> dict:
    """Load one append-only result file, keeping its last record per probe."""
    latest: dict[str, dict] = {}
    record_count = 0
    duplicate_count = 0
    try:
        lines = results_path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read results {results_path}: {exc}") from exc
    for line_no, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid result JSON: {results_path}:{line_no}: {exc}"
            ) from exc
        if not isinstance(row, dict) or not row.get("probe_id"):
            raise ValueError(
                f"result missing probe_id: {results_path}:{line_no}"
            )
        record_count += 1
        probe_id = str(row["probe_id"])
        if probe_id in latest:
            duplicate_count += 1
        row = dict(row)
        row["_record_index"] = record_count
        row["_line_number"] = line_no
        latest[probe_id] = row

    summary = load_json(summary_path) if summary_path else None
    warnings = validate_summary(latest, summary) if summary else []
    return {
        "label": label,
        "results_path": str(results_path.resolve()),
        "summary_path": str(summary_path.resolve()) if summary_path else None,
        "record_count": record_count,
        "unique_probe_count": len(latest),
        "duplicate_record_count": duplicate_count,
        "records": latest,
        "summary": summary,
        "summary_warnings": warnings,
    }


def validate_summary(records: dict[str, dict], summary: dict) -> list[str]:
    """Best-effort consistency checks; mismatches are warnings, not guesses."""
    actual = {
        "n_total": len(records),
        "n_pass": sum(row.get("passed") is True for row in records.values()),
        "n_excluded": sum(bool(row.get("excluded_from_score"))
                          for row in records.values()),
    }
    warnings = []
    for key, value in actual.items():
        declared = summary.get(key)
        if declared is not None and declared != value:
            warnings.append(f"{key}: summary={declared!r}, deduped_results={value!r}")
    return warnings


def scorer_expected(probe: dict) -> Any:
    scorer = probe.get("scorer") or {}
    for key in ("expected", "items"):
        if key in scorer:
            return scorer[key]
    return probe.get("ground_truth")


def clean_wrapper(text: str) -> str:
    value = str(text).strip()
    value = re.sub(r"^[*_`]+[ \t]*", "", value)
    value = re.sub(r"[ \t]*[*_`]+$", "", value)
    value = value.strip(" \t\r\n\"'`")
    latex = re.fullmatch(r"\\(?:text|mathrm)[ \t]*\{([\s\S]*)\}", value)
    return latex.group(1).strip() if latex else value


def balanced_boxed(text: str) -> list[tuple[int, str]]:
    found = []
    for match in re.finditer(r"\\boxed[ \t]*\{", text):
        open_brace = match.end() - 1
        depth = 1
        index = open_brace + 1
        while index < len(text):
            escaped = index > 0 and text[index - 1] == "\\"
            if not escaped and text[index] == "{":
                depth += 1
            elif not escaped and text[index] == "}":
                depth -= 1
                if depth == 0:
                    found.append((open_brace + 1, text[open_brace + 1:index]))
                    break
            index += 1
    return found


def explicit_candidates(response: str) -> list[dict]:
    """Extract answer-position candidates for audit hints, never for scoring."""
    candidates = []
    lines = response.splitlines(keepends=True)
    offsets = []
    offset = 0
    for line in lines:
        raw = line.rstrip("\r\n")
        if raw.strip():
            offsets.append((offset, raw))
        offset += len(line)
    for marker in MARKER_RE.finditer(response):
        line_end = response.find("\n", marker.end())
        if line_end < 0:
            line_end = len(response)
        tail = response[marker.end():line_end].rstrip("\r")
        if tail.strip(" \t*_`"):
            candidates.append({
                "kind": "answer_marker", "position": marker.end(),
                "payload": clean_wrapper(tail),
            })
        else:
            for start, raw in offsets:
                if start > line_end:
                    candidates.append({
                        "kind": "answer_marker_next_line", "position": start,
                        "payload": clean_wrapper(raw),
                    })
                    break
    for position, payload in balanced_boxed(response):
        candidates.append({
            "kind": "boxed", "position": position,
            "payload": clean_wrapper(payload),
        })
    for match in re.finditer(
            r"\b(?:answer|choice|option)[ \t]*(?:is[ \t]*)?"
            r"\(?([A-D])\)?(?=$|[ \t.!?,;:)\-])",
            response, re.IGNORECASE):
        candidates.append({
            "kind": "mcq_option_mention", "position": match.start(1),
            "payload": match.group(1),
        })
    if offsets:
        for kind, (position, raw) in (
            ("first_nonempty_line", offsets[0]),
            ("last_nonempty_line", offsets[-1]),
        ):
            candidates.append({
                "kind": kind, "position": position,
                "payload": clean_wrapper(raw),
            })
    return sorted(candidates, key=lambda item: item["position"])


def _text_contains(text: str, expected: Any, scorer_type: str) -> bool:
    if expected is None:
        return False
    if isinstance(expected, (dict, list)):
        compact = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        compact_text = re.sub(r"\s+", "", text)
        return compact.casefold() in compact_text.casefold()
    if isinstance(expected, bool):
        needle = str(expected).lower()
    else:
        needle = str(expected).strip()
    if not needle:
        return False
    haystack = text.casefold()
    folded = needle.casefold()
    if scorer_type == "mcq_letter" and re.fullmatch(r"[A-D]", needle, re.I):
        # A bare letter in prose is meaningless; MCQ mentions require an
        # answer-position grammar and are handled by candidate matching.
        return False
    if isinstance(expected, (int, float)) or "numeric" in scorer_type:
        normalized = haystack.replace(",", "")
        token = folded.replace(",", "")
        return re.search(r"(?<![0-9.])" + re.escape(token) + r"(?![0-9.])",
                         normalized) is not None
    left = r"(?<![A-Za-z0-9_])" if needle[:1].isascii() \
        and (needle[:1].isalnum() or needle[:1] == "_") else ""
    right = r"(?![A-Za-z0-9_])" if needle[-1:].isascii() \
        and (needle[-1:].isalnum() or needle[-1:] == "_") else ""
    return re.search(left + re.escape(folded) + right, haystack) is not None


def candidate_matches(candidate: str, expected: Any, scorer_type: str) -> bool:
    clean = clean_wrapper(candidate)
    if scorer_type == "mcq_letter":
        match = re.match(
            r"^\(?([A-D])\)?(?:[*_`]*)?(?=$|[ \t.!?,;:)\-])",
            clean, re.IGNORECASE,
        )
        return bool(match and str(expected).strip().upper() == match.group(1).upper())
    return _text_contains(clean, expected, scorer_type)


def judge_snapshot(row: dict) -> dict | None:
    extracted = row.get("score_extracted")
    if not isinstance(extracted, dict):
        return None
    judge_keys = {
        "verdict", "confidence", "criterion_results", "failed_criteria",
        "judge_provider", "judge_model", "judge_requested_model",
        "judge_response_id", "judge_prompt_version", "judge_attempt", "usage",
    }
    if not (judge_keys & set(extracted)):
        return None
    return {key: extracted.get(key) for key in judge_keys if key in extracted}


def occurrence_for(run: dict, row: dict, probe: dict | None) -> dict:
    scorer = (probe or {}).get("scorer") or {}
    scorer_type = str(scorer.get("type") or row.get("scorer_type") or "unknown")
    expected = scorer_expected(probe or {})
    response = str(row.get("response") or "")
    candidates = explicit_candidates(response)
    contains_expected = _text_contains(response, expected, scorer_type)
    matching_candidates = [
        item for item in candidates
        if candidate_matches(item["payload"], expected, scorer_type)
    ]
    strategy = str(row.get("score_strategy") or "")
    error = row.get("error") or row.get("scorer_error")
    judge = judge_snapshot(row)
    flags = set()
    if contains_expected:
        flags.add("response_contains_expected")
    if matching_candidates:
        flags.add("explicit_answer_matches_expected")
    if strategy == "llm_judge_fail" or (judge and judge.get("verdict") == "FAIL"):
        flags.add("judge_fail_review")
    if judge and (judge.get("failed_criteria") or any(
            item.get("status") in {"UNSATISFIED", "TRIGGERED"}
            for item in (judge.get("criterion_results") or [])
            if isinstance(item, dict))):
        flags.add("judge_failed_criteria")
    if scorer_type == "regex" or "regex" in strategy:
        flags.add("regex_boundary_review")
    if scorer_type == "mcq_letter" or "mcq" in strategy:
        flags.add("mcq_boundary_review")
    if ("numeric" in scorer_type or "number" in strategy
            or strategy.startswith(("boxed", "last_num"))):
        flags.add("numeric_boundary_review")
    if error or any(token in strategy for token in (
            "scorer_error", "model_error", "exception", "timeout")):
        flags.add("scorer_error")
    if row.get("deferred") or "deferred" in strategy:
        flags.add("deferred")
    if row.get("excluded_from_score"):
        flags.add("excluded_from_score")
    return {
        "run": run["label"],
        "result_file": run["results_path"],
        "record_index": row.get("_record_index"),
        "line_number": row.get("_line_number"),
        "passed": row.get("passed") is True,
        "excluded_from_score": bool(row.get("excluded_from_score")),
        "deferred": bool(row.get("deferred")),
        "strategy": strategy or "unknown",
        "score_note": row.get("score_note") or "",
        "score_extracted": row.get("score_extracted"),
        "error": error,
        "response": response,
        "reasoning": row.get("reasoning") or "",
        "finish_reason": row.get("finish_reason"),
        "contains_expected": contains_expected,
        "explicit_candidates": candidates,
        "matching_explicit_candidates": matching_candidates,
        "judge": judge,
        "risk_flags": sorted(flags),
    }


def risk_level(flags: set[str]) -> str:
    if flags & {"scorer_error", "deferred", "missing_probe_definition"}:
        return "critical"
    if flags & {
        "explicit_answer_matches_expected", "response_contains_expected",
        "cross_run_disagreement", "judge_fail_review", "judge_failed_criteria",
    }:
        return "high"
    if flags & {
        "regex_boundary_review", "mcq_boundary_review",
        "numeric_boundary_review", "excluded_from_score",
    }:
        return "medium"
    return "low"


def counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items(), key=lambda item: (-item[1], item[0])))


def build_audit(runs: list[dict], probes: dict[str, dict]) -> dict:
    all_probe_ids = sorted({probe_id for run in runs for probe_id in run["records"]})
    items = []
    for probe_id in all_probe_ids:
        probe = probes.get(probe_id)
        occurrences = [
            occurrence_for(run, run["records"][probe_id], probe)
            for run in runs if probe_id in run["records"]
        ]
        failed = [item for item in occurrences if not item["passed"]]
        if not failed:
            continue
        flags = {flag for item in failed for flag in item["risk_flags"]}
        passed_runs = [item["run"] for item in occurrences if item["passed"]]
        failed_runs = [item["run"] for item in failed]
        if passed_runs and failed_runs:
            flags.add("cross_run_disagreement")
        if probe is None:
            flags.add("missing_probe_definition")
        scorer = (probe or {}).get("scorer") or {}
        item = {
            "probe_id": probe_id,
            "canonical_id": (probe or {}).get("canonical_id")
                            or failed[0].get("canonical_id"),
            "category": (probe or {}).get("category") or "unknown",
            "level": (probe or {}).get("level") or "unknown",
            "scorer_type": scorer.get("type") or "unknown",
            "ground_truth": (probe or {}).get("ground_truth"),
            "expected": scorer_expected(probe or {}),
            "prompt": (probe or {}).get("prompt") or "",
            "system_prompt": (probe or {}).get("system_prompt") or "",
            "scorer": scorer,
            "failed_runs": failed_runs,
            "passed_runs": passed_runs,
            "fail_run_count": len(failed_runs),
            "observed_run_count": len(occurrences),
            "risk_level": risk_level(flags),
            "risk_flags": sorted(flags),
            "occurrences": occurrences,
            "missing_probe_definition": probe is None,
        }
        items.append(item)

    items.sort(key=lambda item: (
        RISK_ORDER[item["risk_level"]], item["category"], item["probe_id"],
    ))
    failed_occurrences = [
        occurrence for item in items for occurrence in item["occurrences"]
        if not occurrence["passed"]
    ]
    aggregation = {
        "unique_failed_probes": len(items),
        "fail_occurrences": len(failed_occurrences),
        "effective_fail_occurrences": sum(
            not item["excluded_from_score"] for item in failed_occurrences
        ),
        "excluded_fail_occurrences": sum(
            item["excluded_from_score"] for item in failed_occurrences
        ),
        "by_category": counter_dict(item["category"] for item in items),
        "by_scorer_type": counter_dict(item["scorer_type"] for item in items),
        "by_strategy": counter_dict(
            item["strategy"] for item in failed_occurrences
        ),
        "by_risk_level": counter_dict(item["risk_level"] for item in items),
        "by_risk_flag": counter_dict(
            flag for item in items for flag in item["risk_flags"]
        ),
        "by_run": counter_dict(
            item["run"] for item in failed_occurrences
        ),
    }
    run_metadata = []
    for run in runs:
        run_metadata.append({
            key: run[key] for key in (
                "label", "results_path", "summary_path", "record_count",
                "unique_probe_count", "duplicate_record_count", "summary_warnings",
            )
        } | {"summary": run.get("summary")})
    return {
        "schema_version": "ace-failure-audit-v1",
        "method": {
            "input_mode": "offline_read_only",
            "within_run_dedup": "last JSONL record per probe_id wins",
            "cross_run_dedup": "one review item per probe_id",
            "judge_policy": "only supplied result-record verdict/criteria are reported; no judge is called or inferred",
        },
        "runs": run_metadata,
        "aggregation": aggregation,
        "items": items,
    }


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def json_inline(value: Any, limit: int = 220) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def excerpt(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n… [truncated]"


def markdown_table(title: str, values: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "|---|---:|"]
    lines.extend(f"| {md_escape(key)} | {count} |" for key, count in values.items())
    lines.append("")
    return lines


def render_markdown(audit: dict, excerpt_chars: int = 800) -> str:
    agg = audit["aggregation"]
    lines = [
        "# ACE post-run failure audit", "",
        "> Offline report: no scorer or LLM judge was called. Judge verdicts and "
        "criteria below come only from the supplied result JSONL records.", "",
        "## Overview", "",
        "| Metric | Count |", "|---|---:|",
        f"| Unique failed probes | {agg['unique_failed_probes']} |",
        f"| Failure occurrences | {agg['fail_occurrences']} |",
        f"| Effective failure occurrences | {agg['effective_fail_occurrences']} |",
        f"| Excluded failure occurrences | {agg['excluded_fail_occurrences']} |",
        "", "## Inputs", "",
        "| Run | Unique probes | Duplicate rows | Summary warnings | Results |",
        "|---|---:|---:|---|---|",
    ]
    for run in audit["runs"]:
        lines.append(
            f"| {md_escape(run['label'])} | {run['unique_probe_count']} | "
            f"{run['duplicate_record_count']} | "
            f"{md_escape('; '.join(run['summary_warnings']) or 'none')} | "
            f"`{md_escape(run['results_path'])}` |"
        )
    lines.append("")
    for title, key in (
        ("Failures by category", "by_category"),
        ("Failures by scorer type", "by_scorer_type"),
        ("Failure occurrences by strategy", "by_strategy"),
        ("Review items by risk level", "by_risk_level"),
        ("High-risk flags", "by_risk_flag"),
        ("Failure occurrences by run", "by_run"),
    ):
        lines.extend(markdown_table(title, agg[key]))

    lines.extend(["# Per-probe review", ""])
    for item in audit["items"]:
        lines.extend([
            f"## [{item['risk_level'].upper()}] `{item['probe_id']}`", "",
            f"- Category / level: `{item['category']}` / `{item['level']}`",
            f"- Scorer: `{item['scorer_type']}`",
            f"- Failed runs: {', '.join(f'`{x}`' for x in item['failed_runs'])}",
            f"- Passed runs: {', '.join(f'`{x}`' for x in item['passed_runs']) or 'none'}",
            f"- Risk flags: {', '.join(f'`{x}`' for x in item['risk_flags']) or 'none'}",
            f"- Expected: `{md_escape(json_inline(item['expected']))}`", "",
            "**Prompt excerpt**", "", "```text",
            excerpt(item["prompt"], excerpt_chars), "```", "",
            "| Run | Outcome | Excluded | Strategy | Judge | Failed criteria |",
            "|---|---|---:|---|---|---|",
        ])
        for occurrence in item["occurrences"]:
            judge = occurrence.get("judge") or {}
            lines.append(
                f"| {md_escape(occurrence['run'])} | "
                f"{'PASS' if occurrence['passed'] else 'FAIL'} | "
                f"{'yes' if occurrence['excluded_from_score'] else 'no'} | "
                f"{md_escape(occurrence['strategy'])} | "
                f"{md_escape(judge.get('verdict', ''))} | "
                f"{md_escape(', '.join(judge.get('failed_criteria') or []))} |"
            )
        lines.append("")
        for occurrence in item["occurrences"]:
            if occurrence["passed"]:
                continue
            lines.extend([
                f"### Failure in `{occurrence['run']}`", "",
                f"Flags: {', '.join(f'`{x}`' for x in occurrence['risk_flags']) or 'none'}",
                "", "**Response excerpt**", "", "````text",
                excerpt(occurrence["response"], excerpt_chars), "````", "",
            ])
            if occurrence["score_note"]:
                lines.extend([
                    "**Score note**", "", excerpt(occurrence["score_note"], excerpt_chars), "",
                ])
            if occurrence.get("judge"):
                lines.extend([
                    "**Judge record (from result JSONL)**", "", "```json",
                    json.dumps(occurrence["judge"], ensure_ascii=False, indent=2,
                               default=str), "```", "",
                ])
            elif occurrence.get("score_extracted") is not None:
                lines.extend([
                    "**Score extraction**", "", "```json",
                    json.dumps(occurrence["score_extracted"], ensure_ascii=False,
                               indent=2, default=str), "```", "",
                ])
    return "\n".join(lines).rstrip() + "\n"


def parse_assignment(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"{option} expects LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError(f"{option} expects non-empty LABEL=PATH")
    return label, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", action="append", required=True, metavar="LABEL=PATH",
        help="result JSONL; repeat once per run",
    )
    parser.add_argument(
        "--summary", action="append", default=[], metavar="LABEL=PATH",
        help="optional matching summary JSON; repeat per run",
    )
    parser.add_argument(
        "--probes-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "probes",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default="ace_failure_audit")
    parser.add_argument("--excerpt-chars", type=int, default=800)
    args = parser.parse_args(argv)

    try:
        result_specs = [parse_assignment(value, "--results")
                        for value in args.results]
        summary_specs = dict(parse_assignment(value, "--summary")
                             for value in args.summary)
        labels = [label for label, _ in result_specs]
        if len(labels) != len(set(labels)):
            raise ValueError("duplicate --results label")
        unknown_summaries = set(summary_specs) - set(labels)
        if unknown_summaries:
            raise ValueError(
                f"--summary labels without --results: {sorted(unknown_summaries)}"
            )
        probes = load_probes(args.probes_dir)
        runs = [
            load_run(label, path, summary_specs.get(label))
            for label, path in result_specs
        ]
        audit = build_audit(runs, probes)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.out_dir / f"{args.name}.json"
        md_path = args.out_dir / f"{args.name}.md"
        json_path.write_text(json.dumps(
            audit, ensure_ascii=False, indent=2, default=str,
        ) + "\n")
        md_path.write_text(render_markdown(audit, args.excerpt_chars))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "json": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
        **audit["aggregation"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
