"""Compile every structured probe regex on the supported Python runtime.

This stays deliberately independent of scorer dispatch: a regex hidden in an
LLM-judge rubric must still be valid before the benchmark can be released.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator


ACE = Path(__file__).resolve().parent.parent
PROBE_DIR = ACE / "probes"


def _is_regex_key(key: str) -> bool:
    return (
        key in {"pattern", "patterns", "regex"}
        or key.endswith("_pattern")
        or key.endswith("_patterns")
        or key.endswith("_regex")
    )


def _iter_regexes(value: object, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if _is_regex_key(key):
                if isinstance(child, str):
                    yield child_path, child
                elif isinstance(child, list):
                    for index, pattern in enumerate(child):
                        if isinstance(pattern, str):
                            yield f"{child_path}[{index}]", pattern
            yield from _iter_regexes(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_regexes(child, f"{path}[{index}]")


def _load_probes() -> Iterator[tuple[Path, int, dict]]:
    for probe_path in sorted(PROBE_DIR.glob("*.jsonl")):
        for line_number, raw in enumerate(probe_path.read_text().splitlines(), 1):
            if raw.lstrip().startswith("{"):
                yield probe_path, line_number, json.loads(raw)


def test_all_structured_probe_regexes_compile() -> None:
    checked = 0
    failures = []
    for probe_path, line_number, probe in _load_probes():
        for field_path, pattern in _iter_regexes(probe):
            checked += 1
            location = (
                f"{probe_path.name}:{line_number}:"
                f"{probe.get('probe_id', '<missing-id>')}:{field_path}"
            )
            if r"\z" in pattern:
                failures.append(
                    f"{location}: uses Python-3.14-only \\z; use \\Z for "
                    "the supported Python 3.10 runtime"
                )
                continue
            try:
                re.compile(pattern)
            except re.error as exc:
                failures.append(f"{location}: {exc}: {pattern!r}")

    assert checked >= 500, "regex discovery unexpectedly skipped probe fields"
    assert not failures, "Invalid probe regexes:\n" + "\n".join(failures)


def test_liar_puzzle_has_one_consistent_thief() -> None:
    liar_probe = next(
        probe
        for _, _, probe in _load_probes()
        if probe.get("probe_id") == "logic.liar_puzzle.L1.001"
    )
    assert "Alice says 'Bob is the thief.'" in liar_probe["prompt"]
    assert "Bob says 'Neither Alice nor I is the thief.'" in liar_probe["prompt"]

    consistent = []
    for thief in ("Alice", "Bob"):
        alice_statement = thief == "Bob"
        bob_statement = thief not in {"Alice", "Bob"}
        alice_should_tell_truth = thief != "Alice"
        bob_should_tell_truth = thief != "Bob"
        if (
            alice_statement == alice_should_tell_truth
            and bob_statement == bob_should_tell_truth
        ):
            consistent.append(thief)

    assert consistent == ["Bob"]
    assert liar_probe["ground_truth"] == "Bob"


if __name__ == "__main__":
    test_all_structured_probe_regexes_compile()
    test_liar_puzzle_has_one_consistent_thief()
    print("probe regex compile + liar uniqueness: PASS")
