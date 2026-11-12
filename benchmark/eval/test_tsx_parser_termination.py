"""Termination regressions for the bounded TSX static parser."""

from __future__ import annotations

import subprocess
import signal
import sys
import time
from pathlib import Path


ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))

from eval.scorer import MAX_RESP_BYTES, code_parse_ast  # noqa: E402
from eval.dispatch import _MainThreadAlarm  # noqa: E402


def test_unclosed_nested_tsx_always_terminates() -> None:
    # Run the historical minimal hang in a child process so a future
    # regression fails by timeout instead of wedging the entire test suite.
    source = (
        "import sys; "
        f"sys.path.insert(0, {str(ACE)!r}); "
        "from eval.scorer import code_parse_ast; "
        "result = code_parse_ast('<A><B>', "
        "{'parser': 'tsx', 'checks': {}}); "
        "assert not result.passed; "
        "assert result.strategy == 'parse_failed'"
    )
    subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        timeout=2,
        capture_output=True,
        text=True,
    )


def test_static_parser_rejects_oversized_source_before_parsing() -> None:
    result = code_parse_ast(
        "x" * (MAX_RESP_BYTES + 1),
        {"parser": "tsx", "checks": {}},
    )
    assert not result.passed
    assert result.strategy == "parse_input_too_long"


def test_long_outer_scaffolding_keeps_one_small_fenced_answer() -> None:
    result = code_parse_ast(
        "explanation " * 20_000 + "\n```tsx\nconst x = 1;\n```",
        {"parser": "tsx", "checks": {"bindings": []}},
    )
    assert result.passed, result.to_dict()


def test_many_non_jsx_less_than_signs_are_linear() -> None:
    started = time.monotonic()
    result = code_parse_ast(
        "x<y;" * 49_999,
        {"parser": "tsx", "checks": {}},
    )
    elapsed = time.monotonic() - started
    assert result.passed, result.to_dict()
    assert elapsed < 1.0, elapsed


def test_main_thread_alarm_restores_enclosing_handler_and_deadline() -> None:
    if not all(hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL")):
        return
    original_handler = signal.getsignal(signal.SIGALRM)
    original_timer = signal.getitimer(signal.ITIMER_REAL)

    def enclosing_handler(*_args: object) -> None:
        return None

    try:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, enclosing_handler)
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        with _MainThreadAlarm(seconds=0.2):
            pass
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert signal.getsignal(signal.SIGALRM) is enclosing_handler
        assert 0 < remaining <= 1.0, remaining
        assert interval == 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, *original_timer)


if __name__ == "__main__":
    test_unclosed_nested_tsx_always_terminates()
    test_static_parser_rejects_oversized_source_before_parsing()
    test_long_outer_scaffolding_keeps_one_small_fenced_answer()
    test_many_non_jsx_less_than_signs_are_linear()
    test_main_thread_alarm_restores_enclosing_handler_and_deadline()
    print("TSX parser termination regressions: PASS")
