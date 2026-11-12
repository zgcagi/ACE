"""Authoritative ACE benchmark release version."""

from __future__ import annotations

import re
from pathlib import Path


VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
BENCHMARK_VERSION = VERSION_FILE.read_text(encoding="utf-8").strip()
SCORER_VERSION = BENCHMARK_VERSION
EVALUATION_CONTRACT_VERSION = "ace-eval-contract-v2.6.4-public"
OFFICIAL_TOTAL_PROBES = 2565
OFFICIAL_EFFECTIVE_PROBES = 2489
OFFICIAL_DISABLED_PROBES = 76

if not re.fullmatch(r"\d+\.\d+\.\d+", BENCHMARK_VERSION):
    raise RuntimeError(f"invalid ACE VERSION value: {BENCHMARK_VERSION!r}")

__version__ = BENCHMARK_VERSION
