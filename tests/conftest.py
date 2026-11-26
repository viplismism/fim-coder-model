"""Shared test helpers. Tests run from the repo root (`python -m pytest tests`)."""
import pytest
from core import config as sources

BENCHMARK = sources.ROOT / "work" / "sources" / "real_fim_eval" / "add" / "test.jsonl.gz"
needs_benchmark = pytest.mark.skipif(not BENCHMARK.exists(), reason="Real-FIM-Eval not fetched (bash scripts/fetch.sh rust)")
