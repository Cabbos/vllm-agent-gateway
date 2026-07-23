from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "torture.py"
SPEC = importlib.util.spec_from_file_location("gateway_torture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
torture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = torture
SPEC.loader.exec_module(torture)


def test_parse_prometheus_sums_labelled_series():
    metrics = """
# HELP ignored comment
vllm:num_requests_running{engine="0",model_name="a"} 1.0
vllm:num_requests_running{engine="1",model_name="a"} 2.0
vllm:kv_cache_usage_perc{engine="0"} 0.25
"""

    parsed = torture.parse_prometheus(metrics)

    assert parsed["vllm:num_requests_running"] == 3.0
    assert parsed["vllm:kv_cache_usage_perc"] == 0.25


def test_profiles_are_escalating_and_extreme_includes_192k_dual():
    safe = torture.levels_for("safe")
    extreme = torture.levels_for("extreme")

    assert len(safe) < len(extreme)
    assert extreme[: len(safe)] == safe
    assert any(level.prompt_tokens == 192_000 and level.concurrency == 2 for level in extreme)
    assert max(level.concurrency for level in extreme) == 8


def test_percentile_uses_nearest_rank():
    assert torture.percentile([], 0.95) is None
    assert torture.percentile([1.0, 4.0, 2.0, 3.0], 0.50) == 2.0
    assert torture.percentile([1.0, 4.0, 2.0, 3.0], 0.95) == 4.0
