from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "capability_eval.py"
SPEC = importlib.util.spec_from_file_location("gateway_capability_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capability = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capability
SPEC.loader.exec_module(capability)


def test_grounding_scoring_requires_one_exact_label():
    case = {"expected": "REPLICA_LAG"}

    assert capability.score_grounding(case, "REPLICA_LAG")["passed"] is True
    assert capability.score_grounding(case, "Maybe WRITE_FAILURE or REPLICA_LAG")["passed"] is False


def test_tool_scoring_checks_name_arguments_and_restraint():
    weather = {
        "expected_tool": "get_weather",
        "expected_args": {"location": "Shanghai", "unit": "celsius"},
    }
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location":"shanghai","unit":"Celsius"}',
                }
            }
        ]
    }

    assert capability.score_tool(weather, message)["passed"] is True
    restraint = {"expected_tool": None, "expected_content": "4"}
    assert capability.score_tool(restraint, {"content": "4", "tool_calls": []})["passed"] is True


def test_code_scoring_runs_hidden_tests_and_rejects_imports():
    case = {
        "function": "double",
        "tests": [{"args": [3], "expected": 6}, {"args": [-2], "expected": -4}],
    }

    valid = capability.score_code(case, "def double(value):\n    return value * 2")
    unsafe = capability.score_code(case, "import os\ndef double(value):\n    return value * 2")

    assert valid == {"passed": 2, "total": 2, "error": ""}
    assert unsafe["passed"] == 0
    assert "disallowed syntax" in unsafe["error"]


def test_code_scoring_allows_safe_lambda_and_try():
    case = {
        "function": "ordered_ints",
        "tests": [{"args": [["2", "10", "1"]], "expected": [1, 2, 10]}],
    }
    content = """def ordered_ints(values):
    try:
        return sorted([int(value) for value in values], key=lambda value: value)
    except ValueError:
        return []
"""

    result = capability.score_code(case, content)

    assert result == {"passed": 1, "total": 1, "error": ""}


def test_code_scoring_detects_input_mutation():
    case = {
        "function": "sorted_copy",
        "tests": [{"args": [[3, 1, 2]], "expected": [1, 2, 3]}],
    }

    result = capability.score_code(
        case, "def sorted_copy(values):\n    values.sort()\n    return values"
    )

    assert result["passed"] == 0


def test_frontier_detects_material_drop():
    baseline = {
        "name": "1k-c1",
        "context_tokens": 1_024,
        "concurrency": 1,
        "status": "PASS",
        "overall_score": 100.0,
        "code_score": 100.0,
        "grounding_score": 100.0,
        "tool_score": 100.0,
    }
    degraded = {
        "name": "128k-c1",
        "context_tokens": 131_072,
        "concurrency": 1,
        "status": "PASS",
        "overall_score": 84.6,
        "code_score": 66.7,
        "grounding_score": 100.0,
        "tool_score": 100.0,
    }
    degraded_again = {
        **degraded,
        "name": "192k-c1",
        "context_tokens": 192_000,
    }

    frontier = capability.analyze_frontier([baseline, degraded, degraded_again])

    assert frontier["first_anomaly"] == "128k-c1"
    assert frontier["confirmed_degradation"] == "128k-c1"
    assert frontier["stable_frontier"] == "1k-c1"


def test_frontier_does_not_confirm_a_single_sporadic_anomaly():
    baseline = {
        "name": "1k-c1",
        "context_tokens": 1_024,
        "concurrency": 1,
        "status": "PASS",
        "overall_score": 100.0,
        "code_score": 100.0,
        "grounding_score": 100.0,
        "tool_score": 100.0,
    }
    anomaly = {
        **baseline,
        "name": "32k-c1",
        "context_tokens": 32_768,
        "overall_score": 80.0,
        "grounding_score": 60.0,
    }
    recovered = {**baseline, "name": "128k-c1", "context_tokens": 131_072}

    frontier = capability.analyze_frontier([baseline, anomaly, recovered])

    assert frontier["first_anomaly"] == "32k-c1"
    assert frontier["confirmed_degradation"] is None
    assert frontier["stable_frontier"] == "128k-c1"
