#!/usr/bin/env python3
"""Deterministic capability degradation benchmark for a local vLLM model.

The suite measures executable coding correctness, evidence-grounded answers,
and real auto tool calls while context length and request concurrency increase.
It intentionally avoids model-as-judge scoring. Exit code 75 asks the local
service manager to restart vLLM after a probable engine stall.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torture

MODEL_DEFAULT = "qwen3.6-35b-a3b-nvfp4-fast"
BASE_URL_DEFAULT = "http://127.0.0.1:8000"
METRICS_URL_DEFAULT = "http://127.0.0.1:8001/metrics"
TOKENIZE_URL_DEFAULT = "http://127.0.0.1:8001/tokenize"
STALL_EXIT_CODE = 75


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    context_tokens: int
    concurrency: int


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local text file by its exact path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def code_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "code_dedupe",
            "category": "code",
            "function": "dedupe_keep_order",
            "prompt": (
                "Write Python function dedupe_keep_order(items). Return a new list containing the first "
                "occurrence of each hashable item while preserving input order. Return only Python code, "
                "with no imports and no explanation."
            ),
            "tests": [
                {"args": [[3, 1, 3, 2, 1]], "expected": [3, 1, 2]},
                {"args": [[]], "expected": []},
                {"args": [["a", "a", "b", "a"]], "expected": ["a", "b"]},
                {"args": [[False, True, False]], "expected": [False, True]},
            ],
            "max_tokens": 512,
        },
        {
            "id": "code_merge_intervals",
            "category": "code",
            "function": "merge_intervals",
            "prompt": (
                "Write Python function merge_intervals(intervals). Each interval is [start, end]. Sort and "
                "merge every overlapping or touching interval; do not mutate the input; return a list of "
                "two-item lists. Return only Python code, with no imports and no explanation."
            ),
            "tests": [
                {"args": [[[1, 3], [2, 6], [8, 10], [10, 12]]], "expected": [[1, 6], [8, 12]]},
                {"args": [[]], "expected": []},
                {"args": [[[5, 7]]], "expected": [[5, 7]]},
                {"args": [[[4, 5], [1, 2], [2, 4]]], "expected": [[1, 5]]},
            ],
            "max_tokens": 768,
        },
        {
            "id": "code_parse_duration",
            "category": "code",
            "function": "parse_duration",
            "prompt": (
                "Write Python function parse_duration(text). The input contains zero or more whitespace-"
                "separated nonnegative integer terms ending in h, m, or s, for example '1h 30m 5s'. "
                "Return total seconds. Empty or whitespace-only input returns 0. Return only Python code, "
                "with no imports, eval, regex, or explanation."
            ),
            "tests": [
                {"args": ["1h 30m 5s"], "expected": 5405},
                {"args": ["45m"], "expected": 2700},
                {"args": ["2h 0m 9s"], "expected": 7209},
                {"args": ["   "], "expected": 0},
            ],
            "max_tokens": 768,
        },
        {
            "id": "code_balanced_brackets",
            "category": "code",
            "function": "balanced_brackets",
            "prompt": (
                "Write Python function balanced_brackets(text). Check (), [], and {} nesting while ignoring "
                "all other characters. Return True only when every bracket is correctly matched. Return "
                "only Python code, with no imports and no explanation."
            ),
            "tests": [
                {"args": ["a+(b*[c-{d/e}])"], "expected": True},
                {"args": ["([)]"], "expected": False},
                {"args": ["text without brackets"], "expected": True},
                {"args": ["(()"], "expected": False},
            ],
            "max_tokens": 768,
        },
        {
            "id": "code_longest_run",
            "category": "code",
            "function": "longest_run",
            "prompt": (
                "Write Python function longest_run(values). Return [value, length] for the longest run of "
                "equal consecutive values; on ties keep the earliest run. Empty input returns [None, 0]. "
                "Do not mutate the input. Return only Python code, with no imports and no explanation."
            ),
            "tests": [
                {"args": [[1, 1, 2, 2, 2, 1]], "expected": [2, 3]},
                {"args": [["a", "a", "b", "b"]], "expected": ["a", 2]},
                {"args": [[]], "expected": [None, 0]},
                {"args": [[7]], "expected": [7, 1]},
            ],
            "max_tokens": 768,
        },
    ]


def grounding_cases() -> list[dict[str, Any]]:
    instruction = "Use only the supplied evidence. Return exactly one allowed uppercase label."
    return [
        {
            "id": "ground_release_regression",
            "category": "grounding",
            "prompt": (
                f"{instruction}\nAllowed: R19_REGRESSION, BROKER_CAPACITY, INSUFFICIENT_EVIDENCE.\n"
                "Evidence: acknowledgement p95 rose from 74 ms to 1.61 s immediately after consumer R19 "
                "was released; broker CPU stayed below 31% and backlog stayed between 52 and 57; rolling "
                "back R19 restored p95 to 81 ms. What is the supported cause?"
            ),
            "expected": "R19_REGRESSION",
            "max_tokens": 32,
        },
        {
            "id": "ground_insufficient_energy",
            "category": "grounding",
            "prompt": (
                f"{instruction}\nAllowed: MAP_MEMORY_LEAK, BLUETOOTH_DRAIN, INSUFFICIENT_EVIDENCE.\n"
                "Evidence: one phone lost 23% battery overnight while maps, Bluetooth scanning, and an "
                "always-on screen were all enabled. There is no process-level energy data, heap snapshot, "
                "control device, or repeat. What cause is supported?"
            ),
            "expected": "INSUFFICIENT_EVIDENCE",
            "max_tokens": 32,
        },
        {
            "id": "ground_replica_lag",
            "category": "grounding",
            "prompt": (
                f"{instruction}\nAllowed: WRITE_FAILURE, REPLICA_LAG, BAD_PASSWORD.\n"
                "Evidence: the primary committed revision 9284 and returns the new balance; the API read "
                "revision 9271 from a replica delayed by 6.8 seconds; after replay caught up, the API returned "
                "revision 9284. What explains the stale read?"
            ),
            "expected": "REPLICA_LAG",
            "max_tokens": 32,
        },
        {
            "id": "ground_authorization",
            "category": "grounding",
            "prompt": (
                f"{instruction}\nAllowed: AUTHENTICATION_FAILURE, AUTHORIZATION_FAILURE, NETWORK_FAILURE.\n"
                "Evidence: session authentication succeeded; role analyst-east lacks ledger.export; the "
                "endpoint returned 403 at its permission check. Why was export rejected?"
            ),
            "expected": "AUTHORIZATION_FAILURE",
            "max_tokens": 32,
        },
        {
            "id": "ground_measurement_invalid",
            "category": "grounding",
            "prompt": (
                f"{instruction}\nAllowed: CONVERSION_DROP, MEASUREMENT_INVALID, PAYMENT_OUTAGE.\n"
                "Evidence: purchase_click fell from 12.7% to 9.8%, but at rollout the event field changed "
                "from action_name to event_kind. No server order count, payment success count, or field "
                "coverage is available. What can be concluded about real purchase conversion?"
            ),
            "expected": "MEASUREMENT_INVALID",
            "max_tokens": 32,
        },
    ]


def tool_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "tool_weather",
            "category": "tool",
            "prompt": "What is the weather in Shanghai? Use Celsius. Call the appropriate tool.",
            "expected_tool": "get_weather",
            "expected_args": {"location": "Shanghai", "unit": "celsius"},
            "max_tokens": 128,
        },
        {
            "id": "tool_search",
            "category": "tool",
            "prompt": "Search the web for the current official vLLM stable release. Call the appropriate tool.",
            "expected_tool": "web_search",
            "arg_contains": {"query": ["vllm", "release"]},
            "max_tokens": 128,
        },
        {
            "id": "tool_read_file",
            "category": "tool",
            "prompt": "Read the local file F:/workspace/status.txt. Call the appropriate tool.",
            "expected_tool": "read_file",
            "expected_args": {"path": "F:/workspace/status.txt"},
            "max_tokens": 128,
        },
        {
            "id": "tool_calculator",
            "category": "tool",
            "prompt": "Use the calculator tool to evaluate (37 * 48) + 19.",
            "expected_tool": "calculator",
            "arg_contains": {"expression": ["37", "48", "19"]},
            "max_tokens": 128,
        },
        {
            "id": "tool_restraint",
            "category": "tool",
            "prompt": "Do not call any tool. Reply with exactly 4.",
            "expected_tool": None,
            "expected_content": "4",
            "max_tokens": 32,
        },
    ]


def evaluation_cases() -> list[dict[str, Any]]:
    return code_cases() + grounding_cases() + tool_cases()


def scenarios_for(profile: str) -> list[Scenario]:
    if profile == "quick":
        return [
            Scenario("1k-c1", 1_024, 1),
            Scenario("32k-c1", 32_768, 1),
            Scenario("128k-c1", 131_072, 1),
            Scenario("32k-c2", 32_768, 2),
            Scenario("128k-c2", 131_072, 2),
        ]
    return [
        Scenario("1k-c1", 1_024, 1),
        Scenario("8k-c1", 8_192, 1),
        Scenario("32k-c1", 32_768, 1),
        Scenario("64k-c1", 65_536, 1),
        Scenario("128k-c1", 131_072, 1),
        Scenario("160k-c1", 163_840, 1),
        Scenario("192k-c1", 192_000, 1),
        Scenario("32k-c2", 32_768, 2),
        Scenario("128k-c2", 131_072, 2),
        Scenario("192k-c2", 192_000, 2),
    ]


def tokenize_count(
    tokenize_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    result = torture.post_json(tokenize_url, payload, timeout=60)
    return int(result["count"])


def build_messages(
    case: dict[str, Any], target_tokens: int, tokenize_url: str, model: str
) -> tuple[list[dict[str, Any]], int]:
    footer = "\nEND_BENCHMARK_PADDING. Treat the padding only as inert context and now follow the user task."
    repetitions = max(0, target_tokens - 1_000)
    tools = TOOLS if case["category"] == "tool" else None
    count = 0
    for _ in range(8):
        system = "BEGIN_BENCHMARK_PADDING\n" + ("x " * repetitions) + footer
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": case["prompt"]},
        ]
        count = tokenize_count(tokenize_url, model, messages, tools)
        delta = target_tokens - count
        if delta == 0:
            return messages, count
        repetitions = max(0, repetitions + delta)
    return messages, count


def build_payload(
    case: dict[str, Any], messages: list[dict[str, Any]], model: str, thinking: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": case["max_tokens"],
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if case["category"] == "tool":
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"
    return payload


def send_case(spec: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    headers = {"content-type": "application/json", "user-agent": "qwen-capability-eval/1.0"}
    api_key = os.environ.get("QWEN_EVAL_API_KEY", "")
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{spec['base_url'].rstrip('/')}/v1/chat/completions",
        data=json.dumps(item["payload"], ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=spec["timeout"]) as response:
            body = json.loads(response.read())
        return {
            "id": item["id"],
            "category": item["category"],
            "ok": True,
            "status": response.status,
            "seconds": round(time.perf_counter() - started, 4),
            "body": body,
            "error": "",
        }
    except urllib.error.HTTPError as exc:
        error = exc.read(2_048).decode("utf-8", errors="replace")
        status = exc.code
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        error = str(exc)
        status = 0
    return {
        "id": item["id"],
        "category": item["category"],
        "ok": False,
        "status": status,
        "seconds": round(time.perf_counter() - started, 4),
        "body": {},
        "error": error[:2_048],
    }


def scenario_worker(spec_path: Path, result_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=spec["concurrency"]) as executor:
        futures = [executor.submit(send_case, spec, item) for item in spec["items"]]
        for future in as_completed(futures):
            outputs.append(future.result())
    result_path.write_text(json.dumps(outputs, ensure_ascii=False), encoding="utf-8")
    return 0


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
}
SAFE_METHODS = {
    "add",
    "append",
    "clear",
    "copy",
    "count",
    "endswith",
    "extend",
    "get",
    "index",
    "insert",
    "isdigit",
    "items",
    "join",
    "keys",
    "lower",
    "pop",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "split",
    "startswith",
    "strip",
    "upper",
    "values",
}
DISALLOWED_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


def extract_code(content: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
    return (fenced.group(1) if fenced else content).strip()


def validate_code(tree: ast.AST, expected_function: str) -> None:
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if expected_function not in defined:
        raise ValueError(f"missing function {expected_function}")
    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are disallowed")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr not in SAFE_METHODS
        ):
            raise ValueError(f"disallowed attribute: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS and node.func.id not in defined:
                    raise ValueError(f"disallowed call: {node.func.id}")
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError("indirect calls are disallowed")


def normalize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    return value


def code_worker(spec_path: Path, result_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"passed": 0, "total": len(spec["tests"]), "error": ""}
    try:
        code = extract_code(spec["content"])
        tree = ast.parse(code)
        validate_code(tree, spec["function"])
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        exec(compile(tree, "<model-code>", "exec"), namespace)  # noqa: S102 - AST is restricted
        function = namespace[spec["function"]]
        for test in spec["tests"]:
            arguments = test["args"]
            arguments_before = json.loads(json.dumps(arguments))
            actual = function(*arguments)
            if normalize(actual) == normalize(test["expected"]) and normalize(
                arguments
            ) == normalize(arguments_before):
                result["passed"] += 1
    except Exception as exc:  # noqa: BLE001 - evaluator records model failures
        result["error"] = f"{type(exc).__name__}: {exc}"
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


def response_message(output: dict[str, Any]) -> dict[str, Any]:
    choices = output.get("body", {}).get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def score_code(case: dict[str, Any], content: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qwen-code-score-") as temporary:
        root = Path(temporary)
        spec_path = root / "spec.json"
        result_path = root / "result.json"
        spec_path.write_text(
            json.dumps(
                {
                    "function": case["function"],
                    "tests": case["tests"],
                    "content": content,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--code-spec",
                str(spec_path),
                "--code-result",
                str(result_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return {"passed": 0, "total": len(case["tests"]), "error": "execution timeout"}
        if not result_path.exists():
            return {"passed": 0, "total": len(case["tests"]), "error": "scorer failed"}
        return json.loads(result_path.read_text(encoding="utf-8"))


def score_grounding(case: dict[str, Any], content: str) -> dict[str, Any]:
    labels = re.findall(r"\b[A-Z][A-Z0-9_]+\b", content.upper())
    predicted = labels[-1] if labels else ""
    passed = predicted == case["expected"] and len(set(labels)) == 1
    refusal_terms = ("CANNOT ANSWER", "CANNOT COMPLY", "UNABLE TO", "I'M SORRY")
    failure = ""
    if not passed:
        failure = (
            "refusal"
            if any(term in content.upper() for term in refusal_terms)
            else "wrong_or_unsupported"
        )
    return {
        "passed": passed,
        "expected": case["expected"],
        "predicted": predicted,
        "failure": failure,
    }


def parse_tool_arguments(call: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(arguments, dict):
            return name, {}, "arguments are not an object"
        return name, arguments, ""
    except json.JSONDecodeError as exc:
        return name, {}, f"invalid JSON arguments: {exc}"


def equal_loose(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().casefold() == expected.strip().casefold()
    return actual == expected


def score_tool(case: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    calls = message.get("tool_calls") or []
    if case["expected_tool"] is None:
        content = str(message.get("content") or "").strip()
        passed = not calls and content == case["expected_content"]
        return {
            "passed": passed,
            "expected_tool": None,
            "actual_tool": (calls[0].get("function") or {}).get("name") if calls else None,
            "arguments": {},
            "error": "" if passed else "unexpected tool call or wrong direct answer",
        }
    if len(calls) != 1:
        return {
            "passed": False,
            "expected_tool": case["expected_tool"],
            "actual_tool": None,
            "arguments": {},
            "error": f"expected one tool call, got {len(calls)}",
        }
    name, arguments, error = parse_tool_arguments(calls[0])
    passed = not error and name == case["expected_tool"]
    for key, expected in case.get("expected_args", {}).items():
        passed = passed and key in arguments and equal_loose(arguments[key], expected)
    for key, fragments in case.get("arg_contains", {}).items():
        value = str(arguments.get(key, "")).casefold()
        passed = passed and all(fragment.casefold() in value for fragment in fragments)
    return {
        "passed": passed,
        "expected_tool": case["expected_tool"],
        "actual_tool": name,
        "arguments": arguments,
        "error": error,
    }


def score_output(case: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    base = {
        "id": case["id"],
        "category": case["category"],
        "transport_ok": bool(output.get("ok")),
        "seconds": output.get("seconds"),
        "error": output.get("error", ""),
    }
    if not output.get("ok"):
        return {**base, "passed": False, "detail": {"error": output.get("error", "transport")}}
    message = response_message(output)
    content = str(message.get("content") or "")
    if case["category"] == "code":
        detail = score_code(case, content)
        passed = detail["passed"] == detail["total"]
    elif case["category"] == "grounding":
        detail = score_grounding(case, content)
        passed = bool(detail["passed"])
    else:
        detail = score_tool(case, message)
        passed = bool(detail["passed"])
    return {
        **base,
        "passed": passed,
        "detail": detail,
        "content": content,
        "content_preview": content[:300].replace("\n", "\\n"),
    }


def category_score(results: list[dict[str, Any]], category: str) -> float:
    selected = [item for item in results if item["category"] == category]
    return (
        round(100 * sum(item["passed"] for item in selected) / len(selected), 1)
        if selected
        else 0.0
    )


def summarize_scenario(
    scenario: Scenario,
    results: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    seconds: float,
    stalled: bool,
    error: str,
) -> dict[str, Any]:
    scores = {
        category: category_score(results, category) for category in ("code", "grounding", "tool")
    }
    overall = (
        round(100 * sum(item["passed"] for item in results) / len(results), 1) if results else 0.0
    )
    latencies = [float(item["seconds"]) for item in results if item.get("seconds") is not None]
    return {
        "name": scenario.name,
        "context_tokens": scenario.context_tokens,
        "concurrency": scenario.concurrency,
        "status": "STALL"
        if stalled
        else ("PASS" if results and all(item["transport_ok"] for item in results) else "FAIL"),
        "overall_score": overall,
        "code_score": scores["code"],
        "grounding_score": scores["grounding"],
        "hallucination_rate": round(100 - scores["grounding"], 1),
        "tool_score": scores["tool"],
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "wall_seconds": round(seconds, 3),
        "latency_mean_seconds": round(statistics.fmean(latencies), 3) if latencies else None,
        "latency_p95_seconds": torture.percentile(latencies, 0.95),
        "max_running": max((sample["running"] for sample in samples), default=0),
        "max_waiting": max((sample["waiting"] for sample in samples), default=0),
        "max_kv_cache_percent": max((sample["kv_cache_percent"] for sample in samples), default=0),
        "stalled": stalled,
        "error": error,
        "results": results,
        "samples": samples,
    }


def analyze_frontier(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {
            "baseline_weaknesses": [],
            "first_anomaly": None,
            "confirmed_degradation": None,
            "stable_frontier": None,
        }
    baseline = summaries[0]
    categories = ("code_score", "grounding_score", "tool_score")
    weaknesses = [name.removesuffix("_score") for name in categories if baseline[name] < 80]
    anomalies: list[str] = []
    for summary in summaries[1:]:
        overall_drop = baseline["overall_score"] - summary["overall_score"]
        category_drop = max(baseline[name] - summary[name] for name in categories)
        if summary["status"] != "PASS" or overall_drop >= 10 or category_drop >= 20:
            anomalies.append(summary["name"])

    confirmed = None
    stable_frontier = summaries[-1]["name"]
    summary_index = {summary["name"]: index for index, summary in enumerate(summaries)}
    candidates: list[tuple[int, str, str]] = []
    for concurrency in sorted({int(summary["concurrency"]) for summary in summaries}):
        path = sorted(
            (summary for summary in summaries if int(summary["concurrency"]) == concurrency),
            key=lambda summary: int(summary["context_tokens"]),
        )
        for index in range(len(path) - 1):
            if path[index]["name"] in anomalies and path[index + 1]["name"] in anomalies:
                previous = path[index - 1]["name"] if index > 0 else baseline["name"]
                candidates.append(
                    (summary_index[path[index]["name"]], path[index]["name"], previous)
                )
                break
    if candidates:
        _, confirmed, stable_frontier = min(candidates)
    return {
        "baseline_weaknesses": weaknesses,
        "baseline_overall_score": baseline["overall_score"],
        "first_anomaly": anomalies[0] if anomalies else None,
        "anomalies": anomalies,
        "confirmed_degradation": confirmed,
        "stable_frontier": stable_frontier,
        "rule": (
            "anomaly: overall drop >=10pp, category drop >=20pp, or transport failure; "
            "confirmed: two adjacent context levels at the same concurrency are anomalous"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    frontier = report["frontier"]
    weakness = ", ".join(frontier["baseline_weaknesses"]) or "none"
    lines = [
        "# Qwen capability degradation report",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Thinking: `{report['thinking']}`",
        f"- Repetitions per case: `{report['repetitions']}`",
        f"- Model: `{report['model']}`",
        f"- Baseline score: **{frontier.get('baseline_overall_score', 0):.1f}%**",
        f"- Baseline weaknesses (<80%): **{weakness}**",
        f"- First observed anomaly: **{frontier.get('first_anomaly') or 'not observed'}**",
        f"- Confirmed degradation threshold: **{frontier.get('confirmed_degradation') or 'not observed'}**",
        f"- Stable frontier in this run: **{frontier.get('stable_frontier') or 'none'}**",
        "",
        "| Scenario | Status | Context | C | Overall | Code | Grounded | Hallucination | Tools | P95 latency | KV peak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["scenarios"]:
        latency = (
            "-" if item["latency_p95_seconds"] is None else f"{item['latency_p95_seconds']:.2f}s"
        )
        lines.append(
            f"| {item['name']} | {item['status']} | {item['context_tokens']:,} | {item['concurrency']} | "
            f"{item['overall_score']:.1f}% | {item['code_score']:.1f}% | {item['grounding_score']:.1f}% | "
            f"{item['hallucination_rate']:.1f}% | {item['tool_score']:.1f}% | {latency} | "
            f"{item['max_kv_cache_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Scoring",
            "",
            "- Code is executed against hidden unit tests after AST safety validation.",
            "- Grounding uses closed-book evidence and exact labels; a wrong label counts toward hallucination rate.",
            "- Tool scoring checks auto-selected function names, JSON arguments, and restraint when no tool is allowed.",
            "- The degradation frontier is relative to this run's 1K single-request baseline, not a universal model ranking.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["frontier"] = analyze_frontier(report["scenarios"])
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def run_scenario(
    args: argparse.Namespace,
    scenario: Scenario,
    items: list[dict[str, Any]],
    env: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, bool, str]:
    spec = {
        "base_url": args.base_url,
        "concurrency": scenario.concurrency,
        "timeout": args.request_timeout,
        "items": items,
    }
    samples: list[dict[str, Any]] = []
    stalled = False
    error = ""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="qwen-capability-") as temporary:
        root = Path(temporary)
        spec_path = root / "spec.json"
        result_path = root / "results.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--scenario-spec",
                str(spec_path),
                "--scenario-result",
                str(result_path),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        last_progress = time.monotonic()
        signature: tuple[float, ...] | None = None
        failures = 0
        waves = math.ceil(len(items) / scenario.concurrency)
        hard_deadline = started + args.request_timeout * waves + 30
        while process.poll() is None:
            try:
                sample = torture.fetch_sample(args.metrics_url, started)
                samples.append(sample)
                failures = 0
                current = torture.progress_signature(sample)
                if signature is None or current != signature:
                    signature = current
                    last_progress = time.monotonic()
                if sample["running"] > 0 and time.monotonic() - last_progress >= args.stall_seconds:
                    stalled = True
                    error = f"no vLLM engine progress for {args.stall_seconds:.0f}s"
                    process.kill()
                    break
            except (OSError, urllib.error.URLError, ValueError):
                failures += 1
                if failures >= 3:
                    error = "metrics endpoint failed three consecutive checks"
                    process.kill()
                    break
            if time.monotonic() >= hard_deadline:
                error = "scenario exceeded its hard deadline"
                process.kill()
                break
            time.sleep(args.sample_interval)
        _, stderr = process.communicate(timeout=10)
        outputs: list[dict[str, Any]] = []
        if result_path.exists():
            outputs = json.loads(result_path.read_text(encoding="utf-8"))
        elif not error:
            error = stderr.strip()[:2_048] or f"worker exited {process.returncode}"
    return outputs, samples, time.monotonic() - started, stalled, error


def run_controller(args: argparse.Namespace) -> int:
    scenarios = scenarios_for(args.profile)
    cases = evaluation_cases()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "thinking": args.thinking,
                    "repetitions": args.repetitions,
                    "scenarios": [asdict(item) for item in scenarios],
                    "cases": [{"id": item["id"], "category": item["category"]} for item in cases],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not torture.service_is_ready(args.base_url, args.model):
        print("vLLM/Gateway is not ready or serves another model.", file=sys.stderr)
        return 2
    initial = torture.fetch_sample(args.metrics_url, time.monotonic())
    if not args.force and (initial["running"] > 0 or initial["waiting"] > 0):
        print("Refusing to evaluate while another request is active.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"qwen-capability-{args.profile}-{args.thinking}-{stamp}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    report: dict[str, Any] = {
        "profile": args.profile,
        "thinking": args.thinking,
        "repetitions": args.repetitions,
        "model": args.model,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "finished_at": "",
        "scenarios": [],
        "frontier": {},
    }
    env = os.environ.copy()
    env["QWEN_EVAL_API_KEY"] = args.api_key
    print(
        f"Qwen capability eval profile={args.profile} thinking={args.thinking} "
        f"scenarios={len(scenarios)} cases={len(cases)} repetitions={args.repetitions}"
    )
    print(f"Reports: {markdown_path}")

    final_code = 0
    for index, scenario in enumerate(scenarios, 1):
        print(
            f"[{index}/{len(scenarios)}] {scenario.name}: "
            f"context={scenario.context_tokens:,} concurrency={scenario.concurrency}"
        )
        items: list[dict[str, Any]] = []
        instances: list[tuple[str, dict[str, Any]]] = []
        build_error = ""
        try:
            for case in cases:
                messages, actual_count = build_messages(
                    case, scenario.context_tokens, args.tokenize_url, args.model
                )
                payload = build_payload(case, messages, args.model, args.thinking == "on")
                for repetition in range(1, args.repetitions + 1):
                    instance_id = f"{case['id']}#{repetition}"
                    instances.append((instance_id, case))
                    items.append(
                        {
                            "id": instance_id,
                            "category": case["category"],
                            "prompt_tokens": actual_count,
                            "payload": payload,
                        }
                    )
        except (OSError, urllib.error.URLError, KeyError, ValueError) as exc:
            build_error = f"prompt build failed: {exc}"

        if build_error:
            summary = summarize_scenario(scenario, [], [], 0, False, build_error)
        else:
            outputs, samples, elapsed, stalled, error = run_scenario(args, scenario, items, env)
            by_id = {item["id"]: item for item in outputs}
            scored = []
            for instance_id, case in instances:
                score = score_output(
                    case, by_id.get(instance_id, {"ok": False, "error": "missing output"})
                )
                score["id"] = instance_id
                score["case_id"] = case["id"]
                scored.append(score)
            summary = summarize_scenario(scenario, scored, samples, elapsed, stalled, error)
        report["scenarios"].append(summary)
        print(
            f"  {summary['status']}: overall={summary['overall_score']:.1f}% "
            f"code={summary['code_score']:.1f}% grounded={summary['grounding_score']:.1f}% "
            f"tools={summary['tool_score']:.1f}%"
        )
        write_report(report, json_path, markdown_path)
        if summary["status"] == "STALL":
            final_code = STALL_EXIT_CODE
            break
        if summary["status"] == "FAIL":
            final_code = 1
            break

    write_report(report, json_path, markdown_path)
    frontier = report["frontier"]
    print(f"Baseline: {frontier.get('baseline_overall_score', 0):.1f}%")
    print(f"First anomaly: {frontier.get('first_anomaly') or 'not observed'}")
    print(f"Confirmed degradation: {frontier.get('confirmed_degradation') or 'not observed'}")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")
    return final_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", choices=("quick", "full"), default="quick")
    parser.add_argument("--thinking", choices=("off", "on"), default="off")
    parser.add_argument("--repetitions", type=int, default=0)
    parser.add_argument("--base-url", default=BASE_URL_DEFAULT)
    parser.add_argument("--metrics-url", default=METRICS_URL_DEFAULT)
    parser.add_argument("--tokenize-url", default=TOKENIZE_URL_DEFAULT)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--api-key", default=os.environ.get("GATEWAY_API_KEY", ""))
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "capability"),
    )
    parser.add_argument("--stall-seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scenario-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--code-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--code-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stall_seconds <= 0 or args.sample_interval <= 0 or args.request_timeout <= 0:
        parser.error("timeouts and sample interval must be positive")
    if args.repetitions < 0:
        parser.error("--repetitions cannot be negative")
    args.repetitions = args.repetitions or (2 if args.profile == "full" else 1)
    return args


def main() -> int:
    args = parse_args()
    if args.scenario_spec:
        if not args.scenario_result:
            raise SystemExit("--scenario-result is required")
        return scenario_worker(args.scenario_spec, args.scenario_result)
    if args.code_spec:
        if not args.code_result:
            raise SystemExit("--code-result is required")
        return code_worker(args.code_spec, args.code_result)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
