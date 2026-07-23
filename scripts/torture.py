#!/usr/bin/env python3
"""Escalating, self-observing stress test for a local vLLM chat service.

The controller launches every level in a disposable child process.  This lets
the caller terminate blocked HTTP clients when vLLM is alive but its engine no
longer advances.  Exit code 75 means that a probable engine stall was detected
and the service manager should restart vLLM.
"""

from __future__ import annotations

import argparse
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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

STALL_EXIT_CODE = 75
MODEL_DEFAULT = "qwen3.6-35b-a3b-nvfp4-fast"
BASE_URL_DEFAULT = "http://127.0.0.1:8000"
METRICS_URL_DEFAULT = "http://127.0.0.1:8001/metrics"
TOKENIZE_URL_DEFAULT = "http://127.0.0.1:8001/tokenize"
PROMETHEUS_NUMBER = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)


@dataclass(frozen=True, slots=True)
class Level:
    name: str
    prompt_tokens: int
    concurrency: int
    requests: int
    max_tokens: int
    force_decode: bool = False


@dataclass(frozen=True, slots=True)
class RequestResult:
    request_number: int
    status: int
    ok: bool
    quality_ok: bool | None
    ttft_seconds: float | None
    total_seconds: float
    completion_tokens: int
    output_chars: int
    output_preview: str
    error: str = ""


def levels_for(profile: str) -> list[Level]:
    if profile == "duplex-192k":
        return [
            Level("context-192k-x2", 196_544, 2, 2, 64),
        ]
    if profile == "duplex-256k":
        return [
            Level("context-256k-x2", 262_080, 2, 2, 64),
        ]
    safe = [
        Level("baseline-1", 1_024, 1, 2, 32),
        Level("batch-2", 1_024, 2, 4, 64),
        Level("context-8k-x2", 8_192, 2, 2, 64),
        Level("context-32k-x2", 32_768, 2, 2, 64),
        Level("decode-256-x2", 1_024, 2, 2, 256, True),
    ]
    if profile == "safe":
        return safe
    return safe + [
        Level("queue-4", 1_024, 4, 8, 64),
        Level("queue-8", 1_024, 8, 16, 64),
        Level("context-64k", 65_536, 1, 1, 64),
        Level("context-128k", 131_072, 1, 1, 64),
        Level("context-160k", 163_840, 1, 1, 64),
        Level("context-192k", 192_000, 1, 1, 64),
        Level("context-64k-x2", 65_536, 2, 2, 64),
        Level("context-128k-x2", 131_072, 2, 2, 64),
        Level("context-160k-x2", 163_840, 2, 2, 64),
        Level("context-192k-x2", 192_000, 2, 2, 64),
        Level("decode-1024-x2", 1_024, 2, 2, 1_024, True),
    ]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get_text(url: str, timeout: float = 5.0) -> str:
    request = urllib.request.Request(url, headers={"user-agent": "qwen-torture/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_prometheus(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_NUMBER.match(line)
        if not match:
            continue
        name = match.group("name")
        values[name] = values.get(name, 0.0) + float(match.group("value"))
    return values


def fetch_sample(metrics_url: str, started: float) -> dict[str, Any]:
    values = parse_prometheus(get_text(metrics_url))
    sample: dict[str, Any] = {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "running": values.get("vllm:num_requests_running", 0.0),
        "waiting": values.get("vllm:num_requests_waiting", 0.0),
        "kv_cache_percent": round(values.get("vllm:kv_cache_usage_perc", 0.0) * 100, 3),
        "iteration_count": values.get("vllm:iteration_tokens_total_count", 0.0),
        "iteration_tokens": values.get("vllm:iteration_tokens_total_sum", 0.0),
        "prompt_tokens_total": values.get("vllm:prompt_tokens_total", 0.0),
        "generation_tokens_total": values.get("vllm:generation_tokens_total", 0.0),
    }
    sample.update(gpu_sample())
    return sample


def gpu_sample() -> dict[str, float | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
        return {
            "gpu_util_percent": float(fields[0]),
            "gpu_memory_used_mib": float(fields[1]),
            "gpu_memory_total_mib": float(fields[2]),
            "gpu_power_watts": float(fields[3]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {
            "gpu_util_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
            "gpu_power_watts": None,
        }


def progress_signature(sample: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(sample["iteration_count"]),
        float(sample["iteration_tokens"]),
        float(sample["prompt_tokens_total"]),
        float(sample["generation_tokens_total"]),
        float(sample["kv_cache_percent"]),
    )


def build_exact_prompt(
    tokenize_url: str,
    model: str,
    target_tokens: int,
    tag: str,
    force_decode: bool,
) -> tuple[str, int]:
    prefix = f"Capacity benchmark sample {tag}. The x tokens below are inert padding.\n"
    if force_decode:
        suffix = "\nBegin with OK, then continue producing short numbered items until stopped."
    else:
        suffix = "\nEnd of padding. Return exactly the two uppercase letters OK."
    repetitions = max(0, target_tokens - 48)
    last_count = 0
    for _ in range(8):
        prompt = prefix + ("x " * repetitions) + suffix
        tokenized = post_json(
            tokenize_url,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        last_count = int(tokenized["count"])
        delta = target_tokens - last_count
        if delta == 0:
            return prompt, last_count
        repetitions = max(0, repetitions + delta)
    return prompt, last_count


def extract_content(delta: Any) -> str:
    if isinstance(delta, str):
        return delta
    if not isinstance(delta, list):
        return ""
    parts: list[str] = []
    for item in delta:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)


def send_streaming_request(spec: dict[str, Any], request_number: int) -> RequestResult:
    prompt = spec["prompts"][request_number - 1]
    payload: dict[str, Any] = {
        "model": spec["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": spec["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if spec["force_decode"]:
        payload["ignore_eos"] = True
    headers = {
        "content-type": "application/json",
        "accept": "text/event-stream",
        "user-agent": "qwen-torture/1.0",
    }
    api_key = os.environ.get("QWEN_TORTURE_API_KEY", "")
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{spec['base_url'].rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    ttft: float | None = None
    output: list[str] = []
    completion_tokens = 0
    status = 0
    error = ""
    try:
        with urllib.request.urlopen(request, timeout=spec["timeout"]) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                usage = event.get("usage")
                if isinstance(usage, dict):
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                choices = event.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    content = extract_content((choices[0].get("delta") or {}).get("content"))
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        output.append(content)
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read(1_024).decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        error = str(exc)
    text = "".join(output)
    transport_ok = 200 <= status < 300 and not error
    quality_ok = None if spec["force_decode"] else text.strip() == "OK"
    return RequestResult(
        request_number=request_number,
        status=status,
        ok=transport_ok,
        quality_ok=quality_ok,
        ttft_seconds=round(ttft, 4) if ttft is not None else None,
        total_seconds=round(time.perf_counter() - started, 4),
        completion_tokens=completion_tokens,
        output_chars=len(text),
        output_preview=text[:120].replace("\n", "\\n"),
        error=error[:1_024],
    )


def worker_main(spec_path: Path, result_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=spec["concurrency"]) as executor:
        futures = [
            executor.submit(send_streaming_request, spec, index + 1)
            for index in range(spec["requests"])
        ]
        for future in as_completed(futures):
            results.append(future.result())
    result_path.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 4)


def summarize_level(
    level: Level,
    token_counts: list[int],
    results: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    wall_seconds: float,
    stalled: bool,
    error: str,
) -> dict[str, Any]:
    ttfts = [float(item["ttft_seconds"]) for item in results if item["ttft_seconds"] is not None]
    totals = [float(item["total_seconds"]) for item in results]
    quality_checked = [item["quality_ok"] for item in results if item["quality_ok"] is not None]
    gpu_values = [
        item["gpu_util_percent"] for item in samples if item["gpu_util_percent"] is not None
    ]
    memory_values = [
        item["gpu_memory_used_mib"] for item in samples if item["gpu_memory_used_mib"] is not None
    ]
    status = "STALL" if stalled else "PASS"
    if not stalled and (not results or not all(item["ok"] for item in results)):
        status = "FAIL"
    elif quality_checked and not all(quality_checked):
        status = "DEGRADED"
    return {
        "name": level.name,
        "status": status,
        "prompt_tokens_target": level.prompt_tokens,
        "prompt_tokens_actual": token_counts,
        "concurrency": level.concurrency,
        "requests": level.requests,
        "max_tokens": level.max_tokens,
        "force_decode": level.force_decode,
        "succeeded": sum(bool(item["ok"]) for item in results),
        "quality_passed": sum(item["quality_ok"] is True for item in results),
        "quality_checked": len(quality_checked),
        "wall_seconds": round(wall_seconds, 4),
        "requests_per_second": round(len(results) / wall_seconds, 4) if wall_seconds else 0,
        "ttft_p50_seconds": percentile(ttfts, 0.50),
        "ttft_p95_seconds": percentile(ttfts, 0.95),
        "latency_mean_seconds": round(statistics.fmean(totals), 4) if totals else None,
        "latency_p95_seconds": percentile(totals, 0.95),
        "completion_tokens": sum(int(item["completion_tokens"]) for item in results),
        "max_running": max((item["running"] for item in samples), default=0),
        "max_waiting": max((item["waiting"] for item in samples), default=0),
        "max_kv_cache_percent": max((item["kv_cache_percent"] for item in samples), default=0),
        "max_gpu_util_percent": max(gpu_values, default=None),
        "max_gpu_memory_used_mib": max(memory_values, default=None),
        "stalled": stalled,
        "error": error,
        "results": results,
        "samples": samples,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen torture report",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Started: `{report['started_at']}`",
        f"- Finished: `{report['finished_at']}`",
        f"- Verdict: **{report['verdict']}**",
        f"- Model: `{report['model']}`",
        "",
        "| Level | Status | Prompt | Concurrency | Requests | Output cap | OK | TTFT P95 | Latency P95 | Max KV | Max GPU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in report["levels"]:
        ttft = "-" if level["ttft_p95_seconds"] is None else f"{level['ttft_p95_seconds']:.2f}s"
        latency = (
            "-" if level["latency_p95_seconds"] is None else f"{level['latency_p95_seconds']:.2f}s"
        )
        lines.append(
            f"| {level['name']} | {level['status']} | {level['prompt_tokens_target']:,} | "
            f"{level['concurrency']} | {level['requests']} | {level['max_tokens']:,} | "
            f"{level['succeeded']}/{level['requests']} | {ttft} | {latency} | "
            f"{level['max_kv_cache_percent']:.1f}% | {level['max_gpu_util_percent'] or 0:.0f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PASS`: transport and exact-answer anchor succeeded.",
            "- `DEGRADED`: requests completed, but at least one exact-answer anchor failed.",
            "- `FAIL`: HTTP/client failure occurred; later levels were not attempted.",
            "- `STALL`: vLLM reported work but iteration/KV/token metrics stopped advancing; service restart is recommended.",
            "- Concurrency above vLLM `max-num-seqs` measures queue behavior, not additional GPU execution lanes.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def service_is_ready(base_url: str, model: str) -> bool:
    try:
        payload = json.loads(get_text(f"{base_url.rstrip('/')}/v1/models"))
        return any(item.get("id") == model for item in payload.get("data", []))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def run_preflight(args: argparse.Namespace, levels: list[Level]) -> int | None:
    if args.dry_run:
        print(json.dumps([asdict(level) for level in levels], ensure_ascii=False, indent=2))
        return 0
    if not service_is_ready(args.base_url, args.model):
        print("vLLM/Gateway is not ready or serves a different model.", file=sys.stderr)
        return 2
    initial = fetch_sample(args.metrics_url, time.monotonic())
    if not args.force and (initial["running"] > 0 or initial["waiting"] > 0):
        print(
            "Refusing to start while another request is active. Use --force only in an isolated test window.",
            file=sys.stderr,
        )
        return 2
    return None


def run_controller(args: argparse.Namespace) -> int:
    levels = levels_for(args.profile)
    if (preflight_code := run_preflight(args, levels)) is not None:
        return preflight_code

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"qwen-torture-{args.profile}-{stamp}.json"
    markdown_path = output_dir / f"qwen-torture-{args.profile}-{stamp}.md"
    report: dict[str, Any] = {
        "profile": args.profile,
        "model": args.model,
        "base_url": args.base_url,
        "metrics_url": args.metrics_url,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "finished_at": "",
        "verdict": "RUNNING",
        "stall_seconds": args.stall_seconds,
        "levels": [],
    }
    env = os.environ.copy()
    env["QWEN_TORTURE_API_KEY"] = args.api_key

    print(f"Qwen torture profile={args.profile} levels={len(levels)}")
    print(f"Reports: {markdown_path}")
    final_code = 0
    for position, level in enumerate(levels, 1):
        print(
            f"[{position}/{len(levels)}] {level.name}: prompt={level.prompt_tokens:,} "
            f"concurrency={level.concurrency} requests={level.requests} output={level.max_tokens}"
        )
        token_counts: list[int] = []
        prompts: list[str] = []
        try:
            for request_number in range(level.requests):
                tag = f"{stamp}-{level.name}-{request_number + 1}-{uuid.uuid4().hex[:8]}"
                prompt, count = build_exact_prompt(
                    args.tokenize_url,
                    args.model,
                    level.prompt_tokens,
                    tag,
                    level.force_decode,
                )
                prompts.append(prompt)
                token_counts.append(count)
        except (OSError, urllib.error.URLError, KeyError, ValueError) as exc:
            summary = summarize_level(level, token_counts, [], [], 0, False, f"tokenize: {exc}")
            report["levels"].append(summary)
            report["verdict"] = "FAIL"
            final_code = 1
            break

        spec = {
            "base_url": args.base_url,
            "model": args.model,
            "concurrency": level.concurrency,
            "requests": level.requests,
            "max_tokens": level.max_tokens,
            "force_decode": level.force_decode,
            "timeout": args.request_timeout,
            "prompts": prompts,
        }
        started = time.monotonic()
        samples: list[dict[str, Any]] = []
        stalled = False
        level_error = ""
        with tempfile.TemporaryDirectory(prefix="qwen-torture-") as temporary:
            temp_dir = Path(temporary)
            spec_path = temp_dir / "spec.json"
            result_path = temp_dir / "results.json"
            spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-spec",
                    str(spec_path),
                    "--worker-result",
                    str(result_path),
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            last_progress = time.monotonic()
            last_signature: tuple[float, ...] | None = None
            metric_failures = 0
            hard_deadline = (
                started + args.request_timeout * math.ceil(level.requests / level.concurrency) + 30
            )
            while process.poll() is None:
                try:
                    sample = fetch_sample(args.metrics_url, started)
                    samples.append(sample)
                    metric_failures = 0
                    signature = progress_signature(sample)
                    if last_signature is None or signature != last_signature:
                        last_progress = time.monotonic()
                        last_signature = signature
                    if (
                        sample["running"] > 0
                        and time.monotonic() - last_progress >= args.stall_seconds
                    ):
                        stalled = True
                        level_error = (
                            f"no engine progress for {args.stall_seconds:.0f}s while "
                            f"running={sample['running']:.0f}"
                        )
                        process.kill()
                        break
                except (OSError, urllib.error.URLError, ValueError):
                    metric_failures += 1
                    if metric_failures >= 3:
                        level_error = "vLLM metrics endpoint failed three consecutive checks"
                        process.kill()
                        break
                if time.monotonic() >= hard_deadline:
                    level_error = "level exceeded its hard client deadline"
                    process.kill()
                    break
                time.sleep(args.sample_interval)
            _, worker_stderr = process.communicate(timeout=10)
            results: list[dict[str, Any]] = []
            if result_path.exists():
                results = json.loads(result_path.read_text(encoding="utf-8"))
            elif not level_error:
                level_error = worker_stderr.strip()[:1_024] or f"worker exited {process.returncode}"

        wall_seconds = time.monotonic() - started
        summary = summarize_level(
            level,
            token_counts,
            results,
            samples,
            wall_seconds,
            stalled,
            level_error,
        )
        report["levels"].append(summary)
        print(
            f"  {summary['status']}: ok={summary['succeeded']}/{level.requests} "
            f"ttft_p95={summary['ttft_p95_seconds']}s latency_p95={summary['latency_p95_seconds']}s "
            f"kv={summary['max_kv_cache_percent']:.1f}%"
        )
        if summary["status"] == "STALL":
            report["verdict"] = "STALL"
            final_code = STALL_EXIT_CODE
            break
        if summary["status"] == "FAIL":
            report["verdict"] = "FAIL"
            final_code = 1
            break
        if summary["status"] == "DEGRADED" and report["verdict"] == "RUNNING":
            report["verdict"] = "DEGRADED"
        write_report(report, json_path, markdown_path)

    if report["verdict"] == "RUNNING":
        report["verdict"] = "PASS"
    write_report(report, json_path, markdown_path)
    print(f"Verdict: {report['verdict']}")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")
    return final_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        nargs="?",
        choices=("safe", "extreme", "duplex-192k", "duplex-256k"),
        default="safe",
    )
    parser.add_argument("--base-url", default=BASE_URL_DEFAULT)
    parser.add_argument("--metrics-url", default=METRICS_URL_DEFAULT)
    parser.add_argument("--tokenize-url", default=TOKENIZE_URL_DEFAULT)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GATEWAY_API_KEY", ""),
        help="Prefer the GATEWAY_API_KEY environment variable to command history.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "torture"),
    )
    parser.add_argument("--stall-seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stall_seconds <= 0 or args.sample_interval <= 0 or args.request_timeout <= 0:
        parser.error("timeouts and sample interval must be positive")
    if bool(args.worker_spec) != bool(args.worker_result):
        parser.error("internal worker arguments must be provided together")
    return args


def main() -> int:
    args = parse_args()
    if args.worker_spec:
        return worker_main(args.worker_spec, args.worker_result)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
