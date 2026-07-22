#!/usr/bin/env python3
"""Small dependency-free load smoke for a running vLLM Agent Gateway."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

PROMPT_CHARS = {
    "tiny": 256,
    "4k": 16 * 1024,
    "32k": 128 * 1024,
    "128k": 512 * 1024,
    "192k": 768 * 1024,
}
LARGE_PROMPTS = frozenset({"32k", "128k", "192k"})
PROTOCOLS = ("openai", "anthropic", "ollama", "gemini", "azure")


@dataclass(frozen=True, slots=True)
class RequestResult:
    protocol: str
    concurrency: int
    request_number: int
    status: int
    seconds: float
    response_bytes: int
    error: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and not self.error


def build_prompt(profile: str, request_number: int) -> str:
    target = PROMPT_CHARS[profile]
    marker = f"\n[gateway-smoke prompt-size={profile} request={request_number}] Reply only: OK"
    seed = "Local gateway load smoke input. "
    body_size = max(0, target - len(marker))
    repetitions = math.ceil(body_size / len(seed))
    return (seed * repetitions)[:body_size] + marker


def request_spec(
    protocol: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    headers = {"content-type": "application/json", "user-agent": "gateway-load-smoke/0.2"}
    if protocol == "openai":
        path = "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        }
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
    elif protocol == "azure":
        encoded_model = urllib.parse.quote(model, safe="")
        path = f"/openai/deployments/{encoded_model}/chat/completions?api-version=2025-01-01"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        }
        if api_key:
            headers["api-key"] = api_key
    elif protocol == "anthropic":
        path = "/v1/messages"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        }
        headers["anthropic-version"] = "2023-06-01"
        if api_key:
            headers["x-api-key"] = api_key
    elif protocol == "ollama":
        path = "/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 16, "temperature": 0},
        }
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
    elif protocol == "gemini":
        encoded_model = urllib.parse.quote(model, safe="")
        path = f"/v1beta/models/{encoded_model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 16, "temperature": 0},
        }
        if api_key:
            headers["x-goog-api-key"] = api_key
    else:
        raise ValueError(f"unsupported protocol: {protocol}")
    return f"{base_url}{path}", headers, payload


def send_request(
    protocol: str,
    *,
    concurrency: int,
    request_number: int,
    base_url: str,
    api_key: str,
    model: str,
    prompt_size: str,
    timeout: float,
) -> RequestResult:
    prompt = build_prompt(prompt_size, request_number)
    url, headers, payload = request_spec(
        protocol,
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
    )
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
        error = ""
    except urllib.error.HTTPError as exc:
        body = exc.read(512)
        status = exc.code
        error = body.decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError) as exc:
        body = b""
        status = 0
        error = str(exc)
    return RequestResult(
        protocol=protocol,
        concurrency=concurrency,
        request_number=request_number,
        status=status,
        seconds=time.perf_counter() - started,
        response_bytes=len(body),
        error=error,
    )


def run_level(args: argparse.Namespace, protocol: str, concurrency: int) -> list[RequestResult]:
    request_count = args.requests_per_level or concurrency
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                send_request,
                protocol,
                concurrency=concurrency,
                request_number=index + 1,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt_size=args.prompt_size,
                timeout=args.timeout,
            )
            for index in range(request_count)
        ]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started
    latencies = sorted(result.seconds for result in results)
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    summary = {
        "event": "summary",
        "protocol": protocol,
        "prompt_size": args.prompt_size,
        "prompt_chars": PROMPT_CHARS[args.prompt_size],
        "concurrency": concurrency,
        "requests": len(results),
        "succeeded": sum(result.ok for result in results),
        "failed": sum(not result.ok for result in results),
        "wall_seconds": round(elapsed, 4),
        "requests_per_second": round(len(results) / elapsed, 3) if elapsed else 0,
        "mean_seconds": round(statistics.fmean(latencies), 4),
        "p95_seconds": round(latencies[p95_index], 4),
    }
    print(json.dumps(summary, ensure_ascii=False))
    for result in sorted(results, key=lambda item: item.request_number):
        if not result.ok:
            detail = asdict(result)
            detail["event"] = "request_error"
            detail["error"] = detail["error"][:512]
            print(json.dumps(detail, ensure_ascii=False), file=sys.stderr)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GATEWAY_BASE_URL", "http://127.0.0.1:8000"),
        help="Gateway base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GATEWAY_API_KEY", ""),
        help="Gateway API key; GATEWAY_API_KEY is preferred to command-line history",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SERVED_MODEL", "local-model"),
        help="Served model alias (default: %(default)s)",
    )
    parser.add_argument(
        "--protocol",
        choices=(*PROTOCOLS, "all"),
        default="openai",
        help="Compatibility protocol to exercise (default: %(default)s)",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        choices=(1, 2, 4),
        default=[1, 2, 4],
        help="Concurrency levels (default: 1 2 4)",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=0,
        help="Requests per level; zero sends one wave equal to concurrency",
    )
    parser.add_argument(
        "--prompt-size",
        choices=tuple(PROMPT_CHARS),
        default="tiny",
        help="Approximate token-size marker; default is intentionally lightweight",
    )
    parser.add_argument(
        "--allow-large-prompts",
        action="store_true",
        help="Required for 32k, 128k, and 192k profiles",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the plan without making requests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    if args.requests_per_level < 0:
        raise SystemExit("--requests-per-level must be zero or positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.prompt_size in LARGE_PROMPTS and not args.allow_large_prompts:
        raise SystemExit(f"prompt profile {args.prompt_size!r} requires --allow-large-prompts")

    protocols = PROTOCOLS if args.protocol == "all" else (args.protocol,)
    plan = {
        "event": "plan",
        "base_url": args.base_url,
        "model": args.model,
        "protocols": protocols,
        "concurrency": args.concurrency,
        "requests_per_level": args.requests_per_level or "one-wave",
        "prompt_size": args.prompt_size,
        "prompt_chars": PROMPT_CHARS[args.prompt_size],
        "authenticated": bool(args.api_key),
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, ensure_ascii=False))
    if args.dry_run:
        return 0

    results = [
        result
        for protocol in protocols
        for concurrency in args.concurrency
        for result in run_level(args, protocol, concurrency)
    ]
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
