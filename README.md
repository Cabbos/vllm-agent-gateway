<div align="center">

# vLLM Agent Gateway

**One local model. Five API dialects. Explicit resource boundaries.**

[![CI](https://github.com/Cabbos/vllm-agent-gateway/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Cabbos/vllm-agent-gateway/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/backend-vLLM-6C5CE7)](https://github.com/vllm-project/vllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E.svg)](LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-alpha-F59E0B)](CHANGELOG.md)

A Linux-first compatibility and control layer that lets Claude Code, Codex,
OpenAI clients, Ollama clients, and Gemini-style clients share one private
[vLLM](https://github.com/vllm-project/vllm) deployment.

[Quick start](#quick-start) · [Protocol matrix](#protocol-matrix) ·
[Architecture](#architecture) · [Validated profile](#validated-on-real-hardware) ·
[Security](docs/production-security.md)

</div>

> **Not another inference server.** vLLM owns model execution. This gateway owns
> the boundary around it: protocol translation, model aliasing, admission
> control, document conversion, authentication, streaming, and observability.

## Why this exists

Agent clients speak similar APIs, but not the same language. They disagree on
endpoint paths, thinking controls, tool calls, stream framing, model names,
document blocks, and authentication headers. Pointing every client directly at
vLLM turns those differences into per-client scripts and fragile configuration.

vLLM Agent Gateway gives them one predictable local endpoint:

- **Translate, do not emulate a cloud.** Common agent workflows are mapped onto
  vLLM's OpenAI-compatible backend; unsupported cloud services fail explicitly.
- **Stream end to end.** Responses are forwarded incrementally, including real
  Gemini SSE/JSON-array conversion and cancellation-safe upstream cleanup.
- **Protect the GPU.** Bounded in-flight work, a finite wait queue, queue
  deadlines, per-key rate limits, and body/document limits are available at the
  boundary.
- **Compact multimodal history safely.** The newest images remain intact while
  older visual payloads become explicit text placeholders before they can
  exceed vLLM's per-prompt image limit.
- **Keep credentials separated.** Client keys are consumed by the gateway and
  never forwarded; the vLLM hop can use a different credential.
- **Make behavior inspectable.** Request IDs, bounded-label Prometheus metrics,
  deterministic capability evaluation, and guarded load profiles are built in.

## Protocol matrix

| Client / API family | Supported surface |
|---|---|
| **OpenAI** | Chat Completions, Completions, Responses, model alias routing, reasoning controls |
| **Anthropic** | Messages, token counting, thinking, tools, PDF/document blocks |
| **Ollama** | Chat, Generate, tags/show/ps/version, NDJSON streaming |
| **Gemini-style** | Model listing, `generateContent`, incremental `streamGenerateContent`, token counting, function calls |
| **Azure OpenAI-style** | Deployment-prefixed Chat Completions, Completions, and Responses |

Every requested model ID is routed to `SERVED_MODEL`. A client may insist on a
`gpt-*`, `claude-*`, Gemini, Azure deployment, or Ollama model name while the
backend continues to expose one local model.

## See the compatibility layer in one command

With a gateway already running, exercise every supported generation protocol
using the dependency-free smoke tool:

```bash
GATEWAY_API_KEY=change-me python scripts/load_smoke.py \
  --protocol all --concurrency 1 --prompt-size tiny
```

It emits JSON Lines with per-protocol success counts, wall time, requests per
second, mean latency, and p95 latency, and exits non-zero if any request fails.
Preview the exact plan without sending traffic:

```bash
python scripts/load_smoke.py --dry-run --protocol all --concurrency 1
```

## Architecture

```mermaid
flowchart LR
    Clients["Claude Code · Codex · OpenAI · Ollama · Gemini"] --> Boundary["Request boundary"]
    Boundary --> Adapters["Protocol adapters"]
    Adapters --> Documents["Bounded document service"]
    Adapters --> Stream["Streaming upstream proxy"]
    Documents --> Stream
    Stream --> VLLM["vLLM"]
    Boundary --> Metrics["Request IDs · metrics"]
```

The request boundary applies body limits, authentication, optional rate
limiting, and bounded admission before protocol-specific work reaches the
backend. The application is split into small modules rather than a single proxy
file:

| Area | Responsibility |
|---|---|
| `adapters/` | OpenAI, Anthropic, Ollama, Gemini, and Azure-style conversion |
| `middleware/` | Request IDs, body limits, authentication, queues, rate limits |
| `documents/` | Bounded loading, PDF extraction/rendering, URL security |
| `proxy/` | Header isolation, streaming, disconnect/error cleanup |
| `observability/` | Dependency-free Prometheus request metrics |

See [Architecture](docs/architecture.md) for module ownership and the full
request lifecycle.

## Quick start

### Docker Compose

Requirements: Linux, Docker Compose, the NVIDIA Container Toolkit, and a local
model directory.

```bash
cp .env.example .env
```

Set the deployment-specific values:

```dotenv
MODEL_PATH=/srv/models/my-model
SERVED_MODEL=my-local-model
GATEWAY_API_KEYS=replace-with-a-long-random-secret
TOOL_CALL_PARSER=hermes
```

Start vLLM and the gateway:

```bash
docker compose up -d --build
docker compose ps

curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer replace-with-a-long-random-secret'
```

The gateway container runs as an unprivileged user with a read-only root
filesystem, dropped Linux capabilities, `no-new-privileges`, a PID limit, and a
bounded temporary filesystem.

### Existing vLLM server

Requirements: Python 3.11+ and a reachable vLLM OpenAI-compatible endpoint.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

VLLM_UPSTREAM=http://127.0.0.1:8001 \
SERVED_MODEL=my-local-model \
MODEL_CONTEXT_LENGTH=32768 \
GATEWAY_API_KEYS=change-me \
GATEWAY_MAX_INFLIGHT=2 \
vllm-agent-gateway
```

The process listens on `0.0.0.0:8000` by default. It does not load `.env`
itself; export variables in the shell or use a process manager.

## Connect an agent

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=change-me
export ANTHROPIC_MODEL=my-local-model
export ANTHROPIC_DEFAULT_OPUS_MODEL=my-local-model
export ANTHROPIC_DEFAULT_SONNET_MODEL=my-local-model
export ANTHROPIC_DEFAULT_HAIKU_MODEL=my-local-model
claude
```

### Codex

```toml
model = "my-local-model"
model_provider = "local_vllm"
model_context_window = 32768

[model_providers.local_vllm]
name = "Local vLLM Agent Gateway"
base_url = "http://127.0.0.1:8000/v1"
env_key = "LOCAL_LLM_API_KEY"
wire_api = "responses"
```

```bash
export LOCAL_LLM_API_KEY=change-me
codex
```

### Other clients

| Client type | Base URL |
|---|---|
| OpenAI-compatible | `http://127.0.0.1:8000/v1` |
| Ollama-compatible | `http://127.0.0.1:8000` |
| Gemini-style | `http://127.0.0.1:8000` |

Use a key from `GATEWAY_API_KEYS` and the model name from `SERVED_MODEL`.

The stock Ollama CLI cannot attach arbitrary headers. For authenticated Ollama
traffic, use a private ingress that injects the credential. Disable gateway
keys only for a strictly local, isolated deployment.

## Resource control

The gateway can bound admitted work without pretending to increase GPU
capacity:

```dotenv
GATEWAY_MAX_INFLIGHT=2
GATEWAY_MAX_QUEUE_SIZE=8
GATEWAY_QUEUE_TIMEOUT_SECONDS=30
GATEWAY_REQUESTS_PER_MINUTE=0
GATEWAY_RATE_LIMIT_BURST=10
```

- A streaming request holds its slot until the final response byte.
- A full queue or expired wait returns `429` with `Retry-After`.
- `GATEWAY_MAX_INFLIGHT=0` disables the concurrency limit.
- A positive requests-per-minute value enables a token bucket per API key.
- Limits are process-local; use an ingress or shared limiter for multiple
  workers or replicas.

The Compose example starts conservatively at two vLLM sequences and two gateway
slots for a single 32 GiB GPU. Reduce both to one when long-context latency
predictability matters. Benchmark before increasing either value.

## Multimodal history compaction

Anthropic clients resend the full conversation on every turn, including images
that appeared much earlier in the session. That history can eventually exceed
vLLM's `--limit-mm-per-prompt` setting even when the current request contains
only one new image.

```dotenv
GATEWAY_MAX_PROMPT_IMAGES=4
```

The gateway preserves the newest image blocks and replaces older image payloads
with explicit text placeholders. Message order, text, tool calls, and recent
visual context remain intact. Set this value at or below vLLM's configured image
count; use `0` only when the backend accepts an unbounded image history.

## Documents without an unbounded attack surface

Inline base64 PDF and UTF-8 text inputs are supported. Searchable PDF pages
become text; sparse or scanned pages become JPEG image blocks. Limits cover raw
bytes, request bytes, page count, rendered pages, extracted text, pixel count,
conversion concurrency, and conversion time.

Remote document loading is denied by default:

```dotenv
DOCUMENT_URL_POLICY=deny
DOCUMENT_ALLOWED_HOSTS=
DOCUMENT_EXTRA_ALLOWED_NETWORKS=
```

If remote loading is required, enable only named hosts:

```dotenv
DOCUMENT_URL_POLICY=allowlist
DOCUMENT_ALLOWED_HOSTS=documents.example.com,*.trusted.example
```

Redirects are revalidated. DNS results must be public unless an explicit
network is allowed, and the connected peer is checked against the validated
address when the transport exposes it. Read [Production safety
boundaries](docs/production-security.md) before enabling this feature.

## Authentication and credential isolation

`GATEWAY_API_KEYS` accepts comma-separated client credentials. The gateway
recognizes Bearer, `X-Api-Key`, Azure `Api-Key`, Gemini `X-Goog-Api-Key`, and
Gemini `?key=` credentials. Root and health endpoints remain public.

`VLLM_UPSTREAM_API_KEY` is independent:

```dotenv
GATEWAY_API_KEYS=client-key-one,client-key-two
VLLM_UPSTREAM_API_KEY=different-private-upstream-key
```

Client credential headers and Gemini query keys are stripped before forwarding.
Prefer headers over `?key=` because proxies may log query strings.

## Streaming, thinking, and tools

`streamGenerateContent` incrementally converts fragmented OpenAI SSE into
Gemini-framed SSE (`?alt=sse`) or a streaming JSON array. Text, reasoning,
completed function calls, finish reasons, and usage metadata are preserved.

Thinking is selected per request; no gateway restart is required:

| API | Control |
|---|---|
| Anthropic | `thinking.type = enabled`, `adaptive`, or `disabled` |
| OpenAI Chat | `reasoning_effort` |
| OpenAI Responses | `reasoning.effort` |
| Ollama | `think: true` or `false` |
| Gemini-style | `generationConfig.thinkingConfig` |

The model chat template and the configured vLLM reasoning/tool parsers still
determine whether the model emits valid reasoning and tool calls.

## Observability

Enable gateway-level Prometheus metrics:

```dotenv
GATEWAY_METRICS_ENABLED=true
```

```bash
curl http://127.0.0.1:8000/gateway/metrics \
  -H 'Authorization: Bearer change-me'
```

`/gateway/metrics` reports request counts and end-to-end duration with bounded
`protocol` and `outcome` labels. Secrets, URLs, paths, request IDs, and other
high-cardinality values are forbidden as labels. `/metrics` continues to proxy
the vLLM backend metrics.

## Validated on real hardware

The checked-in [RTX 5090 / Qwen3.6 validation
profile](docs/validated-profile-qwen36-5090.md) records an end-to-end run on a
single 32 GiB GPU, vLLM 0.25.0, and a `Qwen3.6-35B-A3B-NVFP4-Fast` deployment.
It is a reproducible compatibility and capacity smoke, not a vendor benchmark.

The recorded run completed:

- OpenAI, Anthropic, Ollama, Gemini-style, and Azure-style requests;
- real incremental Gemini streaming;
- forced tool calling and searchable-PDF conversion;
- short request waves at client concurrency 1, 2, and 4;
- exact single 131,072- and 192,000-token prompts;
- two different 192,000-token requests admitted together.

The last case completed close to serially. Queueing prevented client rejection;
it did not manufacture extra prefill capacity. That distinction is intentional
and is why the validated profile reports both admission behavior and observed
latency.

## Test and evaluate

The normal suite runs without a GPU or live model:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest -q --cov=vllm_agent_gateway --cov-report=term-missing
pre-commit install
```

It covers protocol transforms, fragmented streams, cancellation, malicious
document inputs, multimodal-history compaction, authentication, admission
controls, and bounded metrics. CI also enforces type checking, 80% source
coverage, and a McCabe complexity ceiling of 18.

### Live protocol/load smoke

```bash
python scripts/load_smoke.py --dry-run --protocol all

GATEWAY_API_KEY=change-me python scripts/load_smoke.py \
  --protocol all --concurrency 1 2 --prompt-size tiny
```

Large prompt profiles require `--allow-large-prompts`.

### Guarded capacity torture

```bash
python scripts/torture.py safe
python scripts/torture.py extreme
```

The safe profile stops at dual 32K prompts plus a short decode stress. The
extreme profile explicitly exercises queued concurrency, single and dual 192K
prompts, and longer decode. The runner refuses to start while unrelated work is
active and records client latency, exact-answer anchors, vLLM scheduler/KV
metrics, and GPU samples under `outputs/torture`.

If vLLM reports active work while iteration, token, and KV metrics remain still
for 60 seconds, the runner exits with code `75` so a service manager can restart
the backend.

### Capability degradation

```bash
python scripts/capability_eval.py quick --thinking off
python scripts/capability_eval.py full --thinking on
```

This suite looks beyond HTTP success. It runs generated Python against hidden
unit tests in a restricted AST environment, scores closed-book evidence
questions without an LLM judge, and verifies auto-selected tool names and JSON
arguments. Reports compare 1K through 192K contexts and single versus dual
request pressure under `outputs/capability`.

## Intentional boundaries

This project translates common local-agent workflows. It does **not** attempt
to recreate cloud control planes.

- No cloud Files API, hosted grounding, cached content, cloud safety service,
  account management, or billing.
- No built-in web search, code execution, or MCP server; tools execute on the
  client side.
- No audio or video API.
- Static in-process authentication is not multi-tenant identity.
- Queues, concurrency limits, rate buckets, and counters are process-local.
- TLS termination, durable audit, distributed quotas, and denial-of-service
  protection belong at a production ingress.

## Documentation

- [Complete configuration reference](docs/configuration.md)
- [Architecture and request flow](docs/architecture.md)
- [Production safety boundaries](docs/production-security.md)
- [Validated 32 GiB Qwen / RTX 5090 profile](docs/validated-profile-qwen36-5090.md)
- [Benchmarking methodology](docs/benchmarking.md)
- [Engineering case studies](docs/engineering-notes.md)
- [Operational roadmap](docs/roadmap.md)
- [v0.1 to v0.2 migration](docs/migration-v0.2.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)

## License

[MIT](LICENSE)
