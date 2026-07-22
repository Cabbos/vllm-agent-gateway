# Configuration reference

All gateway settings are read from environment variables at process import and
stored in an immutable `Settings` object. Restart the process after changing
them. The application does not load `.env` files; Docker Compose loads the
repository `.env` through `compose.yaml`.

## Core gateway settings

| Variable | Code default | Meaning |
|---|---:|---|
| `VLLM_UPSTREAM` | `http://127.0.0.1:8001` | Base URL of the vLLM OpenAI server; trailing slash is removed |
| `VLLM_UPSTREAM_API_KEY` | empty | Private Bearer credential used only for gateway-to-vLLM requests |
| `SERVED_MODEL` | `local-model` | Real model name and target for all client aliases |
| `MODEL_CONTEXT_LENGTH` | `32768` | Context metadata exposed to compatible clients |
| `GATEWAY_HOST` | `0.0.0.0` | Gateway listen address |
| `GATEWAY_PORT` | `8000` | Gateway listen port |
| `GATEWAY_LOG_LEVEL` | `info` | Uvicorn log level |
| `GATEWAY_API_KEYS` | empty | Comma-separated incoming client keys; empty disables authentication |
| `GATEWAY_CORS_ORIGINS` | `*` | Comma-separated CORS origin allowlist |
| `GATEWAY_TRUSTED_HOSTS` | `*` | Comma-separated HTTP Host allowlist |

`LOCAL_SERVED_MODEL` and `LOCAL_MODEL_CONTEXT_LENGTH` remain fallback aliases
for the corresponding v0.1 settings. Prefer the names in the table.

When using the supplied Compose file, `GATEWAY_PORT` selects the host port. The
gateway still listens on fixed container port `8000`, so no Compose edit is
needed when changing the host port.

## Request admission and upstream pool

| Variable | Code default | Meaning |
|---|---:|---|
| `GATEWAY_MAX_REQUEST_BYTES` | derived | Maximum declared and actually received request body |
| `GATEWAY_MAX_INFLIGHT` | `0` | Concurrent non-health requests; `0` disables the limiter |
| `GATEWAY_MAX_QUEUE_SIZE` | `0` | Waiting requests when the concurrency limit is full |
| `GATEWAY_QUEUE_TIMEOUT_SECONDS` | `30` | Maximum queue wait before `429` |
| `GATEWAY_REQUESTS_PER_MINUTE` | `0` | Per-key token refill rate; `0` disables rate limiting |
| `GATEWAY_RATE_LIMIT_BURST` | `10` | Initial/maximum token bucket size |
| `UPSTREAM_CONNECT_TIMEOUT` | `10` | vLLM connection timeout in seconds |
| `UPSTREAM_MAX_CONNECTIONS` | `100` | Maximum connections in the process-local HTTP pool |
| `UPSTREAM_MAX_KEEPALIVE_CONNECTIONS` | `20` | Maximum idle keep-alive connections |

If `GATEWAY_MAX_REQUEST_BYTES` is unset, its value is derived from
`PDF_COMPAT_MAX_BYTES`: approximately base64 expansion plus 4 MiB for JSON,
prompts, tools, and images. With the 50 MiB PDF default, the derived value is
`74099371` bytes.

The request limiter reads actual ASGI chunks, so omitting or falsifying
`Content-Length` does not bypass it. Queue and rate state are local to one
process. Streaming requests hold an in-flight slot until the response finishes
or disconnects.

Recommended starting point for a single 32 GiB GPU:

```dotenv
GATEWAY_MAX_INFLIGHT=2
GATEWAY_MAX_QUEUE_SIZE=8
GATEWAY_QUEUE_TIMEOUT_SECONDS=30
```

For very long context windows or memory pressure, start at `1`. A larger queue
does not add GPU capacity; it only moves waiting clients into gateway memory.

## Document limits and URL policy

| Variable | Default | Meaning |
|---|---:|---|
| `PDF_COMPAT_MAX_BYTES` | `52428800` | Decoded raw bytes for a PDF or plain-text source |
| `PDF_COMPAT_MAX_PAGES` | `64` | Maximum pages in one PDF |
| `PDF_COMPAT_MAX_RENDERED_PAGES` | `24` | Maximum sparse/scanned pages rendered as JPEG |
| `PDF_COMPAT_MAX_CHARS` | `500000` | Maximum extracted text characters; excess text is marked truncated |
| `PDF_MAX_PAGE_PIXELS` | `16000000` | Maximum rendered pixels for each PDF page |
| `PDF_CONVERSION_CONCURRENCY` | `2` | Concurrent CPU-bound document conversions per process |
| `PDF_CONVERSION_TIMEOUT_SECONDS` | `60` | Source load plus conversion deadline; timeout returns `408` |
| `DOCUMENT_URL_POLICY` | `deny` | `deny` or `allowlist` remote document URLs |
| `DOCUMENT_ALLOWED_HOSTS` | empty | Comma-separated exact hosts or `*.example.com` subdomain patterns |
| `DOCUMENT_EXTRA_ALLOWED_NETWORKS` | empty | Explicit comma-separated CIDRs allowed in addition to public IPs |

An allowlist pattern such as `*.example.com` matches subdomains but not the apex
`example.com`; list both if both are required. HTTP(S) ports are limited to 80
and 443.

Secure public-host example:

```dotenv
DOCUMENT_URL_POLICY=allowlist
DOCUMENT_ALLOWED_HOSTS=documents.example.com,*.assets.example.com
DOCUMENT_EXTRA_ALLOWED_NETWORKS=
```

Transparent proxy with benchmark-space DNS, enabled deliberately:

```dotenv
DOCUMENT_URL_POLICY=allowlist
DOCUMENT_ALLOWED_HOSTS=documents.example.com
DOCUMENT_EXTRA_ALLOWED_NETWORKS=198.18.0.0/15
```

The second example widens SSRF reach and should be paired with network-level
egress rules. There is no built-in exception for that network.

## Metrics

| Variable | Default | Meaning |
|---|---:|---|
| `GATEWAY_METRICS_ENABLED` | `true` | Registers the local `/gateway/metrics` exporter |

The exporter currently provides:

- `gateway_requests_total{protocol,outcome}`;
- `gateway_request_duration_seconds{protocol,outcome}`.

Allowed protocols are `openai`, `anthropic`, `ollama`, and `gemini`. Outcomes
are `success`, `rejected`, and `upstream_error`. The metric duration includes
the final streamed response byte.

`/metrics` is not the same endpoint: it is forwarded to vLLM and exposes backend
metrics. Both endpoints require a gateway key when authentication is enabled.

## Model metadata

| Variable | Default | Meaning |
|---|---:|---|
| `MODEL_FAMILY` | `local` | Model family reported to Ollama clients |
| `MODEL_PARAMETER_SIZE` | `unknown` | Displayed parameter size |
| `MODEL_QUANTIZATION` | `unknown` | Displayed quantization |
| `MODEL_FORMAT` | `safetensors` | Displayed artifact format |
| `MODEL_SIZE_BYTES` | `0` | Model size metadata |
| `MODEL_VRAM_BYTES` | `0` | Runtime VRAM metadata |

These values are informational and do not change vLLM execution.

## Compose-only settings

The supplied `compose.yaml` additionally consumes:

| Variable | Example default | Meaning |
|---|---:|---|
| `MODEL_PATH` | required | Absolute host model directory mounted read-only |
| `VLLM_IMAGE` | `vllm/vllm-openai:v0.25.0` | vLLM container image |
| `GPU_MEMORY_UTILIZATION` | `0.90` | vLLM GPU memory utilization flag |
| `MAX_NUM_SEQS` | `2` | vLLM maximum active sequences |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | vLLM scheduler batched-token limit |
| `TOOL_CALL_PARSER` | `hermes` | Model-specific vLLM tool parser |

Reasoning parser values are model-specific and are not automatically wired by
the current Compose file. Add a valid `--reasoning-parser` argument to the vLLM
command when required.
