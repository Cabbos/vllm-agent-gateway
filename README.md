# vLLM Agent Gateway

A Linux-first, multi-protocol compatibility gateway for a local
[vLLM](https://github.com/vllm-project/vllm) server.

It lets Claude Code, Codex, OpenAI-compatible, Ollama-compatible, Gemini-style,
and Azure OpenAI-style clients share one local model. The gateway routes model
aliases, translates protocol-specific tool and thinking fields, streams
responses, and converts PDF/plain-text inputs for multimodal models.

> Status: alpha. v0.2 is designed for local and private-network deployments. It
> is not a complete replacement for the cloud services whose APIs it emulates.

## Supported surfaces

| Client/API family | Endpoints and behavior |
|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, model alias routing |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens`, thinking, tools, PDF blocks |
| Ollama | `/api/chat`, `/api/generate`, tags/show/ps/version, NDJSON streaming |
| Gemini-style | model list, `generateContent`, real incremental `streamGenerateContent`, `countTokens`, function calls |
| Azure OpenAI-style | deployment-prefixed chat completions, completions, and responses paths |
| Documents | searchable PDF to text, scanned PDF pages to JPEG, UTF-8 plain-text attachments |

All requested model IDs are routed to `SERVED_MODEL`. Clients that insist on
`gpt-*`, `claude-*`, Gemini model names, Azure deployment IDs, or Ollama tags can
therefore share one local backend.

## Architecture

```mermaid
flowchart LR
    C["Agent clients"] --> M["Gateway middleware"]
    M --> A["Protocol adapters"]
    A --> D["Bounded document service"]
    A --> P["Streaming upstream proxy"]
    D --> P
    P --> V["vLLM"]
    M --> O["Gateway metrics"]
```

v0.2 separates request orchestration, protocol adapters, document loading,
security middleware, streaming proxy code, and observability. `app.py` remains
a small ASGI entry point and v0.1 compatibility facade. See
[Architecture](docs/architecture.md) for module ownership and request flow.

## One-command Docker start

Requirements: Linux, Docker Compose, the NVIDIA Container Toolkit, and a local
model directory.

```bash
cp .env.example .env
```

Edit at least these values in `.env`:

```dotenv
MODEL_PATH=/srv/models/my-model
SERVED_MODEL=my-local-model
GATEWAY_API_KEYS=replace-with-a-long-random-secret
TOOL_CALL_PARSER=hermes
```

Then start vLLM and the gateway:

```bash
docker compose up -d --build
```

Check readiness:

```bash
docker compose ps
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer replace-with-a-long-random-secret'
```

The gateway image is multi-stage and runs as an unprivileged user. The supplied
Compose service also uses a read-only root filesystem, drops Linux capabilities,
sets `no-new-privileges`, bounds PIDs, and provides a small `/tmp` tmpfs. These
controls complement, but do not replace, host firewalling and TLS ingress.

The example uses `MAX_NUM_SEQS=2` and `GATEWAY_MAX_INFLIGHT=2`, a conservative
starting point for a single 32 GiB GPU. Reduce both to `1` for very long context
windows or memory pressure; benchmark before increasing them. Parser names are
model-specific. `compose.yaml` wires `TOOL_CALL_PARSER`; add a supported vLLM
`--reasoning-parser` flag to the Compose command if your model requires one.

## Local start with an existing vLLM server

Requirements: Python 3.11+ and a reachable vLLM OpenAI-compatible server.

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

The process listens on `0.0.0.0:8000` by default. The application does not load
`.env` files itself; export variables in the shell or use a process manager.

## Client configuration

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

- OpenAI-compatible base URL: `http://127.0.0.1:8000/v1`
- Ollama-compatible host: `http://127.0.0.1:8000`
- Gemini-style base URL: `http://127.0.0.1:8000`
- API key: any value configured in `GATEWAY_API_KEYS`
- Model: the value of `SERVED_MODEL`

The stock Ollama CLI has no general custom-header option. Use a private ingress
that supplies authentication, or disable gateway keys only for a strictly local
Ollama-only deployment.

## Authentication

`GATEWAY_API_KEYS` contains comma-separated credentials accepted from clients.
The gateway recognizes Bearer, `X-Api-Key`, Azure `Api-Key`, Gemini
`X-Goog-Api-Key`, and Gemini `?key=` credentials. Root and health endpoints are
public; model, generation, upstream `/metrics`, and gateway metrics endpoints
require a key when authentication is enabled.

`VLLM_UPSTREAM_API_KEY` is a separate credential for the gateway-to-vLLM hop.
Client credential headers and the Gemini query key are not forwarded. Set the
upstream key only if vLLM or its private ingress requires authentication:

```dotenv
GATEWAY_API_KEYS=client-key-one,client-key-two
VLLM_UPSTREAM_API_KEY=different-private-upstream-key
```

Prefer headers over Gemini's `?key=` form because web-server and proxy access
logs may record query strings.

## Backpressure and rate limiting

Both controls are optional and in-process:

```dotenv
GATEWAY_MAX_INFLIGHT=2
GATEWAY_MAX_QUEUE_SIZE=8
GATEWAY_QUEUE_TIMEOUT_SECONDS=30
GATEWAY_REQUESTS_PER_MINUTE=0
GATEWAY_RATE_LIMIT_BURST=10
```

`GATEWAY_MAX_INFLIGHT=0` disables the gateway concurrency limit. When enabled,
streaming requests hold a slot through the final response byte. A full queue or
expired queue wait returns `429` with `Retry-After`.

`GATEWAY_REQUESTS_PER_MINUTE=0` disables rate limiting. A positive value enables
a token bucket per authenticated API key; unauthenticated requests share the
anonymous bucket. For multiple gateway processes or replicas, enforce a shared
global limit at the ingress or with an external rate-limit service.

## Document security

Inline base64 PDF/plain-text inputs are supported. Searchable PDF pages become
text; sparse or scanned pages become JPEG image blocks. Resource controls cover
raw bytes, request bytes, total PDF pages, rendered scan pages, extracted text,
per-page rendered pixels, conversion concurrency, and conversion timeout.

Remote document URLs are denied by default:

```dotenv
DOCUMENT_URL_POLICY=deny
DOCUMENT_ALLOWED_HOSTS=
DOCUMENT_EXTRA_ALLOWED_NETWORKS=
```

To enable narrowly scoped remote loading:

```dotenv
DOCUMENT_URL_POLICY=allowlist
DOCUMENT_ALLOWED_HOSTS=documents.example.com,*.trusted.example
DOCUMENT_EXTRA_ALLOWED_NETWORKS=
```

Every redirect is revalidated. All DNS results must be public unless covered by
an explicit extra network, and the connected peer IP is checked against the DNS
result when the HTTP transport exposes it. There is no implicit
`198.18.0.0/15` transparent-proxy exception. Only add private or benchmark
networks when the gateway is intentionally allowed to reach them.

See [Configuration](docs/configuration.md) for exact defaults and
[Production safety](docs/production-security.md) before enabling URL loading.

## Gemini streaming

`streamGenerateContent` now converts the upstream OpenAI SSE stream
incrementally. It handles arbitrary network fragmentation and emits text,
thinking parts, completed function calls, finish reasons, and usage metadata.

- `?alt=sse` returns Gemini-framed SSE events.
- Without `alt=sse`, the response is a streaming JSON array.

This is protocol conversion, not Google infrastructure: Files API uploads,
grounding, cached content, safety services, and Vertex IAM are not implemented.

## Multimodal history compaction

Anthropic clients resend the complete conversation, including images from
older turns. `GATEWAY_MAX_PROMPT_IMAGES` defaults to `4`; keep it at or below
vLLM's `--limit-mm-per-prompt` image count. The gateway preserves the newest
images and replaces older visual payloads with text placeholders while keeping
the rest of the conversation intact.

## Metrics

When `GATEWAY_METRICS_ENABLED=true`, `/gateway/metrics` exposes Prometheus text
metrics for request counts and end-to-end duration using bounded `protocol` and
`outcome` labels. API keys, URLs, paths, request IDs, and other high-cardinality
values are forbidden as labels.

The existing `/metrics` path continues to proxy vLLM metrics. Scrape both when
you need gateway-level request behavior and backend scheduler/KV-cache data.

```bash
curl http://127.0.0.1:8000/gateway/metrics \
  -H 'Authorization: Bearer change-me'
```

## Dynamic thinking

The gateway maps client-native controls per request; no gateway restart is
needed.

| API | Control |
|---|---|
| Anthropic | `thinking.type = enabled`, `adaptive`, or `disabled` |
| OpenAI Chat | `reasoning_effort` |
| OpenAI Responses | `reasoning.effort` |
| Ollama | `think: true` or `false` |
| Gemini-style | `generationConfig.thinkingConfig` |

The model chat template and vLLM reasoning parser still determine whether
reasoning is emitted correctly.

## Upgrade and operations

- [v0.1 to v0.2 migration](docs/migration-v0.2.md)
- [Complete configuration reference](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [Production safety boundaries](docs/production-security.md)
- [Validated 32 GiB Qwen/RTX 5090 profile](docs/validated-profile-qwen36-5090.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Intentional limits

- No cloud Files API, embeddings for generative-only models, audio, or video.
- No built-in web search, code execution, or MCP server; tools remain client-side.
- Static in-process authentication is not multi-tenant identity or billing.
- Gateway concurrency, queues, and rate-limit buckets are process-local.
- TLS termination, durable audit, distributed quotas, and denial-of-service
  protection belong at a production ingress.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

The tests exercise transformations, fragmented streaming, malicious document
inputs, and middleware controls without a GPU or running model.

For a running gateway, preview or run the dependency-free protocol/load smoke:

```bash
python scripts/load_smoke.py --dry-run --protocol all

GATEWAY_API_KEY=change-me python scripts/load_smoke.py \
  --protocol all --concurrency 1 2 --prompt-size tiny
```

Large prompt profiles require the explicit `--allow-large-prompts` guard.

## License

MIT
