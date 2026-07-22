# Migrating from v0.1 to v0.2

v0.2 keeps the existing client endpoint families and the
`vllm_agent_gateway.app:app` ASGI target. The important migration changes are
security defaults, credential forwarding, admission controls, and streaming
behavior.

## Migration checklist

1. Back up the current environment and Compose overrides.
2. Keep `GATEWAY_API_KEYS` for incoming clients.
3. If vLLM requires authentication, move its credential to the new
   `VLLM_UPSTREAM_API_KEY`; do not rely on a client key being forwarded.
4. Decide whether remote documents are needed. They are denied by default.
5. Reconcile the request-body limit with base64 PDF expansion.
6. Choose a process-local concurrency limit and bounded queue.
7. Decide whether to enable per-key rate limiting.
8. Add `/gateway/metrics` to monitoring while keeping `/metrics` for vLLM.
9. Test Gemini clients against multi-event streaming.
10. Run protocol smoke tests before exposing the upgraded process.

## Incoming and upstream API keys are separate

In v0.2, incoming client credentials are stripped before generic forwarding.
Configure independent values:

```dotenv
GATEWAY_API_KEYS=client-one,client-two
VLLM_UPSTREAM_API_KEY=private-vllm-ingress-key
```

Leave `VLLM_UPSTREAM_API_KEY` empty when the vLLM service does not require
authentication. Gemini `?key=` and the Azure `api-version` query parameter are
not forwarded to the local vLLM endpoint.

## Remote URLs now fail closed

v0.1 accepted public PDF URLs after address checks. v0.2 defaults to:

```dotenv
DOCUMENT_URL_POLICY=deny
DOCUMENT_ALLOWED_HOSTS=
DOCUMENT_EXTRA_ALLOWED_NETWORKS=
```

Inline base64 PDF and UTF-8 plain text continue to work. To preserve selected
remote workflows, explicitly configure the necessary hosts:

```dotenv
DOCUMENT_URL_POLICY=allowlist
DOCUMENT_ALLOWED_HOSTS=documents.example.com,*.trusted.example
DOCUMENT_EXTRA_ALLOWED_NETWORKS=
```

Each redirect target must also match the host allowlist. Every DNS result must
be public unless an extra CIDR covers it. Connected peer addresses are compared
with validated DNS results when available from the HTTP transport.

v0.1's transparent-proxy treatment of `198.18.0.0/15` is removed. Opt in only
when required:

```dotenv
DOCUMENT_EXTRA_ALLOWED_NETWORKS=198.18.0.0/15
```

The host must still be allowlisted. Prefer narrow egress firewall rules over a
broad process-level network exception.

## Reconcile request and document sizes

Base64 adds roughly one third to a raw PDF. v0.2 derives the request-body
default from `PDF_COMPAT_MAX_BYTES` plus 4 MiB of JSON/prompt overhead. With a
50 MiB raw PDF limit, the derived request limit is `74099371` bytes.

If v0.1 configuration explicitly fixed `GATEWAY_MAX_REQUEST_BYTES=67108864`, a
near-50-MiB PDF can be rejected before decoding. Remove the override to use the
derived value, raise it deliberately, or reduce `PDF_COMPAT_MAX_BYTES`.

The body limiter now counts actual ASGI chunks in addition to checking
`Content-Length`, so chunked requests cannot bypass it.

New PDF controls:

```dotenv
PDF_MAX_PAGE_PIXELS=16000000
PDF_CONVERSION_CONCURRENCY=2
PDF_CONVERSION_TIMEOUT_SECONDS=60
```

Page count, rendered-page count, raw bytes, and extracted-character controls
retain their v0.1 environment names.

## Add bounded admission control

The code defaults keep gateway request concurrency and rate limiting disabled
until configured. For a single 32 GiB GPU, start with:

```dotenv
GATEWAY_MAX_INFLIGHT=2
GATEWAY_MAX_QUEUE_SIZE=8
GATEWAY_QUEUE_TIMEOUT_SECONDS=30
GATEWAY_REQUESTS_PER_MINUTE=0
GATEWAY_RATE_LIMIT_BURST=10
```

Full/expired queues return `429` with `Retry-After`. Streaming requests retain a
slot through the final response byte. Set a positive requests-per-minute value
to enable the per-key token bucket.

These controls are process-local. A multi-worker or replicated deployment
needs an external/global policy.

## Update Gemini streaming clients

v0.1 synthesized one response event for `streamGenerateContent`. v0.2 converts
the upstream OpenAI stream incrementally and may return many events containing
text, thinking, completed function calls, finish reasons, and final usage.

- `?alt=sse` uses Gemini SSE framing.
- Without `alt=sse`, the response is a streaming JSON array.

Clients must not assume one network chunk equals one event or that the complete
response fits in one event. Test cancellation and tool-call flows as well as
plain text.

## Split gateway and vLLM metrics

`/gateway/metrics` is the new gateway exporter. It provides bounded-label
request totals and duration histograms. `/metrics` continues to proxy vLLM.

When gateway authentication is enabled, scrape with a gateway credential. Do
not place keys in Prometheus labels or query strings.

## Python integration changes

New code should construct an application with `create_app(settings)` and import
protocol/document functions from their modules. `vllm_agent_gateway.app`
retains the original ASGI object and selected synchronous helpers as a
compatibility facade, but underscored helpers are not the long-term extension
surface.

See [Architecture](architecture.md) for the module map.

## Smoke-test commands

```bash
curl http://127.0.0.1:8000/healthz

curl http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer client-one'

curl http://127.0.0.1:8000/gateway/metrics \
  -H 'Authorization: Bearer client-one'
```

Also test at least one real request through OpenAI/Responses, Anthropic,
Ollama, and Gemini streaming against the deployed model.
