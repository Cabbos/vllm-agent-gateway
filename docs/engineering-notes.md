# Engineering case studies

This document explains four implementation decisions that are easy to miss in
an endpoint list. Each case starts with a failure mode observed in real agent
workflows and records the boundary the gateway owns.

## Incremental Gemini streaming over OpenAI SSE

### Problem

vLLM emits OpenAI-style Server-Sent Events, while Gemini clients expect either
Gemini-framed SSE or a streaming JSON array. Network chunks do not align with
SSE events, tool arguments may arrive across multiple deltas, and usage can
appear only in the final event. Buffering the entire response would make the
API look non-streaming and increase memory usage for long generations.

### Design

`adapters/gemini_stream.py` separates byte framing, SSE event parsing, tool-call
assembly, candidate conversion, and output framing. The parser retains only an
incomplete event tail and active tool-call fragments. Complete candidates are
emitted immediately in the client-requested framing.

The stream owner closes the upstream response under cancellation shielding, so
a disconnected client does not leave an abandoned vLLM response consuming a
connection indefinitely.

### Verification

Tests split SSE data at arbitrary byte boundaries and cover text, thinking,
function calls, finish reasons, usage metadata, malformed events, SSE output,
and streaming JSON arrays.

Relevant code:

- `src/vllm_agent_gateway/adapters/gemini_stream.py`
- `src/vllm_agent_gateway/proxy/streaming.py`
- `tests/test_gemini_stream_v2.py`
- `tests/test_cancellation_v2.py`

## Multimodal history compaction

### Problem

Agent clients commonly resend the entire conversation on every request. Images
from old turns therefore remain real image inputs even when the user is no
longer discussing them. Once the history exceeds vLLM's per-prompt image limit,
the backend rejects the request before generation.

### Design

The gateway walks the complete normalized conversation, including images nested
inside tool results. It preserves the newest configured number of images and
replaces older payloads with explicit text placeholders. Text, tool calls,
results, ordering, and the newest visual context remain intact.

This policy is deterministic and budget-based. It does not ask a model to
summarize images and does not silently remove entire messages.

### Trade-off

An evicted image can no longer be visually re-examined. The placeholder makes
that loss visible to the model and user. Deployments should set
`GATEWAY_MAX_PROMPT_IMAGES` at or below vLLM's `--limit-mm-per-prompt` value.

Relevant code:

- `src/vllm_agent_gateway/adapters/anthropic.py`
- `src/vllm_agent_gateway/adapters/openai.py`
- `tests/test_anthropic_adapter_v2.py`
- `tests/test_protocols.py`

## Bounded document conversion and SSRF defense

### Problem

PDF conversion combines untrusted bytes, CPU-heavy parsing and rendering, and
optional network fetching. A URL allowlist alone is insufficient: DNS can
resolve to private addresses, redirects can leave the allowed origin, and
large or scanned documents can exhaust memory or worker capacity.

### Design

Document handling applies independent budgets for raw bytes, request size,
pages, rendered pages, extracted text, per-page pixels, worker concurrency, and
end-to-end conversion time. Remote URLs are denied by default. When explicitly
enabled, every DNS result and redirect is validated, and the connected peer IP
is checked when the transport exposes it.

Searchable pages become text; sparse scanned pages are rendered as bounded JPEG
blocks. CPU work runs outside the event loop under a capacity limiter.

Relevant code:

- `src/vllm_agent_gateway/documents/`
- `src/vllm_agent_gateway/document_service.py`
- `tests/test_document_security_v2.py`
- `docs/production-security.md`

## Backpressure that follows the response lifetime

### Problem

Releasing a concurrency slot as soon as response headers are returned makes a
streaming limiter ineffective: the expensive vLLM generation is still running.
Unbounded queueing merely moves overload into memory and unpredictable latency.

### Design

An admitted streaming request owns its slot until the final response byte or a
cancellation/error cleanup path. Queue capacity and wait time are bounded, and
overload returns `429` with `Retry-After`. The gateway-to-vLLM HTTP pool is also
bounded independently.

The controls are intentionally process-local. Multi-replica deployments must
place global quotas at an ingress or shared rate-limit service.

Relevant code:

- `src/vllm_agent_gateway/middleware/concurrency.py`
- `src/vllm_agent_gateway/middleware/rate_limit.py`
- `src/vllm_agent_gateway/proxy/streaming.py`
- `tests/test_gateway_controls_v2.py`

## What the boundaries demonstrate

The gateway does not attempt to recreate cloud identity, billing, hosted tools,
or a distributed scheduler. Its responsibility is narrower: make heterogeneous
agent protocols safe and predictable enough to share a private vLLM backend,
and make overload or unsupported behavior explicit rather than accidental.
