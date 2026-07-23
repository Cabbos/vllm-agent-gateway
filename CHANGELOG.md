# Changelog

All notable project changes are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - Unreleased

### Added

- Modular application factory, protocol adapters, document subsystem,
  middleware, proxy, and observability packages.
- ASGI request-body enforcement that counts actual chunks as well as declared
  `Content-Length`.
- Default-deny remote document policy with exact/wildcard host allowlists,
  redirect revalidation, DNS/peer-IP checks, and explicit extra CIDRs.
- PDF limits for rendered page pixels, conversion concurrency, and conversion
  timeout in addition to raw bytes, page counts, rendered-page counts, and
  extracted characters.
- Optional bounded request concurrency, wait queue, queue deadline, and
  `Retry-After` responses.
- Optional in-process token-bucket rate limiting per authenticated API key.
- `/gateway/metrics` Prometheus exporter with bounded protocol/outcome labels.
- Separate `VLLM_UPSTREAM_API_KEY` for gateway-to-vLLM authentication.
- Incremental Gemini `streamGenerateContent` conversion for fragmented OpenAI
  SSE, including text, thinking, function calls, finish reasons, and usage.
- Request IDs returned through `X-Request-ID`.
- Configurable Anthropic image-history compaction that preserves the newest
  images within vLLM's per-prompt multimodal limit.
- Multi-stage, non-root gateway image and a hardened Compose service with a
  read-only root filesystem, dropped capabilities, `no-new-privileges`, PID
  limit, init process, and bounded temporary filesystem.
- CI checks for Python 3.11/3.12, dependency auditing, Compose validation, and
  gateway image builds.
- Dependency-free multi-protocol load-smoke script with guarded large-prompt
  profiles.
- Guarded capacity torture runner with self-observed vLLM stall detection,
  exact-answer anchors, scheduler/KV metrics, and GPU sampling.
- Deterministic capability-degradation evaluation for executable code,
  evidence-grounded answers, and real tool selection across context pressure.
- A recorded 32 GiB RTX 5090/Qwen validation profile covering protocol,
  concurrency, PDF/tool, Gemini streaming, and exact 192K-context smoke tests.
- Portfolio-oriented project overview, engineering case studies, reproducible
  benchmarking guidance, and an explicit operational roadmap.

### Changed

- `app.py` is now a small ASGI entry point and v0.1 compatibility facade;
  `create_app(settings)` is the primary application construction API.
- Remote document URLs are denied unless `DOCUMENT_URL_POLICY=allowlist` and
  the target host is explicitly listed.
- The former implicit `198.18.0.0/15` transparent-proxy allowance was removed.
- The default request-body limit is derived from the raw PDF limit to account
  for base64 expansion and JSON overhead.
- Gemini request and stream conversion are decomposed into focused helpers;
  CI now enforces type checking, 80% coverage, and bounded function complexity.
- Client API credentials are stripped before upstream forwarding; an optional
  dedicated upstream credential is injected instead.
- Gemini non-SSE streaming uses a streaming JSON array instead of a synthesized
  one-event response.

### Security

- Private, loopback, link-local, reserved, benchmark, multicast, and unspecified
  document addresses now require an explicit network opt-in in addition to an
  allowlisted host.
- Gateway metrics prohibit sensitive and high-cardinality label names such as
  API keys, URLs, paths, and request IDs.
- PDF conversion rejects excessive per-page render dimensions before allocating
  the page image.

### Migration

See [Migrating from v0.1 to v0.2](docs/migration-v0.2.md).

## [0.1.0]

### Added

- Initial OpenAI, Anthropic, Ollama, Gemini-style, and Azure-style compatibility
  gateway for a single local vLLM model.
- Model alias routing, tool/thinking field mapping, streaming proxy behavior,
  and PDF/plain-text compatibility conversion.
