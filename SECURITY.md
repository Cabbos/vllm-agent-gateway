# Security policy

## Reporting a vulnerability

Report security issues privately through GitHub's security advisory feature.
Do not open a public issue containing credentials, private prompts, internal
hostnames, documents, or exploitable request payloads.

Include the affected version, deployment shape, reproduction steps, and impact
when possible. Remove unrelated secrets before attaching logs or captures.

## Supported deployment boundary

The project is intended for a single operator, trusted users, and local or
private-network vLLM deployments. It provides useful application controls, but
it is not an internet-edge security product or a multi-tenant identity system.

Before allowing shared access:

- terminate TLS at a private ingress or authenticated reverse proxy;
- set non-empty, random `GATEWAY_API_KEYS` and rotate them operationally;
- use a different `VLLM_UPSTREAM_API_KEY` for the private gateway-to-vLLM hop;
- restrict `GATEWAY_CORS_ORIGINS` and `GATEWAY_TRUSTED_HOSTS`;
- enable bounded concurrency and an appropriately small queue;
- set ingress request, connection, and idle timeouts;
- restrict network egress from both the gateway and vLLM containers;
- treat prompts, tool results, documents, metrics, and model output as sensitive.

The gateway never executes model-requested tools. Tool execution, filesystem
access, MCP servers, shells, and browsers remain in the calling agent and need
their own sandbox, allowlists, and approval policy.

## Authentication and credentials

`GATEWAY_API_KEYS` authenticates incoming clients. An empty value deliberately
disables gateway authentication and is only appropriate on a strictly local,
trusted endpoint.

`VLLM_UPSTREAM_API_KEY` authenticates the gateway to vLLM or its private
ingress. Client Authorization, `X-Api-Key`, `Api-Key`, `X-Goog-Api-Key`, and
Gemini query-key credentials are stripped before generic upstream forwarding;
the configured upstream key is added separately.

Prefer authentication headers. Gemini's `?key=` form is accepted for client
compatibility, but Uvicorn, ingress, and load-balancer access logs may retain
query strings. Disable or redact such logs if query credentials are unavoidable.

Keys are static configuration values. There is no account database, key
expiration, revocation API, user role model, durable audit log, or billing
boundary. Restart/redeploy after rotating environment-provided keys.

## Request and document controls

The ASGI request-body limiter checks both `Content-Length` and bytes actually
received, including chunked requests. Keep `GATEWAY_MAX_REQUEST_BYTES` large
enough for allowed base64 documents but no larger than operationally necessary.

Document conversion separately limits:

- decoded raw bytes;
- total PDF pages;
- scanned pages rendered to JPEG;
- extracted characters;
- pixels per page at the configured render scale;
- concurrent conversions;
- total load/conversion time.

These controls reduce accidental and malicious resource use; they do not make
arbitrary PDFs risk-free. PyMuPDF remains part of the parsing attack surface.
Run the gateway as an unprivileged user, keep dependencies updated, and apply
container CPU/memory limits appropriate to the host.

## Remote document URLs and SSRF

Remote PDF/plain-text URLs are denied by default. `DOCUMENT_URL_POLICY=allowlist`
also requires an exact or wildcard entry in `DOCUMENT_ALLOWED_HOSTS`.

For allowed URLs, the loader:

- permits only HTTP(S) and ports 80/443;
- rejects embedded credentials;
- checks every DNS answer;
- rejects loopback, private, link-local, reserved, benchmark, multicast, and
  unspecified addresses unless an explicit extra CIDR covers them;
- revalidates every redirect target;
- compares the connected peer with validated DNS results when the HTTP
  transport exposes peer information;
- enforces declared and streamed byte limits.

There is no default `198.18.0.0/15` transparent-proxy exception. Adding a CIDR
to `DOCUMENT_EXTRA_ALLOWED_NETWORKS` deliberately grants access to that network
for allowlisted hosts. Use the narrowest CIDR possible and enforce egress rules
outside the process as the final SSRF boundary.

The policy covers URLs the gateway document service fetches or explicitly
validates. It is not a universal filter for every URL embedded in a payload or
for URLs interpreted later by vLLM, a model plugin, or a calling agent.

## Rate limiting, queues, and metrics

The concurrency queue and token-bucket rate limiter are in memory and local to
one gateway process. They do not coordinate across workers, hosts, or replicas.
Use an ingress or shared external limiter for a global policy.

Per-key rate limiting is meaningful only when gateway authentication is enabled.
Without authentication, all requests share one anonymous bucket; arbitrary
presented credentials do not create additional rate-limit identities.

`/gateway/metrics` is authenticated when `GATEWAY_API_KEYS` is configured. Its
labels are deliberately bounded and reject API keys, URLs, request IDs, paths,
and other sensitive/high-cardinality names. Prometheus data still reveals
traffic volume and latency and should remain on a monitoring network.

The proxied vLLM `/metrics` endpoint may reveal model and scheduler details and
must also be protected.

## Remaining production responsibilities

The operator remains responsible for:

- network firewalling, egress policy, TLS, and denial-of-service protection;
- dependency and base-image patching;
- vLLM/model-specific security and trust decisions;
- log redaction, retention, access control, and incident response;
- shared quotas and admission control across replicas;
- GPU/CPU/memory isolation and capacity planning;
- safe agent tool execution and output handling.

See [Production safety boundaries](docs/production-security.md) for a deployment
checklist.
