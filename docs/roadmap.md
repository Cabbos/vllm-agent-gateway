# Roadmap

The roadmap prioritizes reliability evidence over adding more protocol surface.
Items are directional and do not imply a release date.

## Implemented foundation

- OpenAI, Anthropic, Ollama, Gemini-style, and Azure-style compatibility;
- incremental streaming conversion and cancellation-aware upstream cleanup;
- bounded multimodal-history and document processing;
- authentication, body limits, concurrency admission, queueing, and rate limits;
- gateway and upstream Prometheus metrics;
- typed Python, complexity and coverage gates, dependency audit, and container CI;
- hardened single-GPU Compose profile and reproducible load smoke.

## Next: operational resilience

- explicit end-to-end timeout budgets by request phase;
- retry policy limited to safe, pre-response upstream failures;
- multi-upstream health routing and circuit breaking;
- structured benchmark comparison and machine-readable run metadata;
- release images with provenance, version tags, and rollback verification;
- split protocol route handlers out of `application.py` while preserving the
  public application factory.

## Later: multi-replica governance

- shared tenant quotas and rate-limit state;
- model-aware routing across heterogeneous vLLM pools;
- durable audit events with secret and high-cardinality redaction;
- OpenTelemetry traces across ingress, gateway, and vLLM;
- load-shedding policies informed by backend scheduler and KV-cache pressure.

## Non-goals

- recreating cloud billing, IAM, hosted files, grounding, or safety services;
- executing client tools inside the gateway;
- hiding unsupported protocol behavior behind fabricated responses;
- treating a process-local queue as a distributed GPU scheduler.
