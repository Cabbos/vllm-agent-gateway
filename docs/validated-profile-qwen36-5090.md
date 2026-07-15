# Validated 32 GiB Qwen profile

This is a reproducible compatibility and capacity smoke, not a controlled
throughput benchmark. It records one configuration that was exercised end to
end on 2026-07-15.

## Hardware and runtime

- GPU: NVIDIA GeForce RTX 5090, 32 GiB
- Backend: vLLM 0.25.0 under Linux/WSL
- Model: `Qwen3.6-35B-A3B-NVFP4-Fast`
- Quantization/load format: compressed-tensors NVFP4
- Maximum model length: `196608`
- GPU memory utilization: `0.90`
- Maximum sequences: `2`
- Maximum batched tokens: `4096`
- Prefix caching: enabled
- Reasoning parser: `qwen3`
- Tool-call parser: `qwen3_coder`

At startup, vLLM reported a 22.02 GiB checkpoint, 5.54 GiB available for KV
cache, a 544,604-token GPU KV cache, and estimated maximum concurrency of
`2.77x` at 196,608 tokens per request. Treat these values as version-, model-,
and hardware-specific.

## Compatibility smoke

The gateway was configured with `GATEWAY_MAX_INFLIGHT=2` and a bounded queue.
The following live requests completed successfully:

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages with thinking disabled
- Ollama Chat
- Gemini `generateContent`, incremental SSE, streaming JSON array, and
  `countTokens.generateContentRequest`
- Azure deployment-prefixed Chat Completions
- forced OpenAI function call using the configured Qwen tool parser
- searchable PDF through an OpenAI Responses `input_file`

For short 16-token generations, one wave at concurrency 1, 2, and 4 completed
without failures across OpenAI, Anthropic, Ollama, Gemini, and Azure surfaces.
Concurrency 4 was accepted by the gateway and queued above the two in-flight
slots.

## Long-context smoke

Prompts were tokenized through the same vLLM `/tokenize` endpoint. Single
requests at exactly 131,072 and 192,000 input tokens returned HTTP 200.

Two different, non-shared 192,000-token prefixes were then submitted together.
Both returned HTTP 200. Their observed completion times were approximately 33
and 66 seconds with a 66.6-second wall time, showing that the gateway can admit
two requests while the backend effectively schedules this extreme case close
to serially. A queue therefore avoids client failures but does not create more
GPU prefill capacity.

Practical interpretation for this 32 GiB profile:

- use `GATEWAY_MAX_INFLIGHT=2` for general agent traffic;
- expect ordinary short requests to benefit from batching;
- expect simultaneous near-192K requests to wait substantially;
- use `1` in-flight request when long-context latency predictability matters;
- re-run the smoke after changing vLLM, CUDA kernels, quantization, model, or
  memory utilization.

Use `scripts/load_smoke.py` for protocol/concurrency checks. Its named large
profiles are character-based approximations across tokenizers; use `/tokenize`
when an exact model-specific context boundary matters.
