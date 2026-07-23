# Reproducible benchmarking

This project distinguishes compatibility smokes from controlled performance
benchmarks. A successful request proves that a path works; it does not by itself
measure gateway overhead or GPU throughput.

## What to record

For every result, record enough context to make the number falsifiable:

- date and commit SHA;
- GPU, driver, CUDA, Python, vLLM, and gateway versions;
- model, quantization, context length, memory utilization, maximum sequences,
  maximum batched tokens, and prefix-caching setting;
- protocol, prompt tokens, requested output tokens, concurrency, and request
  count;
- mean, p50, p95, failures, wall time, and requests per second;
- whether the run targeted vLLM directly or traversed the gateway.

Warm the model before recording results. Run one protocol and prompt profile at
a time, repeat the run, and keep the raw JSON Lines output with the report.

## Compatibility and capacity smoke

The repository includes a dependency-free runner:

```bash
python scripts/load_smoke.py --dry-run --protocol all

GATEWAY_API_KEY=change-me python scripts/load_smoke.py \
  --protocol all \
  --concurrency 1 2 4 \
  --requests-per-level 8 \
  --prompt-size tiny
```

Large context profiles are guarded to avoid accidental traffic:

```bash
GATEWAY_API_KEY=change-me python scripts/load_smoke.py \
  --protocol openai \
  --concurrency 1 2 \
  --prompt-size 192k \
  --allow-large-prompts \
  --timeout 600
```

The named prompt sizes are character-based approximations. For an exact model
boundary, tokenize the generated prompt through the same vLLM `/tokenize`
endpoint before sending it.

## Measuring gateway overhead

Use the OpenAI protocol for the controlled comparison because both the gateway
and vLLM expose the same request shape:

1. Run the smoke against the gateway URL and save stdout as JSON Lines.
2. Run the identical command against the private vLLM URL with the same model,
   prompt, output length, request count, and concurrency.
3. Compare paired wall time and latency distributions. Do not compare runs with
   different KV-cache state or simultaneous background traffic.

Example:

```bash
python scripts/load_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --protocol openai --concurrency 1 2 --requests-per-level 20 \
  > gateway.jsonl

python scripts/load_smoke.py \
  --base-url http://127.0.0.1:8001 \
  --protocol openai --concurrency 1 2 --requests-per-level 20 \
  > direct-vllm.jsonl
```

Report absolute latency as well as the difference. For short outputs, GPU
scheduling noise can be larger than gateway overhead, so multiple warmed runs
matter more than a single attractive number.

## Interpreting long-context concurrency

Gateway admission and backend execution are different measurements. Two
requests can be accepted concurrently while vLLM schedules their prefills close
to serially because KV-cache or compute capacity is constrained. Queueing avoids
immediate client failure; it does not create GPU capacity.

The checked-in [validated 32 GiB profile](validated-profile-qwen36-5090.md)
records one real capacity smoke and explicitly limits the conclusions that can
be drawn from it.
