# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

Tests must not require a GPU or a running model unless they are explicitly marked as integration tests. Keep provider-specific behavior behind compatibility transformations and preserve transparent forwarding for unknown vLLM endpoints.

Security-sensitive changes should include tests for authentication, URL validation, body-size limits, or error behavior as applicable. Never commit model weights, prompts containing private data, access tokens, or local `.env` files.

