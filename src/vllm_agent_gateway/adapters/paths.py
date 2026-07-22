from __future__ import annotations

import re

ALIASES = {
    "models": "v1/models",
    "chat/completions": "v1/chat/completions",
    "completions": "v1/completions",
    "responses": "v1/responses",
    "messages": "v1/messages",
    "messages/count_tokens": "v1/messages/count_tokens",
    "openai/v1/models": "v1/models",
    "openai/v1/chat/completions": "v1/chat/completions",
    "openai/v1/completions": "v1/completions",
    "openai/v1/responses": "v1/responses",
}


def normalize_proxy_path(path: str) -> str:
    if path in ALIASES:
        return ALIASES[path]
    azure_match = re.fullmatch(
        r"openai/deployments/[^/]+/(chat/completions|completions|responses)", path
    )
    return f"v1/{azure_match.group(1)}" if azure_match else path
