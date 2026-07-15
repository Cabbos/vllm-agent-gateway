from __future__ import annotations

from collections.abc import Iterable
from typing import Any

CompatibilityEvent = dict[str, Any]


def route_to_model(
    payload: dict[str, Any], served_model: str, events: list[CompatibilityEvent]
) -> None:
    requested_model = payload.get("model")
    if requested_model != served_model:
        payload["model"] = served_model
        events.append(
            {
                "code": "model_routed_local",
                "requested_model": str(requested_model or "<default>"),
            }
        )


def event_codes(events: Iterable[CompatibilityEvent], limit: int = 512) -> str:
    codes: list[str] = []
    for event in events:
        code = str(event.get("code") or "")
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes)[:limit]
