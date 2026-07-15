from __future__ import annotations

import json
from collections.abc import Iterable

from starlette.types import Message, Send


async def send_json_error(
    send: Send,
    *,
    status_code: int,
    message: str,
    headers: Iterable[tuple[bytes, bytes]] = (),
) -> None:
    body = json.dumps(
        {"error": {"message": message}}, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    response_headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
        *headers,
    ]
    start: Message = {
        "type": "http.response.start",
        "status": status_code,
        "headers": response_headers,
    }
    await send(start)
    await send({"type": "http.response.body", "body": body})
