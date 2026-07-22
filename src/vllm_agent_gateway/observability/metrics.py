from __future__ import annotations

import math
import re
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from starlette.types import Receive, Scope, Send

_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_FORBIDDEN_LABELS = frozenset(
    {
        "api_key",
        "client_ip",
        "host",
        "key",
        "path",
        "request_id",
        "secret",
        "token",
        "url",
        "user_id",
    }
)


def _escape_help(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(value)


def _label_text(label_names: Sequence[str], values: Sequence[str]) -> str:
    if not label_names:
        return ""
    pairs = ",".join(
        f'{name}="{_escape_label(value)}"' for name, value in zip(label_names, values, strict=True)
    )
    return "{" + pairs + "}"


class MetricSeriesLimitError(ValueError):
    pass


class _Metric:
    metric_type: str

    def __init__(
        self,
        name: str,
        help_text: str,
        *,
        label_names: Sequence[str],
        allowed_label_values: Mapping[str, Collection[str]] | None,
        max_series: int,
        lock: threading.RLock,
    ) -> None:
        if not _METRIC_NAME.fullmatch(name):
            raise ValueError(f"invalid Prometheus metric name: {name!r}")
        if max_series < 1:
            raise ValueError("max_series must be at least 1")
        names = tuple(label_names)
        if len(set(names)) != len(names):
            raise ValueError("label names must be unique")
        for label_name in names:
            if not _LABEL_NAME.fullmatch(label_name):
                raise ValueError(f"invalid Prometheus label name: {label_name!r}")
            if label_name.lower() in _FORBIDDEN_LABELS:
                raise ValueError(f"sensitive or high-cardinality label is forbidden: {label_name}")
        allowed = allowed_label_values or {}
        unknown_allowed = set(allowed) - set(names)
        if unknown_allowed:
            raise ValueError(f"allowlist provided for unknown labels: {sorted(unknown_allowed)}")

        self.name = name
        self.help_text = help_text
        self.label_names = names
        self.allowed_label_values = {
            key: frozenset(str(value) for value in values) for key, values in allowed.items()
        }
        self.max_series = max_series
        self._lock = lock

    def _key(
        self, labels: Mapping[str, str] | None, known: Collection[tuple[str, ...]]
    ) -> tuple[str, ...]:
        values_by_name = labels or {}
        if set(values_by_name) != set(self.label_names):
            raise ValueError(f"labels must be exactly {self.label_names!r}")
        key = tuple(str(values_by_name[name]) for name in self.label_names)
        for name, value in zip(self.label_names, key, strict=True):
            allowed = self.allowed_label_values.get(name)
            if allowed is not None and value not in allowed:
                raise ValueError(f"label {name!r} value is not allowlisted")
        if key not in known and len(known) >= self.max_series:
            raise MetricSeriesLimitError(
                f"metric {self.name!r} reached its {self.max_series} series limit"
            )
        return key

    def _header(self) -> list[str]:
        return [
            f"# HELP {self.name} {_escape_help(self.help_text)}",
            f"# TYPE {self.name} {self.metric_type}",
        ]

    def render(self) -> list[str]:
        raise NotImplementedError


class Counter(_Metric):
    metric_type = "counter"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        if amount < 0 or not math.isfinite(amount):
            raise ValueError("counter increment must be finite and non-negative")
        with self._lock:
            key = self._key(labels, self._values)
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        with self._lock:
            lines = self._header()
            for key, value in sorted(self._values.items()):
                lines.append(f"{self.name}{_label_text(self.label_names, key)} {_number(value)}")
            return lines


class Gauge(_Metric):
    metric_type = "gauge"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._values: dict[tuple[str, ...], float] = {}

    def set(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        if not math.isfinite(value):
            raise ValueError("gauge value must be finite")
        with self._lock:
            key = self._key(labels, self._values)
            self._values[key] = value

    def inc(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        self._add(amount, labels)

    def dec(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        self._add(-amount, labels)

    def _add(self, amount: float, labels: Mapping[str, str] | None) -> None:
        if not math.isfinite(amount):
            raise ValueError("gauge delta must be finite")
        with self._lock:
            key = self._key(labels, self._values)
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        with self._lock:
            lines = self._header()
            for key, value in sorted(self._values.items()):
                lines.append(f"{self.name}{_label_text(self.label_names, key)} {_number(value)}")
            return lines


@dataclass(slots=True)
class _HistogramValue:
    buckets: list[int]
    count: int = 0
    total: float = 0.0


class Histogram(_Metric):
    metric_type = "histogram"

    def __init__(self, *args, buckets: Sequence[float], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        normalized = tuple(float(bound) for bound in buckets)
        if not normalized or any(not math.isfinite(bound) for bound in normalized):
            raise ValueError("histogram buckets must contain finite bounds")
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("histogram buckets must be strictly increasing")
        self.buckets = normalized
        self._values: dict[tuple[str, ...], _HistogramValue] = {}

    def observe(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        if not math.isfinite(value):
            raise ValueError("histogram observation must be finite")
        with self._lock:
            key = self._key(labels, self._values)
            state = self._values.get(key)
            if state is None:
                state = _HistogramValue([0] * len(self.buckets))
                self._values[key] = state
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    state.buckets[index] += 1
            state.count += 1
            state.total += value

    def render(self) -> list[str]:
        with self._lock:
            lines = self._header()
            for key, state in sorted(self._values.items()):
                base_labels = dict(zip(self.label_names, key, strict=True))
                for bound, count in zip(self.buckets, state.buckets, strict=True):
                    labels = (*key, _number(bound))
                    names = (*self.label_names, "le")
                    lines.append(f"{self.name}_bucket{_label_text(names, labels)} {count}")
                infinity_labels = (*key, "+Inf")
                infinity_names = (*self.label_names, "le")
                lines.append(
                    f"{self.name}_bucket{_label_text(infinity_names, infinity_labels)} {state.count}"
                )
                label_text = _label_text(self.label_names, tuple(base_labels.values()))
                lines.append(f"{self.name}_sum{label_text} {_number(state.total)}")
                lines.append(f"{self.name}_count{label_text} {state.count}")
            return lines


class MetricRegistry:
    """A dependency-free registry with bounded series cardinality."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._metrics: dict[str, _Metric] = {}

    def counter(
        self,
        name: str,
        help_text: str,
        *,
        label_names: Sequence[str] = (),
        allowed_label_values: Mapping[str, Collection[str]] | None = None,
        max_series: int = 100,
    ) -> Counter:
        return self._register(
            Counter(
                name,
                help_text,
                label_names=label_names,
                allowed_label_values=allowed_label_values,
                max_series=max_series,
                lock=self._lock,
            )
        )

    def gauge(
        self,
        name: str,
        help_text: str,
        *,
        label_names: Sequence[str] = (),
        allowed_label_values: Mapping[str, Collection[str]] | None = None,
        max_series: int = 100,
    ) -> Gauge:
        return self._register(
            Gauge(
                name,
                help_text,
                label_names=label_names,
                allowed_label_values=allowed_label_values,
                max_series=max_series,
                lock=self._lock,
            )
        )

    def histogram(
        self,
        name: str,
        help_text: str,
        *,
        buckets: Sequence[float],
        label_names: Sequence[str] = (),
        allowed_label_values: Mapping[str, Collection[str]] | None = None,
        max_series: int = 100,
    ) -> Histogram:
        return self._register(
            Histogram(
                name,
                help_text,
                buckets=buckets,
                label_names=label_names,
                allowed_label_values=allowed_label_values,
                max_series=max_series,
                lock=self._lock,
            )
        )

    def _register(self, metric):
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"metric is already registered: {metric.name}")
            self._metrics[metric.name] = metric
        return metric

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name in sorted(self._metrics):
                lines.extend(self._metrics[name].render())
            return "\n".join(lines) + ("\n" if lines else "")


class MetricsEndpoint:
    content_type = "text/plain; version=0.0.4; charset=utf-8"

    def __init__(self, registry: MetricRegistry) -> None:
        self.registry = registry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        if scope.get("method") not in {"GET", "HEAD"}:
            status = 405
            body = b"Method Not Allowed"
            extra = [(b"allow", b"GET, HEAD")]
        else:
            status = 200
            rendered = self.registry.render().encode("utf-8")
            body = b"" if scope.get("method") == "HEAD" else rendered
            extra = []
        headers = [
            (b"content-type", self.content_type.encode("ascii")),
            (b"content-length", str(len(body)).encode("ascii")),
            *extra,
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
