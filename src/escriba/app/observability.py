"""Structured observability: per-request correlation IDs and latency metrics."""
from __future__ import annotations

import threading
import uuid
from typing import Any

_ctx = threading.local()


def new_correlation_id() -> str:
    return str(uuid.uuid4())


def set_correlation_id(cid: str) -> None:
    _ctx.correlation_id = cid


def get_correlation_id() -> str | None:
    return getattr(_ctx, "correlation_id", None)


class LatencyStore:
    """Rolling P50/P99 latency tracker keyed by operation name.

    Thread-safe, capped at _MAX_SAMPLES entries per key to bound memory.
    """

    _MAX_SAMPLES = 1000

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def record(self, key: str, latency_ms: float) -> None:
        with self._lock:
            bucket = self._data.setdefault(key, [])
            bucket.append(latency_ms)
            if len(bucket) > self._MAX_SAMPLES:
                del bucket[0]

    def percentile(self, key: str, p: float) -> float | None:
        with self._lock:
            samples = sorted(self._data.get(key, []))
        if not samples:
            return None
        idx = max(0, int(len(samples) * p) - 1)
        return samples[idx]

    def p50(self, key: str) -> float | None:
        return self.percentile(key, 0.50)

    def p99(self, key: str) -> float | None:
        return self.percentile(key, 0.99)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            keys = list(self._data.keys())
        return {k: {"p50_ms": self.p50(k), "p99_ms": self.p99(k)} for k in keys}


latency_store = LatencyStore()
