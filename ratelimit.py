"""Small in-memory sliding-window rate limiter (single process)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class RateLimiter:
    def __init__(self) -> None:
        self._hits: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, bucket: str, key: str, limit: int, window_s: float) -> float:
        """Record a hit. Returns 0 if allowed, otherwise seconds until the next slot frees."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[(bucket, key)]
            while q and now - q[0] >= window_s:
                q.popleft()
            if len(q) >= limit:
                return max(1.0, window_s - (now - q[0]))
            q.append(now)
            if len(self._hits) > 50_000:
                self._gc(now)
            return 0.0

    def _gc(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] > 86400]:
            del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
