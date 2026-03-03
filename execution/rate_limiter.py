"""Token bucket rate limiter for API calls."""

from __future__ import annotations

import time
import threading


class RateLimiter:
    """Token bucket rate limiter for Bybit API."""

    def __init__(
        self,
        max_tokens: int = 10,
        refill_rate: float = 2.0,  # tokens per second
    ):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = float(max_tokens)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """Acquire tokens, blocking until available or timeout."""
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            if time.monotonic() >= deadline:
                return False

            time.sleep(0.1)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill = now

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self.tokens
