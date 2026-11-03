"""In-memory sliding-window rate limiter (B1 'rate limiting', FR-4.10 'rate
limit per user and per IP'). Swap for Redis in production without touching
callers — the interface is just allow(key) -> bool.
"""

import threading
import time
from collections import defaultdict, deque

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 30


class SlidingWindowLimiter:
    def __init__(self, window_seconds=WINDOW_SECONDS, max_requests=MAX_REQUESTS_PER_WINDOW):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self._lock = threading.Lock()
        self._hits = defaultdict(deque)

    def allow(self, key):
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window_seconds
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True


generate_limiter = SlidingWindowLimiter()
chat_limiter = SlidingWindowLimiter(max_requests=60)
