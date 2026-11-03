"""NFR-7 observability: in-memory counters exposed at /metrics. No customer data."""

import threading
from collections import defaultdict

_lock = threading.Lock()
_counters = defaultdict(int)


def increment(name, by=1):
    with _lock:
        _counters[name] += by


def snapshot():
    with _lock:
        return dict(_counters)
