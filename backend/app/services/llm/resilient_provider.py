"""NFR-4 / B1 reliability: retries timeouts/429/5xx with exponential backoff
and jitter, honours Retry-After, caps at 3 attempts, then falls back to a
second provider (FR-7.3's fallback chain), then lets the caller send a
polite failure message. Every attempt and every fallback is logged.
"""

import logging
import random
import time

from backend.app.services import metrics
from backend.app.services.llm.provider import LLMError, LLMProvider

logger = logging.getLogger('llm.resilient')


class ResilientProvider(LLMProvider):
    name = 'resilient'

    def __init__(self, primary, fallback=None, max_attempts=3, base_delay_seconds=0.5):
        self.primary = primary
        self.fallback = fallback
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds

    def generate(self, **kwargs):
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.primary.generate(**kwargs)
                result.attempts = attempt
                metrics.increment(f'llm_success_{self.primary.name}')
                return result
            except LLMError as exc:
                last_error = exc
                metrics.increment(f'llm_error_{exc.code}')
                logger.warning('LLM call failed (attempt %d/%d): %s %s',
                                attempt, self.max_attempts, exc.code, exc)
                if not exc.retryable or attempt == self.max_attempts:
                    break
                delay = exc.retry_after if getattr(exc, 'retry_after', None) else (
                    self.base_delay_seconds * (2 ** (attempt - 1)))
                delay += random.uniform(0, delay * 0.25)  # jitter
                time.sleep(delay)

        if self.fallback is not None:
            metrics.increment('llm_fallback_used')
            logger.warning('Falling back to %s after %s', self.fallback.name, last_error)
            result = self.fallback.generate(**kwargs)
            result.attempts = self.max_attempts
            return result

        raise last_error
