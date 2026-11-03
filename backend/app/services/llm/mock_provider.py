"""Two things live here:

1. MockProvider — a deterministic, no-network provider used as the final
   fallback (FR-7.3) when Gemini is unconfigured or exhausted. Reuses the
   rule-based discovery/composer so behaviour is identical to what the system
   did before any LLM was wired in.
2. FaultInjectingProvider — wraps any provider and forces it to time out,
   429, 500 or return invalid JSON on a schedule, so B3's reliability
   requirement ("simulate an LLM provider that randomly times out...") is a
   real, testable object, not just a claim in a report.
"""

import json
import random

from backend.app.services.llm.provider import (
    InvalidOutputLLMError, LLMProvider, LLMResult, RateLimitedLLMError,
    ServerLLMError, TimeoutLLMError,
)


class MockProvider(LLMProvider):
    name = 'mock'

    def generate(self, *, system, messages, response_schema=None, temperature=0.0,
                 max_tokens=1024, timeout=20.0) -> LLMResult:
        structured = None
        if response_schema is not None:
            # Minimal valid-shaped object so callers relying on structured
            # output still get something schema-shaped rather than None —
            # generate_service.py's validation then correctly rejects this
            # (every field empty) and falls through to its own rule-based path.
            structured = {key: None for key in response_schema.get('properties', {})}
        # Free-text mode returns empty text deliberately: this "provider" has
        # no actual language model behind it, so it must not pretend to have
        # written prose — callers (llm_composer.py) check `if text:` and use
        # their own hand-written fallback copy when this is empty.
        return LLMResult(text='', structured=structured, model='mock-rule-based', provider=self.name)


class FaultInjectingProvider(LLMProvider):
    """modes: a list of fault names cycled through on each call, e.g.
    ['timeout', '429', '500', 'invalid_json', 'ok']."""
    name = 'fault-injecting'

    def __init__(self, wrapped, modes, seed=None):
        self.wrapped = wrapped
        self.modes = list(modes)
        self._call_count = 0
        self._rand = random.Random(seed)

    def generate(self, **kwargs):
        mode = self.modes[self._call_count % len(self.modes)]
        self._call_count += 1

        if mode == 'timeout':
            raise TimeoutLLMError('simulated timeout')
        if mode == '429':
            raise RateLimitedLLMError('simulated rate limit', retry_after=1)
        if mode == '500':
            raise ServerLLMError('simulated server error')
        if mode == 'invalid_json':
            schema = kwargs.get('response_schema')
            if schema is not None:
                raise InvalidOutputLLMError('simulated invalid JSON: {not valid')
            return LLMResult(text='{not valid json', provider=self.name)
        return self.wrapped.generate(**kwargs)
