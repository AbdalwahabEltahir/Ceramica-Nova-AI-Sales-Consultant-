"""Provider abstraction (FR-7.1 / B1 'model abstraction'). Every backend —
Gemini, the mock/rule-based fallback, a future HF-served local model — speaks
this one interface, so swapping providers never touches orchestrator code.
"""

from dataclasses import dataclass, field
from typing import Optional


class LLMError(Exception):
    """Base class. .code is one of the B1 error-classification buckets."""
    code = 'PROVIDER_ERROR'
    retryable = True

    def __init__(self, message, code=None, retryable=None, retry_after=None):
        super().__init__(message)
        if code:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.retry_after = retry_after


class TimeoutLLMError(LLMError):
    code = 'PROVIDER_TIMEOUT'
    retryable = True


class RateLimitedLLMError(LLMError):
    code = 'RATE_LIMITED'
    retryable = True


class ServerLLMError(LLMError):
    code = 'PROVIDER_ERROR'
    retryable = True


class InvalidOutputLLMError(LLMError):
    code = 'INVALID_MODEL_OUTPUT'
    retryable = True


@dataclass
class LLMResult:
    text: str
    structured: Optional[dict] = None
    model: str = ''
    provider: str = ''
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    attempts: int = 1


class LLMProvider:
    name = 'base'

    def generate(self, *, system, messages, response_schema=None, temperature=0.0,
                 max_tokens=1024, timeout=20.0) -> LLMResult:
        """messages: list of {'role': 'user'|'assistant', 'content': str}.
        response_schema: optional JSON schema dict — when set, .structured is
        populated with a validated dict or InvalidOutputLLMError is raised."""
        raise NotImplementedError
