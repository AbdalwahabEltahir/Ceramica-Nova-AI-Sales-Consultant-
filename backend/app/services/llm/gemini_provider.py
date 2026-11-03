"""Real Gemini provider via Google AI Studio (aistudio.google.com/api-keys),
using the official google-genai SDK. Reads GOOGLE_API_KEY (and optionally
GOOGLE_API_KEY_2) from the environment only — never hardcoded, never logged
(NFR-5/B1 security requirement: "the application should not expose secrets").

The free tier's daily request quota is per Google Cloud project, not per
process — a second key from a *different* project gives the app a second,
independent daily allowance. api_keys() / MultiKeyGeminiProvider below let
router.py fail over from key 1 to key 2 on a 429 before giving up to the
rule-based mock provider, so one key running dry mid-day doesn't flatten
every reply to template text.
"""

import json
import logging
import os
import time

from backend.app.services.llm.provider import (
    InvalidOutputLLMError, LLMProvider, LLMResult, RateLimitedLLMError,
    ServerLLMError, TimeoutLLMError,
)

logger = logging.getLogger('gemini_provider')

_clients = {}  # api_key -> genai.Client, one per key (Client is bound to a key at construction)

THINKING_HEADROOM_TOKENS = 1024


def api_keys():
    """Every configured Gemini key, in fallback order. GOOGLE_API_KEY_2
    (ideally from a separate Google Cloud project — different quota) is
    optional; only GOOGLE_API_KEY is required."""
    keys = [os.environ.get('GOOGLE_API_KEY'), os.environ.get('GOOGLE_API_KEY_2')]
    return [k for k in keys if k]


def _get_client(api_key):
    if api_key not in _clients:
        from google import genai
        _clients[api_key] = genai.Client(api_key=api_key)
    return _clients[api_key]


def is_configured():
    return bool(api_keys())


class GeminiProvider(LLMProvider):
    name = 'gemini'

    def __init__(self, model='gemini-2.5-flash', api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise RuntimeError('GOOGLE_API_KEY is not set')

    def generate(self, *, system, messages, response_schema=None, temperature=0.0,
                 max_tokens=1024, timeout=20.0) -> LLMResult:
        from google.genai import types
        from google.genai import errors as genai_errors

        client = _get_client(self.api_key)
        contents = '\n'.join(f"{m['role']}: {m['content']}" for m in messages)

        # Thinking tokens count against max_output_tokens on Gemini 2.5/3.x;
        # without headroom a reply gets cut off mid-sentence.
        config_kwargs = {
            'system_instruction': system,
            'temperature': temperature,
            'max_output_tokens': max_tokens + THINKING_HEADROOM_TOKENS,
        }
        if self.model.startswith('gemini-3'):
            config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_level='low')
        if response_schema is not None:
            config_kwargs['response_mime_type'] = 'application/json'
            config_kwargs['response_json_schema'] = response_schema

        start = time.monotonic()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.ClientError as exc:
            status = getattr(exc, 'code', None) or getattr(exc, 'status_code', None)
            if status == 429:
                raise RateLimitedLLMError(str(exc)) from exc
            raise ServerLLMError(str(exc)) from exc
        except genai_errors.ServerError as exc:
            raise ServerLLMError(str(exc)) from exc
        except TimeoutError as exc:
            raise TimeoutLLMError(str(exc)) from exc
        except Exception as exc:  # network errors, etc. — never surface raw internals
            message = str(exc)
            if 'timeout' in message.lower() or 'timed out' in message.lower():
                raise TimeoutLLMError(message) from exc
            raise ServerLLMError(message) from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = getattr(response, 'usage_metadata', None)
        tokens_in = getattr(usage, 'prompt_token_count', 0) or 0
        tokens_out = getattr(usage, 'candidates_token_count', 0) or 0
        text = response.text or ''

        candidates = getattr(response, 'candidates', None) or []
        finish_reason = getattr(candidates[0], 'finish_reason', None) if candidates else None
        if finish_reason == types.FinishReason.MAX_TOKENS:
            raise InvalidOutputLLMError('Reply truncated at max_output_tokens')

        structured = None
        if response_schema is not None:
            try:
                structured = json.loads(text)
            except (json.JSONDecodeError, TypeError) as exc:
                raise InvalidOutputLLMError(f'Model did not return valid JSON: {exc}') from exc

        return LLMResult(
            text=text, structured=structured, model=self.model, provider=self.name,
            tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms,
        )


class MultiKeyGeminiProvider(LLMProvider):
    """Wraps one GeminiProvider per configured key. On RATE_LIMITED (429 —
    daily quota exhausted, not a transient overload) it moves straight to
    the next key with no delay, since retrying the same exhausted key only
    burns ResilientProvider's attempt budget for no gain. Any other error
    (timeout, 5xx, bad output) is raised immediately so ResilientProvider's
    own backoff/retry handles it — rotating keys wouldn't fix those."""
    name = 'gemini'

    def __init__(self, model):
        self.providers = [GeminiProvider(model=model, api_key=key) for key in api_keys()]
        if not self.providers:
            raise RuntimeError('No Gemini API key configured')

    def generate(self, **kwargs) -> LLMResult:
        last_error = None
        for i, provider in enumerate(self.providers):
            try:
                return provider.generate(**kwargs)
            except RateLimitedLLMError as exc:
                last_error = exc
                if i < len(self.providers) - 1:
                    logger.info('Gemini key %d/%d rate-limited, trying next key',
                                i + 1, len(self.providers))
        raise last_error
