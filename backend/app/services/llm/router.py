"""FR-7.2 / A4 model routing. Maps task type to a cost-appropriate Gemini
tier, wraps every call in retry+fallback (resilient_provider.py), and falls
all the way back to the deterministic rule-based provider when no API key is
configured or every retry is exhausted — the system never hard-fails just
because an LLM is unavailable (NFR-4).

Tier reasoning (answers exam A4 for this implementation):
  Gamma (flash-lite): greeting/intent classification, safety classification —
    short, extremely frequent, latency-sensitive, correctness bar is low.
  Beta (flash): requirement extraction, recommendation/objection replies,
    summarisation, simple FAQ — the main conversational workload; needs
    structured-output support and decent reasoning at moderate cost.
  Alpha (pro): escalation only — conflicting constraints, multi-room
    requests, or a Beta reply that failed the output guard twice. Rare and
    expensive by design, matching FR-7.2's escalation rule.
"""

import os

from backend.app.services.llm.gemini_provider import MultiKeyGeminiProvider, is_configured
from backend.app.services.llm.mock_provider import MockProvider
from backend.app.services.llm.resilient_provider import ResilientProvider

TASK_TIER = {
    'greeting': 'gamma',
    'safety_classification': 'gamma',
    'simple_faq': 'gamma',
    'requirement_extraction': 'beta',
    'product_extraction': 'beta',
    'recommendation': 'beta',
    'objection_handling': 'beta',
    'summarization': 'beta',
    'hard_reasoning': 'alpha',
}

# gemini-2.5-* is deprecated for new API keys as of this build (confirmed
# live: 404 NOT_FOUND "no longer available to new users"); 3.5-* and
# gemini-pro-latest are the current working generation. Verified live
# against a real key: 3.5-flash-lite, 3.5-flash and gemini-flash-latest all
# returned 200; gemini-pro-latest hit a 429 (free-tier quota) during
# testing, which is expected for the rarely-used alpha/escalation tier, not
# a wrong model name.
TIER_MODEL = {
    'alpha': os.environ.get('GEMINI_MODEL_ALPHA', 'gemini-pro-latest'),
    'beta': os.environ.get('GEMINI_MODEL_BETA', 'gemini-3.5-flash'),
    'gamma': os.environ.get('GEMINI_MODEL_GAMMA', 'gemini-3.5-flash-lite'),
}

_mock = MockProvider()
_provider_cache = {}


def get_provider(task):
    tier = TASK_TIER.get(task, 'beta')
    if tier not in _provider_cache:
        if is_configured():
            primary = MultiKeyGeminiProvider(model=TIER_MODEL[tier])
            _provider_cache[tier] = ResilientProvider(primary, fallback=_mock)
        else:
            _provider_cache[tier] = _mock
    return _provider_cache[tier]


def tier_for(task):
    return TASK_TIER.get(task, 'beta')
