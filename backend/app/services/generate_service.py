"""Backs POST /api/v1/generate (exam B1/B2). Calls the routed LLM for
structured extraction, validates the result against the exact B2 schema, and
never lets an invalid or missing model response reach the caller unvalidated.
"""

import logging
import time

from pydantic import ValidationError

from backend.app.models.schemas import GENERATE_RESPONSE_JSON_SCHEMA, GenerateResponse
from backend.app.services import metrics
from backend.app.services.discovery import extract_requirements
from backend.app.services.llm.provider import LLMError
from backend.app.services.llm.router import get_provider

logger = logging.getLogger('generate_service')

SYSTEM_PROMPT = (
    'You are the requirement-extraction component of the Ceramica Nova AI Sales '
    'Consultant. Read the customer message and return ONLY the fields that are '
    'actually stated or safely inferable. Do not invent an area, budget, or style '
    'the customer never mentioned — leave those fields empty/null instead. '
    'Respond with JSON matching the given schema exactly.'
)


def _rule_based_fallback(message, language):
    """Deterministic fallback when the LLM is unconfigured or every retry and
    the Gemini fallback both fail — mapped into the exact B2 shape."""
    requirements, _ = extract_requirements(message)
    return GenerateResponse(
        intent=requirements.get('intent', 'product_search'),
        customer_requirements={
            'application': requirements.get('application') or '',
            'area_m2': requirements.get('area_m2'),
            'budget': (requirements.get('budget') or {}).get('amount'),
            'style': requirements.get('preferred_style') or '',
            'color': requirements.get('preferred_color') or '',
            'urgency': requirements.get('urgency') or '',
        },
        missing_information=requirements.get('missing_information', []),
        recommended_actions=(
            ['ask_clarifying_questions'] if requirements.get('missing_information')
            else ['recommend_products']
        ),
        confidence=0.4,  # rule-based heuristic, deliberately lower than an LLM's
    )


def generate(user_id, task, message, language='en'):
    provider = get_provider(task)
    start = time.monotonic()

    messages = [{'role': 'user', 'content': message}]
    last_error = None

    for attempt in (1, 2):  # one retry with the validation error appended (FR-1.9 pattern)
        try:
            result = provider.generate(
                system=SYSTEM_PROMPT if attempt == 1 else
                SYSTEM_PROMPT + f'\n\nYour previous response was invalid: {last_error}. Fix it.',
                messages=messages,
                response_schema=GENERATE_RESPONSE_JSON_SCHEMA,
                temperature=0.0,
                max_tokens=512,
            )
        except LLMError as exc:
            logger.warning('generate() LLM call failed: %s', exc)
            metrics.increment('generate_llm_error')
            response = _rule_based_fallback(message, language)
            return response, {'model': 'mock-rule-based', 'attempts': attempt, 'validation_retried': attempt > 1,
                               'latency_ms': (time.monotonic() - start) * 1000, 'fallback_reason': str(exc)}

        if result.structured is None:
            last_error = 'no structured output returned'
            continue
        try:
            response = GenerateResponse.model_validate(result.structured)
            metrics.increment('generate_llm_success')
            return response, {'model': result.model, 'attempts': attempt,
                               'validation_retried': attempt > 1,
                               'latency_ms': (time.monotonic() - start) * 1000,
                               'tokens_in': result.tokens_in, 'tokens_out': result.tokens_out}
        except ValidationError as exc:
            last_error = str(exc)
            metrics.increment('generate_invalid_output')
            logger.warning('generate() schema validation failed (attempt %d): %s', attempt, exc)
            continue

    # Both attempts produced invalid JSON/schema — fall back rather than
    # letting an unvalidated shape reach the caller (B2's core requirement).
    metrics.increment('generate_fallback_after_invalid')
    response = _rule_based_fallback(message, language)
    return response, {'model': 'mock-rule-based', 'attempts': 2, 'validation_retried': True,
                       'latency_ms': (time.monotonic() - start) * 1000, 'fallback_reason': last_error}
