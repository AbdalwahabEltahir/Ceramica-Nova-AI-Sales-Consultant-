"""LLM-backed gap-filler for requirement extraction (FR-1.1's actual intent:
Gemini understands, deterministic code decides). The regex/lexicon extractor
in discovery.py runs first and is authoritative wherever it matches — a
literal substring match carries zero hallucination risk, so it's never
overridden. This module only fills fields regex left null, which is exactly
where regex structurally can't help: typos ("autdoor"), slang ("4 da
balcny"), or phrasing not in the hand-written trigger lists ("outside" vs.
the specific place-nouns regex looks for). Falls back to doing nothing
(returns {}) when GOOGLE_API_KEY is unset or the call/validation fails —
regex-only extraction is exactly today's behaviour either way.
"""

import json
import logging

from backend.app.services.llm.provider import LLMError
from backend.app.services.llm.router import get_provider

logger = logging.getLogger('llm_discovery')

from backend.app.models.catalog import FINISHES, STYLES

EXTRACTION_SYSTEM = f"""You are the listening layer of Ceramica Nova's tile sales consultant. Your only job is
to turn ONE customer message into structured notes so the consultant remembers what this
customer wants across the conversation. You never talk to the customer.

Read the message the way an experienced showroom consultant would: customers write with typos,
slang, Arabic or mixed language, and rarely use technical terms. Capture what they MEAN, but only
what they actually expressed.

Field rules:
- indoor_outdoor / floor_or_wall: map informal placement words ("outside", "autdoor", "patio",
  "garden path" = outdoor + floor; "backsplash", "behind the sink", "shower wall" = wall).
- area_m2: only when a size is stated in some form. Never estimate a room size yourself.
- preferred_style: one of {', '.join(STYLES)}. Map the customer's words to the closest one
  ("oak look" = wood, "like a hotel lobby" = luxury, "clean and simple" = minimal); null if no
  look/feel was expressed.
- preferred_finish: one of {', '.join(FINISHES)} ("shiny" = polished, "not slippery underfoot
  outside" = textured); null if not expressed.
- preferred_color: the colour or tone they mention, in plain English ("white", "warm beige",
  "dark grey"); null if none.
- slip_requirement: "high" when they mention children, elderly people, pets, wet areas, pools or
  "not slippery".
- traffic_level: "commercial_high" for shops, restaurants, offices, lobbies or heavy foot traffic.
- urgency: "immediate" for this week / asap, "soon" for this month or a near date.
- customer_notes: short third-person notes (max 6 words each) of anything else a good salesperson
  would remember and use later: who lives there, pets, the reason for renovating, things they
  dislike, a deadline, comparisons with other shops. Empty list if nothing new.

Never invent a value to fill a field; null is always a correct answer. Return strict JSON only."""

FILL_SCHEMA = {
    'type': 'object',
    'properties': {
        'indoor_outdoor': {'type': ['string', 'null'], 'enum': ['indoor', 'outdoor', None]},
        'floor_or_wall': {'type': ['string', 'null'], 'enum': ['floor', 'wall', None]},
        'area_m2': {'type': ['number', 'null']},
        'preferred_style': {'type': ['string', 'null']},
        'preferred_finish': {'type': ['string', 'null']},
        'preferred_color': {'type': ['string', 'null']},
        'slip_requirement': {'type': ['string', 'null'], 'enum': ['none', 'standard', 'high', None]},
        'traffic_level': {'type': ['string', 'null'], 'enum': ['residential', 'commercial_high', None]},
        'urgency': {'type': ['string', 'null'], 'enum': ['none', 'soon', 'immediate', None]},
        'customer_notes': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['indoor_outdoor', 'floor_or_wall', 'area_m2', 'preferred_style', 'preferred_finish',
                 'preferred_color', 'slip_requirement', 'traffic_level', 'urgency', 'customer_notes'],
}

VALID_VALUES = {
    'indoor_outdoor': {'indoor', 'outdoor'},
    'floor_or_wall': {'floor', 'wall'},
    'slip_requirement': {'none', 'standard', 'high'},
    'traffic_level': {'residential', 'commercial_high'},
    'urgency': {'none', 'soon', 'immediate'},
    'preferred_style': set(STYLES),
    'preferred_finish': set(FINISHES),
}


def fill_missing_fields(message, already_known):
    """already_known: the requirements dict so far (after regex extraction).
    Returns a dict of {field: value} for fields that were null and the LLM
    could fill — never a field already_known already has a value for."""
    provider = get_provider('requirement_extraction')
    try:
        result = provider.generate(
            system=EXTRACTION_SYSTEM,
            messages=[{'role': 'user', 'content': message}],
            response_schema=FILL_SCHEMA,
            temperature=0.0,
            max_tokens=250,
        )
    except LLMError as exc:
        logger.info('LLM requirement gap-fill failed, regex-only result stands: %s', exc)
        return {}

    if result.structured is None:
        return {}

    try:
        raw = result.structured if isinstance(result.structured, dict) else json.loads(result.text)
    except (json.JSONDecodeError, TypeError):
        return {}

    filled = {}
    notes = raw.get('customer_notes')
    if isinstance(notes, list):
        clean_notes = [str(n).strip()[:60] for n in notes if isinstance(n, str) and n.strip()]
        if clean_notes:
            filled['customer_notes'] = clean_notes[:3]

    for field in FILL_SCHEMA['properties']:
        if field == 'customer_notes':
            continue
        if already_known.get(field):  # regex already found something — keep it, zero-risk wins
            continue
        value = raw.get(field)
        if isinstance(value, str):
            value = value.strip().lower()
        if value is None:
            continue
        if field in VALID_VALUES and value not in VALID_VALUES[field]:
            continue  # model returned something outside the enum — drop it, don't guess
        if field == 'area_m2':
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not (0 < value <= 10000):
                continue
        filled[field] = value

    return filled
