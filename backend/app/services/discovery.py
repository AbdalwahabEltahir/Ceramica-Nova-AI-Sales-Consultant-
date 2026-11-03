"""FR-1 requirement discovery. Rule-based extractor: every field it sets must be
traceable to the customer's own words (the FR-1.2 evidence check), since there is
no LLM key configured in this environment — see services/llm/provider.py for the
pluggable interface this would plug Claude Haiku 5.5 into.
"""

import re

from backend.app.config import MAX_CLARIFYING_QUESTIONS, SQFT_TO_M2

ARABIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

ARABIC_LEXICON = {
    'شرفة': ('balcony', 'outdoor'),
    'تراس': ('terrace', 'outdoor'),
    'مسبح': ('pool_deck', 'outdoor'),
    'حمام': ('bathroom', 'indoor'),
    'مطبخ': ('kitchen', 'indoor'),
    'غرفة المعيشة': ('living_room', 'indoor'),
    'رخام': ('marble', None),
    'مقاوم للانزلاق': ('slip_high', None),
    'زلق': ('slip_high', None),
}


def _is_arabic(text):
    return any('؀' <= ch <= 'ۿ' for ch in text)


def _normalise(message):
    text = message.translate(ARABIC_DIGITS)
    text = text.replace('متر مربع', 'm2').replace('م²', 'm2').replace('م2', 'm2')
    return ' '.join(text.strip().split())


def _evidence_span(pattern, text):
    match = re.search(pattern, text, re.I)
    return match.group(0) if match else None


def _parse_area(text):
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m2|m²|sqm|square\s*meters?)', text, re.I)
    if match:
        return float(match.group(1)), match.group(0), 'm2'
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:sq\s*ft|sqft|square\s*feet)', text, re.I)
    if match:
        value_ft = float(match.group(1))
        return round(value_ft * SQFT_TO_M2, 2), match.group(0), 'sqft'
    return None, None, None


# Matches a message that is *only* a number plus hedging filler — "20",
# "idk maybe 20", "around 20" — and nothing else. Deliberately strict (full
# match, not search) so a number mentioned in passing elsewhere in a longer
# sentence is never misread as an area answer; this only fires for a
# customer giving a bare, unit-less reply to "how many m²?", which a
# non-technical customer does constantly.
_BARE_NUMBER_AREA_RE = re.compile(
    r'^\s*(?:idk,?\s*|i\s*don.?t\s*know,?\s*|maybe\s*|about\s*|around\s*|roughly\s*|'
    r'approximately\s*|probably\s*|its\s*|it.?s\s*)*(\d+(?:\.\d+)?)\s*$', re.I)


def _parse_bare_number_area(text):
    match = _BARE_NUMBER_AREA_RE.match(text.strip())
    if match:
        return float(match.group(1))
    return None


_PER_M2_BUDGET_RE = re.compile(
    r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:/|per|a|an)\s*(?:m2|m²|sqm|square\s*met(?:er|re)s?|meter|metre)', re.I)


def _parse_budget(text):
    match = _PER_M2_BUDGET_RE.search(text)
    if match:
        value = float(match.group(1).replace(',', ''))
        return {'amount': value, 'currency': 'USD', 'basis': 'per_m2'}, match.group(0)
    match = re.search(r'\$\s*([\d,]+(?:\.\d+)?)', text)
    if match:
        value = float(match.group(1).replace(',', ''))
        return {'amount': value, 'currency': 'USD', 'basis': 'total'}, match.group(0)
    return None, None


# Customer wording -> catalogue style value (see models/catalog.STYLES).
STYLE_SYNONYMS = {
    'marble': 'marble', 'calacatta': 'marble',
    'wood': 'wood', 'wooden': 'wood', 'oak': 'wood', 'timber': 'wood', 'parquet': 'wood',
    'stone': 'stone', 'slate': 'slate', 'rustic': 'rustic', 'terracotta': 'rustic',
    'industrial': 'industrial', 'concrete': 'contemporary', 'cement': 'contemporary',
    'luxury': 'luxury', 'luxurious': 'luxury', 'premium': 'luxury',
    'minimal': 'minimal', 'minimalist': 'minimal', 'contemporary': 'contemporary',
    'modern': 'modern', 'subway': 'subway', 'metro': 'subway',
    'natural': 'natural', 'decorative': 'decorative', 'mosaic': 'decorative', 'pattern': 'decorative',
}
FINISH_SYNONYMS = {
    'polished': 'polished', 'shiny': 'polished', 'glossy': 'gloss', 'gloss': 'gloss',
    'matt': 'matt', 'matte': 'matt', 'textured': 'textured', 'rough': 'textured', 'grippy': 'textured',
}
COLOR_WORDS = [
    'white', 'black', 'grey', 'gray', 'beige', 'cream', 'ivory', 'brown', 'blue', 'green',
    'sand', 'terracotta', 'graphite', 'charcoal', 'pearl', 'aqua', 'light', 'dark', 'neutral', 'warm',
]


def _first_word_match(lower, mapping):
    for word, value in mapping.items():
        if re.search(rf'\b{re.escape(word)}\b', lower):
            return word, value
    return None, None


def extract_requirements(message, previous=None, use_llm=True):
    """Returns (requirements: dict, conversion_note: str|None)."""
    previous = previous or {}
    raw = message or ''
    language = 'ar' if _is_arabic(raw) else 'en'
    text = _normalise(raw)
    lower = text.lower()

    evidence = dict(previous.get('evidence', {}))
    requirements = {
        'intent': 'product_search',
        'language': language,
        'application': previous.get('application'),
        'indoor_outdoor': previous.get('indoor_outdoor'),
        'floor_or_wall': previous.get('floor_or_wall'),
        'area_m2': previous.get('area_m2'),
        'budget': previous.get('budget'),
        'preferred_style': previous.get('preferred_style'),
        'preferred_color': previous.get('preferred_color'),
        'preferred_finish': previous.get('preferred_finish'),
        'customer_notes': list(previous.get('customer_notes', [])),
        'slip_requirement': previous.get('slip_requirement'),
        'traffic_level': previous.get('traffic_level'),
        'urgency': previous.get('urgency'),
        'signals': list(previous.get('signals', [])),
        'missing_information': [],
        'clarifying_questions': [],
        'evidence': evidence,
    }

    conversion_note = None
    area_value, area_span, area_unit = _parse_area(lower)
    if area_value is None and requirements['area_m2'] is None:
        # A non-technical customer answering "how many m²?" with just "20"
        # or "idk maybe 20" is common — don't silently drop it just because
        # they left the unit off (the app only ever asks for m² anyway).
        bare = _parse_bare_number_area(text)
        if bare is not None:
            area_value, area_span, area_unit = bare, text.strip(), 'bare_assumed_m2'
    if area_value is not None:
        requirements['area_m2'] = area_value
        evidence['area_m2'] = area_span
        if area_unit == 'sqft':
            conversion_note = f'Converted {area_span} to {area_value} m² (×0.092903).'

    budget, budget_span = _parse_budget(text)
    if budget:
        requirements['budget'] = budget
        evidence['budget'] = budget_span

    # `application` records the space the customer actually named; a bare
    # "indoor"/"outdoor" sets placement only, never an invented room.
    outdoor_words = ['balcony', 'terrace', 'patio', 'pool deck', 'garden', 'driveway', 'outdoor']
    indoor_words = ['kitchen', 'bathroom', 'living room', 'bedroom', 'restaurant', 'office',
                    'hallway', 'shop', 'indoor']
    for word in outdoor_words:
        if word in lower:
            if word != 'outdoor':
                requirements['application'] = word
            requirements['indoor_outdoor'] = 'outdoor'
            requirements['floor_or_wall'] = requirements['floor_or_wall'] or 'floor'
            evidence['indoor_outdoor'] = word
            break
    else:
        for word in indoor_words:
            if word in lower:
                if word != 'indoor':
                    requirements['application'] = word
                requirements['indoor_outdoor'] = 'indoor'
                evidence['indoor_outdoor'] = word
                break

    if any(w in lower for w in ['wall', 'backsplash']):
        requirements['floor_or_wall'] = 'wall'
        evidence['floor_or_wall'] = 'wall'
    elif any(w in lower for w in ['floor', 'flooring']):
        requirements['floor_or_wall'] = 'floor'
        evidence['floor_or_wall'] = 'floor'

    if any(w in lower for w in ['children', 'kids', 'elderly', 'slippery', 'wet', 'anti slip', 'anti-slip', 'not slippery']):
        requirements['slip_requirement'] = 'high'
        evidence['slip_requirement'] = 'children/slippery'

    if any(w in lower for w in ['restaurant', 'shop', 'office', 'high traffic', 'lots of traffic', 'heavy traffic']):
        requirements['traffic_level'] = 'commercial_high'
        evidence['traffic_level'] = 'commercial wording'
    elif requirements['traffic_level'] is None:
        requirements['traffic_level'] = 'residential'

    word, style = _first_word_match(lower, STYLE_SYNONYMS)
    if style:
        requirements['preferred_style'] = style
        evidence['preferred_style'] = word

    word, finish = _first_word_match(lower, FINISH_SYNONYMS)
    if finish:
        requirements['preferred_finish'] = finish
        evidence['preferred_finish'] = word

    for color in COLOR_WORDS:
        if re.search(rf'\b{color}\b', lower):
            requirements['preferred_color'] = 'grey' if color == 'gray' else color
            evidence['preferred_color'] = color
            break

    if any(w in lower for w in ['cheap', 'budget', 'affordable', 'low cost']):
        if 'price_sensitivity' not in requirements['signals']:
            requirements['signals'].append('price_sensitivity')
    if any(w in lower for w in ['today', 'this week', 'urgent', 'asap']):
        requirements['urgency'] = 'immediate'
        if 'urgency' not in requirements['signals']:
            requirements['signals'].append('urgency')
    if any(w in lower for w in ["how much in total", "i'll take it", 'quote', 'purchase']):
        requirements['intent'] = 'quote_request'
        if 'purchase_readiness' not in requirements['signals']:
            requirements['signals'].append('purchase_readiness')

    from backend.app.services import hf_nlp
    if hf_nlp.is_configured():
        sentiment_label, sentiment_score = hf_nlp.analyze_sentiment(text)
        if (sentiment_label == 'negative' and sentiment_score >= hf_nlp.NEGATIVE_SENTIMENT_THRESHOLD
                and 'negative_sentiment' not in requirements['signals']):
            # Ranking/tone input only (E.14/E.2) — never a hard gate, and it
            # never changes a hard constraint. See composer/llm_composer for
            # how this softens upselling language, same handling as the
            # financial-distress rule in section 11.4 of the SRS.
            requirements['signals'].append('negative_sentiment')

    # Arabic lexicon fallback for key fields.
    for term, (value, placement) in ARABIC_LEXICON.items():
        if term in raw:
            if placement == 'outdoor':
                requirements['indoor_outdoor'] = 'outdoor'
                requirements['application'] = requirements['application'] or value
            elif placement == 'indoor':
                requirements['indoor_outdoor'] = 'indoor'
                requirements['application'] = requirements['application'] or value
            elif value == 'slip_high':
                requirements['slip_requirement'] = 'high'
            elif value == 'marble':
                requirements['preferred_style'] = 'marble'

    if requirements['floor_or_wall'] is None and requirements['indoor_outdoor'] == 'outdoor':
        requirements['floor_or_wall'] = 'floor'

    # LLM gap-fill (FR-1.1): regex above is authoritative wherever it hit —
    # zero hallucination risk, never overridden. This only fills fields that
    # are *still* null, which is exactly where regex structurally can't
    # help: typos, slang, or phrasing outside the hand-written trigger
    # lists. No-op (returns {}) when GOOGLE_API_KEY is unset or the call
    # fails — identical to today's regex-only behaviour either way.
    if use_llm:
        from backend.app.services.llm_discovery import fill_missing_fields
        filled = fill_missing_fields(raw, requirements)
        notes = filled.pop('customer_notes', None) or []
        for note in notes:
            if note not in requirements['customer_notes']:
                requirements['customer_notes'].append(note)
        requirements['customer_notes'] = requirements['customer_notes'][-8:]
        for field, value in filled.items():
            requirements[field] = value
            evidence[field] = 'llm-inferred'
        if filled:
            requirements['llm_filled_fields'] = sorted(filled.keys())

    if requirements['floor_or_wall'] is None and requirements['indoor_outdoor'] == 'outdoor':
        requirements['floor_or_wall'] = 'floor'

    requirements['evidence'] = evidence
    return requirements, conversion_note


def essential_fields_for_stage(stage):
    if stage == 'discovery':
        return ['indoor_outdoor', 'floor_or_wall']
    if stage == 'quantity':
        return ['area_m2']
    if stage == 'quote':
        return []
    return []


QUESTION_TEXT = {
    'indoor_outdoor': {
        'en': 'Is this an indoor or outdoor installation?',
        'ar': 'هل هذا تركيب داخلي أم خارجي؟',
    },
    'floor_or_wall': {
        'en': 'Is this for the floor or the wall?',
        'ar': 'هل هذا للأرضية أم للحائط؟',
    },
    'area_m2': {
        'en': 'How many square meters do you need to cover?',
        'ar': 'كم متراً مربعاً تحتاج لتغطيته؟',
    },
}


def missing_essentials(requirements, stage):
    missing = [f for f in essential_fields_for_stage(stage) if not requirements.get(f)]
    questions = [QUESTION_TEXT[f][requirements.get('language', 'en') if requirements.get('language') in ('en', 'ar') else 'en']
                 for f in missing[:MAX_CLARIFYING_QUESTIONS]]
    return missing, questions


TILE_VOCAB = [
    'tile', 'tiles', 'ceramic', 'porcelain', 'floor', 'wall', 'balcony', 'terrace',
    'patio', 'pool', 'outdoor', 'indoor', 'kitchen', 'bathroom', 'living room',
    'bedroom', 'restaurant', 'office', 'retail', 'm2', 'm²', 'sqm', 'sq ft', 'square',
    'budget', '$', 'marble', 'stone', 'wood', 'slate', 'modern', 'slip', 'slippery',
    'traffic', 'commercial', 'quote', 'price', 'box', 'boxes', 'stock', 'available',
    'discount', 'quantity', 'cn0', 'option', 'product', 'collection', 'range', 'catalog',
    'catalogue', 'colour', 'color', 'finish', 'style', 'design', 'size', 'grout',
    'install', 'renovat', 'backsplash', 'shower', 'garden', 'stairs', 'hallway', 'entrance',
] + list(STYLE_SYNONYMS) + list(FINISH_SYNONYMS) + [
    c for c in COLOR_WORDS if c not in ('light', 'dark', 'warm', 'neutral')]
ARABIC_TILE_HINTS = ['بلاط', 'سيراميك', 'شرفة', 'تراس', 'مسبح', 'حمام', 'مطبخ', 'رخام', 'متر', 'عرض سعر']


def _contains_tile_vocab(text):
    lower = text.lower()
    return any(word in lower for word in TILE_VOCAB)


def is_off_topic(message):
    """True when a message contains none of the tile-domain vocabulary at all
    — distinguishes 'what's the capital of France?' from a short clarifying
    answer like 'floor' or '25 m2', which do contain domain vocabulary."""
    if not message or not message.strip():
        return False
    if _contains_tile_vocab(message):
        return False
    if any(hint in message for hint in ARABIC_TILE_HINTS):
        return False
    if re.search(r'\d', message):  # a bare number is likely answering "how many m2?"
        return False

    # The hand-written Arabic hint list above is short; before declaring an
    # Arabic message off-topic, translate it and recheck the (much larger)
    # English vocabulary — catches real tile questions phrased with words
    # not in that short list (e.g. "سعر" / price). No-op when HF is
    # unconfigured or the call fails — same off-topic verdict as before.
    if _is_arabic(message):
        from backend.app.services import hf_nlp
        if hf_nlp.is_configured():
            translated = hf_nlp.translate_arabic_to_english(message)
            if translated and _contains_tile_vocab(translated):
                return False

    return True


# Words/pronouns a customer uses when asking about products just shown to
# them, not a fresh topic — "are they safe?", "is it durable?", "which one's
# best?". Only meaningful (and only checked by the orchestrator) when the
# session actually has recommendations to refer back to; outside that
# context "is it safe" alone is too generic to classify on its own.
FOLLOWUP_HINTS = [
    'safe', 'safety', 'durable', 'durability', 'quality', 'last', 'lasting',
    'scratch', 'resist', 'kid', 'child', 'pet', 'warranty', 'guarantee',
    'good', 'best', 'recommend', 'worth', 'compare', 'better', 'they', 'them',
    'it', 'these', 'those', 'which one', 'this one',
]


def _has_phrase(lower, phrases):
    return any(re.search(rf'\b{re.escape(p)}\b', lower) for p in phrases)


def is_followup_question(message):
    # Word-boundary match: a plain substring check let 'it' match "white" or
    # "with", which hijacked preference answers into the follow-up path.
    if not message:
        return False
    return _has_phrase(message.lower(), FOLLOWUP_HINTS)


MORE_OPTIONS_HINTS = [
    'other option', 'other options', 'more options', 'different options', 'another option',
    'other choices', 'more choices', 'alternatives', 'alternative', 'something else',
    'what else', 'anything else', 'other ones', 'different ones', 'show me more', 'any others',
    'خيارات أخرى', 'بدائل', 'غير ذلك',
]

CATALOG_HINTS = [
    'what are the products', 'what products', 'which products', 'your products', 'all products',
    'the products', 'what do you have', 'what do you sell', 'what do you offer', 'what tiles',
    'which tiles', 'your tiles', 'your range', 'the range', 'collection', 'catalog', 'catalogue',
    'what styles', 'which styles', 'what colors', 'what colours', 'available colors',
    'available colours', 'which colors', 'which colours', 'what finishes', 'what sizes',
    'what are the options', 'what options', 'difference between', 'explain the',
    'ما هي المنتجات', 'المنتجات', 'الألوان', 'الأنواع',
]


_DISCOUNT_RE = re.compile(r'\bdiscount|\d+(?:\.\d+)?\s*(?:%|percent)\s*off\b|\bpercent off\b|\bprice match|خصم', re.I)
_PERCENT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', re.I)
QUOTE_HINTS = ['quote', 'quotation', "i'll take", 'i will take', 'buy it', 'place an order', 'order it',
               'purchase', 'عرض سعر']


def parse_discount_request(message):
    """(asks_for_discount, percent or None). "15% less" alone is a price
    objection, not a discount request, so only discount/off wording counts."""
    if not message or not _DISCOUNT_RE.search(message):
        return False, None
    match = _PERCENT_RE.search(message)
    return True, float(match.group(1)) if match else None


def is_quote_request(message):
    return bool(message) and (parse_discount_request(message)[0] or _has_phrase(message.lower(), QUOTE_HINTS))


def mentions_area(message):
    return bool(message) and (_parse_area(message.lower())[0] is not None
                              or _parse_bare_number_area(message) is not None)


def is_more_options_request(message):
    return bool(message) and _has_phrase(message.lower(), MORE_OPTIONS_HINTS)


def is_catalog_question(message):
    """'What are the products?', 'what colours do you have?' — the customer
    wants the range explained, not a ranked shortlist or a spec lookup."""
    return bool(message) and _has_phrase(message.lower(), CATALOG_HINTS)


def has_preference_signal(message):
    """True when the message states something worth storing (style, finish,
    colour, budget, area, placement) — such a message goes through
    requirement discovery even if it also looks like a follow-up."""
    if not message:
        return False
    lower = message.lower()
    if _first_word_match(lower, STYLE_SYNONYMS)[1] or _first_word_match(lower, FINISH_SYNONYMS)[1]:
        return True
    if _has_phrase(lower, COLOR_WORDS) or '$' in lower:
        return True
    if _parse_area(lower)[0] is not None:
        return True
    return _has_phrase(lower, ['floor', 'wall', 'indoor', 'outdoor', 'kitchen', 'bathroom',
                               'balcony', 'terrace', 'patio', 'living room', 'bedroom', 'restaurant'])


def suitable_products(requirements):
    """Catalogue products that pass every hard constraint for what is known so far."""
    from backend.app.models.catalog import get_catalog
    from backend.app.models.rules import apply_hard_constraints
    return [p for p in get_catalog() if apply_hard_constraints(p, requirements)[0]]


def next_preference_question(requirements):
    """The single most useful thing to ask next, phrased with options that
    actually exist for this customer's placement — so the question is
    grounded in the catalogue, not a generic 'anything else?'."""
    language = requirements.get('language', 'en')
    products = suitable_products(requirements)
    if not products:
        return None

    if not requirements.get('preferred_style'):
        styles = sorted({p['style'] for p in products})
        if len(styles) > 1:
            listed = ', '.join(styles[:-1]) + f' or {styles[-1]}'
            return ('What look do you have in mind — ' + listed + '?' if language != 'ar'
                    else 'ما الطابع الذي تفضله — ' + '، '.join(styles) + '؟')
    if not requirements.get('area_m2'):
        return ('Roughly how many square meters is the space? I can then work out exact boxes and pricing.'
                if language != 'ar' else 'كم تبلغ المساحة تقريباً بالمتر المربع؟ سأحسب لك عدد الصناديق والسعر بدقة.')
    if not requirements.get('preferred_finish'):
        finishes = sorted({p['finish'] for p in products})
        if len(finishes) > 1:
            return (f"Do you prefer a {' or '.join(finishes)} finish?" if language != 'ar'
                    else 'هل تفضل لمسة نهائية ' + ' أم '.join(finishes) + '؟')
    if not requirements.get('budget'):
        return ('Do you have a budget in mind, either in total or per m²?' if language != 'ar'
                else 'هل لديك ميزانية محددة، إجمالية أو لكل متر مربع؟')
    return None


PRODUCT_QUESTION_HINTS = [
    'frost', 'rating', 'care', 'clean', 'maintenance', 'warranty', 'dimension',
    'thickness', 'water absorption', 'what is the', 'tell me about', 'does it have',
    'is it rated', 'مقاس', 'عناية', 'ضمان',
]

# Narrower than PRODUCT_QUESTION_HINTS: specific facts, not generic phrasing
# like "rating" (which matches "slip rating" on every product doc and would
# make the groundedness check below always pass). Used to verify a retrieved
# document actually contains what was asked, not just something about the
# same product.
SPEC_FACT_HINTS = [
    'frost', 'care', 'clean', 'maintenance', 'warranty', 'dimension',
    'thickness', 'water absorption', 'مقاس', 'عناية', 'ضمان',
]


def is_product_question(message):
    """A spec/FAQ-style question about a product, routed to RAG instead of
    the requirement-discovery flow (exam E.16 'simple FAQ' / hallucination
    trap category — e.g. asking for a frost rating the catalogue doesn't have)."""
    if not message:
        return False
    lower = message.lower()
    has_hint = any(h in lower for h in PRODUCT_QUESTION_HINTS)
    # "What is the weather today?" has the same shape as a spec question —
    # it only counts when the message is actually about tiles.
    return (has_hint and ('?' in message or 'tell me' in lower or lower.startswith('what'))
            and not is_off_topic(message))
