"""FR-5 orchestration. One bounded loop: guard, understand, decide, call tools,
compose, guard again (see architecture diagram in the SRS, section 3).

Routing order after the input guard:
  1. product spec question        -> RAG answer
  2. "other options"              -> next-best suitable products (or the range, if none shown yet)
  3. "what are the products?"     -> the suitable range explained, ordered by RAG relevance
  4. follow-up on shown products  -> grounded answer about those SKUs
  5. off-topic                    -> short redirect back to the project
  6. everything else              -> requirement discovery + recommendations
Preferences stated anywhere in the message are stored on the session first,
so no path loses what the customer told us.
"""

from backend.app.models.catalog import find_named_product, get_product_by_sku
from backend.app.models.rules import apply_hard_constraints
from backend.app.models.session_store import session_store
from backend.app.services import composer, llm_composer
from backend.app.services.discovery import (
    _is_arabic, extract_requirements, has_preference_signal, is_catalog_question,
    is_followup_question, is_more_options_request, is_off_topic, is_product_question,
    is_quote_request, mentions_area, missing_essentials, next_preference_question,
    parse_discount_request, suitable_products,
)
from backend.app.services.guardrails import input_guard, output_guard
from backend.app.services.recommender import recommend_products
from backend.app.services.tools.calculator import calculate_tile_quantity
from backend.app.services.tools.inventory import check_inventory
from backend.app.services.tools.quotation import generate_quote


def _next_question(requirements):
    """Missing essentials first (BR-10), then the most useful preference."""
    _, questions = missing_essentials(requirements, 'discovery')
    if questions:
        return questions[0]
    return next_preference_question(requirements)


def _with_question(reply, question):
    return f'{reply}\n{question}' if question else reply


def _guarded(reply, recommendations, tool_results, requirements, quote=None):
    result = output_guard.check_reply(reply, recommendations, tool_results, quote, requirements)
    if result['blocked']:
        return (output_guard.template_fallback(requirements.get('language', 'en')),
                {'triggered': True, 'rule': 'FR-4.7', 'violations': result['violations']})
    return reply, {'triggered': False, 'rule': None}


def _finish(record, message, reply, guardrail, recommendations=None, questions=None,
            missing=None, quote=None, citations=None):
    record.history.append({'role': 'user', 'text': message})
    record.history.append({'role': 'assistant', 'text': reply})
    session_store.save(record)
    payload = {
        'session_id': record.session_id,
        'reply': reply,
        'requirements': record.requirements,
        'recommendations': recommendations or [],
        'missing_information': missing or [],
        'clarifying_questions': questions or [],
        'quote': quote,
        'stage': record.stage,
        'guardrail': guardrail,
    }
    if citations is not None:
        payload['citations'] = citations
    return payload


def _rag_ordered(products, question, requirements):
    """Orders products by retrieval relevance (LanceDB vectors + BM25) to the
    customer's question and stored preferences; unranked ones keep catalogue order."""
    from rag.retriever import retrieve
    query = ' '.join(x for x in (question, requirements.get('preferred_style'),
                                 requirements.get('preferred_finish'), requirements.get('application')) if x)
    try:
        ranked = [r['metadata'].get('sku') for r in retrieve(query, top_k=25)]
    except Exception:
        return products
    position = {sku: i for i, sku in enumerate(ranked) if sku}
    return sorted(products, key=lambda p: position.get(p['sku'], len(position)))


def _explain_catalog(record, message, question, requirements):
    products = _rag_ordered(suitable_products(requirements), question, requirements)
    next_q = _next_question(requirements)
    language = requirements.get('language', 'en')
    reply = llm_composer.explain_catalog(question, products, requirements, next_q)
    if not reply:
        asked_colour = any(w in question.lower() for w in ('colour', 'color', 'لون', 'ألوان'))
        reply = _with_question(composer.compose_catalog_overview(products, language, asked_colour), next_q)
    tool_results = [{'tool_name': 'retrieve', 'result': {'sku': p['sku'], 'price_per_m2': p['price_per_m2']}}
                    for p in products]
    reply, guardrail = _guarded(reply, [], tool_results, requirements)
    return _finish(record, message, reply, guardrail, citations=[p['sku'] for p in products])


def _more_options(record, message, requirements):
    shown = record.shown_skus or list(record.last_recommendations)
    recommendations, _, explanation = recommend_products(requirements, exclude_skus=shown)
    language = requirements.get('language', 'en')
    next_q = _next_question(requirements)
    if not recommendations:
        reply = _with_question(composer.compose_no_more_options(language), next_q)
        reply, guardrail = _guarded(reply, [], [], requirements)
        return _finish(record, message, reply, guardrail)

    skus = [r['sku'] for r in recommendations]
    record.last_recommendations = skus
    record.shown_skus = shown + skus
    reply = (llm_composer.compose_recommendation_reply(requirements, recommendations, explanation,
                                                       next_q, more_options=True)
             or _with_question(composer.compose_recommendations(requirements, recommendations,
                                                                explanation, more_options=True), next_q))
    tool_results = [{'tool_name': 'calculate_tile_quantity', 'result': r} for r in recommendations]
    reply, guardrail = _guarded(reply, recommendations, tool_results, requirements)
    return _finish(record, message, reply, guardrail, recommendations=recommendations)


def _quote(record, message, normalised, named_product):
    """Discount/quote requests (E.10, BR-07). The quote and every figure in it
    come from generate_quote; a discount above 12% is created as
    PENDING_APPROVAL and the reply never claims it is approved. Missing
    details (which product, how many m²) are asked for and remembered."""
    asks_discount, percent = parse_discount_request(normalised)
    pending = record.pending_quote or {}
    asks_discount = asks_discount or pending.get('asks_discount', False)
    if percent is None:
        percent = pending.get('discount_percent')
    if has_preference_signal(normalised) or mentions_area(normalised):
        record.requirements, _ = extract_requirements(normalised, previous=record.requirements, use_llm=False)
    requirements = record.requirements
    language = 'ar' if _is_arabic(message) else requirements.get('language', 'en')

    sku = (named_product or {}).get('sku') or pending.get('sku')
    if not sku and len(record.last_recommendations) == 1:
        sku = record.last_recommendations[0]
    record.pending_quote = {'sku': sku, 'discount_percent': percent, 'asks_discount': asks_discount}
    product = get_product_by_sku(sku) if sku else None
    tool_results, quote = [], None

    if not product:
        options = [get_product_by_sku(s) for s in record.last_recommendations]
        reply = composer.compose_quote_which_product(options, language)
        tool_results = [{'tool_name': 'catalog', 'result': {'sku': p['sku']}} for p in options]
    elif requirements.get('indoor_outdoor') and requirements.get('floor_or_wall') and \
            not apply_hard_constraints(product, requirements)[0]:
        _, violations = apply_hard_constraints(product, requirements)
        record.pending_quote = None
        reply = composer.compose_unsuitable_product(product, violations, language, with_alternatives=False)
        tool_results = [{'tool_name': 'apply_hard_constraints', 'result': {'sku': product['sku']}}]
    elif not requirements.get('area_m2'):
        reply = composer.compose_quote_needs_area(product, language)
        tool_results = [{'tool_name': 'catalog', 'result': {'sku': product['sku']}}]
    else:
        try:
            quote = generate_quote([{'sku': product['sku'], 'area_m2': requirements['area_m2']}],
                                   requested_discount_percent=percent or 0, owner_id=record.session_id)
        except ValueError:
            reply = composer.compose_invalid_discount(language)
            record.pending_quote['discount_percent'] = None
        else:
            record.pending_quote = None
            record.stage = 'quote'
            reply = composer.compose_quote_reply(quote, language, invite_discount=asks_discount and percent is None)
            tool_results = [{'tool_name': 'generate_quote', 'result': {'sku': product['sku']}}]

    reply, guardrail = _guarded(reply, [], tool_results, requirements, quote)
    return _finish(record, message, reply, guardrail, quote=quote)


def handle_turn(session_id, message):
    record = session_store.get_or_create(session_id)
    language_hint = record.requirements.get('language', 'en')
    named_product = find_named_product(message)

    # G1 input guard.
    scan = input_guard.scan_input(message)
    if scan['flagged']:
        reply_parts = [input_guard.safe_reply(language_hint)]
        if named_product:
            # FR-4.3: still help with the legitimate part — the real price.
            reply_parts.append(composer.compose_named_product_fact(named_product, language_hint))
        reply = '\n'.join(reply_parts)
        record.history.append({'role': 'user', 'text': message, 'blocked': True})
        record.history.append({'role': 'assistant', 'text': reply})
        session_store.save(record)
        return {
            'session_id': record.session_id,
            'reply': reply,
            'requirements': record.requirements,
            'recommendations': [],
            'missing_information': [],
            'clarifying_questions': [],
            'quote': None,
            'guardrail': {'triggered': True, 'rule': scan['rule'], 'category': scan['category']},
        }

    normalised = scan['normalised']

    # A pending quote is completed by the detail we asked for ("Terra Rustic", "25 m2").
    if is_quote_request(normalised) or (record.pending_quote and (named_product or mentions_area(normalised))):
        return _quote(record, message, normalised, named_product)

    side_path = (is_product_question(normalised) or is_more_options_request(normalised)
                 or is_catalog_question(normalised))

    # Keep whatever preference the customer states, even when the message is
    # answered by a side path (e.g. "what marble tiles do you have?").
    if side_path and has_preference_signal(normalised):
        updated, _ = extract_requirements(normalised, previous=record.requirements, use_llm=False)
        record.requirements = updated
        record.shown_skus = []
    requirements = dict(record.requirements)
    requirements['language'] = 'ar' if _is_arabic(message) else requirements.get('language', 'en')

    if is_product_question(normalised):
        answer, citations = llm_composer.answer_product_question(
            normalised, requirements, _next_question(requirements))
        rag_tool_results = [{'tool_name': 'retrieve', 'result': {'sku': c}}
                            for c in citations if c.upper().startswith('CN')]
        answer, guardrail = _guarded(answer, [], rag_tool_results, requirements)
        return _finish(record, message, answer, guardrail, citations=citations)

    if is_more_options_request(normalised):
        if record.last_recommendations:
            return _more_options(record, message, requirements)
        return _explain_catalog(record, message, normalised, requirements)

    if is_catalog_question(normalised):
        return _explain_catalog(record, message, normalised, requirements)

    # A short follow-up about products just shown ("are they safe?") — resolve
    # "they"/"it" via the last recommendations. A message that states a new
    # preference ("I'd like something white") goes to discovery instead so
    # the preference is stored and the shortlist re-ranked.
    if (record.last_recommendations and is_followup_question(normalised)
            and not has_preference_signal(normalised)):
        answer, citations = llm_composer.answer_followup_question(
            normalised, record.last_recommendations, requirements, next_preference_question(requirements))
        rag_tool_results = [{'tool_name': 'retrieve', 'result': {'sku': c}} for c in citations]
        answer, guardrail = _guarded(answer, [], rag_tool_results, requirements)
        return _finish(record, message, answer, guardrail, citations=citations)

    # Off-topic messages get one short redirect instead of being forced
    # through the clarifying-question flow (BR-10 is for tile requirements).
    if is_off_topic(normalised):
        _, pending_questions = missing_essentials(
            record.requirements, record.stage if record.stage != 'quote' else 'discovery')
        next_q = None if pending_questions else next_preference_question(requirements)
        reply = llm_composer.redirect_off_topic(normalised, requirements, next_q)
        if pending_questions:
            reply = f'{reply}\n{composer.compose_question_lines(pending_questions)}'
        return _finish(record, message, reply, {'triggered': False, 'rule': None},
                       questions=pending_questions)

    # Requirement discovery (FR-1), merged into the session.
    requirements, conversion_note = extract_requirements(normalised, previous=record.requirements)
    if any(requirements.get(f) != record.requirements.get(f) for f in (
            'indoor_outdoor', 'floor_or_wall', 'preferred_style', 'preferred_finish', 'budget')):
        record.shown_skus = []
    record.requirements = requirements
    record.history.append({'role': 'user', 'text': message})

    stage = record.stage
    missing, questions = missing_essentials(requirements, stage if stage != 'quote' else 'discovery')

    tool_results = []
    recommendations = []
    quote = None
    explanation = None
    unsuitable_note = None

    if missing:
        reply = (llm_composer.compose_clarifying_reply(requirements, questions, conversion_note)
                 or composer.compose_clarifying(requirements, questions, conversion_note))
    else:
        if record.stage == 'discovery':
            record.stage = 'recommendation'

        if named_product and requirements.get('indoor_outdoor') and requirements.get('floor_or_wall'):
            passed, violations = apply_hard_constraints(named_product, requirements)
            if not passed:
                unsuitable_note = composer.compose_unsuitable_product(
                    named_product, violations, requirements.get('language', 'en'))
                tool_results.append({'tool_name': 'apply_hard_constraints', 'result': {'sku': named_product['sku']}})
            elif requirements.get('area_m2'):
                named_quantity = calculate_tile_quantity(requirements['area_m2'], named_product['sku'], layout='simple')
                named_inventory = check_inventory(named_product['sku'], named_quantity['total_purchasable_area_m2'])
                unsuitable_note = composer.compose_named_product_check(
                    named_product, named_quantity, named_inventory, requirements.get('language', 'en'))
                tool_results.append({'tool_name': 'calculate_tile_quantity', 'result': named_quantity})
                tool_results.append({'tool_name': 'check_inventory', 'result': named_inventory})

        recommendations, exclusions, explanation = recommend_products(requirements)
        if recommendations:
            record.last_recommendations = [r['sku'] for r in recommendations]
            record.shown_skus = list(record.last_recommendations)
        for rec in recommendations:
            tool_results.append({'tool_name': 'calculate_tile_quantity', 'result': rec})

        next_q = next_preference_question(requirements) if recommendations else None
        reply = (llm_composer.compose_recommendation_reply(requirements, recommendations, explanation, next_q)
                 or _with_question(composer.compose_recommendations(requirements, recommendations, explanation),
                                   next_q))
        if unsuitable_note:
            reply = f'{unsuitable_note}\n{reply}'
        if conversion_note:
            reply = f'{conversion_note}\n{reply}'
        if requirements.get('intent') == 'quote_request' and recommendations and requirements.get('area_m2'):
            record.stage = 'quote'

    reply, guardrail_info = _guarded(reply, recommendations, tool_results, requirements, quote)

    record.history.append({'role': 'assistant', 'text': reply})
    session_store.save(record)

    return {
        'session_id': record.session_id,
        'reply': reply,
        'requirements': requirements,
        'recommendations': recommendations,
        'missing_information': missing,
        'clarifying_questions': questions,
        'quote': quote,
        'stage': record.stage,
        'guardrail': guardrail_info,
    }
