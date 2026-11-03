"""Deterministic reply composer — builds text from tool results and reason
codes only, no model call. This is the FR-7.3 final fallback ("template
reply built from tool data") and runs whenever GOOGLE_API_KEY is unset or
every LLM retry/fallback is exhausted (see services/llm_composer.py for the
Gemini-backed paths, and services/orchestrator.py for how the two combine).
Either way, the orchestrator only ever sees composed text plus the
structured data it was built from, so FR-6.1 (every number on screen comes
from structured fields) and G4's grounding check hold regardless of source.
"""

from backend.app.models.rules import br08_availability_label


def _label(language, en, ar):
    return ar if language == 'ar' else en


def compose_question_lines(questions):
    return '\n'.join(f'• {q}' for q in questions)


_FAMILY_ORDER = [
    ('Outdoor & anti-slip', 'للخارج ومقاوم للانزلاق', lambda p: p['outdoor_ok']),
    ('Commercial & heavy traffic', 'تجاري وحركة كثيفة', lambda p: p['commercial']),
    ('Indoor floors', 'أرضيات داخلية', lambda p: p['floor_ok']),
    ('Wall tiles', 'بلاط الجدران', lambda p: True),
]


def compose_catalog_overview(products, language='en', asked_about_colour=False):
    """Template fallback for 'what are the products?': the suitable range,
    grouped by where each tile can go, one fact-only line per product."""
    if not products:
        return _label(language,
                      "Nothing in our range meets every requirement you've given — could you tell me more?",
                      'لا يوجد في مجموعتنا ما يلبي جميع متطلباتك — هل يمكنك إخباري بالمزيد؟')
    lines = []
    if asked_about_colour:
        lines.append(_label(language,
                            "Our catalogue doesn't list colours, but the names and styles give a good idea of each look:",
                            'لا يحدد الكتالوج الألوان، لكن الأسماء والأنماط تعطي فكرة جيدة عن كل منتج:'))
    else:
        lines.append(_label(language, 'Here is how our range breaks down:', 'إليك نظرة على مجموعتنا:'))

    remaining = list(products)
    for name_en, name_ar, belongs in _FAMILY_ORDER:
        family = [p for p in remaining if belongs(p)]
        if not family:
            continue
        remaining = [p for p in remaining if p not in family]
        lines.append(_label(language, f'{name_en}:', f'{name_ar}:'))
        for p in family:
            lines.append(f"- {p['name']} ({p['sku']}): {p['style']}, {p['finish']}, {p['size']} cm, "
                         f"${p['price_per_m2']}/m², slip {p['slip_rating']}")
    return '\n'.join(lines)


def compose_clarifying(requirements, questions, conversion_note=None):
    language = requirements.get('language', 'en')
    intro = _label(language,
                    "I'd like to get a couple of details right before suggesting tiles.",
                    'أود التأكد من بعض التفاصيل قبل اقتراح البلاط المناسب.')
    lines = [intro] + [f'• {q}' for q in questions]
    if conversion_note:
        lines.append(conversion_note)
    return '\n'.join(lines)


def compose_no_more_options(language='en'):
    return _label(
        language,
        "Those are all the tiles in our range that meet your requirements. If you can relax one "
        "of them, such as the placement or slip rating, I can show you more.",
        'هذه كل المنتجات في مجموعتنا التي تلبي متطلباتك. إذا أمكنك التخفيف من أحد الشروط، '
        'مثل مكان التركيب أو درجة مقاومة الانزلاق، يمكنني عرض المزيد.')


def compose_recommendations(requirements, recommendations, explanation, more_options=False):
    language = requirements.get('language', 'en')

    if not recommendations:
        return explanation or _label(
            language,
            "I couldn't find a product that meets every requirement — could you tell me more?",
            'لم أتمكن من إيجاد منتج يلبي جميع المتطلبات — هل يمكنك إخباري بالمزيد؟')

    if more_options:
        intro = _label(language, 'Here are more options that suit your project:',
                       'إليك خيارات إضافية تناسب مشروعك:')
    else:
        intro = _label(language,
                       f'I found {len(recommendations)} suitable option(s) for your project:',
                       f'وجدت {len(recommendations)} خيار/خيارات مناسبة لمشروعك:')
    parts = [intro]

    for rec in recommendations:
        availability = br08_availability_label(rec['status']) if rec.get('status') else None
        line = f"- {rec['name']} ({rec['sku']}), ${rec['price_per_m2']}/m², slip {rec['slip_rating']}"
        if rec.get('finish'):
            line += f", {rec['finish']} finish, {rec['size']} cm"
        if 'material_subtotal' in rec:
            line += f", {rec['boxes']} boxes, subtotal ${rec['material_subtotal']}"
        if availability:
            line += f' — {availability}'
        if rec.get('over_budget_by'):
            line += f" (over your budget by ${rec['over_budget_by']})"
        parts.append(line)

    if explanation:
        parts.append(explanation)

    return '\n'.join(parts)


def compose_named_product_fact(product, language='en'):
    return _label(
        language,
        f"The catalogue price for {product['name']} ({product['sku']}) is "
        f"${product['price_per_m2']}/m² — I can't apply an unverified price or approval.",
        f"السعر الحقيقي في الكتالوج لـ {product['name']} ({product['sku']}) هو "
        f"{product['price_per_m2']}$/م² — لا يمكنني تطبيق سعر أو موافقة غير موثقة.")


def compose_unsuitable_product(product, violations, language='en', with_alternatives=True):
    rule_ids = ', '.join(sorted({v['rule_id'] for v in violations}))
    if not with_alternatives:
        return _label(
            language,
            f"I can't quote {product['name']} ({product['sku']}) for this project: it isn't suitable "
            f"(blocked by {rule_ids}). Would you like me to suggest tiles that are?",
            f"لا يمكنني إعداد عرض سعر لـ {product['name']} ({product['sku']}) لهذا المشروع لأنه غير مناسب "
            f"(محظور بموجب {rule_ids}). هل تريد أن أقترح بلاطاً مناسباً؟")
    return _label(
        language,
        f"{product['name']} ({product['sku']}) is not suitable for this request "
        f"(blocked by {rule_ids}). Here are valid alternatives instead:",
        f"{product['name']} ({product['sku']}) غير مناسب لهذا الطلب (محظور بموجب {rule_ids}). "
        'إليك بدائل مناسبة بدلاً منه:')


def compose_named_product_check(product, quantity, inventory, language='en'):
    availability = br08_availability_label(inventory['status'])
    base = (
        f"{product['name']} ({product['sku']}): {quantity['boxes']} boxes needed "
        f"({quantity['total_purchasable_area_m2']} m²), ${quantity['material_subtotal']} subtotal — "
        f"{availability}."
    )
    if inventory['status'] != 'AVAILABLE':
        shortfall = inventory['shortfall_m2']
        base += _label(
            language,
            f" Stock on hand is {inventory['stock_m2']} m², short by {shortfall} m². "
            'I can arrange a staff follow-up or suggest an in-stock alternative below.',
            f" المخزون المتوفر {inventory['stock_m2']} م²، بنقص قدره {shortfall} م². "
            'يمكنني ترتيب متابعة مع فريق المبيعات أو اقتراح بديل متوفر أدناه.')
    return base


def compose_quote_which_product(products, language='en'):
    names = ', '.join(f"{p['name']} ({p['sku']})" for p in products)
    if names:
        return _label(language, f'Happy to prepare a quote. Which product should it be for: {names}?',
                      f'يسعدني إعداد عرض السعر. لأي منتج تريده: {names}؟')
    return _label(language, 'Happy to prepare a quote. Which product would you like it for?',
                  'يسعدني إعداد عرض السعر. لأي منتج تريده؟')


def compose_quote_needs_area(product, language='en'):
    return _label(
        language,
        f"Happy to prepare a quote for {product['name']} ({product['sku']}). How many square meters do you need to cover?",
        f"يسعدني إعداد عرض سعر لـ {product['name']} ({product['sku']}). كم متراً مربعاً تحتاج لتغطيته؟")


def compose_invalid_discount(language='en'):
    return _label(language,
                  "I can't put that discount on a quote. Tell me a discount percentage below 100% and I'll add the request.",
                  'لا يمكنني إضافة هذا الخصم إلى عرض السعر. أخبرني بنسبة خصم أقل من 100% وسأضيف الطلب.')


def compose_quote_reply(quote, language='en', invite_discount=False):
    """Every figure comes from the generate_quote tool; approval is never
    claimed (BR-07) - a discount above 12% is shown as pending staff review."""
    lines = [compose_quote_summary(quote, language)]
    for line in quote['lines']:
        lines.append(f"- {line['product_name']} ({line['sku']}): {line['boxes']} boxes at "
                     f"${_money(line['unit_price'])}/m², subtotal ${_money(line['subtotal'])} — "
                     f"{br08_availability_label(line['availability'])}")
    if quote['approval_required']:
        lines.append(_label(language,
                            'Our team will review the discount and get back to you; until then the discounted total is not confirmed.',
                            'سيراجع فريقنا الخصم ويعود إليك؛ وحتى ذلك الحين لا يعتبر الإجمالي بعد الخصم مؤكداً.'))
    elif invite_discount:
        lines.append(_label(language,
                            'These are our catalogue prices. If you would like to request a discount, tell me the '
                            'percentage and I will add it to the quote; requests above 12% are reviewed by our team.',
                            'هذه أسعار الكتالوج. إذا أردت طلب خصم، أخبرني بالنسبة وسأضيفها إلى عرض السعر؛ '
                            'الطلبات التي تزيد عن 12% يراجعها فريقنا.'))
    return '\n'.join(lines)


def _money(value):
    return f'{float(value):,.2f}'


def compose_quote_summary(quote, language='en'):
    if quote['status'] == 'PENDING_APPROVAL':
        percent = f"{quote['discount_percent']:g}"
        status_line = _label(language,
                              f'This quote needs staff approval because the requested discount ({percent}%) '
                              'is above the 12% limit (BR-07).',
                              f'يحتاج هذا العرض إلى موافقة الموظفين لأن الخصم المطلوب ({percent}%) يتجاوز حد 12%.')
    else:
        status_line = _label(language, 'Here is your quote.', 'إليك عرض السعر الخاص بك.')
    return (
        f"{status_line}\n"
        f"Quote {quote['quotation_id']}: subtotal ${_money(quote['subtotal'])}, "
        f"discount ${_money(quote['discount_amount'])}, total ${_money(quote['total'])}."
    )
