"""G4 output guard — FR-4.7..FR-4.9. Numbers and claims must trace back to tool
results in code, not merely by prompt instruction — this matters more once an
LLM (not just the template composer) is allowed to write the reply text.
"""

import re

from backend.app.config import SCARCITY_PHRASES, FORBIDDEN_DELIVERY_PHRASES, MAX_PRIMARY_PRODUCTS

SKU_PATTERN = re.compile(r'\bCN0?\d{2,3}\b', re.I)
MONEY_PATTERN = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)')
NUMBER_TOLERANCE = 0.015  # covers rounding/formatting drift (e.g. 777.6 vs 777.60)


def _collect_known_numbers(recommendations, tool_results, quote, requirements):
    """Every dollar figure the reply is allowed to quote — sourced only from
    tool output and the customer's own stated budget, never invented."""
    numbers = set()

    def add(value):
        if value is None:
            return
        try:
            numbers.add(round(float(value), 2))
        except (TypeError, ValueError):
            pass

    for rec in recommendations or []:
        for field in ('price_per_m2', 'material_subtotal', 'over_budget_by'):
            add(rec.get(field))

    for tr in tool_results or []:
        result = tr.get('result', {}) if isinstance(tr, dict) else {}
        for field in ('price_per_m2', 'material_subtotal', 'subtotal', 'total',
                      'discount_amount', 'over_budget_by'):
            add(result.get(field))

    if quote:
        for field in ('subtotal', 'discount_amount', 'total'):
            add(quote.get(field))
        for line in quote.get('lines', []):
            add(line.get('unit_price'))
            add(line.get('subtotal'))

    if requirements and requirements.get('budget'):
        add(requirements['budget'].get('amount'))

    return numbers


def _collect_valid_skus(recommendations, tool_results):
    valid_skus = {r['sku'] for r in (recommendations or [])}
    for tr in (tool_results or []):
        sku = tr.get('result', {}).get('sku') if isinstance(tr, dict) else None
        if sku:
            valid_skus.add(sku.upper())
    return valid_skus


def check_reply(reply_text, recommendations, tool_results, quote=None, requirements=None):
    """Returns {'blocked': bool, 'violations': [str]}."""
    violations = []
    text = reply_text or ''
    lower = text.lower()

    if len(recommendations or []) > MAX_PRIMARY_PRODUCTS:
        violations.append('TOO_MANY_PRODUCTS')

    valid_skus = _collect_valid_skus(recommendations, tool_results)
    for match in SKU_PATTERN.finditer(text):
        token = match.group(0).upper()
        if not token.startswith('CN0'):
            token = 'CN0' + token[2:]
        if token not in valid_skus:
            violations.append(f'UNGROUNDED_SKU:{match.group(0)}')

    known_numbers = _collect_known_numbers(recommendations, tool_results, quote, requirements)
    if known_numbers:  # only enforce once there's something to ground against
        for match in MONEY_PATTERN.finditer(text):
            value = round(float(match.group(1).replace(',', '')), 2)
            if not any(abs(value - known) <= NUMBER_TOLERANCE for known in known_numbers):
                violations.append(f'UNGROUNDED_NUMBER:${match.group(1)}')

    for phrase in SCARCITY_PHRASES:
        if phrase in lower:
            violations.append(f'SCARCITY_PHRASE:{phrase}')

    for phrase in FORBIDDEN_DELIVERY_PHRASES:
        if phrase in lower:
            violations.append(f'DELIVERY_PROMISE:{phrase}')

    if 'approved' in lower or 'discount applied' in lower:
        if not (quote and quote.get('status') == 'APPROVED'):
            violations.append('UNAUTHORISED_APPROVAL_CLAIM')

    return {'blocked': len(violations) > 0, 'violations': violations}


TEMPLATE_FALLBACK_EN = "Here is what I can confirm from the catalogue and current stock."
TEMPLATE_FALLBACK_AR = "هذا ما يمكنني تأكيده من الكتالوج والمخزون الحالي."


def template_fallback(language='en'):
    return TEMPLATE_FALLBACK_AR if language == 'ar' else TEMPLATE_FALLBACK_EN
