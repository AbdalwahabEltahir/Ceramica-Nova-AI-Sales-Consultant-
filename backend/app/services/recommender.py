"""FR-3 hard constraints, retrieval and ranking. Hard constraints run first, in
code; only survivors are ever scored or shown (BR-09: at most 3 primary products).
"""

from backend.app.config import MAX_PRIMARY_PRODUCTS, RANKING_WEIGHTS
from backend.app.models.catalog import get_catalog
from backend.app.models.rules import apply_hard_constraints, hc4_commercial_priority, hc5_stock_covers_requirement
from backend.app.services.tools.calculator import calculate_tile_quantity
from backend.app.services.tools.inventory import check_inventory


def _retrieval_query(requirements):
    """Builds a free-text query from the structured requirements so retrieval
    relevance (RAG, FR-3.2) reflects what the customer actually asked for,
    not just hard-constraint fields."""
    parts = []
    for field in ('application', 'preferred_style', 'preferred_finish', 'preferred_color',
                  'floor_or_wall', 'indoor_outdoor'):
        value = requirements.get(field)
        if value:
            parts.append(str(value))
    if requirements.get('slip_requirement') == 'high':
        parts.append('anti-slip safe')
    if requirements.get('traffic_level') == 'commercial_high':
        parts.append('commercial heavy traffic')
    return ' '.join(parts) or 'tile'


def _soft_score(product, requirements, inventory, retrieval_score=0.5):
    score = 0.0
    reasons = []

    style = requirements.get('preferred_style')
    if style:
        if style in product['style']:  # 'marble' also matches 'luxury marble'
            score += RANKING_WEIGHTS['style_match']
            reasons.append(f"STYLE_{style.upper().replace(' ', '_')}")
    else:
        score += RANKING_WEIGHTS['style_match'] * 0.3

    finish = requirements.get('preferred_finish')
    if finish and product.get('finish') == finish:
        score += RANKING_WEIGHTS['finish_match']
        reasons.append(f'FINISH_{finish.upper()}')

    budget = requirements.get('budget')
    area = requirements.get('area_m2')
    within_budget = True
    if budget and budget.get('basis') == 'per_m2':
        if product['price_per_m2'] <= budget['amount']:
            score += RANKING_WEIGHTS['price_fit']
            reasons.append('WITHIN_BUDGET')
        else:
            within_budget = False
            score += RANKING_WEIGHTS['price_fit'] * max(
                0, 1 - (product['price_per_m2'] - budget['amount']) / budget['amount'])
    elif budget and area:
        estimate = product['price_per_m2'] * area * 1.1
        if estimate <= budget['amount']:
            score += RANKING_WEIGHTS['price_fit']
            reasons.append('WITHIN_BUDGET')
        else:
            within_budget = False
            score += RANKING_WEIGHTS['price_fit'] * max(0, 1 - (estimate - budget['amount']) / budget['amount'])
    else:
        score += RANKING_WEIGHTS['price_fit'] * 0.5

    if requirements.get('preferred_color'):
        color = requirements['preferred_color'].lower()
        if color in product['name'].lower():
            score += RANKING_WEIGHTS['colour_match']
            reasons.append('COLOUR_MATCH')
    else:
        score += RANKING_WEIGHTS['colour_match'] * 0.5

    score += RANKING_WEIGHTS['retrieval_relevance'] * retrieval_score

    if inventory:
        if inventory['status'] == 'AVAILABLE':
            score += RANKING_WEIGHTS['stock_margin']
            reasons.append('STOCK_OK')
        elif inventory['status'] == 'PARTIALLY_AVAILABLE':
            score += RANKING_WEIGHTS['stock_margin'] * 0.3
    else:
        score += RANKING_WEIGHTS['stock_margin'] * 0.5

    if hc4_commercial_priority(product, requirements):
        score += 0.5
        reasons.append('COMMERCIAL_RATED')

    if requirements.get('indoor_outdoor') == 'outdoor' and product.get('outdoor_ok'):
        reasons.append('OUTDOOR_RATED')
    if requirements.get('slip_requirement') == 'high' and product.get('slip_rating') in ('R11', 'R12'):
        reasons.append('SLIP_SAFE')

    return score, reasons, within_budget


def _retrieval_scores_by_sku(requirements):
    """One retrieval call per turn (not per product) — avoids hammering the
    HF embeddings API once per catalogue row."""
    from rag.retriever import retrieve
    query = _retrieval_query(requirements)
    try:
        results = retrieve(query, top_k=15)
    except Exception:
        # Retrieval is a ranking signal only, never a hard dependency — if it
        # fails for any reason (network, bad cache file, etc.) ranking just
        # falls back to the other four soft-score signals.
        return {}
    return {
        r['metadata']['sku']: max(0.0, 1.0 - rank / 15)
        for rank, r in enumerate(results) if r['metadata'].get('sku')
    }


def recommend_products(requirements, exclude_skus=None):
    """Returns (recommendations: list, exclusions: list, explanation: str|None).
    exclude_skus: products already shown, skipped so "other options" gets
    the next-best suitable products instead of the same three again."""
    area = requirements.get('area_m2')
    scored = []
    exclusions = []
    retrieval_scores = _retrieval_scores_by_sku(requirements)
    skip = set(exclude_skus or [])

    for product in get_catalog():
        if product['sku'] in skip:
            continue
        passed, violations = apply_hard_constraints(product, requirements)
        if not passed:
            exclusions.append({'sku': product['sku'], 'violations': violations})
            continue

        quantity = None
        inventory = None
        if area:
            quantity = calculate_tile_quantity(area, product['sku'], layout='simple')
            inventory = check_inventory(product['sku'], quantity['total_purchasable_area_m2'])
            stock_ok, rule_id, reason = hc5_stock_covers_requirement(product, quantity['total_purchasable_area_m2'])
            if inventory['status'] == 'OUT_OF_STOCK':
                exclusions.append({'sku': product['sku'], 'violations': [{'rule_id': 'HC-5', 'reason_code': 'OUT_OF_STOCK'}]})
                continue

        retrieval_score = retrieval_scores.get(product['sku'], 0.0)
        score, reasons, within_budget = _soft_score(product, requirements, inventory, retrieval_score)
        scored.append({
            'product': product,
            'score': score,
            'reasons': reasons,
            'quantity': quantity,
            'inventory': inventory,
            'within_budget': within_budget,
            'partial_stock': inventory is not None and inventory['status'] == 'PARTIALLY_AVAILABLE',
        })

    # BR-03/HC-4: commercial-rated porcelain first for commercial high-traffic floors.
    scored.sort(key=lambda item: (
        0 if hc4_commercial_priority(item['product'], requirements) else 1,
        item['partial_stock'],
        -item['score'],
        item['product']['price_per_m2'],
        item['product']['sku'],
    ))

    recommendations = []
    for entry in scored[:MAX_PRIMARY_PRODUCTS]:
        product = entry['product']
        quantity = entry['quantity']
        inventory = entry['inventory']
        rec = {
            'sku': product['sku'],
            'name': product['name'],
            'style': product['style'],
            'type': product['type'],
            'finish': product['finish'],
            'size': product['size'],
            'application': product['application'],
            'price_per_m2': product['price_per_m2'],
            'stock_m2': product['stock_m2'],
            'slip_rating': product['slip_rating'],
            'commercial': product['commercial'],
            'reason_codes': entry['reasons'],
            'status': inventory['status'] if inventory else None,
            'score': round(entry['score'], 4),
        }
        budget = requirements.get('budget')
        if not entry['within_budget'] and budget and budget.get('basis') != 'per_m2' and quantity:
            over_by = round(quantity['material_subtotal'] - requirements['budget']['amount'], 2)
            if over_by > 0:
                rec['over_budget_by'] = over_by
        if quantity:
            rec.update({
                'boxes': quantity['boxes'],
                'purchasable_area_m2': quantity['total_purchasable_area_m2'],
                'material_subtotal': quantity['material_subtotal'],
                'wastage_percent': quantity['wastage_percent'],
            })
        recommendations.append(rec)

    explanation = None
    if not recommendations and exclusions:
        rule_ids = sorted({v['rule_id'] for ex in exclusions for v in ex['violations']})
        explanation = (
            'No catalogue product satisfies all of your requirements '
            f"(blocked by {', '.join(rule_ids)}). Try relaxing one constraint, "
            'such as the indoor/outdoor placement or the slip rating.'
        )

    return recommendations, exclusions, explanation
