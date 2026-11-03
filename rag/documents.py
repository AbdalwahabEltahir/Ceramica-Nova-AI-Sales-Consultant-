"""Builds the RAG source corpus: one document per SKU (specs + a plain-language
descriptor line) and one per business rule (exam C1 / SRS FR-3.2). Metadata lets
the retriever stay inside the hard-constraint boundary already enforced by
models/rules.py — retrieval never overrides it, it only ranks survivors.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.models.catalog import get_catalog  # noqa: E402

BUSINESS_RULES = [
    ('BR-01', 'Outdoor flooring should normally use slip-resistant products rated R11 or above.'),
    ('BR-02', 'Wall-only ceramic products must not be recommended as flooring.'),
    ('BR-03', 'Commercial high-traffic flooring should prioritize porcelain products designed for commercial use.'),
    ('BR-04', 'Stock quantities are authoritative. The assistant must never fabricate inventory.'),
    ('BR-05', 'Standard wastage: simple layout 10%, diagonal or complex layout 15%.'),
    ('BR-06', 'Price calculations must use deterministic software, never LLM arithmetic.'),
    ('BR-07', 'No discount above 12% without human approval.'),
    ('BR-08', 'Products below the required quantity must not be described as immediately available.'),
    ('BR-09', 'Recommendations should normally contain no more than three primary products.'),
    ('BR-10', 'If essential customer information is missing, ask targeted questions instead of guessing.'),
]

# Approved care notes (assumption A-9): the catalogue has no maintenance data,
# so this is the only place "easy to clean" / durability claims may come from.
CARE_NOTES = {
    'CN001': 'Polished porcelain. Wipe with a damp mop; avoid acidic cleaners which dull the polish.',
    'CN005': 'Matt commercial porcelain. Resists heavy foot traffic; sweep and mop with neutral detergent.',
    'CN009': 'Textured outdoor porcelain. Rinse with water; textured surface resists algae build-up.',
    'CN012': 'Textured commercial porcelain, R11. Scrub brush recommended for grout lines in kitchens.',
}


def _placement_text(product):
    placement = []
    if product['floor_ok']:
        placement.append('floors')
    if product['wall_ok']:
        placement.append('walls')
    if product['outdoor_ok']:
        placement.append('outdoor areas')
    return ' and '.join(placement) if placement else 'walls only'


def _slip_text(rating):
    if rating in ('R11', 'R12'):
        return f'slip rating {rating} (slip-resistant, suitable for wet or outdoor floors)'
    if rating == 'N/A':
        return 'no slip rating (wall tile, not for floors)'
    return f'slip rating {rating} (standard indoor grip, not for wet or outdoor floors)'


def _descriptor(product):
    # Stock is deliberately left out of the embedded text: it changes daily and
    # must only ever be quoted from check_inventory (BR-04), never from RAG.
    commercial_text = 'rated for commercial high-traffic use' if product['commercial'] else 'for residential use'
    return (
        f"{product['name']} ({product['sku']}) is a {product['style']} style {product['type']} tile "
        f"with a {product['finish']} finish, size {product['size']} cm, {product['tiles_per_box']} tiles per box. "
        f"Application: {product['application']} — suitable for {_placement_text(product)}, {commercial_text}. "
        f"It has a {_slip_text(product['slip_rating'])}. Price ${product['price_per_m2']}/m²."
    )


def build_documents():
    """Returns a list of {'id', 'type', 'text', 'metadata'} dicts — the corpus
    before ingestion-time sanitisation (see rag/index.py's quarantine step)."""
    documents = []

    for product in get_catalog():
        text = _descriptor(product)
        if product['sku'] in CARE_NOTES:
            text += f" Care: {CARE_NOTES[product['sku']]}"
        documents.append({
            'id': f"product:{product['sku']}",
            'type': 'product',
            'text': text,
            'metadata': {
                'sku': product['sku'],
                'name': product['name'],
                'tile_type': product['type'],
                'size': product['size'],
                'application': product['application'],
                'finish': product['finish'],
                'slip_rating': product['slip_rating'],
                'price_per_m2': product['price_per_m2'],
                'stock_m2': product['stock_m2'],
                'tiles_per_box': product['tiles_per_box'],
                'style': product['style'],
                'floor_ok': product['floor_ok'],
                'wall_ok': product['wall_ok'],
                'outdoor_ok': product['outdoor_ok'],
                'commercial': product['commercial'],
            },
        })

    for rule_id, text in BUSINESS_RULES:
        documents.append({
            'id': f"rule:{rule_id}",
            'type': 'business_rule',
            'text': f'{rule_id}: {text}',
            'metadata': {'rule_id': rule_id},
        })

    return documents
