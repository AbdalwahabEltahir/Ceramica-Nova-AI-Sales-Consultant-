"""Ceramica Nova product catalogue — the single source of truth for specs,
prices and stock. rag/documents.py indexes it into LanceDB for retrieval;
the hard-constraint flags (floor_ok / wall_ok / outdoor_ok / commercial) are
derived from the `application` column so rules.py can stay data-driven.
"""

# (sku, name, type, width_cm, length_cm, application, finish, slip, price, stock_m2, tiles_per_box, style)
_ROWS = [
    ('CN001', 'Arctic Pearl', 'porcelain', 60, 60, 'Indoor floor', 'Polished', 'R9', 31, 820, 4, 'Marble'),
    ('CN002', 'Desert Sand', 'porcelain', 60, 120, 'Indoor floor', 'Matt', 'R10', 36, 430, 2, 'Stone'),
    ('CN003', 'Nordic Oak', 'porcelain', 20, 120, 'Indoor floor', 'Matt', 'R10', 39, 510, 5, 'Wood'),
    ('CN004', 'Ocean Grip', 'porcelain', 60, 60, 'Outdoor', 'Textured', 'R12', 34, 300, 4, 'Modern'),
    ('CN005', 'Urban Graphite', 'porcelain', 60, 60, 'Indoor/Commercial', 'Matt', 'R10', 29, 740, 4, 'Industrial'),
    ('CN006', 'Royal Calacatta', 'porcelain', 120, 120, 'Premium indoor', 'Polished', 'R9', 61, 190, 1, 'Luxury marble'),
    ('CN007', 'Ivory Mist', 'ceramic', 30, 60, 'Wall', 'Gloss', 'N/A', 19, 980, 8, 'Minimal'),
    ('CN008', 'Aqua Shield', 'ceramic', 30, 60, 'Bathroom wall', 'Gloss', 'N/A', 23, 620, 8, 'Contemporary'),
    ('CN009', 'Terra Rustic', 'porcelain', 30, 60, 'Outdoor', 'Textured', 'R12', 27, 350, 8, 'Rustic'),
    ('CN010', 'Metro White', 'ceramic', 10, 30, 'Kitchen wall', 'Gloss', 'N/A', 17, 1200, 30, 'Subway'),
    ('CN011', 'Black Onyx', 'porcelain', 60, 120, 'Indoor', 'Polished', 'R9', 48, 260, 2, 'Luxury'),
    ('CN012', 'Sandstone Pro', 'porcelain', 60, 60, 'Commercial', 'Textured', 'R11', 42, 400, 4, 'Natural'),
    ('CN013', 'Pearl Mosaic', 'ceramic', 30, 30, 'Bathroom wall', 'Gloss', 'N/A', 26, 220, 12, 'Decorative'),
    ('CN014', 'Garden Slate', 'porcelain', 40, 80, 'Outdoor', 'Textured', 'R12', 33, 270, 4, 'Slate'),
    ('CN015', 'Concrete Light', 'porcelain', 60, 120, 'Indoor', 'Matt', 'R10', 35, 560, 2, 'Contemporary'),
]

# Placement flags per application value. "Indoor floor" products are floor
# only; general "Indoor"/"Premium indoor" porcelain is floor and wall.
_PLACEMENT = {
    'Indoor floor': dict(floor_ok=True, wall_ok=False, outdoor_ok=False, commercial=False),
    'Indoor': dict(floor_ok=True, wall_ok=True, outdoor_ok=False, commercial=False),
    'Premium indoor': dict(floor_ok=True, wall_ok=True, outdoor_ok=False, commercial=False),
    'Indoor/Commercial': dict(floor_ok=True, wall_ok=False, outdoor_ok=False, commercial=True),
    'Commercial': dict(floor_ok=True, wall_ok=False, outdoor_ok=False, commercial=True),
    'Outdoor': dict(floor_ok=True, wall_ok=False, outdoor_ok=True, commercial=False),
    'Wall': dict(floor_ok=False, wall_ok=True, outdoor_ok=False, commercial=False),
    'Bathroom wall': dict(floor_ok=False, wall_ok=True, outdoor_ok=False, commercial=False),
    'Kitchen wall': dict(floor_ok=False, wall_ok=True, outdoor_ok=False, commercial=False),
}


def _build(row):
    sku, name, tile_type, width, length, application, finish, slip, price, stock, per_box, style = row
    product = {
        'sku': sku,
        'name': name,
        'type': tile_type,
        'size': f'{width}×{length}',
        'application': application,
        'finish': finish.lower(),
        'slip_rating': slip,
        'style': style.lower(),
        'price_per_m2': price,
        'stock_m2': stock,
        'tile_width_cm': width,
        'tile_length_cm': length,
        'tiles_per_box': per_box,
    }
    product.update(_PLACEMENT[application])
    return product


CATALOG = [_build(row) for row in _ROWS]

STYLES = sorted({p['style'] for p in CATALOG})
FINISHES = sorted({p['finish'] for p in CATALOG})


def get_catalog():
    return CATALOG


def get_product_by_sku(sku):
    if not sku:
        return None
    sku = sku.upper()
    for product in CATALOG:
        if product['sku'] == sku:
            return product
    return None


def find_named_product(message):
    """Finds a product the customer named explicitly, by SKU or product name
    (used so a named product still gets its real price/suitability stated,
    even when the surrounding message is blocked by the input guard)."""
    import re

    if not message:
        return None
    match = re.search(r'\bCN0\d{2}\b', message, re.I)
    if match:
        return get_product_by_sku(match.group(0))

    lower = message.lower()
    best = None
    for product in CATALOG:
        name_lower = product['name'].lower()
        if name_lower in lower and (best is None or len(name_lower) > len(best['name'])):
            best = product
    return best
