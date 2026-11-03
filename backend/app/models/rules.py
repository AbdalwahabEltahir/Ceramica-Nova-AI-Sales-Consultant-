"""BR-01..BR-10 as pure functions with reason codes (exam section E.4).

Each hard-constraint check returns (passed: bool, rule_id: str | None, reason_code: str).
These are the single source of truth for the constraint engine (FR-3) and the
output guard (FR-4.7) — nothing here depends on Flask or the LLM.
"""

from backend.app.config import OUTDOOR_MIN_SLIP, INDOOR_HIGH_SLIP_MIN

SLIP_ORDER = ['N/A', 'R9', 'R10', 'R11', 'R12']


def _slip_rank(rating):
    try:
        return SLIP_ORDER.index(rating)
    except ValueError:
        return 0


def slip_at_least(rating, minimum):
    return _slip_rank(rating) >= _slip_rank(minimum)


def hc1_floor_excludes_wall_only(product, requirements):
    """BR-02: a floor request must exclude wall-only products."""
    if requirements.get('floor_or_wall') == 'floor' and not product.get('floor_ok'):
        return False, 'HC-1', 'WALL_ONLY_PRODUCT'
    return True, None, None


def hc1_wall_excludes_non_wall(product, requirements):
    if requirements.get('floor_or_wall') == 'wall' and not product.get('wall_ok'):
        return False, 'HC-1', 'FLOOR_ONLY_PRODUCT'
    return True, None, None


def hc2_outdoor_rated_only(product, requirements):
    """Outdoor requests use only outdoor-rated products."""
    if requirements.get('indoor_outdoor') == 'outdoor' and not product.get('outdoor_ok'):
        return False, 'HC-2', 'NOT_OUTDOOR_RATED'
    return True, None, None


def hc3_slip_requirement(product, requirements):
    """BR-01 / A-3: outdoor floors need R11+, slip_requirement=high needs
    R11+ outdoors and R10+ indoors."""
    outdoor = requirements.get('indoor_outdoor') == 'outdoor'
    floor = requirements.get('floor_or_wall') == 'floor'
    high_slip = requirements.get('slip_requirement') == 'high'

    if outdoor and floor and not slip_at_least(product.get('slip_rating'), OUTDOOR_MIN_SLIP):
        return False, 'HC-3', 'SLIP_BELOW_R11_OUTDOOR'
    if high_slip:
        minimum = OUTDOOR_MIN_SLIP if outdoor else INDOOR_HIGH_SLIP_MIN
        if not slip_at_least(product.get('slip_rating'), minimum):
            return False, 'HC-3', 'SLIP_BELOW_REQUIRED'
    return True, None, None


def hc5_stock_covers_requirement(product, required_area_m2):
    """BR-04 / BR-08: out-of-stock products are excluded as primary options."""
    stock = product.get('stock_m2', 0)
    if required_area_m2 and stock <= 0:
        return False, 'HC-5', 'OUT_OF_STOCK'
    return True, None, None


HARD_CONSTRAINTS = [
    hc1_floor_excludes_wall_only,
    hc1_wall_excludes_non_wall,
    hc2_outdoor_rated_only,
    hc3_slip_requirement,
]


def apply_hard_constraints(product, requirements):
    """Run every hard constraint against one product. Returns (passed, violations)."""
    violations = []
    for check in HARD_CONSTRAINTS:
        passed, rule_id, reason = check(product, requirements)
        if not passed:
            violations.append({'rule_id': rule_id, 'reason_code': reason})
    return (len(violations) == 0), violations


def hc4_commercial_priority(product, requirements):
    """BR-03: commercial high-traffic floors list commercial-rated porcelain first."""
    return bool(product.get('commercial')) and requirements.get('traffic_level') == 'commercial_high'


def br05_wastage_percent(layout):
    from backend.app.config import WASTAGE_SIMPLE_PERCENT, WASTAGE_COMPLEX_PERCENT
    if layout in ('diagonal', 'complex'):
        return WASTAGE_COMPLEX_PERCENT
    return WASTAGE_SIMPLE_PERCENT


def br07_requires_approval(discount_percent):
    from backend.app.config import DISCOUNT_APPROVAL_THRESHOLD_PERCENT
    return discount_percent > DISCOUNT_APPROVAL_THRESHOLD_PERCENT


def br08_availability_label(status):
    """Only AVAILABLE may be described as 'in stock for your quantity'."""
    return {
        'AVAILABLE': 'in stock for your quantity',
        'PARTIALLY_AVAILABLE': 'only partially in stock',
        'OUT_OF_STOCK': 'out of stock',
    }.get(status, 'unknown')
