"""check_inventory — FR-2.6..FR-2.10. Stock is authoritative (BR-04)."""

from datetime import datetime, timezone

from backend.app.models.catalog import get_product_by_sku
from backend.app.models.money import to_decimal


def check_inventory(sku, required_area_m2):
    product = get_product_by_sku(sku)
    if not product:
        raise ValueError('SKU_NOT_FOUND')

    required = to_decimal(required_area_m2)
    if required <= 0:
        raise ValueError('VALIDATION_ERROR')

    stock = to_decimal(product['stock_m2'])
    if stock >= required:
        status = 'AVAILABLE'
        shortfall = to_decimal('0')
    elif stock > 0:
        status = 'PARTIALLY_AVAILABLE'
        shortfall = required - stock
    else:
        status = 'OUT_OF_STOCK'
        shortfall = required

    return {
        'sku': sku,
        'product_name': product['name'],
        'stock_m2': float(stock),
        'required_area_m2': float(required),
        'status': status,
        'shortfall_m2': float(shortfall),
        'checked_at': datetime.now(timezone.utc).isoformat(),
    }
