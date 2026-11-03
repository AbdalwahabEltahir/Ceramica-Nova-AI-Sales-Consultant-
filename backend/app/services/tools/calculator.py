"""calculate_tile_quantity — FR-2.1..FR-2.5. Deterministic, decimal-only arithmetic."""

from decimal import Decimal

from backend.app.models.catalog import get_product_by_sku
from backend.app.models.money import to_decimal, round_money, ceil_decimal
from backend.app.models.rules import br05_wastage_percent


def calculate_tile_quantity(area_m2, sku, layout='simple', wastage_percent=None,
                             tile_width_cm=None, tile_length_cm=None, tiles_per_box=None):
    product = get_product_by_sku(sku) if sku else None
    if sku and not product:
        raise ValueError('SKU_NOT_FOUND')

    area = to_decimal(area_m2)
    if area <= 0:
        raise ValueError('VALIDATION_ERROR')

    if product:
        tile_width_cm = to_decimal(tile_width_cm) if tile_width_cm is not None else to_decimal(product['tile_width_cm'])
        tile_length_cm = to_decimal(tile_length_cm) if tile_length_cm is not None else to_decimal(product['tile_length_cm'])
        tiles_per_box = to_decimal(tiles_per_box) if tiles_per_box is not None else to_decimal(product['tiles_per_box'])
        price_per_m2 = to_decimal(product['price_per_m2'])
    else:
        if tile_width_cm is None or tile_length_cm is None or tiles_per_box is None:
            raise ValueError('VALIDATION_ERROR')
        tile_width_cm = to_decimal(tile_width_cm)
        tile_length_cm = to_decimal(tile_length_cm)
        tiles_per_box = to_decimal(tiles_per_box)
        price_per_m2 = Decimal('0')

    if wastage_percent is not None:
        waste_percent = to_decimal(wastage_percent)
        if not (Decimal('0') <= waste_percent <= Decimal('30')):
            raise ValueError('VALIDATION_ERROR')
    else:
        waste_percent = Decimal(str(br05_wastage_percent(layout)))

    area_with_wastage = area * (Decimal('1') + (waste_percent / Decimal('100')))
    tile_area_m2 = (tile_width_cm * tile_length_cm) / Decimal('10000')
    tiles_exact = area_with_wastage / tile_area_m2
    tiles_rounded_up = ceil_decimal(tiles_exact)
    boxes = ceil_decimal(tiles_rounded_up / tiles_per_box)
    purchasable_tiles = boxes * tiles_per_box
    total_purchasable_area_m2 = purchasable_tiles * tile_area_m2
    material_subtotal = round_money(total_purchasable_area_m2 * price_per_m2)

    return {
        'sku': sku,
        'layout': layout,
        'required_area_m2': float(area),
        'wastage_percent': float(waste_percent),
        'area_with_wastage_m2': float(round_money(area_with_wastage)),
        'tile_area_m2': float(tile_area_m2),
        'tiles_exact': float(tiles_exact),
        'tiles_rounded_up': int(tiles_rounded_up),
        'boxes': int(boxes),
        'purchasable_tiles': int(purchasable_tiles),
        'total_purchasable_area_m2': float(round_money(total_purchasable_area_m2)),
        'price_per_m2': float(price_per_m2),
        'material_subtotal': float(material_subtotal),
        'currency': 'USD',
        'pricing_basis': 'priced on full boxes',
    }
