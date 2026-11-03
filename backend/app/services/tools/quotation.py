"""generate_quote — FR-2.11..FR-2.15. Prices and owner always come from the server."""

from decimal import Decimal

from backend.app.models.catalog import get_product_by_sku
from backend.app.models.money import to_decimal, round_money
from backend.app.models.rules import br07_requires_approval
from backend.app.models.session_store import quote_store
from backend.app.services.tools.calculator import calculate_tile_quantity
from backend.app.services.tools.inventory import check_inventory


def generate_quote(lines, requested_discount_percent=0, owner_id='guest'):
    if not lines:
        raise ValueError('VALIDATION_ERROR')

    subtotal = Decimal('0')
    line_results = []
    currency = 'USD'

    for line in lines:
        sku = line.get('sku')
        area = line.get('area_m2')
        if not sku or not area:
            raise ValueError('VALIDATION_ERROR')
        area = to_decimal(area)
        if area <= 0:
            raise ValueError('VALIDATION_ERROR')

        product = get_product_by_sku(sku)
        if not product:
            raise ValueError('SKU_NOT_FOUND')
        if product.get('currency', 'USD') != currency:
            raise ValueError('CURRENCY_MISMATCH')

        calc = calculate_tile_quantity(area, sku, layout=line.get('layout', 'simple'))
        inventory = check_inventory(sku, calc['total_purchasable_area_m2'])
        subtotal += to_decimal(calc['material_subtotal'])
        line_results.append({
            'sku': sku,
            'product_name': product['name'],
            'area_m2': calc['required_area_m2'],
            'boxes': calc['boxes'],
            'unit_price': calc['price_per_m2'],
            'subtotal': calc['material_subtotal'],
            'availability': inventory['status'],
        })

    discount_percent = to_decimal(requested_discount_percent or 0)
    if not (Decimal('0') <= discount_percent < Decimal('100')):
        raise ValueError('VALIDATION_ERROR')

    discount_amount = subtotal * (discount_percent / Decimal('100'))
    total = subtotal - discount_amount
    approval_required = br07_requires_approval(discount_percent)

    quote = {
        'quotation_id': quote_store.new_id(),
        'owner_id': owner_id,
        'lines': line_results,
        'subtotal': float(round_money(subtotal)),
        'discount_percent': float(discount_percent),
        'discount_amount': float(round_money(discount_amount)),
        'total': float(round_money(total)),
        'currency': currency,
        'approval_required': approval_required,
        'approval_reason': (
            f'BR-07: discount of {discount_percent}% exceeds the 12% threshold and needs staff approval.'
            if approval_required else ''
        ),
        'status': 'PENDING_APPROVAL' if approval_required else 'DRAFT',
    }
    quote_store.save(quote)
    return quote


def approve_quote(quotation_id):
    quote = quote_store.get(quotation_id)
    if not quote:
        raise ValueError('NOT_FOUND')
    quote['status'] = 'APPROVED'
    quote['approval_required'] = False
    quote_store.save(quote)
    return quote
