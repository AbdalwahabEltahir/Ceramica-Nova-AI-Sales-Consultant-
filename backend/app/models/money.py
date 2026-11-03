"""Decimal and rounding helpers (FR-2.0a). Never use float for money or quantities."""

from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING


def to_decimal(value):
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def round_money(value):
    return to_decimal(value).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def round_area(value):
    return to_decimal(value).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def ceil_decimal(value):
    return to_decimal(value).to_integral_value(rounding=ROUND_CEILING)
