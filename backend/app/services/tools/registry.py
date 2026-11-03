"""Whitelisted tools the orchestrator / LLM may call (G3 tool guard, FR-4.6)."""

from backend.app.services.tools.calculator import calculate_tile_quantity
from backend.app.services.tools.inventory import check_inventory
from backend.app.services.tools.quotation import generate_quote

TOOL_REGISTRY = {
    'calculate_tile_quantity': calculate_tile_quantity,
    'check_inventory': check_inventory,
    'generate_quote': generate_quote,
}

# Arguments an untrusted caller (LLM) must never be allowed to supply directly.
FORBIDDEN_ARGS = {'price_per_m2', 'owner_id', 'approved', 'status', 'stock_m2'}


def is_whitelisted(tool_name):
    return tool_name in TOOL_REGISTRY


def call_tool(tool_name, arguments):
    if not is_whitelisted(tool_name):
        raise ValueError('TOOL_NOT_WHITELISTED')
    forbidden = FORBIDDEN_ARGS & set(arguments.keys())
    if forbidden:
        raise ValueError(f'FORBIDDEN_ARGS:{",".join(sorted(forbidden))}')
    return TOOL_REGISTRY[tool_name](**arguments)
