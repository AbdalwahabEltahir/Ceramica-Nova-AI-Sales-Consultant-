"""G3 tool guard — FR-4.6, FR-4.6a. Schema validation + ownership checks."""

from backend.app.services.tools.registry import call_tool, is_whitelisted


def safe_call_tool(tool_name, arguments):
    if not is_whitelisted(tool_name):
        raise ValueError('TOOL_NOT_WHITELISTED')
    try:
        return call_tool(tool_name, arguments)
    except TypeError as exc:
        raise ValueError('VALIDATION_ERROR') from exc


def owner_matches(resource_owner_id, session_owner_id):
    """FR-4.6a: 'not found' and 'not yours' return the same response."""
    return resource_owner_id == session_owner_id
