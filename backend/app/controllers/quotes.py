from flask import Blueprint, jsonify, request

from backend.app.config import SESSION_COOKIE_NAME
from backend.app.models.session_store import quote_store, session_store
from backend.app.services.guardrails.tool_guard import owner_matches
from backend.app.services.tools.quotation import approve_quote, generate_quote
from backend.app.views.errors import error_response

quotes_bp = Blueprint('quotes_bp', __name__)


def _resolve_session_id():
    """Owner id always comes from a real session, never a shared default
    (a missing cookie used to collapse every anonymous visitor onto one
    'guest' owner, letting them read each other's quotes)."""
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    record = session_store.get_or_create(existing)
    session_store.save(record)
    return record.session_id


def _set_session_cookie(response, session_id):
    response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True, samesite='Lax')
    return response


@quotes_bp.route('/api/v1/quotes', methods=['POST'])
def create_quote():
    data = request.get_json(silent=True) or {}
    session_id = _resolve_session_id()
    try:
        quote = generate_quote(
            data.get('lines', []),
            data.get('requested_discount_percent', 0),
            owner_id=session_id,
        )
        return _set_session_cookie(jsonify(quote), session_id)
    except ValueError as exc:
        body, status = error_response(str(exc))
        return _set_session_cookie(body, session_id), status


@quotes_bp.route('/api/v1/quotes/<quotation_id>', methods=['GET'])
def get_quote(quotation_id):
    session_id = _resolve_session_id()
    quote = quote_store.get(quotation_id)
    # Same response whether the quote is missing or belongs to someone else (FR-4.6a).
    if not quote or not owner_matches(quote['owner_id'], session_id):
        body, status = error_response('NOT_FOUND', 'Quote not found.')
        return _set_session_cookie(body, session_id), status
    return _set_session_cookie(jsonify(quote), session_id)


@quotes_bp.route('/api/v1/quotes/<quotation_id>/approve', methods=['POST'])
def approve(quotation_id):
    # Staff-only in production; this MVP has no staff auth yet (see SRS section 15).
    try:
        quote = approve_quote(quotation_id)
        return jsonify(quote)
    except ValueError as exc:
        return error_response(str(exc))
