from flask import Blueprint, jsonify, request

from backend.app.config import SESSION_COOKIE_NAME
from backend.app.services import metrics
from backend.app.services.orchestrator import handle_turn
from backend.app.views.errors import error_response

chat_bp = Blueprint('chat_bp', __name__)


@chat_bp.route('/api/v1/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or data.get('prompt') or '').strip()
    if not message:
        return error_response('VALIDATION_ERROR', 'Message is required.')

    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    result = handle_turn(session_id, message)
    metrics.increment('chat_turns')
    if result.get('guardrail', {}).get('triggered'):
        metrics.increment('guardrail_triggered')

    response = jsonify({
        'reply': result['reply'],
        'response': result['reply'],  # kept for the existing frontend key
        'requirements': result['requirements'],
        'recommendations': result['recommendations'],
        'missing_information': result['missing_information'],
        'clarifying_questions': result['clarifying_questions'],
        'quote': result.get('quote'),
        'stage': result.get('stage'),
        'guardrail': result.get('guardrail', {'triggered': False}),
    })
    response.set_cookie(
        SESSION_COOKIE_NAME,
        result['session_id'],
        httponly=True,
        samesite='Lax',
    )
    return response
