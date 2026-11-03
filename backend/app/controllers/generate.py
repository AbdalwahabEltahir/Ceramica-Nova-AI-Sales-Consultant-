"""POST /api/v1/generate — the exam B1 production LLM gateway."""

import logging

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from backend.app.models.schemas import GenerateRequest
from backend.app.services import metrics
from backend.app.services.generate_service import generate
from backend.app.services.rate_limiter import generate_limiter
from backend.app.views.errors import error_response

generate_bp = Blueprint('generate_bp', __name__)
logger = logging.getLogger('generate_controller')


@generate_bp.route('/api/v1/generate', methods=['POST'])
def generate_endpoint():
    data = request.get_json(silent=True) or {}

    try:
        req = GenerateRequest.model_validate(data)
    except ValidationError as exc:
        metrics.increment('generate_validation_error')
        return error_response('VALIDATION_ERROR', str(exc))

    if not generate_limiter.allow(req.user_id):
        metrics.increment('generate_rate_limited')
        return error_response('RATE_LIMITED', 'Too many requests, try again shortly.')

    metrics.increment('generate_requests')
    response, debug = generate(req.user_id, req.task, req.message, req.language)

    # Structured logging (B1 'logging', B5 observability) — no message content
    # or secrets, just what's needed to debug/cost-track a call.
    logger.info('generate task=%s model=%s attempts=%s latency_ms=%.1f',
                req.task, debug.get('model'), debug.get('attempts'), debug.get('latency_ms', 0))

    body = response.model_dump()
    body['_meta'] = {
        'model': debug.get('model'),
        'attempts': debug.get('attempts'),
        'latency_ms': round(debug.get('latency_ms', 0), 1),
        'tokens_in': debug.get('tokens_in', 0),
        'tokens_out': debug.get('tokens_out', 0),
    }
    return jsonify(body)
