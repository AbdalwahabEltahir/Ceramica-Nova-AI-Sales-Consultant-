"""FR-5.4 error model. Every error returns the same shape; never a stack trace."""

import uuid

from flask import jsonify

ERROR_HTTP_STATUS = {
    'VALIDATION_ERROR': (422, False),
    'SKU_NOT_FOUND': (404, False),
    'NOT_FOUND': (404, False),
    'CURRENCY_MISMATCH': (422, False),
    'RATE_LIMITED': (429, True),
    'TOOL_NOT_WHITELISTED': (403, False),
    'FORBIDDEN_ARGS': (403, False),
    'INTERNAL_ERROR': (500, False),
}


def error_response(code, message=None, request_id=None):
    status, retryable = ERROR_HTTP_STATUS.get(code, (400, False))
    body = {
        'error': {
            'code': code,
            'message': message or code.replace('_', ' ').title(),
            'request_id': request_id or str(uuid.uuid4()),
            'retryable': retryable,
        }
    }
    return jsonify(body), status
