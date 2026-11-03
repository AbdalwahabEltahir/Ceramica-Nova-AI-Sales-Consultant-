from flask import Blueprint, jsonify

from backend.app.services import metrics as metrics_service

health_bp = Blueprint('health_bp', __name__)


@health_bp.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'ceramica-nova-ai'})


@health_bp.route('/metrics')
def metrics():
    return jsonify(metrics_service.snapshot())
