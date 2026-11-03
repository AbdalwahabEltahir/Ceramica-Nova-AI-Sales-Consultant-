from flask import Blueprint, jsonify, request

from backend.app.services.tools.calculator import calculate_tile_quantity
from backend.app.services.tools.inventory import check_inventory
from backend.app.views.errors import error_response

tools_bp = Blueprint('tools_bp', __name__)


@tools_bp.route('/api/v1/tools/calculate-quantity', methods=['POST'])
def calculate_quantity():
    data = request.get_json(silent=True) or {}
    try:
        result = calculate_tile_quantity(
            area_m2=data.get('area_m2'),
            sku=data.get('sku'),
            layout=data.get('layout', 'simple'),
            wastage_percent=data.get('wastage_percent'),
            tile_width_cm=data.get('tile_width_cm'),
            tile_length_cm=data.get('tile_length_cm'),
            tiles_per_box=data.get('tiles_per_box'),
        )
        return jsonify(result)
    except ValueError as exc:
        return error_response(str(exc))


@tools_bp.route('/api/v1/tools/check-inventory', methods=['POST'])
def inventory_check():
    data = request.get_json(silent=True) or {}
    try:
        result = check_inventory(data.get('sku'), data.get('required_area_m2'))
        return jsonify(result)
    except ValueError as exc:
        return error_response(str(exc))
