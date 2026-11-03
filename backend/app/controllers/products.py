import os
import re

from flask import Blueprint, current_app, jsonify, url_for

from backend.app.models.catalog import get_catalog, get_product_by_sku
from backend.app.views.errors import error_response

products_bp = Blueprint('products_bp', __name__)

# CN001.jpg = flat texture (tiled onto the room surfaces);
# CN001-room.jpg = lifestyle shot (cards, side-by-side, chat picks).
_IMAGE_NAME = re.compile(r'^(CN\d{3})(-room)?\.(jpe?g|png|webp)$', re.I)


@products_bp.route('/api/v1/products/<sku>', methods=['GET'])
def get_product(sku):
    product = get_product_by_sku(sku)
    if not product:
        return error_response('SKU_NOT_FOUND', 'Unknown SKU.')
    return jsonify(product)


@products_bp.route('/api/v1/products', methods=['GET'])
def list_products():
    return jsonify(get_catalog())


@products_bp.route('/api/v1/product-images', methods=['GET'])
def list_product_images():
    """Which product photos exist on disk, so the UI only asks for real files
    and falls back to generated textures for the rest."""
    folder = os.path.join(current_app.static_folder, 'img', 'products')
    images = {}
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            match = _IMAGE_NAME.match(name)
            if not match:
                continue
            sku = match.group(1).upper()
            kind = 'room' if match.group(2) else 'texture'
            images.setdefault(sku, {})[kind] = url_for('static', filename=f'img/products/{name}')
    return jsonify(images)
