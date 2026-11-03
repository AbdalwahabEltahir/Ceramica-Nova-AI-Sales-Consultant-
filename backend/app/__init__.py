import os

from flask import Flask


def create_app():
    app = Flask(__name__, template_folder='../../frontend/templates', static_folder='../../frontend/static')
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

    from backend.app.controllers.pages import pages_bp
    from backend.app.controllers.health import health_bp
    from backend.app.controllers.chat import chat_bp
    from backend.app.controllers.tools import tools_bp
    from backend.app.controllers.quotes import quotes_bp
    from backend.app.controllers.products import products_bp
    from backend.app.controllers.generate import generate_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(tools_bp)
    app.register_blueprint(quotes_bp)
    app.register_blueprint(products_bp)
    app.register_blueprint(generate_bp)

    return app
