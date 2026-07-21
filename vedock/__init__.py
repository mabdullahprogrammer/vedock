from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from flask import Flask, abort, g, jsonify, render_template, request, session
from flask_login import current_user
from sqlalchemy import event
from sqlalchemy.engine import Engine

from .config import Config
from .extensions import db, login_manager
from .models import User


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(connection: Any, _record: Any) -> None:
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
    except Exception:
        pass


def create_app(config_object: Any = None, *, register_legacy: bool = True) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)
    if config_object:
        if isinstance(config_object, dict):
            app.config.update(config_object)
        else:
            app.config.from_object(config_object)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str) -> User | None:
        try:
            return db.session.get(User, int(user_id))
        except (TypeError, ValueError):
            return None

    from .auth.routes import bp as auth_bp
    from .main.routes import bp as main_bp
    from .web.routes import bp as web_bp
    from .api.routes import bp as api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    @app.before_request
    def csrf_protect() -> None:
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return
        if app.config.get("TESTING") and not app.config.get("CSRF_IN_TESTS", False):
            return
        if request.path == "/api/v1/auth/login":
            return
        bearer = request.headers.get("Authorization", "").startswith("Bearer ")
        if request.path.startswith("/api/") and bearer:
            return
        expected = session.get("csrf_token")
        received = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not expected or not received or not secrets.compare_digest(expected, received):
            abort(400, description="The form expired or its security token is invalid. Please retry.")

    @app.before_request
    def expose_request_id() -> None:
        g.request_id = secrets.token_hex(6)

    @app.context_processor
    def inject_branding() -> dict[str, Any]:
        token = session.get("csrf_token")
        if not token:
            token = secrets.token_urlsafe(24)
            session["csrf_token"] = token
        return {
            "branding": {
                "name": app.config["APP_NAME"],
                "short_name": app.config["APP_SHORT_NAME"],
                "cli_name": app.config["CLI_NAME"],
                "tagline": app.config["APP_TAGLINE"],
            },
            "csrf_token": token,
        }

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(413)
    @app.errorhandler(422)
    @app.errorhandler(500)
    @app.errorhandler(503)
    def handle_error(error: Any):
        status = getattr(error, "code", 500) or 500
        message = getattr(error, "description", "An unexpected error occurred.")
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": {"code": status, "message": message}, "request_id": g.get("request_id")}), status
        return render_template("errors/error.html", status=status, message=message), status

    with app.app_context():
        from .services.paths import ensure_storage_layout

        ensure_storage_layout()
        db.create_all()
        from .services.database import ensure_schema_compatibility

        ensure_schema_compatibility()
        if register_legacy:
            from .services.model_registry import register_legacy_models

            register_legacy_models()

    return app
