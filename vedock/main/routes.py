from __future__ import annotations

from flask import Blueprint, current_app, render_template, send_file
from pathlib import Path
from flask_login import current_user, login_required

from vedock.extensions import db
from vedock.models import Conversation, DatasetVersion, Job, ModelRecord, RawDataset
from vedock.services.hardware import system_report
from vedock.services.model_registry import fork_count, visible_models


bp = Blueprint("main", __name__)


@bp.get("/brand/logo")
def brand_logo():
    """Serve the configured Vedock logo without copying user assets into source."""
    logo = Path(current_app.config["STORAGE_ROOT"]) / "logos" / "logo.png"
    if logo.is_file():
        return send_file(logo, conditional=True, max_age=3600)
    packaged = Path(__file__).resolve().parents[2] / "vedock_cli" / "assets" / "logo.png"
    if packaged.is_file():
        return send_file(packaged, conditional=True, max_age=86400)
    return ("", 404)


@bp.get("/")
def landing():
    models = visible_models(None)[:8]
    return render_template(
        "landing.html",
        featured_models=models,
        fork_counts={model.id: fork_count(model) for model in models},
    )


@bp.get("/health")
def health():
    return {"ok": True, "service": "vedock"}


@bp.get("/dashboard")
@login_required
def dashboard():
    counts = {
        "datasets": RawDataset.query.filter_by(owner_id=current_user.id).count(),
        "versions": DatasetVersion.query.filter_by(owner_id=current_user.id).count(),
        "models": len(visible_models(current_user.id)),
        "jobs": Job.query.filter_by(owner_id=current_user.id).count(),
        "conversations": Conversation.query.filter_by(owner_id=current_user.id).count(),
    }
    recent_jobs = Job.query.filter_by(owner_id=current_user.id).order_by(Job.created_at.desc()).limit(5).all()
    models = visible_models(current_user.id)
    community_models = [model for model in models if model.visibility == "public"]
    user_models = [model for model in models if model.owner_id == current_user.id]
    return render_template("dashboard.html", counts=counts, recent_jobs=recent_jobs, models=models[:12], community_models=community_models, user_models=user_models, fork_counts={model.id: fork_count(model) for model in models})
