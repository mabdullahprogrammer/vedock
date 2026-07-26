from __future__ import annotations

import shutil
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from flask import Blueprint, abort, current_app, render_template, request
from flask_login import current_user, login_required

from vedock.models import ApiToken, ConnectedDevice, Job, ModelRecord, RawDataset, User
from vedock.services.jobs import read_job_logs
from vedock.services.operations import read_application_log, read_error_events


bp = Blueprint("admin", __name__, url_prefix="/admin")
F = TypeVar("F", bound=Callable[..., Any])


def _admin_names() -> set[str]:
    configured = current_app.config.get("ADMIN_USERNAMES") or ()
    if isinstance(configured, str):
        configured = tuple(item.strip().lower() for item in configured.split(",") if item.strip())
    return {str(item).strip().lower() for item in configured if str(item).strip()}


def is_admin_user(user: Any) -> bool:
    return bool(
        getattr(user, "is_authenticated", False)
        and str(getattr(user, "username", "")).lower() in _admin_names()
    )


def admin_required(function: F) -> F:
    @wraps(function)
    @login_required
    def wrapper(*args: Any, **kwargs: Any):
        if not is_admin_user(current_user):
            abort(403, description="Administrator access is required.")
        return function(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


@bp.get("")
@bp.get("/")
@admin_required
def index():
    tab = request.args.get("tab", "overview")
    if tab not in {"overview", "users", "jobs", "errors", "logs"}:
        tab = "overview"

    users = User.query.order_by(User.created_at.desc()).limit(200).all()
    recent_jobs = Job.query.order_by(Job.created_at.desc()).limit(100).all()
    errors = read_error_events(250)
    admin_names = _admin_names()
    user_rows = [
        {
            "user": user,
            "admin": user.username.lower() in admin_names,
            "models": ModelRecord.query.filter_by(owner_id=user.id).count(),
            "datasets": RawDataset.query.filter_by(owner_id=user.id).count(),
            "jobs": Job.query.filter_by(owner_id=user.id).count(),
            "devices": ConnectedDevice.query.filter_by(owner_id=user.id).count(),
            "tokens": ApiToken.query.filter_by(user_id=user.id, revoked_at=None).count(),
        }
        for user in users
    ]
    job_rows = []
    for job in recent_jobs:
        entries = read_job_logs(job, limit=8)
        job_rows.append({"job": job, "owner": job.owner, "logs": entries, "last_log": entries[-1] if entries else None})

    storage = Path(current_app.config["STORAGE_ROOT"])
    disk = shutil.disk_usage(storage)
    counts = {
        "users": User.query.count(),
        "models": ModelRecord.query.count(),
        "datasets": RawDataset.query.count(),
        "jobs": Job.query.count(),
        "failed_jobs": Job.query.filter_by(status="failed").count(),
        "connected_devices": ConnectedDevice.query.count(),
        "recent_errors": len(errors),
    }
    return render_template(
        "admin/index.html",
        tab=tab,
        counts=counts,
        user_rows=user_rows,
        job_rows=job_rows,
        errors=errors,
        application_log=read_application_log(500),
        disk={"free": disk.free, "total": disk.total},
    )
