from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user

from vedock.extensions import db
from vedock.models import User


bp = Blueprint("auth", __name__, url_prefix="/auth")


def _safe_next(target: str | None) -> bool:
    if not target:
        return False
    host = urlparse(request.host_url)
    candidate = urlparse(urljoin(request.host_url, target))
    return candidate.scheme in {"http", "https"} and host.netloc == candidate.netloc


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirmation = request.form.get("password_confirmation", "")
        errors = []
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", username):
            errors.append("Username must be 3–64 letters, numbers, underscores, or hyphens.")
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            errors.append("Enter a valid email address.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password != confirmation:
            errors.append("Password confirmation does not match.")
        if User.query.filter(db.or_(User.username == username, User.email == email)).first():
            errors.append("That username or email is already registered.")
        if errors:
            for error in errors:
                flash(error, "error")
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Your workspace is ready.", "success")
            return redirect(url_for("main.dashboard"))
    return render_template("auth/register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        identity = request.form.get("identity", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter(db.or_(User.username == identity, User.email == identity.lower())).first()
        if user and user.check_password(password):
            login_user(user, remember=bool(request.form.get("remember")))
            destination = request.args.get("next")
            flash("Welcome back.", "success")
            return redirect(destination if _safe_next(destination) else url_for("main.dashboard"))
        flash("The username/email or password is incorrect.", "error")
    return render_template("auth/login.html")


@bp.post("/logout")
def logout():
    logout_user()
    flash("You have signed out.", "info")
    return redirect(url_for("main.landing"))
