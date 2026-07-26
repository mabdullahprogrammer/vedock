from __future__ import annotations

from vedock.extensions import db
from vedock.models import Job, User


def test_configured_admin_can_view_accounts_jobs_errors_and_logs(registered_client, app, tmp_path):
    with app.app_context():
        owner = User.query.filter_by(username="tester").one()
        job = Job(
            owner=owner,
            job_type="training",
            status="failed",
            progress=12,
            current_stage="failed",
            config_json={},
            logs_path=str(tmp_path / "admin-job.jsonl"),
            error_message="Synthetic test failure",
        )
        db.session.add(job)
        db.session.commit()

    # Record a privacy-bounded 404 diagnostic through the normal error handler.
    assert registered_client.get("/admin-test-missing").status_code == 404

    for tab, expected in [
        ("overview", b"Vedock operations"),
        ("users", b"tester@example.com"),
        ("jobs", b"Synthetic test failure"),
        ("errors", b"/admin-test-missing"),
        ("logs", b"request id="),
    ]:
        response = registered_client.get(f"/admin?tab={tab}")
        assert response.status_code == 200
        assert expected in response.data
        assert b"password_hash" not in response.data
        assert b"token_hash" not in response.data


def test_non_admin_account_is_forbidden(client, app):
    with app.app_context():
        user = User(username="ordinary-member", email="member@example.com")
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
    login = client.post("/auth/login", data={"identity": "ordinary-member", "password": "password123"})
    assert login.status_code == 302
    response = client.get("/admin")
    assert response.status_code == 403
    assert b"Administrator access is required" in response.data
