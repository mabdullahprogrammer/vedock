from __future__ import annotations

from pathlib import Path

import pytest

from vedock import create_app
from vedock.extensions import db


LEGACY_ROOT = Path(r"D:\LLM\new-llm\LLM-2025\StoryMaker")


@pytest.fixture()
def app(tmp_path: Path):
    application = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "STORAGE_ROOT": tmp_path / "storage",
            "STORYMAKER_ROOT": LEGACY_ROOT,
            "STORYMAKER_FINAL_PATH": LEGACY_ROOT / "gpt-storygen-final",
            "STORYMAKER_FINETUNED_PATH": LEGACY_ROOT / "gpt2fintuned_storymaker",
            "PROTECTED_ROOTS": (Path(r"D:\LLM\StoryMaker"), LEGACY_ROOT),
            "LAUNCH_JOBS": False,
            "NODE_MODE": "local_compute",
            "APP_NAME": "Vedock Test",
            "APP_SHORT_NAME": "VTest",
            "CLI_NAME": "vtest",
            "APP_TAGLINE": "Test all controls.",
        }
    )
    yield application
    with application.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def registered_client(client):
    response = client.post(
        "/auth/register",
        data={
            "username": "tester",
            "email": "tester@example.com",
            "password": "password123",
            "password_confirmation": "password123",
        },
    )
    assert response.status_code == 302
    return client
