from __future__ import annotations

import io
from pathlib import Path

from werkzeug.datastructures import FileStorage

from vedock.extensions import db
from vedock.models import User
from vedock.services.datasets import import_upload, preview_transform, revalidate_version, save_dataset_version, transform_record


CSV = b"prompt,story\nMoon prompt,Moon story\nRobot prompt,Robot story\nRobot prompt,Robot story\n,Missing prompt\n"


def test_privacy_and_quality_transformations_are_reproducible():
    transformed = transform_record(
        {"text": "  José emailed me@example.com or +1 (555) 123-4567  ", "owner": None},
        [
            {"type": "trim_whitespace", "config": {"fields": ["text"]}},
            {"type": "strip_accents", "config": {"fields": ["text"]}},
            {"type": "redact_emails", "config": {"fields": ["text"]}},
            {"type": "redact_phone_numbers", "config": {"fields": ["text"]}},
            {"type": "fill_missing", "config": {"fields": ["owner"], "value": "unknown"}},
        ],
    )
    assert transformed == {"text": "Jose emailed [EMAIL] or [PHONE]", "owner": "unknown"}


def test_raw_upload_transform_validation_and_immutability(app):
    with app.app_context():
        user = User(username="datauser", email="data@example.com")
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
        upload = FileStorage(stream=io.BytesIO(CSV), filename="stories.csv", content_type="text/csv")
        raw = import_upload(upload, user, "Stories")
        raw_path = Path(raw.storage_path)
        original_bytes = raw_path.read_bytes()
        assert raw.row_count == 4
        assert raw.statistics_json["duplicate_count"] == 1
        operations = [
            {"type": "trim_whitespace", "config": {"fields": ["prompt", "story"]}},
            {"type": "remove_empty_records", "config": {"fields": ["prompt", "story"]}},
            {"type": "remove_duplicates", "config": {}},
        ]
        preview = preview_transform(raw, operations, "prompt_response", {"prompt": "prompt", "response": "story"})
        assert preview["rows"][0]["output"] == {"prompt": "Moon prompt", "response": "Moon story"}
        first = save_dataset_version(raw, user, operations, "prompt_response", {"prompt": "prompt", "response": "story"})
        assert first.validation_status in {"valid", "warning"}
        assert first.row_count == 2
        assert Path(first.storage_path).is_file()
        assert raw_path.read_bytes() == original_bytes
        assert revalidate_version(first)["status"] in {"valid", "warning"}
        second = save_dataset_version(raw, user, operations, "prompt_response", {"prompt": "prompt", "response": "story"}, limit=1)
        assert second.id != first.id
        assert second.storage_path != first.storage_path
        assert Path(first.storage_path).read_bytes() != b""
        assert raw_path.read_bytes() == original_bytes


def test_api_dataset_import_and_transform(registered_client):
    response = registered_client.post(
        "/api/v1/datasets/import",
        data={"file": (io.BytesIO(CSV), "stories.csv"), "name": "API Stories"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    dataset_id = response.get_json()["data"]["id"]
    transformed = registered_client.post(
        f"/api/v1/datasets/{dataset_id}/transform",
        json={
            "operations": [{"type": "remove_empty_records", "config": {"fields": ["prompt", "story"]}}, {"type": "remove_duplicates", "config": {}}],
            "output_schema": "prompt_response",
            "field_mapping": {"prompt": "prompt", "response": "story"},
        },
    )
    assert transformed.status_code == 200
    assert transformed.get_json()["data"]["row_count"] == 2


def test_uploaded_dataset_builder_renders_and_scratch_project_stays_draft(registered_client, app):
    imported = registered_client.post(
        "/api/v1/datasets/import",
        data={"file": (io.BytesIO(CSV), "builder-stories.csv"), "name": "Builder Stories"},
        content_type="multipart/form-data",
    )
    dataset_id = imported.get_json()["data"]["id"]
    builder = registered_client.get(f"/datasets/{dataset_id}/builder")
    assert builder.status_code == 200
    assert b"Choose schema and map fields" in builder.data

    transformed = registered_client.post(
        f"/api/v1/datasets/{dataset_id}/transform",
        json={
            "operations": [{"type": "remove_empty_records", "config": {"fields": ["prompt", "story"]}}],
            "output_schema": "prompt_response",
            "field_mapping": {"prompt": "prompt", "response": "story"},
        },
    )
    version_id = transformed.get_json()["data"]["id"]
    project = registered_client.post(
        "/create-model",
        data={
                "task_type": "causal_lm",
            "build_mode": "scratch",
            "source_type": "scratch",
                "scratch_preset": "tiny",
                "architecture_family": "gpt2",
            "tokenizer_repository": "gpt2",
            "base_model_name": "Tiny New Story Model",
            "project_name": "Scratch Story Project",
            "dataset_version_id": version_id,
            "output_model_name": "tiny-story-output",
        },
    )
    assert project.status_code == 302
    assert "/projects/" in project.headers["Location"]
    review = registered_client.get(project.headers["Location"])
    assert review.status_code == 200
    assert b"random" not in review.data.lower() or b"scratch" in review.data.lower()
    assert b"Delete project" in review.data
    with app.app_context():
        from vedock.models import Job, ModelProject

        saved = ModelProject.query.filter_by(name="Scratch Story Project").one()
        assert saved.status == "draft"
        assert saved.training_method == "scratch"
        assert Job.query.count() == 0
    listing = registered_client.get("/datasets")
    assert listing.status_code == 200
    assert b"Download JSONL" in listing.data
    assert b"processed" in listing.data.lower()
    deleted = registered_client.post(f"{project.headers['Location']}/delete")
    assert deleted.status_code == 302
    with app.app_context():
        assert ModelProject.query.filter_by(name="Scratch Story Project").first() is None
