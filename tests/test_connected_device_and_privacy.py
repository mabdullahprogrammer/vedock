from __future__ import annotations

import io

from vedock.extensions import db
from vedock.models import Conversation, Job, Message, ModelRecord, User


def test_hosted_system_hides_server_hardware_paths_and_packages(registered_client, app):
    app.config["NODE_MODE"] = "hosted_inference"
    page = registered_client.get("/system")
    assert page.status_code == 200
    assert b"Host hardware" in page.data
    assert str(app.config["STORAGE_ROOT"]).encode() not in page.data
    report = registered_client.get("/api/v1/system/doctor").get_json()["data"]
    assert report["private_details_hidden"] is True
    assert report["node"]["training_location"] == "connected_user_device"
    assert "python_executable" not in report
    assert "packages" not in report
    model = registered_client.get("/api/v1/models/storymaker-final").get_json()["data"]
    assert "loaded_model_path" not in model["capabilities"]
    assert "storage_path" not in str(model)


def test_claimed_task_can_be_released_by_its_device(registered_client, app, tmp_path):
    with app.app_context():
        owner = User.query.filter_by(username="tester").one()
        job = Job(
            owner=owner,
            job_type="training",
            status="claimed",
            progress=0,
            current_stage="claimed_by_local_device",
            config_json={},
            logs_path=str(tmp_path / "release-job.jsonl"),
            claimed_by_device="device-123",
            device_name="Test device",
        )
        db.session.add(job)
        db.session.commit()
        job_id = job.id
    released = registered_client.post(f"/api/v1/jobs/{job_id}/release", json={"device_id": "device-123"})
    assert released.status_code == 200
    assert released.get_json()["data"]["status"] == "awaiting_device"
    with app.app_context():
        saved = db.session.get(Job, job_id)
        assert saved.claimed_by_device is None
        assert saved.current_stage == "waiting_for_device"


def test_outdated_connected_client_is_stopped_before_claim(registered_client, app, tmp_path):
    with app.app_context():
        owner = User.query.filter_by(username="tester").one()
        job = Job(
            owner=owner,
            job_type="training",
            status="awaiting_device",
            progress=0,
            current_stage="waiting_for_device",
            config_json={},
            logs_path=str(tmp_path / "outdated-client.jsonl"),
        )
        db.session.add(job)
        db.session.commit()
        job_id = job.id
    response = registered_client.post(
        f"/api/v1/jobs/{job_id}/claim",
        json={"device_id": "old-device", "device_name": "Old Vedock"},
        headers={"X-Vedock-Client-Version": "2026.07.20.4"},
    )
    assert response.status_code == 426
    error = response.get_json()["error"]
    assert error["code"] == "client_update_required"
    assert error["details"]["minimum"] == app.config["MIN_CONNECTED_CLIENT_VERSION"]
    with app.app_context():
        saved = db.session.get(Job, job_id)
        assert saved.status == "awaiting_device"
        assert saved.claimed_by_device is None


def test_dataset_recommendations_and_portable_exports(registered_client):
    source = b"prompt,response\n Hello ,World\n Hello ,World\n,Missing prompt\n"
    imported = registered_client.post(
        "/api/v1/datasets/import",
        data={"file": (io.BytesIO(source), "health.csv"), "name": "Health check"},
        content_type="multipart/form-data",
    )
    dataset = imported.get_json()["data"]
    titles = {item["title"] for item in dataset["statistics"]["recommendations"]}
    assert "Remove duplicate examples" in titles
    assert "Remove incomplete examples" in titles
    transformed = registered_client.post(
        f"/api/v1/datasets/{dataset['id']}/transform",
        json={
            "operations": [
                {"type": "trim_whitespace", "config": {"fields": ["prompt", "response"]}},
                {"type": "remove_empty_records", "config": {"fields": ["prompt", "response"]}},
                {"type": "remove_duplicates", "config": {}},
            ],
            "output_schema": "prompt_response",
            "field_mapping": {"prompt": "prompt", "response": "response"},
        },
    )
    version_id = transformed.get_json()["data"]["id"]
    spreadsheet = registered_client.get(f"/dataset-versions/{version_id}/export/xlsx")
    assert spreadsheet.status_code == 200
    assert spreadsheet.data[:2] == b"PK"
    assert "spreadsheetml" in spreadsheet.mimetype
    csv_export = registered_client.get(f"/dataset-versions/{version_id}/export/csv")
    assert csv_export.status_code == 200
    assert b"prompt,response" in csv_export.data


def test_owner_can_delete_a_chat(registered_client, app):
    with app.app_context():
        owner = User.query.filter_by(username="tester").one()
        model = ModelRecord.query.filter_by(slug="storymaker-final").one()
        conversation = Conversation(owner=owner, selected_model=model, model_version=model.versions[-1], model_id=model.id, title="Delete me")
        db.session.add(conversation)
        db.session.flush()
        db.session.add(Message(conversation=conversation, role="user", content="Temporary"))
        db.session.commit()
        conversation_id = conversation.id
    response = registered_client.post(f"/conversations/{conversation_id}/delete")
    assert response.status_code == 302
    with app.app_context():
        assert db.session.get(Conversation, conversation_id) is None
