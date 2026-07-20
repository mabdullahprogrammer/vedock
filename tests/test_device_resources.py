from __future__ import annotations

from pathlib import Path
import json
import zipfile

import pytest

from vedock.extensions import db
from vedock.models import DatasetVersion, DeviceResource, Job, ModelRecord, User
from vedock.services.jobs import JobError, delete_job, resume_job
from vedock.services.remote_jobs import claim_job, job_manifest
from vedock_cli.resources import prepare_dataset_file


def _connect(client, device_id: str = "owner-device"):
    response = client.post(
        "/api/v1/devices/connect",
        json={"device_id": device_id, "device_name": "Owner laptop", "details": {"platform": "Windows 11"}},
    )
    assert response.status_code == 200


def test_web_path_is_relayed_only_to_matching_device_and_becomes_opaque(registered_client, app):
    _connect(registered_client)
    created = registered_client.post(
        "/api/v1/device-resources/requests",
        json={
            "device_id": "owner-device",
            "kind": "model",
            "path": r"D:\private\models\assistant-one",
            "name": "Assistant One",
            "runtime": "transformers_text",
            "task_type": "causal_lm",
        },
    )
    assert created.status_code == 200
    resource_id = created.get_json()["data"]["id"]
    assert "private" not in str(created.get_json()["data"])

    public_metadata = registered_client.get("/api/v1/device-resources").get_json()["data"]
    assert all("requested_path" not in record for record in public_metadata)
    pending = registered_client.get(
        "/api/v1/device-resources/requests?device_id=owner-device",
        headers={"X-Vedock-Device": "owner-device"},
    ).get_json()["data"]
    assert pending[0]["requested_path"] == r"D:\private\models\assistant-one"

    verified = registered_client.post(
        f"/api/v1/device-resources/{resource_id}/verify",
        json={
            "device_id": "owner-device",
            "kind": "model",
            "status": "ready",
            "name": "Assistant One",
            "path_hint": "assistant-one",
            "runtime": "transformers_text",
            "task_type": "causal_lm",
            "size_bytes": 42,
            "sha256": "a" * 64,
            "metadata": {"validated_on_device": True},
        },
    )
    assert verified.status_code == 200
    with app.app_context():
        resource = db.session.get(DeviceResource, resource_id)
        model = ModelRecord.query.filter_by(source_path=f"device://{resource_id}").one()
        slug = model.slug
        assert resource.pending_locator is None
        assert model.source_type == "device_local"
        assert model.versions[0].status == "available"
    run = registered_client.post(f"/api/v1/models/{slug}/run", json={"inputs": {"prompt": "Hello"}, "parameters": {}})
    assert run.status_code == 409
    assert run.get_json()["error"]["code"] == "connected_device_required"


def test_connected_dataset_metadata_creates_compatible_version_without_hosted_file(registered_client, app):
    _connect(registered_client)
    response = registered_client.post(
        "/api/v1/device-resources",
        json={
            "device_id": "owner-device",
            "kind": "dataset",
            "status": "ready",
            "name": "Private prompts",
            "path_hint": "prompts.csv",
            "output_schema": "prompt_response",
            "size_bytes": 100,
            "sha256": "b" * 64,
            "metadata": {
                "file_format": "jsonl",
                "columns": ["prompt", "response"],
                "row_count": 12,
                "validation_status": "valid",
                "field_mapping": {"prompt": "prompt", "response": "response"},
            },
        },
    )
    assert response.status_code == 200
    resource_id = response.get_json()["data"]["id"]
    with app.app_context():
        version = DatasetVersion.query.filter_by(storage_path=f"device://{resource_id}/processed").one()
        assert version.output_format == "prompt_response"
        assert version.row_count == 12
        assert not Path(version.storage_path).exists()
    studio = registered_client.get("/create-model?task=causal_lm")
    assert studio.status_code == 200
    assert b"1 compatible" in studio.data
    assert b"Private prompts" in studio.data
    assert b"WHAT THIS MODEL ACCEPTS" in studio.data


def test_local_model_form_never_checks_the_host_filesystem(registered_client, app):
    app.config["NODE_MODE"] = "hosted_inference"
    _connect(registered_client)
    response = registered_client.post(
        "/create-model?task=causal_lm",
        data={
            "task_type": "causal_lm",
            "build_mode": "inference_only",
            "source_type": "local",
            "device_id": "owner-device",
            "local_path": r"D:\a-folder-that-does-not-exist-on-the-server\my-model",
            "base_model_name": "My local model",
            "project_name": "Local inference",
            "output_model_name": "local-output",
        },
    )
    assert response.status_code == 302
    with app.app_context():
        model = ModelRecord.query.filter_by(name="My local model").one()
        assert model.source_path.startswith("device://")
        assert "D:\\" not in model.source_path


def test_terminal_job_can_resume_then_be_cancelled_and_deleted(app, tmp_path):
    with app.app_context():
        app.config["NODE_MODE"] = "hosted_inference"
        owner = User(username="job-owner", email="jobs@example.com")
        owner.set_password("password123")
        job = Job(owner=owner, job_type="training", status="cancelled", current_stage="cancelled", config_json={}, logs_path=str(tmp_path / "job.log"))
        db.session.add_all([owner, job])
        db.session.commit()
        # Empty training configuration is intentionally not resumable.
        try:
            resume_job(job, owner)
        except ValueError as exc:
            assert "base model" in str(exc)
        job.job_type = "dataset_inspection"
        resumed = resume_job(job, owner)
        assert resumed.status == "awaiting_device"
        resumed.status = "cancelled"
        db.session.commit()
        deleted_id = delete_job(resumed, owner)
        assert db.session.get(Job, deleted_id) is None


def test_device_job_manifest_never_requires_private_artifact_download(registered_client, app):
    _connect(registered_client, "device-a")
    model_resource = registered_client.post(
        "/api/v1/device-resources",
        json={"device_id": "device-a", "kind": "model", "status": "ready", "name": "Local base", "path_hint": "base", "runtime": "transformers_text", "task_type": "causal_lm", "size_bytes": 20, "sha256": "c" * 64},
    ).get_json()["data"]
    dataset_resource = registered_client.post(
        "/api/v1/device-resources",
        json={"device_id": "device-a", "kind": "dataset", "status": "ready", "name": "Local data", "path_hint": "data.jsonl", "output_schema": "prompt_response", "size_bytes": 30, "sha256": "d" * 64, "metadata": {"row_count": 2, "validation_status": "valid"}},
    ).get_json()["data"]
    with app.app_context():
        owner = User.query.filter_by(username="tester").one()
        model = ModelRecord.query.filter_by(source_path=f"device://{model_resource['id']}").one()
        dataset = DatasetVersion.query.filter_by(storage_path=f"device://{dataset_resource['id']}/processed").one()
        job = Job(
            owner=owner,
            job_type="training",
            status="awaiting_device",
            current_stage="waiting_for_device",
            logs_path=str(Path(app.config["STORAGE_ROOT"]) / "job-device.jsonl"),
            config_json={"model_id": model.id, "base_model_version_id": model.versions[0].id, "dataset_version_id": dataset.id, "runtime": model.runtime_key, "task_type": model.task_type, "parameters": {}},
        )
        db.session.add(job)
        db.session.commit()
        manifest = job_manifest(job)
        assert manifest["model"]["artifact_required"] is False
        assert manifest["model"]["device_resource_id"] == model_resource["id"]
        assert manifest["dataset"]["device_resource_id"] == dataset_resource["id"]
        with pytest.raises(JobError, match="another connected device"):
            claim_job(job, "device-b", "Wrong device")
        assert claim_job(job, "device-a", "Owner device").status == "claimed"


def test_local_dataset_preparation_is_immutable_and_supports_image_zip(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    source = tmp_path / "prompts.csv"
    source.write_text("prompt,response\nHello,World\nHello,World\n,Missing\n", encoding="utf-8")
    before = source.read_bytes()
    prepared, metadata = prepare_dataset_file(str(source), "prompt_response")
    assert source.read_bytes() == before
    assert prepared.is_file()
    assert prepared.parent != source.parent
    assert metadata["metadata"]["row_count"] == 1
    assert json.loads(prepared.read_text(encoding="utf-8").strip()) == {"prompt": "Hello", "response": "World"}

    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("cats/one.png", b"not-decoded-during-preparation")
        bundle.writestr("dogs/two.jpg", b"not-decoded-during-preparation")
    image_data, image_metadata = prepare_dataset_file(str(archive), "image_classification")
    rows = [json.loads(line) for line in image_data.read_text(encoding="utf-8").splitlines()]
    assert {row["label"] for row in rows} == {"cats", "dogs"}
    assert all(Path(row["image"]).is_file() for row in rows)
    assert image_metadata["metadata"]["row_count"] == 2
