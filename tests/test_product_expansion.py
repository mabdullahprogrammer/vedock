from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

from PIL import Image

from vedock.extensions import db
from vedock.models import DatasetVersion, ModelFork, ModelRecord, ModelWorkspaceState, RawDataset
from vedock.runtimes.registry import get_runtime


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (12, 10), color).save(output, format="PNG")
    return output.getvalue()


def test_chat_exposes_stop_and_saved_storymaker_pattern(registered_client):
    response = registered_client.get("/playground/storymaker-final")
    assert response.status_code == 200
    assert b"data-stop-button" in response.data
    assert b"output_pattern" in response.data
    assert b"&lt;|start_of_input|&gt;" in response.data


def test_stream_stop_endpoint_dispatches_generation_id(monkeypatch, registered_client):
    runtime = get_runtime("storymaker")
    calls = []
    monkeypatch.setattr(runtime, "cancel", lambda generation_id: calls.append(generation_id) or True)
    response = registered_client.post(
        "/api/v1/models/storymaker-final/stop",
        json={"generation_id": "generation-123"},
    )
    assert response.status_code == 200
    assert response.get_json()["data"]["stopping"] is True
    assert calls == ["generation-123"]


def test_real_fast_task_schemas_are_selectable_without_training(registered_client):
    pattern = registered_client.get("/create-model?task=pattern_sequence")
    assert pattern.status_code == 200
    assert b"Pattern sequence model" in pattern.data
    assert b'name="order"' in pattern.data
    image = registered_client.get("/create-model?task=image_classification")
    assert image.status_code == 200
    assert b"Image classification" in image.data
    assert b'name="algorithm"' in image.data
    assert b"MODEL_TRAINING_ENABLED" not in image.data


def test_image_folder_zip_inspects_and_saves_immutable_classification_version(registered_client, app):
    archive = BytesIO()
    with ZipFile(archive, "w") as output:
        output.writestr("cats/one.png", _image_bytes((255, 0, 0)))
        output.writestr("cats/two.png", _image_bytes((200, 0, 0)))
        output.writestr("dogs/one.png", _image_bytes((0, 0, 255)))
        output.writestr("dogs/two.png", _image_bytes((0, 0, 200)))
    archive.seek(0)
    imported = registered_client.post(
        "/datasets",
        data={"source_type": "upload", "name": "Image folders", "file": (archive, "images.zip")},
        content_type="multipart/form-data",
    )
    assert imported.status_code == 302
    with app.app_context():
        dataset = RawDataset.query.filter_by(name="Image folders").one()
        dataset_id = dataset.id
        assert dataset.file_format == "zip"
        assert dataset.row_count == 4
        assert (dataset.detected_schema_json or {})["columns"] == ["image", "label"]
    saved = registered_client.post(
        f"/datasets/{dataset_id}/builder",
        data={"action": "save", "output_schema": "image_classification", "map_image": "image", "map_label": "label", "shuffle_seed": "42"},
    )
    assert saved.status_code == 302
    with app.app_context():
        version = DatasetVersion.query.filter_by(raw_dataset_id=dataset_id).one()
        assert version.output_format == "image_classification"
        assert version.row_count == 4
        assert version.validation_status == "warning"
        assert version.validation_json["warnings"][0]["code"] == "small_dataset"
        assert version.storage_path != db.session.get(RawDataset, dataset_id).storage_path


def test_public_landing_is_an_honest_product_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"MODEL STUDIO" in response.data
    assert b"The web coordinates" in response.data
    assert b"StoryMaker Final" not in response.data


def test_model_fork_edit_and_recoverable_remove_use_api(registered_client, app):
    forked = registered_client.post("/api/v1/models/storymaker-final/fork")
    assert forked.status_code == 200
    slug = forked.get_json()["data"]["slug"]
    updated = registered_client.patch(
        f"/api/v1/models/{slug}",
        json={"name": "Editable local fork", "output_pattern": "[input]{prompt}[/input][output]{response}[/output]"},
    )
    assert updated.status_code == 200
    removed = registered_client.delete(f"/api/v1/models/{slug}")
    assert removed.status_code == 200
    with app.app_context():
        model = ModelRecord.query.filter_by(slug=slug).one()
        assert model.name == "Editable local fork"
        assert ModelFork.query.filter_by(child_model_id=model.id).one()
        assert ModelWorkspaceState.query.filter_by(model_id=model.id, archived=True).one()


def test_one_file_installer_and_internal_node_payload_are_installable(client):
    installer = client.get("/downloads/vedock-installer.exe")
    assert installer.status_code == 200
    assert installer.data[:2] == b"MZ"
    assert len(installer.data) > 1_000_000
    node = client.get("/downloads/vedock-node.zip")
    assert node.status_code == 200
    with ZipFile(BytesIO(node.data)) as archive:
        names = set(archive.namelist())
        assert "vedock-node/serve.py" in names
        assert "vedock-node/scripts/setup-portable.ps1" in names
        assert "vedock-node/INSTALL.txt" in names
        assert "vedock-node/requirements-core.txt" in names
        assert "vedock-node/requirements-text.txt" in names

    connected = client.get("/downloads/vedock-client.zip")
    assert connected.status_code == 200
    with ZipFile(BytesIO(connected.data)) as archive:
        names = set(archive.namelist())
        assert "vedock-client/requirements-client.txt" in names
        assert "vedock-client/requirements-local-core.txt" in names
        assert "vedock-client/vedock_cli/desktop.py" in names
        assert "vedock-client/vedock_cli/assets/logo.png" in names


def test_public_models_are_discoverable_but_only_remixes_are_editable(registered_client, app):
    first_fork = registered_client.post("/api/v1/models/storymaker-final/fork")
    slug = first_fork.get_json()["data"]["slug"]
    published = registered_client.patch(f"/api/v1/models/{slug}", json={"visibility": "public", "name": "Public remix"})
    assert published.status_code == 200
    assert published.get_json()["data"]["visibility"] == "public"
    registered_client.post("/auth/logout")
    second = registered_client.post(
        "/auth/register",
        data={"username": "second", "email": "second@example.com", "password": "password123", "password_confirmation": "password123"},
    )
    assert second.status_code == 302
    listed = registered_client.get("/api/v1/models").get_json()["data"]
    assert slug in {item["slug"] for item in listed}
    blocked = registered_client.patch(f"/api/v1/models/{slug}", json={"name": "Stolen edit"})
    assert blocked.status_code == 403
    remixed = registered_client.post(f"/api/v1/models/{slug}/fork")
    assert remixed.status_code == 200
    with app.app_context():
        source = ModelRecord.query.filter_by(slug=slug).one()
        assert len({fork.owner_id for fork in source.direct_forks}) == 1
        assert source.direct_forks[0].source_model_id == source.id


def test_model_cover_is_generated_and_model_aware_dataset_link_is_explained(registered_client):
    cover = registered_client.get("/model-media/storymaker-final/cover")
    assert cover.status_code == 200
    assert cover.mimetype == "image/svg+xml"
    datasets = registered_client.get("/datasets?for_model=storymaker-final")
    assert datasets.status_code == 200
    assert b"Modify data for StoryMaker Final" in datasets.data
