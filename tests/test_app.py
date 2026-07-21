from __future__ import annotations

import inspect
from pathlib import Path

from flask import Response

from vedock.extensions import db
from vedock.models import Job, ModelProject, ModelRecord, ModelVersion, User
from vedock.runtimes import get_runtime


def test_desktop_page_renders_complete_html():
    from vedock_cli.desktop import DesktopBridge, _page

    page = _page()
    assert "Run on this computer" in page
    assert "Publish after final review" in page
    assert "vedock.ecorims.com" in page
    assert "api.dispatch(name,args)" in page
    assert "Connecting securely to this device" in page
    # PyWebView generates JavaScript formal parameters from this signature.
    # Naming one of them `arguments` shadows JavaScript's built-in arguments
    # object and makes dispatch receive no action at runtime.
    assert "arguments" not in inspect.signature(DesktopBridge.dispatch).parameters


def test_installer_folder_result_and_html_use_a_real_input_name():
    from installer.vedock_installer import InstallerBridge, installer_html

    class FolderDialog:
        def create_file_dialog(self, *_args, **_kwargs):
            return r"E:\Users\Ada\Vedock"

    bridge = InstallerBridge()
    bridge.window = FolderDialog()
    assert bridge.choose_folder() == r"E:\Users\Ada\Vedock"
    page = installer_html()
    assert 'id="installLocation"' in page
    assert "locationInput().value" in page
    assert "const installButton=()=>byId('install')" in page
    assert "install.disabled" not in page
    assert "api.dispatch(action,payload)" in page
    assert "api.choose_folder" not in page
    assert bridge.dispatch("choose_folder") == r"E:\Users\Ada\Vedock"


def test_installer_reuses_a_saved_custom_install(monkeypatch, tmp_path):
    import json
    from installer.vedock_installer import CLIENT_VERSION, InstallerBridge
    from vedock_cli import CONNECTED_CLIENT_VERSION

    assert CLIENT_VERSION == CONNECTED_CLIENT_VERSION

    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    location = tmp_path / "custom-vedock"
    executable = location / "runtime" / "Scripts" / "vedock.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"installed")
    config = tmp_path / "appdata" / "vedock" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"install_location": str(location), "client_version": CLIENT_VERSION}), encoding="utf-8")
    state = InstallerBridge().state()
    assert state["installed"] is True
    assert state["location"] == str(location.resolve())


def test_owner_can_publish_runtime_defaults(registered_client, app, tmp_path):
    model_path = tmp_path / "published-model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    with app.app_context():
        owner = User.query.filter_by(username="tester").one()
        model = ModelRecord(owner=owner, slug="publisher-defaults", name="Publisher defaults", task_type="causal_lm", runtime_key="transformers_text", source_type="training", source_path=str(model_path), visibility="public")
        model.versions.append(ModelVersion(version_number=1, label="Published", storage_path=str(model_path), status="completed"))
        db.session.add(model)
        db.session.commit()

    saved = registered_client.post(
        "/models/publisher-defaults/edit",
        data={
            "name": "Publisher defaults",
            "description": "Owner-selected starting controls.",
            "visibility": "public",
            "output_pattern": "[input]{prompt}[/input][output]{response}[/output]",
            "publisher_system_prompt": "Answer with short, direct language.",
            "publisher_temperature": "0.65",
            "publisher_max_new_tokens": "144",
            "publisher_use_history": "true",
            "publisher_context_limit": "12000",
            "publisher_allow_overrides": "true",
        },
    )
    assert saved.status_code == 302
    with app.app_context():
        version = ModelRecord.query.filter_by(slug="publisher-defaults").one().versions[-1]
        defaults = version.metadata_json["publisher_defaults"]
        assert defaults["inference_parameters"]["system_prompt"] == "Answer with short, direct language."
        assert defaults["inference_parameters"]["temperature"] == 0.65
        assert defaults["inference_parameters"]["output_pattern"].startswith("[input]")
        assert defaults["chat"] == {"use_history": True, "context_limit": 12000}

    details = registered_client.get("/api/v1/models/publisher-defaults")
    assert details.status_code == 200
    assert details.get_json()["data"]["publisher_defaults"]["inference_parameters"]["max_new_tokens"] == 144


def test_branding_is_environment_driven(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Vedock Test" in response.data
    assert b"Test all controls." in response.data


def test_favicon_and_unknown_routes_do_not_raise_500(registered_client):
    favicon = registered_client.get("/favicon.ico")
    assert favicon.status_code in {200, 204}
    missing = registered_client.get("/this-route-does-not-exist")
    assert missing.status_code == 404
    assert b"requested URL was not found" in missing.data


def test_installer_download_falls_back_when_current_file_disappears(client, app, monkeypatch, tmp_path):
    from vedock.web import routes

    distribution = tmp_path / "distribution"
    distribution.mkdir()
    current = distribution / "VedockInstaller-current.exe"
    stable = distribution / "VedockInstaller.exe"
    current.write_bytes(b"current")
    stable.write_bytes(b"stable")
    app.config["DISTRIBUTION_ROOT"] = distribution
    requested: list[str] = []

    def fake_send_file(path, **kwargs):
        requested.append(Path(path).name)
        if Path(path) == current:
            raise OSError(22, "quarantined during open")
        return Response(Path(path).read_bytes(), mimetype=kwargs.get("mimetype"))

    monkeypatch.setattr(routes, "send_file", fake_send_file)
    response = client.get("/downloads/vedock-installer.exe")
    assert response.status_code == 200
    assert response.data == b"stable"
    assert requested == [current.name, stable.name]


def test_register_login_and_protected_pages(registered_client):
    for path in ["/dashboard", "/datasets", "/models", "/playground", "/create-model", "/jobs", "/merge", "/developer", "/settings", "/system"]:
        response = registered_client.get(path)
        assert response.status_code == 200, path
    conversations = registered_client.get("/conversations")
    assert conversations.status_code == 302
    assert "/playground" in conversations.headers["Location"]


def test_legacy_models_register_without_loading(app):
    runtime = get_runtime("storymaker")
    runtime.unload_model()
    with app.app_context():
        assert ModelRecord.query.filter_by(slug="storymaker-final").one()
        assert ModelRecord.query.filter_by(slug="storymaker-finetuned").one()
        assert runtime.get_model_capabilities()["loaded_model_path"] is None


def test_api_login_and_model_list(client):
    client.post(
        "/auth/register",
        data={"username": "apiuser", "email": "api@example.com", "password": "password123", "password_confirmation": "password123"},
    )
    login = client.post("/api/v1/auth/login", json={"username": "apiuser", "password": "password123"})
    assert login.status_code == 200
    token = login.get_json()["data"]["token"]
    models = client.get("/api/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert models.status_code == 200
    assert {item["slug"] for item in models.get_json()["data"]} >= {"storymaker-final", "storymaker-finetuned"}


def test_every_template_compiles(app):
    with app.app_context():
        for template_name in app.jinja_env.list_templates():
            app.jinja_env.get_template(template_name)


def test_catalog_model_can_be_registered_without_loading_or_training(registered_client, app):
    response = registered_client.post(
        "/create-model",
        data={
            "task_type": "causal_lm",
            "build_mode": "inference_only",
            "source_type": "catalog",
            "catalog_model": "gpt2-medium",
            "revision": "main",
            "base_model_name": "My GPT-2 Medium",
            "project_name": "Imported model",
        },
    )
    assert response.status_code == 302
    with app.app_context():
        model = ModelRecord.query.filter_by(name="My GPT-2 Medium").one()
        assert model.source_type == "community_online"
        assert model.versions[0].storage_path.startswith("hf://gpt2-medium")
        assert Job.query.count() == 0
        assert ModelProject.query.count() == 0


def test_default_chat_post_is_allowed_and_labels_are_capability_neutral(registered_client):
    response = registered_client.post("/playground", data={"prompt": ""})
    assert response.status_code == 200
    assert b"Method Not Allowed" not in response.data
    assert b"Generate story" not in response.data
    assert b">Send<" in response.data
    studio = registered_client.get("/create-model")
    assert b"Causal language model" in studio.data
    assert b"story_generation" not in studio.data
    assert b'name="learning_rate"' in studio.data
    assert b'step="any"' in studio.data


def test_streaming_chat_uses_api_and_persists_context_without_reloading(monkeypatch, registered_client, app):
    runtime = get_runtime("storymaker")
    calls = []

    def fake_stream(model_path, prompt, parameters):
        calls.append((model_path, prompt, parameters))
        yield "Hello"
        yield " there"

    monkeypatch.setattr(runtime, "stream_infer", fake_stream)
    response = registered_client.post(
        "/api/v1/models/storymaker-final/stream",
        json={"prompt": "First message", "parameters": {}},
    )
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Hello" in body and "event: done" in body
    assert len(calls) == 1
    with app.app_context():
        from vedock.models import Conversation

        conversation = Conversation.query.one()
        conversation_id = conversation.id
        assert [message.role for message in conversation.messages] == ["user", "assistant"]
        assert conversation.messages[-1].content == "Hello there"
    follow_up = registered_client.post(
        "/api/v1/models/storymaker-final/stream",
        json={"prompt": "Second message", "conversation_id": conversation_id, "parameters": {}},
    )
    assert follow_up.status_code == 200
    follow_up.get_data()
    assert len(calls) == 2
    assert "First message" in calls[1][1]
    assert "Hello there" in calls[1][1]
    assert "Second message" in calls[1][1]


def test_model_training_is_enabled_but_never_starts_without_complete_user_submission(registered_client, app):
    assert app.config["MODEL_TRAINING_ENABLED"] is True
    response = registered_client.post("/api/v1/train/storymaker-final", json={})
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"
