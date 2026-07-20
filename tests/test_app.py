from __future__ import annotations

from vedock.models import Job, ModelProject, ModelRecord
from vedock.runtimes import get_runtime


def test_desktop_page_renders_complete_html():
    from vedock_cli.desktop import _page

    page = _page()
    assert "Run on this computer" in page
    assert "Publish after final review" in page
    assert "vedock.ecorims.com" in page


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


def test_installer_reuses_a_saved_custom_install(monkeypatch, tmp_path):
    import json
    from installer.vedock_installer import CLIENT_VERSION, InstallerBridge

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


def test_branding_is_environment_driven(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Vedock Test" in response.data
    assert b"Test all controls." in response.data


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
