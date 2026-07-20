from __future__ import annotations

import json
import shutil
import zipfile
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import requests
from flask import Blueprint, Response, abort, current_app, g, jsonify, request, send_file, stream_with_context
from flask_login import current_user

from vedock.extensions import db
from vedock.models import ApiToken, ConnectedDevice, Conversation, DatasetVersion, DeviceResource, Job, Message, ModelFork, ModelRecord, ModelVersion, ModelWorkspaceState, RawDataset, User, new_id, utcnow
from vedock.runtimes import get_runtime
from vedock.runtimes.parameters import ParameterValidationError, validate_parameters
from vedock.services.datasets import DatasetError, import_upload, import_url, inspect_dataset, preview_transform, revalidate_version, save_dataset_version
from vedock.services.dataset_catalog import COMMUNITY_DATASETS, import_community_dataset
from vedock.services.hardware import system_report
from vedock.services.inference import RunnerValidationError, normalize_runtime_result, runner_contract, validate_runner_inputs
from vedock.services.jobs import JobError, delete_job, enqueue_dataset_transform, enqueue_training, read_job_logs, request_cancellation, resume_job
from vedock.services.merges import MergeError, compatibility_report, execute_linear_merge, execute_weighted_adapter_merge, record_failed_merge_attempt, resolve_latest_pair
from vedock.services.model_registry import fork_count, latest_version, visible_models
from vedock.services.model_profiles import model_output_pattern, publisher_defaults, schema_with_model_defaults, set_publisher_defaults, submitted_with_model_defaults, validate_output_pattern
from vedock.services.model_media import save_model_cover
from vedock.services.paths import assert_writable_path
from vedock.services.device_resources import (
    DeviceResourceError,
    owner_devices,
    record_device,
    register_device_resource,
    request_device_path,
)
from vedock.services.remote_jobs import (
    claim_job,
    edit_waiting_job,
    finalize_remote_job,
    job_manifest,
    model_artifact_archive,
    release_job,
    update_remote_job,
)


bp = Blueprint("api", __name__)
F = TypeVar("F", bound=Callable[..., Any])


def ok(data: Any = None, **meta: Any):
    return jsonify({"ok": True, "data": data, "meta": meta or {}})


def failure(message: str, status: int = 400, code: str = "invalid_request", details: Any = None):
    return jsonify({"ok": False, "error": {"code": code, "message": message, "details": details}}), status


def _version_parts(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.strip().split("."))
    except (TypeError, ValueError):
        return ()


def local_compute_required():
    if current_app.config.get("NODE_MODE") == "local_compute":
        return None
    return failure(
        "This operation uses developer storage or compute. Download Vedock and call the API on your local node.",
        409,
        "local_compute_required",
    )


def api_user_required(function: F) -> F:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any):
        user = current_user if current_user.is_authenticated else None
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            token = ApiToken.authenticate(authorization[7:].strip())
            user = token.user if token else None
        if not user:
            return failure("Authentication is required.", 401, "authentication_required")
        g.api_user = user
        return function(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _model(identifier: str) -> ModelRecord | None:
    model = ModelRecord.query.filter(db.or_(ModelRecord.slug == identifier, ModelRecord.id == identifier)).first()
    if not model or not (model.visibility == "public" or model.owner_id == g.api_user.id):
        return None
    return model


def _raw_dataset(dataset_id: str) -> RawDataset | None:
    return RawDataset.query.filter_by(id=dataset_id, owner_id=g.api_user.id).first()


def _dataset_version(version_id: str) -> DatasetVersion | None:
    return DatasetVersion.query.filter_by(id=version_id, owner_id=g.api_user.id).first()


def _conversation(conversation_id: str | None) -> Conversation | None:
    if not conversation_id:
        return None
    return Conversation.query.filter_by(id=conversation_id, owner_id=g.api_user.id).first()


def _device_resource(resource_id: str) -> DeviceResource | None:
    return DeviceResource.query.filter_by(id=resource_id, owner_id=g.api_user.id).first()


def _device_local_inference(model: ModelRecord):
    if model.source_type != "device_local":
        return None
    return failure(
        "This private model is stored on a connected device. Open it in Vedock Desktop, which resolves and runs the local folder without exposing its path to the hosted server.",
        409,
        "connected_device_required",
    )


def _chat_context(messages: list[Message], prompt: str | None = None, maximum_characters: int = 16_000) -> str:
    lines = []
    for message in messages:
        lines.append(f"{'Assistant' if message.role == 'assistant' else 'User'}: {message.content.strip()}")
    if prompt is not None:
        lines.append(f"User: {prompt.strip()}")
    lines.append("Assistant:")
    return "\n\n".join(lines)[-maximum_characters:]


def _conversation_matches(conversation: Conversation, model: ModelRecord, version: ModelVersion) -> bool:
    return (conversation.model_id or conversation.model_version.model_id) == model.id and conversation.model_version_id == version.id


def _context_settings(payload: dict[str, Any]) -> tuple[bool, str, int]:
    use_history = payload.get("use_history", True) not in {False, "false", "0", 0}
    override = str(payload.get("context_override") or "").strip()
    try:
        maximum = int(payload.get("context_limit") or 16_000)
    except (TypeError, ValueError):
        maximum = 16_000
    return use_history, override[:64_000], min(64_000, max(1_000, maximum))


def _model_input(messages: list[Message], prompt: str, use_history: bool, override: str, maximum: int) -> str:
    if override:
        return (override.replace("{prompt}", prompt) if "{prompt}" in override else f"{override}\n\nUser: {prompt}\n\nAssistant:")[-maximum:]
    if use_history:
        return _chat_context(messages, prompt, maximum)
    return prompt


@bp.get("")
@bp.get("/")
def metadata():
    return ok(
        {
            "name": current_app.config["APP_NAME"],
            "short_name": current_app.config["APP_SHORT_NAME"],
            "cli_name": current_app.config["CLI_NAME"],
            "tagline": current_app.config["APP_TAGLINE"],
            "api_version": "v1",
            "node_mode": current_app.config["NODE_MODE"],
            "compute_location": "hosted_inference_and_connected_devices" if current_app.config["NODE_MODE"] != "local_compute" else "local_device",
            "storage_location": "private_host_storage" if current_app.config["NODE_MODE"] != "local_compute" else str(current_app.config["STORAGE_ROOT"]),
            "documentation": "/developer",
        }
    )


@bp.get("/openapi.json")
def openapi():
    name = current_app.config["APP_NAME"]
    return jsonify(
        {
            "openapi": "3.1.0",
            "info": {"title": f"{name} API", "version": "0.1.0", "description": current_app.config["APP_TAGLINE"]},
            "servers": [{"url": "/api/v1"}],
            "paths": {
                "/auth/login": {"post": {"summary": f"Log in to {name}"}},
                "/system/doctor": {"get": {"summary": "Inspect runtime and hardware"}},
                "/models": {"get": {"summary": "List accessible models"}},
                "/models/{model}/infer": {"post": {"summary": "Run text inference"}},
                "/models/{model}/run": {"post": {"summary": "Run any model through its typed runtime contract"}},
                "/models/{model}/stream": {"post": {"summary": "Stream persistent inference with stoppable generation"}},
                "/models/{model}/stop": {"post": {"summary": "Stop an active generation by generation ID"}},
                "/models/{model}/classify-image": {"post": {"summary": "Classify an uploaded local image"}},
                "/models/{model}/cover": {"post": {"summary": "Set an owned model's community cover image"}},
                "/models/{model}/fork": {"post": {"summary": "Create an editable local model fork"}},
                "/models/{model}": {"get": {"summary": "Inspect model capabilities"}, "patch": {"summary": "Edit an owned model"}, "delete": {"summary": "Recoverably remove a model from the workspace"}},
                "/datasets/import": {"post": {"summary": "Import a local file or direct URL"}},
                "/community-datasets": {"get": {"summary": "List locally importable community starter datasets"}},
                "/datasets/{dataset}/transform": {"post": {"summary": "Create an immutable training-ready version"}},
                "/train/{model}": {"post": {"summary": "Queue a background fine-tuning job"}},
                "/merge/compatibility": {"post": {"summary": "Check model merge compatibility"}},
            },
        }
    )


@bp.post("/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    identity = str(payload.get("username") or payload.get("email") or "").strip()
    password = str(payload.get("password") or "")
    user = User.query.filter(db.or_(User.username == identity, User.email == identity.lower())).first()
    if not user or not user.check_password(password):
        return failure("The username/email or password is incorrect.", 401, "invalid_credentials")
    token, plain = ApiToken.issue(user, str(payload.get("token_name") or "CLI"))
    db.session.add(token)
    db.session.commit()
    return ok({"token": plain, "token_prefix": token.prefix, "user": {"id": user.id, "username": user.username, "email": user.email}})


@bp.get("/whoami")
@api_user_required
def whoami():
    user = g.api_user
    return ok({"id": user.id, "username": user.username, "email": user.email})


@bp.post("/devices/connect")
@api_user_required
def device_connect():
    payload = request.get_json(silent=True) or {}
    try:
        device = record_device(
            g.api_user,
            str(payload.get("device_id") or ""),
            str(payload.get("device_name") or "Vedock device"),
            payload.get("details") if isinstance(payload.get("details"), dict) else {},
        )
        return ok(device.to_dict())
    except DeviceResourceError as exc:
        return failure(str(exc), 422, "device_registration_failed")


@bp.get("/devices")
@api_user_required
def devices_list():
    return ok([device.to_dict() for device in owner_devices(g.api_user.id)])


@bp.get("/device-resources")
@api_user_required
def device_resources_list():
    records = DeviceResource.query.filter_by(owner_id=g.api_user.id).order_by(DeviceResource.created_at.desc()).all()
    return ok([record.to_dict() for record in records])


@bp.post("/device-resources")
@api_user_required
def device_resource_register():
    payload = request.get_json(silent=True) or {}
    try:
        resource = register_device_resource(g.api_user, str(payload.get("device_id") or ""), payload)
        return ok(resource.to_dict())
    except (DeviceResourceError, TypeError, ValueError) as exc:
        db.session.rollback()
        return failure(str(exc), 422, "device_resource_invalid")


@bp.post("/device-resources/requests")
@api_user_required
def device_resource_request():
    payload = request.get_json(silent=True) or {}
    try:
        resource = request_device_path(
            g.api_user,
            str(payload.get("device_id") or ""),
            str(payload.get("kind") or ""),
            str(payload.get("path") or ""),
            display_name=payload.get("name"),
            runtime_key=payload.get("runtime"),
            task_type=payload.get("task_type"),
            output_schema=payload.get("output_schema"),
        )
        return ok(resource.to_dict())
    except DeviceResourceError as exc:
        return failure(str(exc), 422, "device_resource_request_failed")


@bp.get("/device-resources/requests")
@api_user_required
def device_resource_requests():
    device_id = str(request.args.get("device_id") or request.headers.get("X-Vedock-Device") or "")
    if not ConnectedDevice.query.filter_by(owner_id=g.api_user.id, device_uid=device_id).first():
        return failure("This connected device is not registered to the account.", 403, "wrong_device")
    records = DeviceResource.query.filter_by(owner_id=g.api_user.id, device_uid=device_id, status="pending_device").order_by(DeviceResource.created_at).all()
    return ok([record.to_dict(include_locator=True) for record in records])


@bp.post("/device-resources/<resource_id>/verify")
@api_user_required
def device_resource_verify(resource_id: str):
    resource = _device_resource(resource_id)
    if not resource:
        return failure("Device resource not found.", 404, "not_found")
    payload = request.get_json(silent=True) or {}
    try:
        verified = register_device_resource(g.api_user, str(payload.get("device_id") or ""), payload, resource=resource)
        return ok(verified.to_dict())
    except (DeviceResourceError, TypeError, ValueError) as exc:
        db.session.rollback()
        return failure(str(exc), 422, "device_resource_verification_failed")


@bp.get("/system/doctor")
@api_user_required
def doctor():
    return ok(system_report(), app={"name": current_app.config["APP_NAME"], "cli": current_app.config["CLI_NAME"], "tagline": current_app.config["APP_TAGLINE"]})


@bp.get("/models")
@api_user_required
def models_list():
    return ok([model.to_dict(include_versions=True) for model in visible_models(g.api_user.id)])


@bp.get("/models/<identifier>")
@api_user_required
def model_info(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    version = latest_version(model)
    runtime = get_runtime(model.runtime_key)
    data = model.to_dict(include_versions=True)
    capabilities = runtime.get_model_capabilities(version.storage_path if version else None)
    if current_app.config["NODE_MODE"] != "local_compute":
        capabilities["loaded"] = bool(capabilities.pop("loaded_model_path", None))
    data["capabilities"] = capabilities
    data["inference_parameters"] = schema_with_model_defaults(runtime.get_inference_parameter_schema(), model, version, g.api_user.id)
    data["training_parameters"] = schema_with_model_defaults(runtime.get_training_parameter_schema(), model, version, g.api_user.id)
    data["dataset_schemas"] = runtime.get_dataset_schema()
    data["creator"] = model.owner.username if model.owner else "Vedock / legacy"
    data["fork_count"] = fork_count(model)
    data["output_pattern"] = model_output_pattern(model, version, g.api_user.id) if model.runtime_key in {"transformers_text", "storymaker"} else None
    data["publisher_defaults"] = publisher_defaults(version)
    return ok(data)


@bp.post("/models/<identifier>/fork")
@api_user_required
def model_fork(identifier: str):
    source = _model(identifier)
    if not source:
        return failure("Model not found.", 404, "not_found")
    version = latest_version(source)
    if not version:
        return failure("The source model has no version to fork.", 409, "no_version")
    child_id = new_id()
    child = ModelRecord(id=child_id, owner_id=g.api_user.id, slug=f"{source.slug[:120]}-fork-{child_id[:8]}", name=f"{source.name} fork"[:160], description=f"Local editable fork of {source.name}.", task_type=source.task_type, runtime_key=source.runtime_key, source_type="fork_reference", source_path=version.storage_path, visibility="private", cover_image_path=source.cover_image_path)
    origin = ModelFork(owner_id=g.api_user.id, child_model=child, source_model=source, source_version=version, configuration_json={"output_pattern": model_output_pattern(source, version, g.api_user.id)})
    db.session.add_all([child, origin])
    db.session.commit()
    return ok(child.to_dict(include_versions=True))


@bp.patch("/models/<identifier>")
@api_user_required
def model_update(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    if model.owner_id != g.api_user.id:
        return failure("Fork this shared model before editing it.", 403, "fork_required")
    payload = request.get_json(silent=True) or {}
    try:
        name = str(payload.get("name") or model.name).strip()
        if not name:
            return failure("Model name is required.")
        model.name = name[:160]
        model.description = str(payload.get("description") if "description" in payload else model.description)[:5000]
        visibility = str(payload.get("visibility") if "visibility" in payload else model.visibility)
        if visibility not in {"public", "private"}:
            return failure("Visibility must be public or private.", 422, "validation_error")
        model.visibility = visibility
        pattern = str(payload.get("output_pattern") or "").strip()
        if pattern and model.runtime_key in {"transformers_text", "storymaker"}:
            pattern = validate_output_pattern(pattern)
            origin = ModelFork.query.filter_by(child_model_id=model.id).first()
            if origin:
                configuration = dict(origin.configuration_json or {})
                configuration["output_pattern"] = pattern
                origin.configuration_json = configuration
            else:
                version = latest_version(model)
                if version:
                    configuration = dict(version.config_json or {})
                    configuration["output_pattern"] = pattern
                    version.config_json = configuration
        if "publisher_defaults" in payload:
            version = latest_version(model)
            if not version:
                return failure("A completed model version is required before publication defaults can be saved.", 409, "no_version")
            supplied = payload.get("publisher_defaults") or {}
            raw_parameters = dict(supplied.get("inference_parameters") or {})
            if "output_pattern" in raw_parameters and model.runtime_key in {"transformers_text", "storymaker"}:
                raw_parameters["output_pattern"] = validate_output_pattern(raw_parameters["output_pattern"])
            normalized = validate_parameters(raw_parameters, get_runtime(model.runtime_key).get_inference_parameter_schema(), include_defaults=False)
            set_publisher_defaults(
                version,
                normalized,
                dict(supplied.get("chat") or {}),
                allow_user_overrides=bool(supplied.get("allow_user_overrides", True)),
            )
        db.session.commit()
        return ok(model.to_dict(include_versions=True))
    except ValueError as exc:
        db.session.rollback()
        return failure(str(exc), 422, "validation_error")


@bp.post("/models/<identifier>/cover")
@api_user_required
def model_cover_upload(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    if model.owner_id != g.api_user.id:
        return failure("Fork this public model before changing its image.", 403, "fork_required")
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return failure("Choose a model image.")
    try:
        model.cover_image_path = save_model_cover(model, upload)
        db.session.commit()
        return ok(model.to_dict())
    except (ValueError, OSError) as exc:
        db.session.rollback()
        return failure(str(exc), 422, "invalid_image")


@bp.delete("/models/<identifier>")
@api_user_required
def model_remove(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    state = ModelWorkspaceState.query.filter_by(owner_id=g.api_user.id, model_id=model.id).first()
    if not state:
        state = ModelWorkspaceState(owner_id=g.api_user.id, model_id=model.id)
        db.session.add(state)
    state.archived = True
    db.session.commit()
    get_runtime(model.runtime_key).unload_model()
    return ok({"model": model.slug, "archived": True, "recoverable": True})


@bp.post("/models/<identifier>/infer")
@api_user_required
def model_infer(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    if blocked := _device_local_inference(model):
        return blocked
    version = latest_version(model)
    if not version or model.source_type == "scratch_definition":
        return failure("Model has no completed version.", 409, "no_version")
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt") or "")
    if not prompt.strip():
        return failure("Prompt is required.")
    if len(prompt) > current_app.config["MAX_PROMPT_CHARS"]:
        return failure("Prompt exceeds the configured length limit.", 413, "prompt_too_large")
    runtime = get_runtime(model.runtime_key)
    try:
        parameter_schema = schema_with_model_defaults(runtime.get_inference_parameter_schema(), model, version, g.api_user.id)
        parameters = validate_parameters(submitted_with_model_defaults(payload.get("parameters") or {}, model, version, g.api_user.id), parameter_schema)
        conversation = _conversation(str(payload.get("conversation_id") or ""))
        if payload.get("conversation_id") and not conversation:
            return failure("Conversation not found.", 404, "not_found")
        if conversation and not _conversation_matches(conversation, model, version):
            return failure("The conversation belongs to another model version.", 409, "model_mismatch")
        save_conversation = bool(payload.get("save_conversation") or conversation)
        use_history, context_override, context_limit = _context_settings(payload)
        model_input = _model_input(conversation.messages if conversation else [], prompt, use_history, context_override, context_limit) if save_conversation else prompt
        result = runtime.infer(version.storage_path, model_input, parameters)
        if save_conversation:
            if conversation is None:
                conversation = Conversation(owner_id=g.api_user.id, model_version=version, selected_model=model, title=str(payload.get("title") or prompt[:80]), parameters_json=parameters)
                db.session.add(conversation)
                db.session.flush()
            conversation.parameters_json = {**parameters, "_chat": {"use_history": use_history, "context_override": context_override, "context_limit": context_limit}}
            conversation.updated_at = utcnow()
            db.session.add_all([Message(conversation=conversation, role="user", content=prompt), Message(conversation=conversation, role="assistant", content=result["text"])])
            db.session.commit()
        result["conversation_id"] = conversation.id if conversation else None
        return ok(result)
    except ParameterValidationError as exc:
        return failure("Inference parameters are invalid.", 422, "validation_error", exc.errors)
    except (ValueError, RuntimeError, OSError) as exc:
        return failure(str(exc), 422, "inference_failed")


@bp.post("/models/<identifier>/run")
@api_user_required
def model_run(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    if blocked := _device_local_inference(model):
        return blocked
    version = latest_version(model)
    if not version or model.source_type == "scratch_definition":
        return failure("Model has no completed version.", 409, "no_version")
    runtime = get_runtime(model.runtime_key)
    temporary = None
    try:
        contract = runner_contract(runtime, version.storage_path)
        payload = request.get_json(silent=True) if request.is_json else None
        submitted_inputs = dict((payload or {}).get("inputs") or {}) if payload is not None else {}
        if payload is None:
            for field in contract["inputs"]:
                name = field["name"]
                if field["type"] in {"image", "file"}:
                    upload = request.files.get(name)
                    if not upload or not upload.filename:
                        submitted_inputs[name] = None
                        continue
                    suffix = Path(upload.filename).suffix.lower()
                    accepted = {str(item).lower() for item in field.get("accept") or []}
                    if accepted and suffix not in accepted:
                        raise RunnerValidationError({name: f"Choose one of: {', '.join(sorted(accepted))}"})
                    temporary = temporary or assert_writable_path(Path(current_app.config["STORAGE_ROOT"]) / "temporary" / "api-inference" / str(g.api_user.id) / new_id())
                    temporary.mkdir(parents=True, exist_ok=True)
                    saved = assert_writable_path(temporary / f"{len(submitted_inputs)}{suffix}")
                    upload.save(saved)
                    submitted_inputs[name] = str(saved)
                elif field["type"] == "boolean":
                    submitted_inputs[name] = request.form.get(name, "false")
                else:
                    submitted_inputs[name] = request.form.get(name)
        inputs = validate_runner_inputs(submitted_inputs, contract)
        if payload is not None:
            submitted_parameters = (payload or {}).get("parameters") or {}
        else:
            input_names = {item["name"] for item in contract["inputs"]}
            submitted_parameters = {key: value for key, value in request.form.items() if key not in input_names and key != "csrf_token"}
        parameter_schema = schema_with_model_defaults(runtime.get_inference_parameter_schema(), model, version, g.api_user.id)
        prepared_parameters = (
            submitted_with_model_defaults(submitted_parameters, model, version, g.api_user.id)
            if any(field["name"] == "output_pattern" for field in parameter_schema)
            else dict(submitted_parameters)
        )
        parameters = validate_parameters(prepared_parameters, parameter_schema)
        return ok(normalize_runtime_result(runtime.run(version.storage_path, inputs, parameters), contract), runner=contract)
    except RunnerValidationError as exc:
        return failure("Model inputs are invalid.", 422, "validation_error", exc.errors)
    except ParameterValidationError as exc:
        return failure("Inference parameters are invalid.", 422, "validation_error", exc.errors)
    except (ValueError, RuntimeError, OSError) as exc:
        return failure(str(exc), 422, "inference_failed")
    finally:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)


@bp.post("/models/<identifier>/classify-image")
@api_user_required
def model_classify_image(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    if blocked := _device_local_inference(model):
        return blocked
    version = latest_version(model)
    if not version:
        return failure("Model has no completed version.", 409, "no_version")
    runtime = get_runtime(model.runtime_key)
    capabilities = runtime.get_model_capabilities(version.storage_path)
    if capabilities.get("interaction") != "image_classification":
        return failure("The selected model does not accept image-classification input.", 409, "wrong_interaction")
    upload = request.files.get("image")
    local_path = str(request.form.get("local_path") or "").strip()
    temporary = None
    try:
        if upload and upload.filename:
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                return failure("Upload a PNG, JPEG, WebP, or BMP image.", 422, "invalid_image")
            temporary = assert_writable_path(Path(current_app.config["STORAGE_ROOT"]) / "temporary" / "api-inference" / str(g.api_user.id) / new_id())
            temporary.mkdir(parents=True)
            image_path = assert_writable_path(temporary / f"input{suffix}")
            upload.save(image_path)
        elif local_path:
            image_path = Path(local_path).resolve()
        else:
            return failure("Upload image or provide local_path.")
        submitted = {key: value for key, value in request.form.items() if key not in {"local_path", "csrf_token"}}
        schema = runtime.get_inference_parameter_schema()
        parameters = validate_parameters(submitted, schema)
        return ok(runtime.infer(version.storage_path, str(image_path), parameters))
    except (ParameterValidationError, ValueError, RuntimeError, OSError) as exc:
        return failure(str(getattr(exc, "errors", exc)), 422, "inference_failed")
    finally:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)


@bp.post("/models/<identifier>/stream")
@api_user_required
def model_stream(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    if blocked := _device_local_inference(model):
        return blocked
    version = latest_version(model)
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt") or "")
    if not version or model.source_type == "scratch_definition" or not prompt.strip():
        return failure("A completed model version and prompt are required.")
    if len(prompt) > current_app.config["MAX_PROMPT_CHARS"]:
        return failure("Prompt exceeds the configured length limit.", 413, "prompt_too_large")
    runtime = get_runtime(model.runtime_key)
    try:
        parameter_schema = schema_with_model_defaults(runtime.get_inference_parameter_schema(), model, version, g.api_user.id)
        parameters = validate_parameters(submitted_with_model_defaults(payload.get("parameters") or {}, model, version, g.api_user.id), parameter_schema)
    except ParameterValidationError as exc:
        return failure("Inference parameters are invalid.", 422, "validation_error", exc.errors)
    conversation = _conversation(str(payload.get("conversation_id") or ""))
    if payload.get("conversation_id") and not conversation:
        return failure("Conversation not found.", 404, "not_found")
    if conversation and not _conversation_matches(conversation, model, version):
        return failure("The conversation belongs to another model version.", 409, "model_mismatch")
    use_history, context_override, context_limit = _context_settings(payload)
    model_input = _model_input(conversation.messages if conversation else [], prompt, use_history, context_override, context_limit)
    if conversation is None:
        conversation = Conversation(owner_id=g.api_user.id, model_version=version, selected_model=model, title=prompt[:80] or "New chat", parameters_json=parameters)
        db.session.add(conversation)
        db.session.flush()
    conversation.parameters_json = {**parameters, "_chat": {"use_history": use_history, "context_override": context_override, "context_limit": context_limit}}
    conversation.updated_at = utcnow()
    db.session.add(Message(conversation=conversation, role="user", content=prompt))
    db.session.commit()
    conversation_id = conversation.id
    model_path = version.storage_path
    generation_id = new_id()
    runtime_parameters = dict(parameters)
    runtime_parameters["_generation_id"] = generation_id

    @stream_with_context
    def events():
        pieces: list[str] = []
        try:
            yield f"event: started\ndata: {json.dumps({'generation_id': generation_id, 'conversation_id': conversation_id})}\n\n"
            for token in runtime.stream_infer(model_path, model_input, runtime_parameters):
                pieces.append(token)
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            saved = db.session.get(Conversation, conversation_id)
            if saved:
                db.session.add(Message(conversation=saved, role="assistant", content="".join(pieces).strip()))
                saved.updated_at = utcnow()
                db.session.commit()
            yield f"event: done\ndata: {json.dumps({'ok': True, 'conversation_id': conversation_id})}\n\n"
        except Exception as exc:
            db.session.rollback()
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return Response(events(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp.post("/models/<identifier>/stop")
@api_user_required
def model_stop(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    payload = request.get_json(silent=True) or {}
    generation_id = str(payload.get("generation_id") or "").strip()
    if not generation_id:
        return failure("generation_id is required.")
    stopped = get_runtime(model.runtime_key).cancel(generation_id)
    return ok({"generation_id": generation_id, "stopping": stopped})


@bp.post("/models/<identifier>/unload")
@api_user_required
def model_unload(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    get_runtime(model.runtime_key).unload_model()
    return ok({"unloaded": model.slug})


@bp.get("/versions/<identifier>")
@api_user_required
def versions_list(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    return ok([version.to_dict() for version in model.versions])


@bp.get("/datasets")
@api_user_required
def datasets_list():
    records = RawDataset.query.filter_by(owner_id=g.api_user.id).order_by(RawDataset.created_at.desc()).all()
    return ok([record.to_dict() for record in records])


@bp.get("/community-datasets")
@api_user_required
def community_datasets_list():
    return ok([{key: value for key, value in item.items() if key != "path"} for item in COMMUNITY_DATASETS])


@bp.post("/community-datasets/<identifier>/import")
@api_user_required
def community_dataset_import(identifier: str):
    try:
        return ok(import_community_dataset(identifier, g.api_user).to_dict())
    except (DatasetError, OSError) as exc:
        return failure(str(exc), 422, "dataset_import_failed")


@bp.post("/datasets/import")
@api_user_required
def datasets_import():
    try:
        if request.is_json:
            payload = request.get_json() or {}
            record = import_url(str(payload.get("url") or ""), g.api_user, payload.get("name"), str(payload.get("description") or ""))
        else:
            upload = request.files.get("file")
            if not upload:
                return failure("Upload a file or send a JSON URL.")
            record = import_upload(upload, g.api_user, request.form.get("name"), request.form.get("description", ""))
        return ok(record.to_dict())
    except (DatasetError, OSError, requests.RequestException) as exc:
        return failure(str(exc), 422, "dataset_import_failed")


@bp.get("/datasets/<dataset_id>")
@api_user_required
def dataset_info(dataset_id: str):
    dataset = _raw_dataset(dataset_id)
    if not dataset:
        return failure("Dataset not found.", 404, "not_found")
    data = dataset.to_dict()
    data["versions"] = [version.to_dict() for version in dataset.versions]
    return ok(data)


@bp.post("/datasets/<dataset_id>/inspect")
@api_user_required
def dataset_inspect(dataset_id: str):
    dataset = _raw_dataset(dataset_id)
    if not dataset:
        return failure("Dataset not found.", 404, "not_found")
    try:
        return ok(inspect_dataset(dataset))
    except (DatasetError, OSError, ValueError) as exc:
        return failure(str(exc), 422, "inspection_failed")


@bp.post("/datasets/<dataset_id>/preview")
@api_user_required
def dataset_preview(dataset_id: str):
    dataset = _raw_dataset(dataset_id)
    if not dataset:
        return failure("Dataset not found.", 404, "not_found")
    payload = request.get_json(silent=True) or {}
    try:
        return ok(preview_transform(dataset, payload.get("operations") or [], payload.get("output_schema", "prompt_response"), payload.get("field_mapping") or {}, str(payload.get("template") or ""), min(int(payload.get("limit") or 20), 100)))
    except (DatasetError, ValueError, OSError) as exc:
        return failure(str(exc), 422, "transform_failed")


@bp.post("/datasets/<dataset_id>/transform")
@api_user_required
def dataset_transform(dataset_id: str):
    dataset = _raw_dataset(dataset_id)
    if not dataset:
        return failure("Dataset not found.", 404, "not_found")
    payload = request.get_json(silent=True) or {}
    try:
        if dataset.size_bytes > current_app.config["DATASET_SYNC_MAX_BYTES"]:
            job = enqueue_dataset_transform(
                g.api_user,
                dataset,
                payload.get("operations") or [],
                payload.get("output_schema", "prompt_response"),
                payload.get("field_mapping") or {},
                str(payload.get("template") or ""),
                int(payload.get("limit_examples") or 0),
                bool(payload.get("shuffle")),
                int(payload.get("shuffle_seed") or 42),
            )
            return ok({"job": job.to_dict()}, asynchronous=True)
        version = save_dataset_version(dataset, g.api_user, payload.get("operations") or [], payload.get("output_schema", "prompt_response"), payload.get("field_mapping") or {}, str(payload.get("template") or ""), int(payload.get("limit_examples") or 0), bool(payload.get("shuffle")), int(payload.get("shuffle_seed") or 42))
        return ok(version.to_dict())
    except (DatasetError, ValueError, OSError) as exc:
        return failure(str(exc), 422, "transform_failed")


@bp.post("/dataset-versions/<version_id>/validate")
@api_user_required
def version_validate(version_id: str):
    version = _dataset_version(version_id)
    if not version:
        return failure("Dataset version not found.", 404, "not_found")
    try:
        return ok(revalidate_version(version))
    except (DatasetError, OSError, ValueError) as exc:
        return failure(str(exc), 422, "validation_failed")


@bp.post("/train/<identifier>")
@api_user_required
def train(identifier: str):
    if not current_app.config["MODEL_TRAINING_ENABLED"]:
        return failure(
            "Model training is disabled on this Vedock installation.",
            409,
            "training_disabled",
        )
    model = _model(identifier)
    payload = request.get_json(silent=True) or {}
    dataset = _dataset_version(str(payload.get("dataset") or payload.get("dataset_version_id") or ""))
    if not model or not dataset:
        return failure("An accessible model and owned dataset version are required.", 404, "not_found")
    try:
        job = enqueue_training(g.api_user, model, dataset, payload.get("parameters") or {})
        return ok(job.to_dict())
    except ParameterValidationError as exc:
        return failure("Training parameters are invalid.", 422, "validation_error", exc.errors)
    except JobError as exc:
        return failure(str(exc), 422, "training_blocked")


@bp.get("/jobs")
@api_user_required
def jobs_list():
    records = Job.query.filter_by(owner_id=g.api_user.id).order_by(Job.created_at.desc()).all()
    return ok([job.to_dict() for job in records])


@bp.get("/jobs/<job_id>")
@api_user_required
def job_info(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    return ok(job.to_dict())


@bp.patch("/jobs/<job_id>")
@api_user_required
def job_edit(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    try:
        payload = request.get_json(silent=True) or {}
        return ok(edit_waiting_job(job, payload.get("parameters") or payload).to_dict())
    except ParameterValidationError as exc:
        return failure("Training parameters are invalid.", 422, "validation_error", exc.errors)
    except JobError as exc:
        return failure(str(exc), 409, "cannot_edit")


@bp.post("/jobs/<job_id>/claim")
@api_user_required
def job_claim(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    required_version = str(current_app.config.get("MIN_CONNECTED_CLIENT_VERSION") or "")
    reported_version = request.headers.get("X-Vedock-Client-Version", "")
    if required_version and _version_parts(reported_version) < _version_parts(required_version):
        control_plane = str(current_app.config.get("CONTROL_PLANE_URL") or "").rstrip("/")
        download_url = f"{control_plane}/downloads/vedock-installer.exe" if control_plane else "/downloads/vedock-installer.exe"
        return failure(
            "This Vedock connected client is outdated. Install the latest client before running a training task.",
            426,
            "client_update_required",
            {"installed": reported_version or "unknown", "minimum": required_version, "download_url": download_url},
        )
    payload = request.get_json(silent=True) or {}
    try:
        claimed = claim_job(job, str(payload.get("device_id") or ""), str(payload.get("device_name") or "Vedock device"))
        return ok({"job": claimed.to_dict(), "manifest": job_manifest(claimed)})
    except JobError as exc:
        return failure(str(exc), 409, "cannot_claim")


@bp.post("/jobs/<job_id>/release")
@api_user_required
def job_release(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    payload = request.get_json(silent=True) or {}
    try:
        return ok(release_job(job, str(payload.get("device_id") or ""), str(payload.get("reason") or "")).to_dict())
    except JobError as exc:
        return failure(str(exc), 409, "cannot_release")


@bp.get("/jobs/<job_id>/manifest")
@api_user_required
def job_compute_manifest(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    device_id = request.headers.get("X-Vedock-Device", "")
    if job.claimed_by_device and job.claimed_by_device != device_id:
        return failure("This task is claimed by another device.", 409, "wrong_device")
    try:
        return ok(job_manifest(job))
    except JobError as exc:
        return failure(str(exc), 409, "manifest_unavailable")


@bp.post("/jobs/<job_id>/events")
@api_user_required
def job_events(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    payload = request.get_json(silent=True) or {}
    try:
        return ok(update_remote_job(job, str(payload.pop("device_id", "")), payload).to_dict())
    except (JobError, ValueError) as exc:
        return failure(str(exc), 409, "event_rejected")


@bp.get("/jobs/<job_id>/dataset")
@api_user_required
def job_dataset_download(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    if job.claimed_by_device != request.headers.get("X-Vedock-Device", ""):
        return failure("This device did not claim the task.", 409, "wrong_device")
    version = db.session.get(DatasetVersion, (job.config_json or {}).get("dataset_version_id"))
    path = Path(version.storage_path) if version else Path()
    if not version or not path.is_file():
        return failure("The task dataset artifact is unavailable.", 404, "artifact_missing")
    return send_file(path, as_attachment=True, download_name=f"{version.id}.jsonl")


@bp.get("/jobs/<job_id>/base-model")
@api_user_required
def job_base_model_download(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    if job.claimed_by_device != request.headers.get("X-Vedock-Device", ""):
        return failure("This device did not claim the task.", 409, "wrong_device")
    try:
        path = model_artifact_archive(job)
        return send_file(path, as_attachment=True, download_name=f"vedock-base-{job.id}.zip")
    except JobError as exc:
        return failure(str(exc), 409, "artifact_unavailable")


@bp.post("/jobs/<job_id>/finalize")
@api_user_required
def job_finalize(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    try:
        metadata = json.loads(request.form.get("metadata") or "{}")
        version = finalize_remote_job(
            job,
            request.form.get("device_id", ""),
            request.files.get("artifact"),
            metadata,
        )
        return ok({"job": job.to_dict(), "model": version.model.to_dict(include_versions=True)})
    except (JobError, ValueError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        return failure(str(exc), 422, "finalize_failed")


@bp.get("/jobs/<job_id>/logs")
@api_user_required
def job_logs(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    return ok(read_job_logs(job, int(request.args.get("limit", 500))))


@bp.post("/jobs/<job_id>/cancel")
@api_user_required
def job_cancel(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    try:
        return ok(request_cancellation(job, g.api_user).to_dict())
    except JobError as exc:
        return failure(str(exc), 409, "cannot_cancel")


@bp.post("/jobs/<job_id>/resume")
@api_user_required
def job_resume(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    try:
        return ok(resume_job(job, g.api_user).to_dict())
    except JobError as exc:
        return failure(str(exc), 409, "cannot_resume")


@bp.delete("/jobs/<job_id>")
@api_user_required
def job_delete(job_id: str):
    job = Job.query.filter_by(id=job_id, owner_id=g.api_user.id).first()
    if not job:
        return failure("Job not found.", 404, "not_found")
    try:
        deleted_id = delete_job(job, g.api_user)
        return ok({"deleted": True, "job_id": deleted_id, "result_model_preserved": True})
    except JobError as exc:
        return failure(str(exc), 409, "cannot_delete")


@bp.get("/conversations")
@api_user_required
def conversations_list():
    records = Conversation.query.filter_by(owner_id=g.api_user.id).order_by(Conversation.updated_at.desc()).all()
    return ok([record.to_dict() for record in records])


@bp.get("/conversations/<conversation_id>")
@api_user_required
def conversation_info(conversation_id: str):
    conversation = Conversation.query.filter_by(id=conversation_id, owner_id=g.api_user.id).first()
    if not conversation:
        return failure("Conversation not found.", 404, "not_found")
    return ok(conversation.to_dict(include_messages=True))


@bp.delete("/conversations/<conversation_id>")
@api_user_required
def conversation_delete(conversation_id: str):
    conversation = Conversation.query.filter_by(id=conversation_id, owner_id=g.api_user.id).first()
    if not conversation:
        return failure("Conversation not found.", 404, "not_found")
    db.session.delete(conversation)
    db.session.commit()
    return ok({"deleted": True, "conversation_id": conversation_id})


@bp.post("/merge/compatibility")
@api_user_required
def merge_compatibility():
    payload = request.get_json(silent=True) or {}
    first = _model(str(payload.get("model_a") or ""))
    second = _model(str(payload.get("model_b") or ""))
    if not first or not second or first.id == second.id:
        return failure("Select two different accessible models.")
    try:
        version_a, version_b = resolve_latest_pair(first, second)
        return ok(compatibility_report(version_a, version_b))
    except MergeError as exc:
        return failure(str(exc), 422, "merge_blocked")


@bp.post("/merge")
@api_user_required
def merge_execute():
    payload = request.get_json(silent=True) or {}
    first = _model(str(payload.get("model_a") or ""))
    second = _model(str(payload.get("model_b") or ""))
    if not first or not second or first.id == second.id:
        return failure("Select two different accessible models.")
    version_a = version_b = None
    report: dict[str, Any] = {}
    method = str(payload.get("method") or "auto")
    try:
        version_a, version_b = resolve_latest_pair(first, second)
        report = compatibility_report(version_a, version_b)
        if method == "auto":
            method = "weighted_adapter" if report.get("lora_safe") else "linear"
        executor = execute_weighted_adapter_merge if method == "weighted_adapter" else execute_linear_merge
        merge, output = executor(version_a, version_b, float(payload.get("weight_a", 0.5)), float(payload.get("weight_b", 0.5)), g.api_user, str(payload.get("output_name") or "Merged model"))
        return ok({"merge_id": merge.id, "model_version": output.to_dict()})
    except (MergeError, ValueError, OSError, RuntimeError) as exc:
        attempt = None
        if version_a and version_b:
            try:
                attempt = record_failed_merge_attempt(version_a, version_b, method, [float(payload.get("weight_a", 0.5)), float(payload.get("weight_b", 0.5))], g.api_user, report, str(exc))
            except Exception:
                db.session.rollback()
        return failure(str(exc), 422, "merge_attempt_failed", {"attempt_id": attempt.id if attempt else None, "compatibility": report})


@bp.post("/export/<identifier>")
@api_user_required
def export_model(identifier: str):
    model = _model(identifier)
    if not model:
        return failure("Model not found.", 404, "not_found")
    version = latest_version(model)
    if not version:
        return failure("Model has no completed version.", 409, "no_version")
    export_root = assert_writable_path(Path(current_app.config["STORAGE_ROOT"]) / "exports" / str(g.api_user.id))
    export_root.mkdir(parents=True, exist_ok=True)
    base = assert_writable_path(export_root / f"{model.slug}-{new_id()[:8]}")
    archive = Path(shutil.make_archive(str(base), "zip", root_dir=version.storage_path))
    return send_file(archive, as_attachment=True, download_name=f"{model.slug}.zip")
