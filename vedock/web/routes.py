from __future__ import annotations

import json
import csv
import io
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import requests
from flask import Blueprint, Response, abort, current_app, flash, redirect, render_template, request, send_file, stream_with_context, url_for
from flask_login import current_user, login_required

from vedock.extensions import db
from vedock.models import (
    Conversation,
    DatasetVersion,
    DeviceResource,
    Job,
    Message,
    ModelFork,
    ModelRecord,
    ModelProject,
    ModelReaction,
    ModelVersion,
    ModelWorkspaceState,
    RawDataset,
    TrainingRecipe,
    new_id,
    utcnow,
)
from vedock.runtimes import get_runtime
from vedock.runtimes.parameters import ParameterValidationError, schema_groups, validate_parameters
from vedock.services.datasets import (
    DatasetError,
    import_upload,
    import_url,
    inspect_dataset,
    preview_transform,
    revalidate_version,
    save_dataset_version,
)
from vedock.services.dataset_catalog import COMMUNITY_DATASETS, import_community_dataset
from vedock.services.cli_distribution import build_client_archive, build_node_archive
from vedock.services.hardware import system_report
from vedock.services.inference import RunnerValidationError, normalize_runtime_result, runner_contract, validate_runner_inputs
from vedock.services.jobs import JobError, assert_training_enabled, delete_job, enqueue_dataset_transform, enqueue_training, read_job_logs, request_cancellation, resume_job
from vedock.services.merges import MergeError, compatibility_report, execute_linear_merge, execute_weighted_adapter_merge, record_failed_merge_attempt, resolve_latest_pair
from vedock.services.model_registry import fork_count, latest_version, recent_chat_models, runnable_models, visible_models
from vedock.services.model_profiles import model_output_pattern, publisher_defaults, schema_with_model_defaults, set_publisher_defaults, submitted_with_model_defaults, validate_output_pattern
from vedock.services.model_media import generated_cover_svg, save_model_cover
from vedock.services.model_sources import (
    BUILD_MODES,
    PRETRAINED_MODEL_CATALOG,
    SCRATCH_PRESETS,
    TASK_OPTIONS,
    ModelSourceError,
    resolve_model_source,
)
from vedock.services.paths import assert_writable_path
from vedock.services.device_resources import DeviceResourceError, owner_devices, request_device_path


bp = Blueprint("web", __name__)


def _owned_dataset(dataset_id: str) -> RawDataset:
    dataset = db.session.get(RawDataset, dataset_id)
    if not dataset or dataset.owner_id != current_user.id:
        abort(404)
    return dataset


def _owned_version(version_id: str) -> DatasetVersion:
    version = db.session.get(DatasetVersion, version_id)
    if not version or version.owner_id != current_user.id:
        abort(404)
    return version


def _visible_model(slug_or_id: str) -> ModelRecord:
    model = ModelRecord.query.filter(db.or_(ModelRecord.slug == slug_or_id, ModelRecord.id == slug_or_id)).first()
    if not model or not (model.visibility == "public" or model.owner_id == current_user.id):
        abort(404)
    return model


@bp.get("/model-media/<slug>/cover")
def model_cover(slug: str):
    model = ModelRecord.query.filter_by(slug=slug).first()
    is_owner = current_user.is_authenticated and model and model.owner_id == current_user.id
    if not model or (model.visibility != "public" and not is_owner):
        abort(404)
    path = Path(model.cover_image_path) if model.cover_image_path else None
    if path and path.is_file():
        return send_file(path, conditional=True, max_age=3600)
    return Response(generated_cover_svg(model), mimetype="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


def _owned_conversation(conversation_id: str | None) -> Conversation | None:
    if not conversation_id:
        return None
    conversation = db.session.get(Conversation, conversation_id)
    if not conversation or conversation.owner_id != current_user.id:
        abort(404)
    return conversation


def _chat_prompt(messages: list[Message], prompt: str | None = None, maximum_characters: int = 16_000, use_history: bool = True, override: str = "") -> str:
    if override:
        value = override.replace("{prompt}", prompt or "") if "{prompt}" in override else f"{override}\n\nUser: {prompt or ''}\n\nAssistant:"
        return value[-maximum_characters:]
    if not use_history:
        return prompt or ""
    lines = []
    for message in messages:
        role = "Assistant" if message.role == "assistant" else "User"
        lines.append(f"{role}: {message.content.strip()}")
    if prompt is not None:
        lines.append(f"User: {prompt.strip()}")
    lines.append("Assistant:")
    combined = "\n\n".join(lines)
    return combined[-maximum_characters:]


def _web_context_settings() -> tuple[bool, str, int]:
    use_history = request.form.get("use_history") == "true"
    override = request.form.get("context_override", "").strip()[:64_000]
    try:
        maximum = int(request.form.get("context_limit") or 16_000)
    except ValueError:
        maximum = 16_000
    return use_history, override, min(64_000, max(1_000, maximum))


def _form_parameters(schema: list[dict]) -> dict:
    submitted = {}
    for field in schema:
        name = field["name"]
        if field["type"] == "boolean":
            if name in request.form:
                submitted[name] = True
            elif not field.get("depends_on"):
                submitted[name] = False
        elif name in request.form:
            submitted[name] = request.form.get(name)
    return submitted


def _builder_configuration() -> tuple[list[dict], str, dict[str, str], str, int, bool, int]:
    operations_text = request.form.get("operations_json", "").strip()
    if operations_text:
        operations = json.loads(operations_text)
        if not isinstance(operations, list):
            raise DatasetError("Advanced operations JSON must be an array.")
    else:
        operations = []
        fields = [value for value in request.form.getlist("clean_fields") if value]
        if request.form.get("trim_whitespace"):
            operations.append({"type": "trim_whitespace", "config": {"fields": fields}})
        if request.form.get("normalize_unicode"):
            operations.append({"type": "normalize_unicode", "config": {"fields": fields, "form": "NFKC"}})
        if request.form.get("remove_html"):
            operations.append({"type": "remove_html", "config": {"fields": fields}})
        if request.form.get("remove_urls"):
            operations.append({"type": "remove_urls", "config": {"fields": fields}})
        if request.form.get("remove_control_characters"):
            operations.append({"type": "remove_control_characters", "config": {"fields": fields}})
        if request.form.get("lowercase"):
            operations.append({"type": "lowercase", "config": {"fields": fields}})
        if request.form.get("strip_accents"):
            operations.append({"type": "strip_accents", "config": {"fields": fields}})
        if request.form.get("redact_emails"):
            operations.append({"type": "redact_emails", "config": {"fields": fields, "replacement": "[EMAIL]"}})
        if request.form.get("redact_phone_numbers"):
            operations.append({"type": "redact_phone_numbers", "config": {"fields": fields, "replacement": "[PHONE]"}})
        if request.form.get("remove_empty_records"):
            operations.append({"type": "remove_empty_records", "config": {"fields": fields}})
        if request.form.get("remove_duplicates"):
            operations.append({"type": "remove_duplicates", "config": {}})
        minimum = request.form.get("minimum_length", "").strip()
        maximum = request.form.get("maximum_length", "").strip()
        length_field = request.form.get("length_field", "").strip()
        if length_field and (minimum or maximum):
            operations.append({"type": "filter_length", "config": {"field": length_field, "minimum": int(minimum) if minimum else None, "maximum": int(maximum) if maximum else None}})
    output_schema = request.form.get("output_schema", "prompt_response")
    mapping = {}
    for target in [
        "text",
        "prompt",
        "response",
        "instruction",
        "input",
        "output",
        "label",
        "system",
        "image",
        "caption",
        "target",
    ]:
        source = request.form.get(f"map_{target}", "").strip()
        if source:
            mapping[target] = source
    feature_sources = [value.strip() for value in request.form.getlist("map_features") if value.strip()]
    if feature_sources:
        mapping["features"] = feature_sources
    template = request.form.get("formatting_template", "")
    limit = max(0, int(request.form.get("limit_examples") or 0))
    shuffle = bool(request.form.get("shuffle"))
    shuffle_seed = int(request.form.get("shuffle_seed") or 42)
    return operations, output_schema, mapping, template, limit, shuffle, shuffle_seed


def _model_dataset_recommendation(model: ModelRecord | None, columns: list[str]) -> tuple[str | None, dict[str, str], list[str]]:
    if not model:
        return None, {}, []
    schemas = [item["name"] for item in get_runtime(model.runtime_key).get_dataset_schema()]
    normalized = {str(column).lower(): str(column) for column in columns}

    def first(*names: str) -> str | None:
        return next((normalized[name] for name in names if name in normalized), None)

    prompt = first("prompt", "question", "instruction", "input")
    response = first("response", "answer", "story", "completion", "output", "target")
    if "image_classification" in schemas:
        selected = "image_classification"
    elif "tabular_supervised" in schemas:
        selected = "tabular_supervised"
    elif "prompt_response" in schemas and prompt and response:
        selected = "prompt_response"
    elif "instruction" in schemas and first("instruction") and response:
        selected = "instruction"
    else:
        selected = schemas[0] if schemas else None
    aliases = {
        "text": first("text", "content", "document", "body"),
        "prompt": prompt,
        "response": response,
        "instruction": first("instruction", "prompt", "question"),
        "input": first("input", "context"),
        "output": first("output", "response", "answer", "story"),
        "label": first("label", "class", "category", "target"),
        "image": first("image", "image_path", "file", "path"),
        "caption": first("caption", "description"),
        "target": first("target", "sales", "revenue", "price", "demand", "label", "outcome"),
    }
    if selected == "tabular_supervised":
        target = aliases["target"]
        aliases["features"] = [str(column) for column in columns if str(column) != target]
    return selected, {target: source for target, source in aliases.items() if source}, schemas


@bp.route("/datasets", methods=["GET", "POST"])
@login_required
def datasets():
    target_model = _visible_model(request.values.get("for_model")) if request.values.get("for_model") else None
    if request.method == "POST":
        try:
            source_type = request.form.get("source_type")
            if source_type == "device_path":
                resource = request_device_path(
                    current_user,
                    request.form.get("device_id", ""),
                    "dataset",
                    request.form.get("local_path", ""),
                    display_name=request.form.get("name"),
                    output_schema=request.form.get("output_schema") or None,
                )
                flash(f"Sent {resource.path_hint} to the connected device for private inspection. Keep the Vedock app open, then refresh.", "success")
                return redirect(url_for("web.datasets", for_model=target_model.slug if target_model else None))
            if current_app.config.get("NODE_MODE") != "local_compute":
                raise DatasetError("Hosted Vedock does not copy private datasets to the server. Choose a path on a connected device or add the file from the Vedock desktop app.")
            if source_type == "community":
                dataset = import_community_dataset(request.form.get("community_id", ""), current_user)
            elif source_type == "url":
                dataset = import_url(request.form.get("url", ""), current_user, request.form.get("name"), request.form.get("description", ""))
            else:
                upload = request.files.get("file")
                if not upload:
                    raise DatasetError("Choose a dataset file.")
                dataset = import_upload(upload, current_user, request.form.get("name"), request.form.get("description", ""))
            flash(f"Imported {dataset.name} without changing the raw source.", "success")
            return redirect(url_for("web.dataset_builder", dataset_id=dataset.id, model=target_model.slug if target_model else None))
        except (DatasetError, DeviceResourceError, requests.RequestException, OSError, ValueError) as exc:
            flash(str(exc), "error")
    records = RawDataset.query.filter_by(owner_id=current_user.id).order_by(RawDataset.created_at.desc()).all()
    resources = DeviceResource.query.filter_by(owner_id=current_user.id, kind="dataset").order_by(DeviceResource.created_at.desc()).all()
    return render_template("web/datasets.html", datasets=records, device_resources=resources, devices=owner_devices(current_user.id), storage_root=current_app.config["STORAGE_ROOT"] / "datasets", community_datasets=COMMUNITY_DATASETS, target_model=target_model)


@bp.post("/datasets/<dataset_id>/inspect")
@login_required
def dataset_inspect(dataset_id: str):
    dataset = _owned_dataset(dataset_id)
    try:
        inspect_dataset(dataset)
        flash("Dataset inspection completed.", "success")
    except (DatasetError, OSError, ValueError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.dataset_builder", dataset_id=dataset.id))


@bp.route("/datasets/<dataset_id>/builder", methods=["GET", "POST"])
@login_required
def dataset_builder(dataset_id: str):
    dataset = _owned_dataset(dataset_id)
    target_model = _visible_model(request.values.get("model")) if request.values.get("model") else None
    preview = None
    if request.method == "POST":
        try:
            operations, output_schema, mapping, template, limit, shuffle, shuffle_seed = _builder_configuration()
            if request.form.get("action") == "preview":
                preview = preview_transform(dataset, operations, output_schema, mapping, template)
                flash("Preview generated from raw data; nothing was saved.", "info")
            else:
                if dataset.size_bytes > current_app.config["DATASET_SYNC_MAX_BYTES"]:
                    job = enqueue_dataset_transform(current_user, dataset, operations, output_schema, mapping, template, limit, shuffle, shuffle_seed)
                    flash("Large dataset transformation is running in a separate worker.", "success")
                    return redirect(url_for("web.job_details", job_id=job.id))
                version = save_dataset_version(dataset, current_user, operations, output_schema, mapping, template, limit, shuffle, shuffle_seed)
                flash(f"Immutable dataset version {version.version_number} saved with status {version.validation_status}.", "success" if version.validation_status != "invalid" else "warning")
                return redirect(url_for("web.dataset_builder", dataset_id=dataset.id, version=version.id, model=target_model.slug if target_model else None))
        except (DatasetError, ValueError, json.JSONDecodeError, OSError) as exc:
            flash(str(exc), "error")
    selected_version = None
    if request.args.get("version"):
        selected_version = _owned_version(request.args["version"])
        if selected_version.raw_dataset_id != dataset.id:
            abort(404)
    columns = (dataset.detected_schema_json or {}).get("columns", [])
    recommended_schema, recommended_mapping, compatible_schemas = _model_dataset_recommendation(target_model, columns)
    return render_template("web/dataset_builder.html", dataset=dataset, columns=columns, preview=preview, selected_version=selected_version, target_model=target_model, recommended_schema=recommended_schema, recommended_mapping=recommended_mapping, compatible_schemas=compatible_schemas)


@bp.post("/dataset-versions/<version_id>/validate")
@login_required
def dataset_validate(version_id: str):
    version = _owned_version(version_id)
    try:
        report = revalidate_version(version)
        flash(f"Validation finished with status {report['status']}.", "success" if report["status"] != "invalid" else "error")
    except (DatasetError, OSError, ValueError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.dataset_builder", dataset_id=version.raw_dataset_id, version=version.id))


@bp.get("/datasets/<dataset_id>/raw/download")
@login_required
def dataset_raw_download(dataset_id: str):
    dataset = _owned_dataset(dataset_id)
    path = Path(dataset.storage_path)
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name=dataset.original_filename)


@bp.get("/dataset-versions/<version_id>/download")
@login_required
def dataset_version_download(version_id: str):
    version = _owned_version(version_id)
    path = Path(version.storage_path)
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"{version.raw_dataset.name}-v{version.version_number}.jsonl")


def _version_rows(version: DatasetVersion) -> list[dict]:
    rows: list[dict] = []
    with Path(version.storage_path).open("r", encoding="utf-8-sig", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(value if isinstance(value, dict) else {"value": value})
            if number > 1_000_000:
                raise ValueError("Use JSONL export for datasets larger than one million rows.")
    return rows


def _xlsx_bytes(rows: list[dict]) -> bytes:
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["value"]

    def column_name(index: int) -> str:
        value = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            value = chr(65 + remainder) + value
        return value

    def cell(reference: str, value: object) -> str:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        text = xml_escape(str("" if value is None else value)[:32_767])
        return f'<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'

    sheet_rows = [f'<row r="1">{"".join(cell(f"{column_name(i)}1", name) for i, name in enumerate(columns, 1))}</row>']
    for row_number, row in enumerate(rows, 2):
        sheet_rows.append(f'<row r="{row_number}">{"".join(cell(f"{column_name(i)}{row_number}", row.get(name)) for i, name in enumerate(columns, 1))}</row>')
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        archive.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Dataset" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(sheet_rows) + "</sheetData></worksheet>")
    return output.getvalue()


@bp.get("/dataset-versions/<version_id>/export/<file_format>")
@login_required
def dataset_version_export(version_id: str, file_format: str):
    version = _owned_version(version_id)
    file_format = file_format.lower()
    if file_format == "jsonl":
        return dataset_version_download(version_id)
    if file_format not in {"json", "csv", "txt", "xlsx"}:
        abort(404)
    try:
        rows = _version_rows(version)
        stem = f"{version.raw_dataset.name}-v{version.version_number}"
        if file_format == "json":
            payload, mimetype = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"), "application/json"
        elif file_format == "txt":
            lines = []
            for row in rows:
                simple = next(iter(row.values())) if len(row) == 1 else row
                lines.append(simple if isinstance(simple, str) else json.dumps(simple, ensure_ascii=False))
            payload, mimetype = ("\n\n".join(lines) + "\n").encode("utf-8"), "text/plain"
        elif file_format == "xlsx":
            payload, mimetype = _xlsx_bytes(rows), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            text_stream = io.StringIO(newline="")
            writer = csv.DictWriter(text_stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
            payload, mimetype = ("\ufeff" + text_stream.getvalue()).encode("utf-8"), "text/csv"
        return send_file(io.BytesIO(payload), as_attachment=True, download_name=f"{stem}.{file_format}", mimetype=mimetype)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        flash(f"Export failed: {exc}", "error")
        return redirect(url_for("web.dataset_builder", dataset_id=version.raw_dataset_id, version=version.id))


@bp.get("/models")
@login_required
def models():
    records = visible_models(current_user.id)
    recent_model_ids = []
    for conversation in Conversation.query.filter_by(owner_id=current_user.id).order_by(Conversation.updated_at.desc()).limit(30):
        identifier = conversation.model_id or conversation.model_version.model_id
        if identifier not in recent_model_ids:
            recent_model_ids.append(identifier)
    archived = (
        ModelRecord.query.join(ModelWorkspaceState, ModelWorkspaceState.model_id == ModelRecord.id)
        .filter(ModelWorkspaceState.owner_id == current_user.id, ModelWorkspaceState.archived.is_(True))
        .order_by(ModelWorkspaceState.created_at.desc())
        .all()
    )
    return render_template("web/models.html", models=records, recent_model_ids=recent_model_ids[:6], archived_models=archived, fork_counts={model.id: fork_count(model) for model in records})


@bp.get("/models/<slug>")
@login_required
def model_details(slug: str):
    model = _visible_model(slug)
    runtime = get_runtime(model.runtime_key)
    version = latest_version(model)
    capabilities = runtime.get_model_capabilities(version.storage_path if version else None)
    inference_schema = schema_with_model_defaults(runtime.get_inference_parameter_schema(), model, version, current_user.id)
    return render_template("web/model_details.html", model=model, version=version, capabilities=capabilities, output_pattern=model_output_pattern(model, version, current_user.id), publisher_defaults=publisher_defaults(version), publisher_fields=[field for field in inference_schema if field["name"] not in {"system_prompt", "output_pattern"}], forks=fork_count(model))


@bp.post("/models/<slug>/fork")
@login_required
def model_fork(slug: str):
    source = _visible_model(slug)
    version = latest_version(source)
    if not version:
        flash("The source model has no version to fork.", "error")
        return redirect(url_for("web.model_details", slug=source.slug))
    child_id = new_id()
    child = ModelRecord(
        id=child_id,
        owner_id=current_user.id,
        slug=f"{source.slug[:120]}-fork-{child_id[:8]}",
        name=f"{source.name} fork"[:160],
        description=f"Local editable fork of {source.name}.",
        task_type=source.task_type,
        runtime_key=source.runtime_key,
        source_type="fork_reference",
        source_path=version.storage_path,
        visibility="private",
        cover_image_path=source.cover_image_path,
    )
    origin = ModelFork(owner_id=current_user.id, child_model=child, source_model=source, source_version=version, configuration_json={"output_pattern": model_output_pattern(source, version, current_user.id)})
    db.session.add_all([child, origin])
    db.session.commit()
    flash("Editable local fork created. The source weights remain unchanged.", "success")
    return redirect(url_for("web.model_details", slug=child.slug))


@bp.post("/models/<slug>/edit")
@login_required
def model_edit(slug: str):
    model = _visible_model(slug)
    if model.owner_id != current_user.id:
        flash("Fork this shared model before editing it.", "error")
        return redirect(url_for("web.model_details", slug=model.slug))
    name = request.form.get("name", "").strip()
    if not name:
        flash("Model name is required.", "error")
        return redirect(url_for("web.model_details", slug=model.slug))
    try:
        pattern = request.form.get("output_pattern", "").strip()
        if model.runtime_key in {"transformers_text", "storymaker"}:
            pattern = validate_output_pattern(pattern)
        model.name = name[:160]
        model.description = request.form.get("description", "")[:5000]
        visibility = request.form.get("visibility", model.visibility)
        if visibility not in {"public", "private"}:
            raise ValueError("Visibility must be public or private.")
        model.visibility = visibility
        cover = request.files.get("cover_image")
        if cover and cover.filename:
            model.cover_image_path = save_model_cover(model, cover)
        origin = ModelFork.query.filter_by(child_model_id=model.id).first()
        if origin:
            configuration = dict(origin.configuration_json or {})
            if pattern:
                configuration["output_pattern"] = pattern
            origin.configuration_json = configuration
        else:
            version = latest_version(model)
            if version and pattern:
                configuration = dict(version.config_json or {})
                configuration["output_pattern"] = pattern
                version.config_json = configuration
        version = latest_version(model)
        if version:
            inference_schema = get_runtime(model.runtime_key).get_inference_parameter_schema()
            submitted_defaults: dict[str, object] = {}
            for field in inference_schema:
                field_name = field["name"]
                form_name = f"publisher_{field_name}"
                if field_name == "output_pattern":
                    submitted_defaults[field_name] = pattern
                elif field["type"] == "boolean":
                    submitted_defaults[field_name] = form_name in request.form
                elif form_name in request.form:
                    submitted_defaults[field_name] = request.form.get(form_name)
            normalized_defaults = validate_parameters(submitted_defaults, inference_schema, include_defaults=False)
            chat_defaults = {
                "use_history": "publisher_use_history" in request.form,
                "context_limit": max(1, int(request.form.get("publisher_context_limit") or 16_000)),
            }
            set_publisher_defaults(
                version,
                normalized_defaults,
                chat_defaults,
                allow_user_overrides="publisher_allow_overrides" in request.form,
            )
        db.session.commit()
        flash("Model metadata and runtime formatting were updated.", "success")
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), "error")
    return redirect(url_for("web.model_details", slug=model.slug))


@bp.post("/models/<slug>/reaction/<value>")
@login_required
def model_reaction(slug: str, value: str):
    model = _visible_model(slug)
    if value not in {"up", "down"}:
        abort(404)
    reaction = ModelReaction.query.filter_by(owner_id=current_user.id, model_id=model.id).first()
    if reaction and reaction.value == value:
        db.session.delete(reaction)
    elif reaction:
        reaction.value = value
    else:
        db.session.add(ModelReaction(owner_id=current_user.id, model_id=model.id, value=value))
    db.session.commit()
    return redirect(request.referrer or url_for("web.playground", slug=model.slug))


@bp.post("/models/<slug>/archive")
@login_required
def model_archive(slug: str):
    model = _visible_model(slug)
    state = ModelWorkspaceState.query.filter_by(owner_id=current_user.id, model_id=model.id).first()
    if not state:
        state = ModelWorkspaceState(owner_id=current_user.id, model_id=model.id)
        db.session.add(state)
    state.archived = True
    db.session.commit()
    get_runtime(model.runtime_key).unload_model()
    flash("Model removed from the active workspace. Its source and artifacts remain recoverable.", "success")
    return redirect(url_for("web.models"))


@bp.post("/models/<slug>/restore")
@login_required
def model_restore(slug: str):
    model = ModelRecord.query.filter_by(slug=slug).first()
    if not model or not (model.visibility == "public" or model.owner_id == current_user.id):
        abort(404)
    state = ModelWorkspaceState.query.filter_by(owner_id=current_user.id, model_id=model.id).first()
    if state:
        state.archived = False
        db.session.commit()
    flash("Model restored to the workspace.", "success")
    return redirect(url_for("web.models"))


@bp.post("/models/<slug>/unload")
@login_required
def model_unload(slug: str):
    model = _visible_model(slug)
    get_runtime(model.runtime_key).unload_model()
    flash(f"{model.name} was unloaded from memory.", "success")
    return redirect(url_for("web.model_details", slug=model.slug))


@bp.route("/playground", defaults={"slug": None}, methods=["GET", "POST"])
@bp.route("/playground/<slug>", methods=["GET", "POST"])
@login_required
def playground(slug: str | None):
    available_models = recent_chat_models(current_user.id)
    if not available_models:
        flash("Register or import a runnable model first.", "warning")
        return redirect(url_for("web.models"))
    if slug is None:
        preferred = next((item for item in available_models if item.slug == "storymaker-final"), available_models[0])
        slug = preferred.slug
    model = _visible_model(slug)
    if model.source_type == "device_local" and current_app.config.get("NODE_MODE") != "local_compute":
        flash("This private model runs in Vedock Desktop on the device that registered its folder.", "info")
        return redirect(url_for("web.model_details", slug=model.slug))
    version = latest_version(model)
    if not version or model not in available_models:
        abort(404)
    runtime = get_runtime(model.runtime_key)
    schema = schema_with_model_defaults(runtime.get_inference_parameter_schema(), model, version, current_user.id)
    capabilities = runtime.get_model_capabilities(version.storage_path)
    if capabilities.get("interaction") == "image_classification":
        result = None
        if request.method == "POST":
            upload = request.files.get("image")
            if not upload or not upload.filename:
                flash("Choose an image to classify.", "error")
            else:
                suffix = Path(upload.filename).suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                    flash("Choose a PNG, JPEG, WebP, or BMP image.", "error")
                else:
                    temporary = assert_writable_path(Path(current_app.config["STORAGE_ROOT"]) / "temporary" / "inference" / str(current_user.id) / new_id())
                    temporary.mkdir(parents=True)
                    image_path = assert_writable_path(temporary / f"input{suffix}")
                    try:
                        upload.save(image_path)
                        parameters = validate_parameters(_form_parameters(schema), schema)
                        result = runtime.infer(version.storage_path, str(image_path), parameters)
                    except (ParameterValidationError, ValueError, RuntimeError, OSError) as exc:
                        flash(str(getattr(exc, "errors", exc)), "error")
                    finally:
                        shutil.rmtree(temporary, ignore_errors=True)
        return render_template("web/image_runner.html", model=model, version=version, all_models=visible_models(current_user.id), schema_groups=schema_groups(schema), capabilities=capabilities, result=result)
    if capabilities.get("interaction") != "chat":
        contract = runner_contract(runtime, version.storage_path)
        result = None
        submitted_inputs: dict[str, object] = {}
        temporary = None
        if request.method == "POST":
            try:
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
                        temporary = temporary or assert_writable_path(Path(current_app.config["STORAGE_ROOT"]) / "temporary" / "inference" / str(current_user.id) / new_id())
                        temporary.mkdir(parents=True, exist_ok=True)
                        saved = assert_writable_path(temporary / f"{len(submitted_inputs)}{suffix}")
                        upload.save(saved)
                        submitted_inputs[name] = str(saved)
                    elif field["type"] == "boolean":
                        submitted_inputs[name] = name in request.form
                    else:
                        submitted_inputs[name] = request.form.get(name)
                inputs = validate_runner_inputs(submitted_inputs, contract)
                parameters = validate_parameters(_form_parameters(schema), schema)
                result = normalize_runtime_result(runtime.run(version.storage_path, inputs, parameters), contract)
            except (RunnerValidationError, ParameterValidationError, ValueError, RuntimeError, OSError) as exc:
                flash(str(getattr(exc, "errors", exc)), "error")
            finally:
                if temporary:
                    shutil.rmtree(temporary, ignore_errors=True)
        return render_template("web/task_runner.html", model=model, version=version, schema_groups=schema_groups(schema), capabilities=capabilities, contract=contract, result=result, submitted_inputs=submitted_inputs)
    conversation = _owned_conversation(request.values.get("conversation_id") or request.args.get("conversation"))
    if conversation and ((conversation.model_id or conversation.model_version.model_id) != model.id or conversation.model_version_id != version.id):
        abort(404)
    published_chat = publisher_defaults(version)["chat"]
    stored_chat = {**published_chat, **(((conversation.parameters_json or {}).get("_chat") or {}) if conversation else {})}
    model_conversations = (
        Conversation.query.filter_by(owner_id=current_user.id, model_id=model.id)
        .order_by(Conversation.updated_at.desc())
        .limit(20)
        .all()
    )
    all_conversations = (
        Conversation.query.filter_by(owner_id=current_user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(100)
        .all()
    )
    reaction = ModelReaction.query.filter_by(owner_id=current_user.id, model_id=model.id).first()
    reaction_counts = {
        "up": ModelReaction.query.filter_by(model_id=model.id, value="up").count(),
        "down": ModelReaction.query.filter_by(model_id=model.id, value="down").count(),
    }
    prompt = request.form.get("prompt", "") if request.method == "POST" else ""
    if request.method == "POST":
        if not prompt.strip():
            flash("Enter a message.", "error")
        elif len(prompt) > current_app.config["MAX_PROMPT_CHARS"]:
            flash("The message exceeds the configured length limit.", "error")
        else:
            try:
                parameters = validate_parameters(submitted_with_model_defaults(_form_parameters(schema), model, version, current_user.id), schema)
                use_history, context_override, context_limit = _web_context_settings()
                model_input = _chat_prompt(conversation.messages if conversation else [], prompt, context_limit, use_history, context_override)
                result = runtime.infer(version.storage_path, model_input, parameters)
                if conversation is None:
                    conversation = Conversation(owner_id=current_user.id, model_version=version, selected_model=model, title=(prompt.strip()[:80] or "New chat"), parameters_json=parameters)
                    db.session.add(conversation)
                    db.session.flush()
                conversation.parameters_json = {**parameters, "_chat": {"use_history": use_history, "context_override": context_override, "context_limit": context_limit}}
                conversation.updated_at = utcnow()
                db.session.add_all([Message(conversation=conversation, role="user", content=prompt), Message(conversation=conversation, role="assistant", content=result["text"])])
                db.session.commit()
                return redirect(url_for("web.playground", slug=model.slug, conversation=conversation.id))
            except (ParameterValidationError, ValueError, RuntimeError, OSError) as exc:
                flash(str(getattr(exc, "errors", exc)), "error")
    context_preview = _chat_prompt(conversation.messages if conversation else [], None, int(stored_chat.get("context_limit") or 16_000)) if conversation else ""
    return render_template("web/playground.html", model=model, version=version, schema_groups=schema_groups(schema), conversation=conversation, model_conversations=model_conversations, all_conversations=all_conversations, prompt=prompt, capabilities=capabilities, chat_settings=stored_chat, context_preview=context_preview, reaction=reaction, reaction_counts=reaction_counts)


@bp.post("/playground/<slug>/stream")
@login_required
def playground_stream(slug: str):
    model = _visible_model(slug)
    version = latest_version(model)
    if not version or model.source_type == "scratch_definition":
        abort(404)
    prompt = request.form.get("prompt", "").strip()
    if not prompt or len(prompt) > current_app.config["MAX_PROMPT_CHARS"]:
        return Response("event: error\ndata: {\"message\":\"Enter a valid message.\"}\n\n", status=422, mimetype="text/event-stream")
    runtime = get_runtime(model.runtime_key)
    schema = schema_with_model_defaults(runtime.get_inference_parameter_schema(), model, version, current_user.id)
    try:
        parameters = validate_parameters(submitted_with_model_defaults(_form_parameters(schema), model, version, current_user.id), schema)
    except ParameterValidationError as exc:
        return Response(f"event: error\ndata: {json.dumps({'message': str(exc.errors)})}\n\n", status=422, mimetype="text/event-stream")
    conversation = _owned_conversation(request.form.get("conversation_id"))
    if conversation and ((conversation.model_id or conversation.model_version.model_id) != model.id or conversation.model_version_id != version.id):
        abort(404)
    use_history, context_override, context_limit = _web_context_settings()
    if conversation is None:
        conversation = Conversation(owner_id=current_user.id, model_version=version, selected_model=model, title=prompt[:80] or "New chat", parameters_json=parameters)
        db.session.add(conversation)
        db.session.flush()
    conversation.parameters_json = {**parameters, "_chat": {"use_history": use_history, "context_override": context_override, "context_limit": context_limit}}
    conversation.updated_at = utcnow()
    db.session.add(Message(conversation=conversation, role="user", content=prompt))
    db.session.commit()
    model_input = _chat_prompt(conversation.messages, None, context_limit, use_history, context_override)
    conversation_id = conversation.id
    model_path = version.storage_path

    @stream_with_context
    def events():
        pieces: list[str] = []
        try:
            for token in runtime.stream_infer(model_path, model_input, parameters):
                pieces.append(token)
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            output = "".join(pieces).strip()
            saved = db.session.get(Conversation, conversation_id)
            if saved:
                db.session.add(Message(conversation=saved, role="assistant", content=output))
                saved.updated_at = utcnow()
                db.session.commit()
            yield f"event: done\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"
        except Exception as exc:
            db.session.rollback()
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return Response(events(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp.route("/create-model", methods=["GET", "POST"])
@login_required
def create_model():
    requested_task = request.values.get("task_type") or request.args.get("task") or "causal_lm"
    task_definition = next((task for task in TASK_OPTIONS if task["id"] == requested_task and task["available"]), None)
    if not task_definition:
        requested_task = "causal_lm"
        task_definition = next(task for task in TASK_OPTIONS if task["id"] == requested_task)
    runtime_key = str(task_definition["runtime"])
    runtime = get_runtime(runtime_key)
    accepted_schemas = {item["name"] for item in runtime.get_dataset_schema()}
    models = [model for model in visible_models(current_user.id) if model.task_type == requested_task]
    all_versions = DatasetVersion.query.filter(DatasetVersion.owner_id == current_user.id, DatasetVersion.validation_status.in_(["valid", "warning"])).order_by(DatasetVersion.created_at.desc()).all()
    versions = [version for version in all_versions if version.output_format in accepted_schemas]
    selected_model = next((model for model in models if model.slug == "storymaker-final"), models[0] if models else None)
    selected_model_reference = request.form.get("base_model_id") or request.args.get("base_model")
    if selected_model_reference:
        selected_model = next((model for model in models if model.id == selected_model_reference or model.slug == selected_model_reference), selected_model)
    schema = runtime.get_training_parameter_schema()
    if requested_task in {"tabular_regression", "tabular_classification"}:
        objective = "classification" if requested_task == "tabular_classification" else "regression"
        method = "logistic_fit" if objective == "classification" else "linear_fit"
        for field in schema:
            if field["name"] == "objective":
                field["default"] = objective
                field["choices"] = [objective]
            elif field["name"] == "training_method":
                field["default"] = method
                field["choices"] = [method]
    if selected_model:
        schema = schema_with_model_defaults(schema, selected_model, latest_version(selected_model), current_user.id)
    if request.method == "POST":
        try:
            build_mode = request.form.get("build_mode", "fine_tune")
            if build_mode == "merge":
                return redirect(url_for("web.merge_models"))
            if requested_task in {"pattern_sequence", "image_classification", "tabular_regression", "tabular_classification"}:
                build_mode = "scratch"
                source_type = "fast"
            else:
                source_type = "scratch" if build_mode == "scratch" else request.form.get("source_type", "existing")
            model = resolve_model_source(
                current_user,
                requested_task,
                source_type,
                {
                    "base_model_id": request.form.get("base_model_id"),
                    "repository": request.form.get("catalog_model") if source_type == "catalog" else request.form.get("repository"),
                    "revision": request.form.get("revision", "main"),
                    "local_path": request.form.get("local_path"),
                    "device_id": request.form.get("device_id"),
                    "model_name": request.form.get("base_model_name"),
                    "scratch_preset": request.form.get("scratch_preset"),
                    "architecture_family": request.form.get("architecture_family"),
                    "tokenizer_repository": request.form.get("tokenizer_repository"),
                    "n_layer": request.form.get("n_layer"),
                    "n_head": request.form.get("n_head"),
                    "n_embd": request.form.get("n_embd"),
                    "n_positions": request.form.get("n_positions"),
                },
            )
            if build_mode == "inference_only":
                db.session.commit()
                flash(f"{model.name} was registered without loading weights or starting training.", "success")
                return redirect(url_for("web.model_details", slug=model.slug))
            dataset = db.session.get(DatasetVersion, request.form.get("dataset_version_id"))
            if not dataset or dataset.owner_id != current_user.id or dataset.validation_status not in {"valid", "warning"}:
                raise JobError("Select one of your validated dataset versions.")
            if requested_task == "pattern_sequence":
                training_method = "pattern_fit"
            elif requested_task == "image_classification":
                training_method = "classifier_fit"
            elif requested_task == "tabular_classification":
                training_method = "logistic_fit"
            elif requested_task == "tabular_regression":
                training_method = "linear_fit"
            elif build_mode == "scratch":
                training_method = "scratch"
            elif build_mode == "continue_pretraining":
                training_method = "continue_pretraining"
            else:
                training_method = request.form.get("training_method", "lora")
            submitted = _form_parameters(schema)
            submitted["output_model_name"] = request.form.get("output_model_name", "model-output")
            submitted["training_method"] = training_method
            if requested_task in {"tabular_regression", "tabular_classification"}:
                submitted["objective"] = "classification" if requested_task == "tabular_classification" else "regression"
            normalized = validate_parameters(submitted, get_runtime(model.runtime_key).get_training_parameter_schema())
            project = ModelProject(
                owner=current_user,
                name=request.form.get("project_name", normalized["output_model_name"])[:160],
                task_type=requested_task,
                base_model=model,
                dataset_version=dataset,
                training_method=training_method,
                status="draft",
                config_json={
                    "build_mode": build_mode,
                    "source_type": source_type,
                    "parameters": normalized,
                },
            )
            db.session.add(project)
            if request.form.get("save_recipe"):
                db.session.add(TrainingRecipe(owner_id=current_user.id, name=f"{normalized['output_model_name']} recipe", runtime_key=model.runtime_key, config_json=normalized))
            db.session.commit()
            flash("Draft project saved. Review it before starting the final training step.", "success")
            return redirect(url_for("web.project_details", project_id=project.id))
        except (JobError, ModelSourceError, ParameterValidationError, ValueError) as exc:
            db.session.rollback()
            flash(str(getattr(exc, "errors", exc)), "error")
    display_schema = [field for field in schema if field["name"] not in {"output_model_name", "training_method"}]
    projects = ModelProject.query.filter_by(owner_id=current_user.id).order_by(ModelProject.created_at.desc()).limit(8).all()
    return render_template(
        "web/create_model.html",
        models=models,
        dataset_versions=versions,
        schema_groups=schema_groups(display_schema),
        selected_model=selected_model,
        task_options=TASK_OPTIONS,
        build_modes=BUILD_MODES if requested_task == "causal_lm" else [next(mode for mode in BUILD_MODES if mode["id"] == "scratch")],
        model_catalog=PRETRAINED_MODEL_CATALOG,
        scratch_presets=SCRATCH_PRESETS,
        devices=owner_devices(current_user.id),
        accepted_dataset_schemas=runtime.get_dataset_schema(),
        all_dataset_versions=all_versions,
        projects=projects,
        requested_task=requested_task,
    )


@bp.route("/projects/<project_id>", methods=["GET", "POST"])
@login_required
def project_details(project_id: str):
    project = db.session.get(ModelProject, project_id)
    if not project or project.owner_id != current_user.id:
        abort(404)
    if request.method == "POST":
        try:
            assert_training_enabled()
            if project.status not in {"draft", "failed"}:
                raise JobError(f"A project with status {project.status!r} cannot start another training job.")
            parameters = dict((project.config_json or {}).get("parameters") or {})
            job = enqueue_training(current_user, project.base_model, project.dataset_version, parameters, task_type=project.task_type)
            configuration = dict(project.config_json or {})
            configuration["job_id"] = job.id
            project.config_json = configuration
            project.status = "queued"
            db.session.commit()
            flash("Training task created. Claim it from the Vedock CLI or desktop app to use your own hardware.", "success")
            return redirect(url_for("web.job_details", job_id=job.id))
        except (JobError, ParameterValidationError, ValueError, OSError) as exc:
            db.session.rollback()
            flash(str(getattr(exc, "errors", exc)), "error")
    return render_template("web/project_details.html", project=project, report=system_report(), version=latest_version(project.base_model))


@bp.post("/projects/<project_id>/delete")
@login_required
def project_delete(project_id: str):
    project = db.session.get(ModelProject, project_id)
    if not project or project.owner_id != current_user.id:
        abort(404)
    name = project.name
    job_id = str((project.config_json or {}).get("job_id") or "")
    db.session.delete(project)
    db.session.commit()
    if job_id:
        flash(f"Project {name} was removed from drafts. Its separate training task and finalized model were preserved.", "success")
    else:
        flash(f"Draft project {name} was deleted.", "success")
    return redirect(url_for("web.create_model"))


@bp.get("/jobs")
@login_required
def jobs():
    records = Job.query.filter_by(owner_id=current_user.id).order_by(Job.created_at.desc()).all()
    return render_template("web/jobs.html", jobs=records)


@bp.get("/jobs/<job_id>")
@login_required
def job_details(job_id: str):
    job = db.session.get(Job, job_id)
    if not job or job.owner_id != current_user.id:
        abort(404)
    return render_template("web/job_details.html", job=job, logs=read_job_logs(job))


@bp.post("/jobs/<job_id>/cancel")
@login_required
def job_cancel(job_id: str):
    job = db.session.get(Job, job_id)
    if not job or job.owner_id != current_user.id:
        abort(404)
    try:
        request_cancellation(job, current_user)
        flash("Cancellation requested.", "info")
    except JobError as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.job_details", job_id=job.id))


@bp.post("/jobs/<job_id>/resume")
@login_required
def job_resume(job_id: str):
    job = db.session.get(Job, job_id)
    if not job or job.owner_id != current_user.id:
        abort(404)
    try:
        resume_job(job, current_user)
        flash("Task returned to the connected-device queue. Nothing starts until you press Run.", "success")
    except JobError as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.job_details", job_id=job.id))


@bp.post("/jobs/<job_id>/delete")
@login_required
def job_delete(job_id: str):
    job = db.session.get(Job, job_id)
    if not job or job.owner_id != current_user.id:
        abort(404)
    try:
        delete_job(job, current_user)
        flash("Task and logs deleted. Any finalized model version was preserved.", "success")
        return redirect(url_for("web.jobs"))
    except JobError as exc:
        flash(str(exc), "error")
        return redirect(url_for("web.job_details", job_id=job.id))


@bp.get("/conversations")
@login_required
def conversations():
    return redirect(url_for("web.playground"))


@bp.get("/conversations/<conversation_id>")
@login_required
def conversation_details(conversation_id: str):
    conversation = db.session.get(Conversation, conversation_id)
    if not conversation or conversation.owner_id != current_user.id:
        abort(404)
    return redirect(url_for("web.playground", slug=conversation.chat_model.slug, conversation=conversation.id))


@bp.post("/conversations/<conversation_id>/delete")
@login_required
def conversation_delete(conversation_id: str):
    conversation = db.session.get(Conversation, conversation_id)
    if not conversation or conversation.owner_id != current_user.id:
        abort(404)
    slug = conversation.chat_model.slug
    db.session.delete(conversation)
    db.session.commit()
    flash("Chat deleted.", "success")
    return redirect(url_for("web.playground", slug=slug))


@bp.route("/merge", methods=["GET", "POST"])
@login_required
def merge_models():
    models = visible_models(current_user.id)
    report = None
    version_a = version_b = None
    if request.method == "POST":
        try:
            first = next((model for model in models if model.id == request.form.get("model_a")), None)
            second = next((model for model in models if model.id == request.form.get("model_b")), None)
            if not first or not second or first.id == second.id:
                raise MergeError("Select two different accessible models.")
            version_a, version_b = resolve_latest_pair(first, second)
            report = compatibility_report(version_a, version_b)
            if request.form.get("action") == "merge":
                method = request.form.get("merge_method", "auto")
                if method == "auto":
                    method = "weighted_adapter" if report.get("lora_safe") else "linear"
                executor = execute_weighted_adapter_merge if method == "weighted_adapter" else execute_linear_merge
                merge, output = executor(version_a, version_b, float(request.form.get("weight_a", 0.5)), float(request.form.get("weight_b", 0.5)), current_user, request.form.get("output_name", "Merged model"))
                flash("Compatible models were merged into a new immutable version.", "success")
                return redirect(url_for("web.model_details", slug=output.model.slug))
        except (MergeError, ValueError, OSError, RuntimeError) as exc:
            if request.form.get("action") == "merge" and version_a and version_b:
                try:
                    record_failed_merge_attempt(version_a, version_b, request.form.get("merge_method", "auto"), [float(request.form.get("weight_a", 0.5)), float(request.form.get("weight_b", 0.5))], current_user, report or {}, str(exc))
                except Exception:
                    db.session.rollback()
            flash(str(exc), "error")
        except Exception as exc:
            db.session.rollback()
            flash(f"Experimental merge attempt failed safely: {exc}", "error")
    return render_template("web/merge.html", models=models, report=report)


@bp.get("/developer")
@login_required
def developer():
    control_plane = current_app.config.get("CONTROL_PLANE_URL", "").rstrip(":/")
    api_base = f"{control_plane}/api/v1" if control_plane else f"http://127.0.0.1:{current_app.config['APP_PORT']}/api/v1"
    return render_template("web/developer.html", api_base=api_base)


@bp.get("/downloads/vedock-cli.zip")
def cli_download():
    return redirect(url_for("web.installer_download"))


@bp.get("/downloads/vedock-installer.exe")
def installer_download():
    root = Path(current_app.config["DISTRIBUTION_ROOT"])
    # A running Windows executable locks its own filename. Builds are first
    # published to the unlocked current slot so web downloads can be updated
    # without killing an installer window the user may still be viewing.
    current = root / "VedockInstaller-current.exe"
    path = current if current.is_file() else root / "VedockInstaller.exe"
    if not path.is_file():
        abort(404, description="The Windows installer has not been built on this node.")
    return send_file(path, as_attachment=True, download_name=f"{current_app.config['APP_SHORT_NAME']}-Installer.exe", mimetype="application/vnd.microsoft.portable-executable")


@bp.get("/downloads/vedock-node.zip")
def node_download():
    return send_file(build_node_archive(), as_attachment=True, download_name="vedock-local-node.zip", mimetype="application/zip")


@bp.get("/downloads/vedock-client.zip")
def client_download():
    return send_file(build_client_archive(), as_attachment=True, download_name="vedock-connected-client.zip", mimetype="application/zip")


@bp.get("/downloads/install-linux.sh")
def linux_installer_download():
    path = Path(__file__).resolve().parents[2] / "installer" / "install-linux.sh"
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name="install-vedock.sh", mimetype="text/x-shellscript")


@bp.get("/settings")
@login_required
def settings():
    return render_template("web/settings.html")


@bp.get("/system")
@login_required
def system():
    return render_template("web/system.html", report=system_report())
