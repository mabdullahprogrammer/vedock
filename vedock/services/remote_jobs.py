from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from flask import current_app
from werkzeug.datastructures import FileStorage

from vedock.extensions import db
from vedock.models import DatasetVersion, Job, ModelRecord, ModelVersion, new_id, utcnow
from vedock.runtimes import get_runtime
from vedock.runtimes.parameters import validate_parameters

from .jobs import JobError, append_job_log
from .model_references import parse_model_reference
from .paths import allocate_directory, assert_writable_path
from .training import re_safe_slug
from .device_resources import required_job_resources, resource_for_reference


FINAL_ARTIFACT_NAMES = {
    "config.json",
    "generation_config.json",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "model.safetensors",
    "model.safetensors.index.json",
    "adapter_model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "vedock_training.json",
    "README.md",
    "metadata.json",
    "pattern_model.json",
    "predictor.json",
    "classifier.json",
}


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def job_manifest(job: Job) -> dict[str, Any]:
    if job.job_type != "training":
        raise JobError("Only training tasks have a compute manifest.")
    configuration = job.config_json or {}
    model = db.session.get(ModelRecord, configuration.get("model_id"))
    version = db.session.get(ModelVersion, configuration.get("base_model_version_id"))
    dataset = db.session.get(DatasetVersion, configuration.get("dataset_version_id"))
    if not model or not version or not dataset:
        raise JobError("This task references a model or dataset version that no longer exists.")
    model_resource = resource_for_reference(version.storage_path)
    if model_resource:
        reference_kind = "device"
        artifact_required = False
        artifact_size = model_resource.size_bytes
        reference_value = None
    else:
        reference = parse_model_reference(version.storage_path)
        reference_kind = reference.kind
        artifact_required = reference.kind == "local"
        artifact_size = _directory_size(Path(reference.source)) if artifact_required and Path(reference.source).is_dir() else 0
        reference_value = version.storage_path if reference.kind in {"huggingface", "scratch"} else None
    dataset_reference = str(dataset.storage_path).split("/processed", 1)[0]
    dataset_resource = resource_for_reference(dataset_reference)
    dataset_size = dataset_resource.size_bytes if dataset_resource else (Path(dataset.storage_path).stat().st_size if Path(dataset.storage_path).is_file() else 0)
    return {
        "job": job.to_dict(),
        "runtime": configuration.get("runtime"),
        "task_type": configuration.get("task_type"),
        "parameters": configuration.get("parameters") or {},
        "model": {
            "id": model.id,
            "slug": model.slug,
            "name": model.name,
            "runtime": model.runtime_key,
            "task_type": model.task_type,
            "source_type": model.source_type,
            "reference_kind": reference_kind,
            "reference": reference_value,
            "device_resource_id": model_resource.id if model_resource else None,
            "required_device_id": model_resource.device_uid if model_resource else None,
            "resource_status": model_resource.status if model_resource else None,
            "version_id": version.id,
            "version_status": version.status,
            "version_config": version.config_json or {},
            "artifact_required": artifact_required,
            "artifact_size_bytes": artifact_size,
        },
        "dataset": {
            "id": dataset.id,
            "name": dataset.raw_dataset.name,
            "schema": dataset.output_format,
            "rows": dataset.row_count,
            "sha256": dataset.sha256,
            "size_bytes": dataset_size,
            "device_resource_id": dataset_resource.id if dataset_resource else None,
            "required_device_id": dataset_resource.device_uid if dataset_resource else None,
            "resource_status": dataset_resource.status if dataset_resource else None,
        },
    }


def edit_waiting_job(job: Job, submitted: dict[str, Any]) -> Job:
    if job.status != "awaiting_device":
        raise JobError("Only a task that is still waiting for a device can be edited.")
    configuration = dict(job.config_json or {})
    runtime = get_runtime(str(configuration.get("runtime") or ""))
    current = dict(configuration.get("parameters") or {})
    current.update(submitted)
    configuration["parameters"] = validate_parameters(current, runtime.get_training_parameter_schema())
    job.config_json = configuration
    db.session.commit()
    append_job_log(job, "Training parameters edited before device claim", parameters=configuration["parameters"])
    return job


def claim_job(job: Job, device_id: str, device_name: str) -> Job:
    device_id = str(device_id or "").strip()[:120]
    if not device_id:
        raise JobError("A stable device ID is required.")
    if job.status == "claimed" and job.claimed_by_device == device_id:
        job.last_heartbeat_at = utcnow()
        db.session.commit()
        return job
    if job.status != "awaiting_device":
        raise JobError(f"A task with status {job.status!r} cannot be claimed.")
    configuration = job.config_json or {}
    version = db.session.get(ModelVersion, configuration.get("base_model_version_id"))
    dataset = db.session.get(DatasetVersion, configuration.get("dataset_version_id"))
    if not version or not dataset:
        raise JobError("This task references a model or dataset version that no longer exists.")
    resources = required_job_resources(version, dataset)
    for resource in resources:
        if resource.device_uid != device_id:
            raise JobError(f"{resource.display_name} is private to another connected device. Run this task on that device.")
        if resource.status != "ready":
            raise JobError(f"{resource.display_name} has not been verified on this device yet. Open Vedock Desktop and sync local resources.")
    job.status = "claimed"
    job.current_stage = "claimed_by_local_device"
    job.claimed_by_device = device_id
    job.device_name = str(device_name or "Vedock device").strip()[:160]
    job.last_heartbeat_at = utcnow()
    db.session.commit()
    append_job_log(job, "Task claimed by local device", device_id=device_id, device_name=job.device_name)
    return job


def release_job(job: Job, device_id: str, reason: str = "") -> Job:
    """Return a claimed-but-not-running task to its owner queue."""
    if job.status != "claimed":
        raise JobError("Only a claimed task that has not started can be released.")
    if job.claimed_by_device != str(device_id or ""):
        raise JobError("Only the device that claimed this task can release it.")
    previous_device = job.device_name or "Vedock device"
    job.status = "awaiting_device"
    job.current_stage = "waiting_for_device"
    job.claimed_by_device = None
    job.device_name = None
    job.last_heartbeat_at = None
    job.cancel_requested = False
    db.session.commit()
    append_job_log(job, "Task released back to the device queue", device_name=previous_device, reason=str(reason or "")[:500])
    return job


def update_remote_job(job: Job, device_id: str, payload: dict[str, Any]) -> Job:
    if not job.claimed_by_device or job.claimed_by_device != str(device_id or ""):
        raise JobError("This device did not claim the task.")
    allowed_statuses = {"claimed", "running", "awaiting_publish", "completed", "failed", "cancelled"}
    status = str(payload.get("status") or job.status)
    if status not in allowed_statuses:
        raise JobError("The reported job status is not supported.")
    job.status = status
    job.current_stage = str(payload.get("stage") or job.current_stage)[:80]
    if payload.get("progress") is not None:
        job.progress = min(100, max(0, int(payload["progress"])))
    job.last_heartbeat_at = utcnow()
    if status == "running" and not job.started_at:
        job.started_at = utcnow()
    if status in {"completed", "failed", "cancelled"}:
        job.finished_at = utcnow()
    if payload.get("error"):
        job.error_message = str(payload["error"])[:20_000]
    db.session.commit()
    message = str(payload.get("message") or "Local device status updated")[:2_000]
    append_job_log(job, message, device_id=device_id, stage=job.current_stage, progress=job.progress, metrics=payload.get("metrics"))
    return job


def model_artifact_archive(job: Job) -> Path:
    manifest = job_manifest(job)
    if not manifest["model"]["artifact_required"]:
        raise JobError("This base model is resolved from a repository or scratch definition and has no hosted artifact download.")
    version = db.session.get(ModelVersion, manifest["model"]["version_id"])
    source = Path(version.storage_path) if version else Path()
    if not source.is_dir():
        raise JobError("The base-model artifact directory is unavailable.")
    cache = assert_writable_path(Path(current_app.config["STORAGE_ROOT"]) / "remote-cache" / "models")
    cache.mkdir(parents=True, exist_ok=True)
    output = assert_writable_path(cache / f"{version.id}.zip")
    if output.is_file() and output.stat().st_mtime >= max(item.stat().st_mtime for item in source.rglob("*") if item.is_file()):
        return output
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for item in sorted(source.rglob("*")):
            if item.is_file() and "checkpoints" not in item.parts:
                archive.write(item, item.relative_to(source).as_posix())
    temporary.replace(output)
    return output


def _allowed_final_file(path: PurePosixPath) -> bool:
    name = path.name
    if any(part.lower().startswith("checkpoint") for part in path.parts):
        return False
    if name in FINAL_ARTIFACT_NAMES:
        return True
    return (
        (name.startswith("model-") and name.endswith(".safetensors"))
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
        or name.endswith(".tiktoken")
    )


def _hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def finalize_remote_job(job: Job, device_id: str, upload: FileStorage, metadata: dict[str, Any]) -> ModelVersion:
    if job.claimed_by_device != str(device_id or ""):
        raise JobError("This device did not claim the task.")
    if job.status not in {"running", "awaiting_publish"}:
        raise JobError("The task is not waiting for a finalized artifact.")
    if not upload or not upload.filename:
        raise JobError("Upload the finalized model artifact ZIP.")
    model_id, version_id = new_id(), new_id()
    destination = allocate_directory("models", str(job.owner_id), model_id, version_id)
    archive_path = assert_writable_path(Path(current_app.config["STORAGE_ROOT"]) / "temporary" / f"publish-{job.id}.zip")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    upload.save(archive_path)
    extracted = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if not members:
                raise JobError("The finalized model archive is empty.")
            for member in members:
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise JobError("The finalized archive contains an unsafe path.")
                # A single wrapper directory is ignored to keep artifacts flat.
                if len(relative.parts) > 1 and all(PurePosixPath(item.filename).parts[:1] == relative.parts[:1] for item in members):
                    relative = PurePosixPath(*relative.parts[1:])
                if not relative.parts or not _allowed_final_file(relative):
                    continue
                target = assert_writable_path(destination / Path(*relative.parts))
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                extracted += 1
        runtime_key = str((job.config_json or {}).get("runtime") or "transformers_text")
        runtime_artifacts = {
            "pattern_sequence": ("pattern_model.json",),
            "tabular_prediction": ("predictor.json",),
            "sklearn_image": ("classifier.json",),
        }
        required_artifacts = runtime_artifacts.get(runtime_key, ("config.json", "adapter_config.json"))
        if not extracted or not any((destination / name).is_file() for name in required_artifacts):
            raise JobError("No usable model configuration and inference artifact files were found.")
        configuration = job.config_json or {}
        base = db.session.get(ModelRecord, configuration.get("model_id"))
        params = configuration.get("parameters") or {}
        published = dict(metadata.get("publisher_defaults") or {})
        if published:
            inference_schema = get_runtime(str(configuration.get("runtime") or "transformers_text")).get_inference_parameter_schema()
            published_parameters = validate_parameters(dict(published.get("inference_parameters") or {}), inference_schema, include_defaults=False)
            chat = dict(published.get("chat") or {})
            if "context_limit" in chat:
                chat["context_limit"] = max(1, int(chat["context_limit"]))
            if "use_history" in chat:
                chat["use_history"] = bool(chat["use_history"])
            metadata = {
                **metadata,
                "publisher_defaults": {
                    "inference_parameters": published_parameters,
                    "chat": chat,
                    "allow_user_overrides": bool(published.get("allow_user_overrides", True)),
                },
            }
        name = str(metadata.get("name") or params.get("output_model_name") or "Finalized model")[:160]
        model = ModelRecord(
            id=model_id,
            owner_id=job.owner_id,
            slug=f"{re_safe_slug(name)}-{model_id[:8]}",
            name=name,
            description=str(metadata.get("description") or f"Trained on the owner's local Vedock device from {base.name if base else 'a Vedock base model'}.")[:5000],
            task_type=str(configuration.get("task_type") or (base.task_type if base else "causal_lm")),
            runtime_key=str(configuration.get("runtime") or (base.runtime_key if base else "transformers_text")),
            source_type="remote_local_training",
            source_path=str(destination),
            visibility="public" if metadata.get("publish") else "private",
        )
        version = ModelVersion(
            id=version_id,
            model=model,
            version_number=1,
            label="Local-device training",
            storage_path=str(destination),
            base_model_path=base.source_path if base else None,
            status="completed",
            config_json=params,
            metadata_json={"training_location": "owner_device", "device_name": job.device_name, "job_id": job.id, **metadata},
            sha256=_hash_directory(destination),
        )
        db.session.add_all([model, version])
        db.session.flush()
        job.result_model_version_id = version.id
        job.status = "completed"
        job.current_stage = "published" if model.visibility == "public" else "finalized_private"
        job.progress = 100
        job.finished_at = utcnow()
        db.session.commit()
        append_job_log(job, "Finalized inference artifact received", files=extracted, model_id=model.id, visibility=model.visibility, sha256=version.sha256)
        return version
    except Exception:
        db.session.rollback()
        import shutil

        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        archive_path.unlink(missing_ok=True)
