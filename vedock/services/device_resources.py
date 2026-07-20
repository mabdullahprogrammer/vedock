from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from sqlalchemy import select

from vedock.extensions import db
from vedock.models import (
    ConnectedDevice,
    DatasetVersion,
    DeviceResource,
    ModelRecord,
    ModelVersion,
    RawDataset,
    User,
    new_id,
    utcnow,
)


class DeviceResourceError(ValueError):
    pass


def device_reference(resource_id: str) -> str:
    return f"device://{resource_id}"


def resource_id_from_reference(value: str | None) -> str | None:
    raw = str(value or "")
    return raw[9:] if raw.startswith("device://") and len(raw) > 9 else None


def record_device(owner: User, device_uid: str, name: str, details: dict[str, Any] | None = None) -> ConnectedDevice:
    uid = str(device_uid or "").strip()[:120]
    if not uid:
        raise DeviceResourceError("A stable connected-device ID is required.")
    record = ConnectedDevice.query.filter_by(owner_id=owner.id, device_uid=uid).first()
    if not record:
        record = ConnectedDevice(owner=owner, device_uid=uid, name="Vedock device")
        db.session.add(record)
    details = details or {}
    record.name = str(name or "Vedock device").strip()[:160]
    record.platform = str(details.get("platform") or "")[:160]
    # Do not accept paths, package lists, or server-derived hardware data here.
    record.capabilities_json = {
        key: details[key]
        for key in ("cuda_available", "gpu_count", "ram_total_bytes", "runtime_status")
        if key in details
    }
    record.last_seen_at = utcnow()
    db.session.commit()
    return record


def owner_devices(owner_id: int) -> list[ConnectedDevice]:
    return list(
        db.session.scalars(
            select(ConnectedDevice)
            .where(ConnectedDevice.owner_id == owner_id)
            .order_by(ConnectedDevice.last_seen_at.desc())
        )
    )


def _owned_device(owner: User, device_uid: str) -> ConnectedDevice:
    device = ConnectedDevice.query.filter_by(owner_id=owner.id, device_uid=str(device_uid or "")).first()
    if not device:
        raise DeviceResourceError("Connect and sign in to the Vedock desktop app before choosing a device-local path.")
    return device


def _path_hint(locator: str) -> str:
    windows = PureWindowsPath(locator)
    posix = PurePosixPath(locator)
    if not (windows.is_absolute() or posix.is_absolute()):
        raise DeviceResourceError("Enter an absolute path from the connected device.")
    return (windows.name or posix.name or "Local resource")[:255]


def request_device_path(
    owner: User,
    device_uid: str,
    kind: str,
    locator: str,
    *,
    display_name: str | None = None,
    runtime_key: str | None = None,
    task_type: str | None = None,
    output_schema: str | None = None,
) -> DeviceResource:
    _owned_device(owner, device_uid)
    if kind not in {"model", "dataset", "checkpoint"}:
        raise DeviceResourceError("Unsupported connected-device resource type.")
    raw = str(locator or "").strip().strip('"')
    if not raw or len(raw) > 4096:
        raise DeviceResourceError("Enter a valid device-local path.")
    hint = _path_hint(raw)
    resource = DeviceResource(
        owner=owner,
        device_uid=device_uid,
        kind=kind,
        status="pending_device",
        display_name=str(display_name or hint)[:160],
        path_hint=hint,
        pending_locator=raw,
        runtime_key=runtime_key,
        task_type=task_type,
        output_schema=output_schema,
        metadata_json={"created_from": "web_path_request"},
    )
    db.session.add(resource)
    db.session.flush()
    _ensure_resource_records(resource)
    db.session.commit()
    return resource


def register_device_resource(
    owner: User,
    device_uid: str,
    payload: dict[str, Any],
    *,
    resource: DeviceResource | None = None,
) -> DeviceResource:
    _owned_device(owner, device_uid)
    kind = str(payload.get("kind") or (resource.kind if resource else ""))
    if kind not in {"model", "dataset", "checkpoint"}:
        raise DeviceResourceError("Resource kind must be model, checkpoint, or dataset.")
    if resource and (resource.owner_id != owner.id or resource.device_uid != device_uid):
        raise DeviceResourceError("This resource request belongs to another device.")
    if not resource:
        resource = DeviceResource(owner=owner, device_uid=device_uid, kind=kind, display_name="Local resource")
        db.session.add(resource)
    status = str(payload.get("status") or "ready")
    if status not in {"ready", "missing", "invalid"}:
        raise DeviceResourceError("Resource verification status is not supported.")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    resource.kind = kind
    resource.status = status
    resource.display_name = str(payload.get("name") or resource.display_name or resource.path_hint or "Local resource")[:160]
    resource.path_hint = str(payload.get("path_hint") or resource.path_hint or "Private local path")[:255]
    resource.runtime_key = str(payload.get("runtime") or resource.runtime_key or "")[:80] or None
    resource.task_type = str(payload.get("task_type") or resource.task_type or "")[:50] or None
    resource.output_schema = str(payload.get("output_schema") or resource.output_schema or "")[:40] or None
    resource.size_bytes = max(0, int(payload.get("size_bytes") or 0))
    sha256 = str(payload.get("sha256") or "").lower()
    resource.sha256 = sha256[:64] if len(sha256) == 64 else None
    resource.metadata_json = {**(resource.metadata_json or {}), **metadata}
    resource.pending_locator = None
    resource.last_verified_at = utcnow()
    db.session.flush()
    _ensure_resource_records(resource)
    db.session.commit()
    return resource


def _ensure_resource_records(resource: DeviceResource) -> None:
    """Materialize searchable metadata records without copying the artifact."""
    if resource.kind in {"model", "checkpoint"}:
        version = ModelVersion.query.filter_by(storage_path=resource.reference).first()
        if version:
            version.status = "available" if resource.status == "ready" else "unavailable"
            version.sha256 = resource.sha256
            version.config_json = resource.metadata_json or {}
            version.metadata_json = {**(version.metadata_json or {}), "device_resource_id": resource.id, "device_id": resource.device_uid}
            version.model.name = resource.display_name
            version.model.runtime_key = resource.runtime_key or version.model.runtime_key
            version.model.task_type = resource.task_type or version.model.task_type
            return
        model_id = new_id()
        slug_base = "".join(character.lower() if character.isalnum() else "-" for character in resource.display_name).strip("-") or "local-model"
        model = ModelRecord(
            id=model_id,
            owner_id=resource.owner_id,
            slug=f"{slug_base[:120]}-{model_id[:8]}",
            name=resource.display_name,
            description="Private model registered from the owner's connected Vedock device.",
            task_type=resource.task_type or "causal_lm",
            runtime_key=resource.runtime_key or "transformers_text",
            source_type="device_local",
            source_path=resource.reference,
            visibility="private",
        )
        version = ModelVersion(
            model=model,
            version_number=1,
            label="Connected-device source",
            storage_path=resource.reference,
            status="available" if resource.status == "ready" else "unavailable",
            config_json=resource.metadata_json or {},
            metadata_json={"device_resource_id": resource.id, "device_id": resource.device_uid, "artifact_location": "owner_device"},
            sha256=resource.sha256,
        )
        db.session.add_all([model, version])
        return

    raw = RawDataset.query.filter_by(storage_path=f"{resource.reference}/raw").first()
    metadata = resource.metadata_json or {}
    if not raw:
        raw = RawDataset(
            id=new_id(),
            owner_id=resource.owner_id,
            name=resource.display_name,
            description="Private dataset held by the owner's connected Vedock device.",
            source_type="device_local",
            original_filename=resource.path_hint or resource.display_name,
            storage_path=f"{resource.reference}/raw",
            file_format=str(metadata.get("file_format") or "jsonl")[:20],
            mime_type=str(metadata.get("mime_type") or "application/x-ndjson")[:120],
            size_bytes=resource.size_bytes,
            sha256=resource.sha256 or "0" * 64,
            inspection_status="completed" if resource.status == "ready" else resource.status,
            detected_schema_json={"columns": metadata.get("columns") or [], "device_resource_id": resource.id},
            statistics_json={"recommendations": metadata.get("recommendations") or [], "artifact_location": "owner_device"},
            row_count=int(metadata.get("row_count") or 0),
        )
        db.session.add(raw)
        db.session.flush()
    else:
        raw.inspection_status = "completed" if resource.status == "ready" else resource.status
        raw.row_count = int(metadata.get("row_count") or raw.row_count or 0)
        raw.size_bytes = resource.size_bytes
        raw.sha256 = resource.sha256 or raw.sha256
        raw.file_format = str(metadata.get("file_format") or raw.file_format)[:20]
        raw.detected_schema_json = {"columns": metadata.get("columns") or [], "device_resource_id": resource.id}
        raw.statistics_json = {"recommendations": metadata.get("recommendations") or [], "artifact_location": "owner_device"}
    if resource.output_schema and resource.status == "ready":
        version_path = f"{resource.reference}/processed"
        version = DatasetVersion.query.filter_by(storage_path=version_path).first()
        if not version:
            version = DatasetVersion(
                raw_dataset=raw,
                owner_id=resource.owner_id,
                version_number=1,
                output_format=resource.output_schema,
                storage_path=version_path,
                transformation_config=metadata.get("transformations") or [],
                field_mapping=metadata.get("field_mapping") or {},
                validation_status=str(metadata.get("validation_status") or "valid"),
                validation_json=metadata.get("validation") or {"status": "valid", "source": "connected_device"},
                row_count=int(metadata.get("row_count") or 0),
                invalid_row_count=int(metadata.get("invalid_row_count") or 0),
                token_estimate=int(metadata.get("token_estimate") or 0),
                sha256=resource.sha256 or "0" * 64,
            )
            db.session.add(version)


def resource_for_reference(reference: str | None) -> DeviceResource | None:
    resource_id = resource_id_from_reference(reference)
    return db.session.get(DeviceResource, resource_id) if resource_id else None


def required_job_resources(model_version: ModelVersion, dataset_version: DatasetVersion) -> list[DeviceResource]:
    output = []
    for reference in (model_version.storage_path, dataset_version.storage_path):
        resource = resource_for_reference(reference.split("/processed", 1)[0].split("/raw", 1)[0])
        if resource:
            output.append(resource)
    return output
