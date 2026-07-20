from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable


SCHEMA_FIELDS = {
    "text_completion": ("text",),
    "prompt_response": ("prompt", "response"),
    "instruction": ("instruction", "input", "output"),
    "chat": ("messages",),
    "classification": ("text", "label"),
    "image_classification": ("image", "label"),
    "tabular_supervised": ("features", "target"),
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

ALIASES = {
    "text": ("text", "content", "story", "document"),
    "prompt": ("prompt", "instruction", "question", "input"),
    "response": ("response", "output", "answer", "story", "completion"),
    "instruction": ("instruction", "prompt", "question"),
    "input": ("input", "context"),
    "output": ("output", "response", "answer", "story"),
    "label": ("label", "class", "category", "target"),
    "image": ("image", "image_path", "path", "file"),
    "target": ("target", "label", "sales", "price", "revenue", "outcome"),
}


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _hash_directory(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    return digest.hexdigest(), size


def inspect_model_folder(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Model folder does not exist on this device: {path}")
    configuration = {}
    for name in ("config.json", "adapter_config.json", "pattern_model.json", "predictor.json", "classifier.json"):
        candidate = path / name
        if candidate.is_file():
            try:
                configuration = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                configuration = {}
            break
    markers = {
        "config.json", "adapter_config.json", "pattern_model.json", "predictor.json", "classifier.json",
        "model.safetensors", "pytorch_model.bin", "adapter_model.safetensors",
    }
    filenames = {item.name for item in path.iterdir() if item.is_file()}
    if not filenames.intersection(markers) and not any(name.startswith(("model-", "pytorch_model-")) for name in filenames):
        raise ValueError("The selected folder has no recognized Vedock, Transformers, adapter, pattern, image, or tabular model artifact.")
    runtime = "transformers_text"
    task_type = "causal_lm"
    if "pattern_model.json" in filenames:
        runtime, task_type = "pattern_sequence", "pattern_sequence"
    elif "classifier.json" in filenames:
        runtime, task_type = "sklearn_image", "image_classification"
    elif "predictor.json" in filenames:
        runtime = "tabular_prediction"
        task_type = "tabular_classification" if configuration.get("objective") == "classification" else "tabular_regression"
    digest, size = _hash_directory(path)
    return {
        "kind": "model",
        "name": path.name,
        "path_hint": path.name,
        "runtime": runtime,
        "task_type": task_type,
        "size_bytes": size,
        "sha256": digest,
        "metadata": {
            "model_type": configuration.get("model_type") or configuration.get("architecture_family"),
            "architectures": configuration.get("architectures") or [],
            "files": sorted(filenames)[:200],
            "validated_on_device": True,
        },
    }


def _records(path: Path, limit: int = 100_000) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            yield from (dict(row) for _, row in zip(range(limit), csv.DictReader(stream)))
        return
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for _, line in zip(range(limit), stream):
                if line.strip():
                    value = json.loads(line)
                    yield value if isinstance(value, dict) else {"text": value}
        return
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
        rows = value if isinstance(value, list) else value.get("data", [value]) if isinstance(value, dict) else [value]
        for row in rows[:limit]:
            yield row if isinstance(row, dict) else {"text": row}
        return
    if suffix == ".txt":
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for _, line in zip(range(limit), stream):
                if line.strip():
                    yield {"text": line.strip()}
        return
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ValueError("Parquet inspection needs the text-training runtime (pandas/pyarrow) on this device.") from exc
        for row in pd.read_parquet(path).head(limit).to_dict(orient="records"):
            yield row
        return
    raise ValueError("Choose CSV, JSON, JSONL, NDJSON, TXT, or Parquet. Image ZIP preparation is handled by the image dataset workflow.")


def inspect_dataset_file(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Dataset file does not exist on this device: {path}")
    if path.suffix.lower() == ".zip":
        sample = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if member.is_dir() or relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label = relative.parts[-2] if len(relative.parts) > 1 else "unlabeled"
                sample.append({"image": relative.as_posix(), "label": label})
                if len(sample) >= 1000:
                    break
        if not sample:
            raise ValueError("The ZIP contains no supported image files. Use class/image.ext folders for image classification.")
    else:
        sample = list(_records(path, 1000))
    if not sample:
        raise ValueError("The selected dataset contains no readable records.")
    columns = list(dict.fromkeys(str(key) for row in sample for key in row))
    lower = {column.lower(): column for column in columns}
    mapping = {field: next((lower[name] for name in aliases if name in lower), "") for field, aliases in ALIASES.items()}
    if path.suffix.lower() == ".zip":
        schema = "image_classification"
    elif "messages" in lower:
        schema = "chat"
    elif mapping["prompt"] and mapping["response"]:
        schema = "prompt_response"
    elif mapping["text"] and mapping["label"]:
        schema = "classification"
    elif mapping["text"]:
        schema = "text_completion"
    elif mapping["target"] and len(columns) > 1:
        schema = "tabular_supervised"
    else:
        schema = ""
    duplicate_count = len(sample) - len({json.dumps(row, sort_keys=True, default=str) for row in sample})
    empty_rows = sum(not any(str(value or "").strip() for value in row.values()) for row in sample)
    recommendations = []
    if duplicate_count:
        recommendations.append({"level": "quality", "title": "Remove duplicate examples", "rows": duplicate_count})
    if empty_rows:
        recommendations.append({"level": "quality", "title": "Remove empty records", "rows": empty_rows})
    return {"path": str(path), "name": path.stem, "file_format": path.suffix.lower().lstrip("."), "columns": columns, "sample": sample[:8], "recommended_schema": schema, "recommended_mapping": mapping, "recommendations": recommendations}


def _mapped_row(row: dict[str, Any], schema: str, mapping: dict[str, Any], columns: list[str]) -> dict[str, Any] | None:
    if schema == "tabular_supervised":
        target_field = str(mapping.get("target") or "")
        feature_fields = mapping.get("features") or [column for column in columns if column != target_field]
        if isinstance(feature_fields, str):
            feature_fields = [item.strip() for item in feature_fields.split(",") if item.strip()]
        if not target_field or row.get(target_field) in {None, ""}:
            return None
        return {"features": {field: row.get(field) for field in feature_fields}, "target": row.get(target_field)}
    if schema == "chat":
        messages_field = str(mapping.get("messages") or "messages")
        if isinstance(row.get(messages_field), list):
            return {"messages": row[messages_field]}
        prompt_field, response_field = str(mapping.get("prompt") or ""), str(mapping.get("response") or "")
        if not prompt_field or not response_field:
            return None
        return {"messages": [{"role": "user", "content": str(row.get(prompt_field) or "").strip()}, {"role": "assistant", "content": str(row.get(response_field) or "").strip()}]}
    output = {}
    for field in SCHEMA_FIELDS[schema]:
        source = str(mapping.get(field) or "")
        if not source:
            return None
        value = row.get(source)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        output[field] = value.strip() if isinstance(value, str) else value
    return output


def prepare_dataset_file(path_value: str, schema: str = "auto", field_mapping: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
    inspection = inspect_dataset_file(path_value)
    selected = inspection["recommended_schema"] if schema in {"", "auto"} else schema
    if selected not in SCHEMA_FIELDS:
        raise ValueError("Vedock could not infer a training format. Choose a schema and field mapping in the desktop app.")
    mapping = dict(inspection["recommended_mapping"])
    mapping.update(field_mapping or {})
    if selected == "tabular_supervised":
        target = mapping.get("target")
        mapping["features"] = mapping.get("features") or [column for column in inspection["columns"] if column != target]
    root = Path(os.getenv("LOCALAPPDATA", Path.home())) / "Vedock" / "resources" / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=False)
    destination = root / "data.jsonl"
    rows, invalid = 0, 0
    seen: set[str] = set()
    try:
        if Path(inspection["path"]).suffix.lower() == ".zip":
            if selected != "image_classification":
                raise ValueError("An image ZIP must use the image classification schema in this workflow.")
            images = root / "images"
            images.mkdir()
            with zipfile.ZipFile(inspection["path"]) as archive, destination.open("x", encoding="utf-8", newline="\n") as output:
                for member in archive.infolist():
                    relative = PurePosixPath(member.filename)
                    if member.is_dir() or relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    label = relative.parts[-2] if len(relative.parts) > 1 else "unlabeled"
                    target = images / f"{rows:08d}{relative.suffix.lower()}"
                    with archive.open(member) as source, target.open("xb") as image_output:
                        shutil.copyfileobj(source, image_output)
                    output.write(json.dumps({"image": str(target.resolve()), "label": label}, ensure_ascii=False) + "\n")
                    rows += 1
            if not rows:
                raise ValueError("The ZIP contains no supported images.")
            mapping = {"image": "archive member", "label": "parent folder"}
        else:
            with destination.open("x", encoding="utf-8", newline="\n") as output:
                for row in _records(Path(inspection["path"])):
                    mapped = _mapped_row(row, selected, mapping, inspection["columns"])
                    if mapped is None:
                        invalid += 1
                        continue
                    if selected == "image_classification":
                        image = Path(str(mapped["image"]))
                        mapped["image"] = str((image if image.is_absolute() else Path(inspection["path"]).parent / image).resolve())
                    encoded = json.dumps(mapped, ensure_ascii=False, sort_keys=True, default=str)
                    if encoded in seen:
                        continue
                    seen.add(encoded)
                    output.write(encoded + "\n")
                    rows += 1
        if not rows:
            raise ValueError("No valid rows remain after applying the selected schema and field mapping.")
        digest, size = _hash_file(destination)
        metadata = {
            "file_format": "jsonl",
            "mime_type": mimetypes.guess_type(destination.name)[0] or "application/x-ndjson",
            "columns": inspection["columns"],
            "row_count": rows,
            "invalid_row_count": invalid,
            "recommendations": inspection["recommendations"],
            "field_mapping": mapping,
            "transformations": [{"type": "map_to_schema", "config": {"schema": selected, "field_mapping": mapping}}, {"type": "remove_empty_records"}, {"type": "remove_duplicates"}],
            "validation_status": "warning" if invalid else "valid",
            "validation": {"status": "warning" if invalid else "valid", "invalid_rows": invalid, "source": "connected_device"},
            "original_name": Path(inspection["path"]).name,
        }
        return destination, {"kind": "dataset", "name": inspection["name"], "path_hint": Path(inspection["path"]).name, "output_schema": selected, "size_bytes": size, "sha256": digest, "metadata": metadata}
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def remember_resource(client: Any, resource: dict[str, Any], local_path: Path) -> None:
    from vedock_cli.main import save_config

    records = dict(client.config.get("device_resources") or {})
    records[str(resource["id"])] = {"path": str(local_path.resolve()), "kind": resource["kind"], "sha256": resource.get("sha256"), "name": resource.get("name")}
    client.config["device_resources"] = records
    save_config(client.config)


def resolve_local_resource(client: Any, resource_id: str, kind: str) -> Path:
    record = (client.config.get("device_resources") or {}).get(str(resource_id))
    if not record or record.get("kind") not in {kind, "checkpoint" if kind == "model" else kind}:
        raise ValueError(f"This device has no local mapping for {kind} resource {resource_id}.")
    path = Path(record.get("path") or "").expanduser().resolve()
    if (kind == "model" and not path.is_dir()) or (kind == "dataset" and not path.is_file()):
        raise ValueError(f"The registered {kind} path is no longer available on this device: {path}")
    return path


def register_model(client: Any, device_id: str, path_value: str, name: str | None = None, request_id: str | None = None) -> dict[str, Any]:
    payload = inspect_model_folder(path_value)
    payload.update({"device_id": device_id, "status": "ready"})
    if name:
        payload["name"] = name
    endpoint = f"/device-resources/{request_id}/verify" if request_id else "/device-resources"
    resource = client.request("POST", endpoint, json=payload)
    remember_resource(client, resource, Path(path_value))
    return resource


def register_dataset(client: Any, device_id: str, path_value: str, schema: str = "auto", mapping: dict[str, Any] | None = None, request_id: str | None = None) -> dict[str, Any]:
    prepared, payload = prepare_dataset_file(path_value, schema, mapping)
    payload.update({"device_id": device_id, "status": "ready"})
    endpoint = f"/device-resources/{request_id}/verify" if request_id else "/device-resources"
    resource = client.request("POST", endpoint, json=payload)
    remember_resource(client, resource, prepared)
    return resource


def sync_pending_requests(client: Any, device_id: str) -> list[dict[str, Any]]:
    records = client.request("GET", f"/device-resources/requests?device_id={device_id}", headers={"X-Vedock-Device": device_id})
    results = []
    for request in records:
        path = str(request.get("requested_path") or "")
        try:
            if request["kind"] in {"model", "checkpoint"}:
                results.append(register_model(client, device_id, path, request.get("name"), request["id"]))
            else:
                results.append(register_dataset(client, device_id, path, request.get("output_schema") or "auto", request_id=request["id"]))
        except Exception as exc:
            client.request("POST", f"/device-resources/{request['id']}/verify", json={"device_id": device_id, "kind": request["kind"], "status": "invalid", "name": request.get("name"), "path_hint": request.get("path_hint"), "metadata": {"error": str(exc)}})
            results.append({"id": request["id"], "status": "invalid", "error": str(exc)})
    return results
