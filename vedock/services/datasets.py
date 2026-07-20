from __future__ import annotations

import csv
import hashlib
import html
import ipaddress
import json
import mimetypes
import random
import re
import shutil
import socket
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import requests
from flask import current_app
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from vedock.extensions import db
from vedock.models import (
    DatasetTransformation,
    DatasetVersion,
    RawDataset,
    User,
    new_id,
)

from .paths import allocate_directory, assert_writable_path, atomic_write_json


SUPPORTED_FORMATS = {"csv", "json", "jsonl", "txt", "parquet", "zip"}
EXTENSION_MAP = {".csv": "csv", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl", ".txt": "txt", ".parquet": "parquet", ".zip": "zip"}
SCHEMA_REQUIRED_FIELDS = {
    "text_completion": ["text"],
    "prompt_response": ["prompt", "response"],
    "instruction": ["instruction", "input", "output"],
    "chat": ["messages"],
    "classification": ["text", "label"],
    "image_classification": ["image", "label"],
    "tabular_supervised": ["features", "target"],
}
SUPPORTED_TRANSFORMATIONS = {
    "select_columns", "rename_columns", "remove_columns", "join_columns", "split_column", "add_constant",
    "trim_whitespace", "normalize_unicode", "remove_html", "remove_urls", "remove_control_characters",
    "replace_text", "regex_replace", "lowercase", "remove_empty_records", "remove_duplicates",
    "filter_length", "filter_numeric", "filter_regex", "map_labels", "convert_type", "fill_missing",
    "strip_accents", "redact_emails", "redact_phone_numbers", "hash_field", "truncate_text",
    "prepend_text", "append_text", "shuffle", "limit_examples",
}


class DatasetError(ValueError):
    pass


def detect_format(filename: str, content_type: str | None = None) -> str:
    extension = Path(filename).suffix.lower()
    detected = EXTENSION_MAP.get(extension)
    if not detected and content_type:
        normalized = content_type.split(";", 1)[0].strip().lower()
        detected = {
            "text/csv": "csv",
            "application/json": "json",
            "application/x-ndjson": "jsonl",
            "text/plain": "txt",
            "application/vnd.apache.parquet": "parquet",
        }.get(normalized)
    if detected not in SUPPORTED_FORMATS:
        raise DatasetError("Supported dataset formats are CSV, JSON, JSONL, TXT, Parquet, and ZIP image folders.")
    return detected


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _source_path(dataset: RawDataset) -> Path:
    path = Path(dataset.storage_path)
    if not path.is_file():
        raise DatasetError("The raw dataset artifact is missing.")
    return path


def import_upload(file: FileStorage, owner: User, name: str | None = None, description: str = "") -> RawDataset:
    original = secure_filename(file.filename or "")
    if not original:
        raise DatasetError("Choose a file with a valid filename.")
    file_format = detect_format(original, file.mimetype)
    dataset_id = new_id()
    directory = allocate_directory("datasets", "raw", str(owner.id), dataset_id)
    suffix = Path(original).suffix.lower()
    destination = assert_writable_path(directory / f"source_file{suffix}")
    digest = hashlib.sha256()
    size = 0
    try:
        with destination.open("xb") as output:
            while chunk := file.stream.read(1024 * 1024):
                size += len(chunk)
                if size > current_app.config["MAX_CONTENT_LENGTH"]:
                    raise DatasetError("The upload exceeds the configured size limit.")
                digest.update(chunk)
                output.write(chunk)
        record = RawDataset(
            id=dataset_id,
            owner=owner,
            name=(name or Path(original).stem)[:160],
            description=description[:5000],
            source_type="upload",
            original_filename=original,
            storage_path=str(destination),
            file_format=file_format,
            mime_type=file.mimetype or mimetypes.guess_type(original)[0],
            size_bytes=size,
            sha256=digest.hexdigest(),
            inspection_status="pending",
        )
        atomic_write_json(
            directory / "source_metadata.json",
            {
                "source_type": "upload",
                "original_filename": original,
                "mime_type": record.mime_type,
                "file_format": file_format,
                "size_bytes": size,
                "sha256": record.sha256,
            },
        )
        db.session.add(record)
        db.session.commit()
        if size <= current_app.config["DATASET_SYNC_MAX_BYTES"]:
            inspect_dataset(record)
        else:
            from .jobs import enqueue_dataset_inspection

            record.inspection_status = "queued"
            db.session.commit()
            enqueue_dataset_inspection(owner, record)
        return record
    except Exception:
        db.session.rollback()
        persisted = db.session.get(RawDataset, dataset_id)
        if persisted is not None:
            db.session.delete(persisted)
            db.session.commit()
        if directory.exists():
            shutil.rmtree(directory)
        raise


def _validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise DatasetError("Only http and https dataset URLs are supported.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise DatasetError("Dataset URLs must have a public hostname and cannot include credentials.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise DatasetError(f"The dataset hostname could not be resolved: {exc}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise DatasetError("The dataset URL resolves to a private or reserved network address.")
    return url


def _filename_from_response(url: str, response: requests.Response) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", disposition, flags=re.I)
    if match:
        candidate = secure_filename(unquote(match.group(1).strip()))
        if candidate:
            return candidate
    candidate = secure_filename(Path(urlparse(url).path).name)
    if candidate:
        return candidate
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
    extension = mimetypes.guess_extension(content_type) or ".json"
    return f"download{extension}"


def import_url(url: str, owner: User, name: str | None = None, description: str = "") -> RawDataset:
    current_url = _validate_public_url(url.strip())
    session = requests.Session()
    response: requests.Response | None = None
    for _redirect in range(current_app.config["URL_MAX_REDIRECTS"] + 1):
        response = session.get(
            current_url,
            stream=True,
            allow_redirects=False,
            timeout=(current_app.config["URL_CONNECT_TIMEOUT"], current_app.config["URL_READ_TIMEOUT"]),
            headers={"User-Agent": f"{current_app.config['APP_NAME']}/0.1 dataset importer"},
        )
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise DatasetError("The dataset URL redirected without a destination.")
            current_url = _validate_public_url(urljoin(current_url, location))
            continue
        break
    else:
        raise DatasetError("The dataset URL exceeded the redirect limit.")
    assert response is not None
    if response.status_code >= 400:
        response.close()
        raise DatasetError(f"The dataset server returned HTTP {response.status_code}.")

    filename = _filename_from_response(current_url, response)
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
    file_format = detect_format(filename, content_type)
    dataset_id = new_id()
    directory = allocate_directory("datasets", "raw", str(owner.id), dataset_id)
    destination = assert_writable_path(directory / f"source_file{Path(filename).suffix.lower()}")
    digest = hashlib.sha256()
    size = 0
    try:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > current_app.config["URL_DOWNLOAD_MAX_BYTES"]:
            raise DatasetError("The remote dataset exceeds the configured size limit.")
        with destination.open("xb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > current_app.config["URL_DOWNLOAD_MAX_BYTES"]:
                    raise DatasetError("The remote dataset exceeded the configured size limit while downloading.")
                digest.update(chunk)
                output.write(chunk)
        record = RawDataset(
            id=dataset_id,
            owner=owner,
            name=(name or Path(filename).stem)[:160],
            description=description[:5000],
            source_type="url",
            source_url=current_url,
            original_filename=filename,
            storage_path=str(destination),
            file_format=file_format,
            mime_type=content_type,
            size_bytes=size,
            sha256=digest.hexdigest(),
            inspection_status="pending",
        )
        atomic_write_json(
            directory / "source_metadata.json",
            {
                "source_type": "url",
                "requested_url": url,
                "final_url": current_url,
                "original_filename": filename,
                "mime_type": content_type,
                "file_format": file_format,
                "size_bytes": size,
                "sha256": record.sha256,
            },
        )
        db.session.add(record)
        db.session.commit()
        if size <= current_app.config["DATASET_SYNC_MAX_BYTES"]:
            inspect_dataset(record)
        else:
            from .jobs import enqueue_dataset_inspection

            record.inspection_status = "queued"
            db.session.commit()
            enqueue_dataset_inspection(owner, record)
        return record
    except Exception:
        db.session.rollback()
        persisted = db.session.get(RawDataset, dataset_id)
        if persisted is not None:
            db.session.delete(persisted)
            db.session.commit()
        if directory.exists():
            shutil.rmtree(directory)
        raise
    finally:
        response.close()


def iter_records(path: Path, file_format: str) -> Iterator[dict[str, Any]]:
    if file_format == "zip":
        supported_images = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) > 100_000:
                raise DatasetError("Image archives are limited to 100,000 files.")
            expanded = sum(item.file_size for item in members)
            if expanded > current_app.config["MAX_CONTENT_LENGTH"] * 10:
                raise DatasetError("The expanded image archive exceeds the safety limit.")
            for member in members:
                member_path = Path(member.filename.replace("\\", "/"))
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise DatasetError("The ZIP contains an unsafe path.")
                if member_path.suffix.lower() not in supported_images:
                    continue
                parts = [part for part in member_path.parts if part not in {".", ""}]
                label = parts[-2] if len(parts) >= 2 else "unlabeled"
                yield {"image": member.filename, "label": label}
        return
    if file_format == "csv":
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise DatasetError("The CSV has no header row.")
            for row in reader:
                yield {str(key): value for key, value in row.items() if key is not None}
        return
    if file_format == "jsonl":
        with path.open("r", encoding="utf-8-sig", errors="strict") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
                if not isinstance(value, dict):
                    raise DatasetError(f"JSONL line {line_number} is not an object.")
                yield value
        return
    if file_format == "json":
        if path.stat().st_size > current_app.config["DATASET_SYNC_MAX_BYTES"]:
            raise DatasetError("Large JSON arrays must be converted to JSONL before synchronous inspection.")
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            value = value.get("data", [value])
        if not isinstance(value, list):
            raise DatasetError("JSON datasets must contain an array of objects or a data array.")
        for index, row in enumerate(value, 1):
            if not isinstance(row, dict):
                raise DatasetError(f"JSON record {index} is not an object.")
            yield row
        return
    if file_format == "txt":
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                value = line.strip()
                if value:
                    yield {"text": value}
        return
    if file_format == "parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1_000):
            for row in batch.to_pylist():
                yield row
        return
    raise DatasetError(f"Unsupported dataset format: {file_format}")


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _inspection_recommendations(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn inspection evidence into transparent, optional cleanup actions."""
    recommendations: list[dict[str, Any]] = []

    def add(operation: str | None, title: str, detail: str, severity: str = "suggestion", affected: int | None = None) -> None:
        recommendations.append(
            {
                "operation": operation,
                "title": title,
                "detail": detail,
                "severity": severity,
                "affected": affected,
                "auto_applicable": bool(operation),
            }
        )

    duplicates = int(result.get("duplicate_count") or 0)
    if duplicates:
        add("remove_duplicates", "Remove duplicate examples", f"{duplicates} repeated rows can over-weight the same example during training.", "warning", duplicates)
    column_stats = result.get("column_statistics") or {}
    empty_count = sum(int(item.get("empty_count") or 0) + int(item.get("null_count") or 0) for item in column_stats.values())
    if empty_count:
        add("remove_empty_records", "Remove incomplete examples", f"Detected {empty_count} empty or null field values. Review required fields before saving.", "warning", empty_count)
    samples = result.get("sample_rows") or []
    text_values = [value for row in samples for value in row.values() if isinstance(value, str)]
    if any(value != value.strip() or "  " in value for value in text_values):
        add("trim_whitespace", "Normalize whitespace", "Some sampled text has leading, trailing, or repeated spacing.")
    if any(re.search(r"<[^>]+>", value) for value in text_values):
        add("remove_html", "Remove HTML markup", "Markup tags were found in the sample and may become unwanted model output.")
    if any(re.search(r"https?://|www\.", value, re.I) for value in text_values):
        add("remove_urls", "Review web addresses", "URLs were found. Remove them when link memorization is not part of the task.")
    if any(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", value) for value in text_values):
        add("remove_control_characters", "Remove control characters", "Invisible control bytes can cause tokenization and parsing failures.", "warning")
    if any(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value) for value in text_values):
        add("redact_emails", "Review personal email addresses", "Email-like values were found. Redact them unless they are essential and authorized.", "privacy")
    maximum = max((int(item.get("maximum_text_length") or 0) for item in column_stats.values()), default=0)
    if maximum > 16_000:
        add(None, "Very long text detected", f"Some values reach {maximum:,} characters. Choose a length filter or a runtime sequence limit before training.", "warning")
    rows = int(result.get("row_count") or 0)
    if rows and rows < 20:
        add(None, "Small training set", f"Only {rows} rows were inspected. Fine-tuning may overfit; add more varied examples if possible.", "warning", rows)
    possible = result.get("possible_fields") or {}
    if not any(possible.get(name) for name in ("text", "prompt", "image")):
        add(None, "Choose input fields manually", "Vedock could not confidently identify a text, prompt, or image input column.", "warning")
    image_stats = result.get("image_statistics") or {}
    if int(image_stats.get("invalid_images_in_sample") or 0):
        add(None, "Repair invalid image references", "Some sampled images could not be opened. Fix or remove them before image training.", "warning")
    if not recommendations:
        add(None, "No obvious cleanup required", "The inspected sample has no empty rows, duplicates, markup, URLs, or control-character warnings.", "ready")
    return recommendations


def inspect_dataset(dataset: RawDataset) -> dict[str, Any]:
    dataset.inspection_status = "inspecting"
    db.session.commit()
    columns: list[str] = []
    types: dict[str, set[str]] = defaultdict(set)
    nulls: dict[str, int] = defaultdict(int)
    empties: dict[str, int] = defaultdict(int)
    text_lengths: dict[str, list[int]] = defaultdict(list)
    samples: list[dict[str, Any]] = []
    row_hashes: set[str] = set()
    duplicates = 0
    row_count = 0
    truncated = False
    try:
        for row in iter_records(_source_path(dataset), dataset.file_format):
            row_count += 1
            if row_count > current_app.config["DATASET_INSPECT_MAX_ROWS"]:
                truncated = True
                row_count -= 1
                break
            for key in row:
                if key not in columns:
                    columns.append(key)
            canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
            row_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if row_hash in row_hashes:
                duplicates += 1
            else:
                row_hashes.add(row_hash)
            for key in columns:
                value = row.get(key)
                types[key].add(_value_type(value))
                if value is None:
                    nulls[key] += 1
                elif isinstance(value, str):
                    if not value.strip():
                        empties[key] += 1
                    text_lengths[key].append(len(value))
            if len(samples) < current_app.config["DATASET_PREVIEW_ROWS"]:
                samples.append({key: (value[:500] + "…" if isinstance(value, str) and len(value) > 500 else value) for key, value in row.items()})

        column_stats = {}
        for key in columns:
            lengths = text_lengths.get(key, [])
            column_stats[key] = {
                "types": sorted(types[key]),
                "null_count": nulls[key],
                "empty_count": empties[key],
                "minimum_text_length": min(lengths) if lengths else None,
                "maximum_text_length": max(lengths) if lengths else None,
                "average_text_length": round(sum(lengths) / len(lengths), 2) if lengths else None,
            }
        lowered = {column.lower(): column for column in columns}
        possible = {
            "prompt": [lowered[name] for name in ["prompt", "instruction", "question", "input"] if name in lowered],
            "response": [lowered[name] for name in ["response", "story", "completion", "output", "answer"] if name in lowered],
            "text": [lowered[name] for name in ["text", "content", "body"] if name in lowered],
            "label": [lowered[name] for name in ["label", "category", "class", "target"] if name in lowered],
            "image": [lowered[name] for name in ["image", "image_path", "file"] if name in lowered],
            "caption": [lowered[name] for name in ["caption", "description"] if name in lowered],
        }
        result = {
            "file_type": dataset.file_format,
            "encoding": "utf-8-sig/replace" if dataset.file_format in {"csv", "txt"} else "utf-8",
            "size_bytes": dataset.size_bytes,
            "row_count": row_count,
            "truncated": truncated,
            "columns": columns,
            "column_statistics": column_stats,
            "duplicate_count": duplicates,
            "sample_rows": samples,
            "possible_fields": possible,
        }
        if dataset.file_format == "zip":
            from PIL import Image

            dimensions: list[list[int]] = []
            invalid_images = 0
            with zipfile.ZipFile(_source_path(dataset)) as archive:
                for record in samples[: min(len(samples), 20)]:
                    try:
                        with archive.open(record["image"]) as stream, Image.open(stream) as image:
                            dimensions.append([int(image.width), int(image.height)])
                            image.verify()
                    except Exception:
                        invalid_images += 1
            result["image_statistics"] = {
                "sample_dimensions": dimensions,
                "invalid_images_in_sample": invalid_images,
                "label_count": len({str(row.get("label")) for row in samples}),
            }
        result["recommendations"] = _inspection_recommendations(result)
        dataset.detected_schema_json = {"columns": columns, "types": {key: sorted(value) for key, value in types.items()}}
        dataset.statistics_json = result
        dataset.row_count = row_count
        dataset.inspection_status = "completed"
        db.session.commit()
        return result
    except Exception:
        dataset.inspection_status = "failed"
        db.session.commit()
        raise


def _fields(operation: dict[str, Any], row: dict[str, Any]) -> list[str]:
    configured = operation.get("config", {}).get("fields") or []
    return configured if configured else [key for key, value in row.items() if isinstance(value, str)]


def transform_record(row: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any] | None:
    transformed = dict(row)
    for operation in operations:
        kind = operation.get("type")
        config = operation.get("config") or {}
        if kind == "select_columns":
            selected = config.get("columns") or []
            transformed = {key: transformed.get(key) for key in selected}
        elif kind == "rename_columns":
            for old, new in (config.get("mapping") or {}).items():
                if old in transformed and new:
                    transformed[new] = transformed.pop(old)
        elif kind == "remove_columns":
            for key in config.get("columns") or []:
                transformed.pop(key, None)
        elif kind == "join_columns":
            transformed[config.get("target", "joined")] = config.get("separator", " ").join(
                str(transformed.get(key, "")) for key in config.get("columns") or []
            )
        elif kind == "split_column":
            source = str(transformed.get(config.get("field"), ""))
            separator = str(config.get("separator", ","))
            targets = [str(item) for item in config.get("targets") or [] if str(item)]
            pieces = source.split(separator, max(0, len(targets) - 1))
            for index, target in enumerate(targets):
                transformed[target] = pieces[index] if index < len(pieces) else ""
        elif kind == "add_constant":
            transformed[str(config.get("name", "constant"))] = config.get("value", "")
        elif kind == "trim_whitespace":
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = " ".join(transformed[key].strip().split())
        elif kind == "normalize_unicode":
            form = config.get("form", "NFKC")
            if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
                raise DatasetError("Unsupported Unicode normalization form.")
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = unicodedata.normalize(form, transformed[key])
        elif kind == "remove_html":
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = html.unescape(re.sub(r"<[^>]{0,2000}>", " ", transformed[key]))
        elif kind == "remove_urls":
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = re.sub(r"https?://\S+|www\.\S+", "", transformed[key], flags=re.I)
        elif kind == "remove_control_characters":
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = "".join(char for char in transformed[key] if char in "\n\t" or unicodedata.category(char) != "Cc")
        elif kind == "replace_text":
            old = str(config.get("old", ""))
            new = str(config.get("new", ""))
            if old:
                for key in _fields(operation, transformed):
                    if isinstance(transformed.get(key), str):
                        transformed[key] = transformed[key].replace(old, new)
        elif kind == "regex_replace":
            pattern = str(config.get("pattern", ""))
            if len(pattern) > 300:
                raise DatasetError("Regular-expression patterns are limited to 300 characters.")
            compiled = re.compile(pattern)
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = compiled.sub(str(config.get("replacement", "")), transformed[key][:200_000])
        elif kind == "lowercase":
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = transformed[key].lower()
        elif kind == "strip_accents":
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    normalized = unicodedata.normalize("NFKD", transformed[key])
                    transformed[key] = "".join(char for char in normalized if not unicodedata.combining(char))
        elif kind == "redact_emails":
            replacement = str(config.get("replacement", "[EMAIL]"))
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = re.sub(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])", replacement, transformed[key])
        elif kind == "redact_phone_numbers":
            replacement = str(config.get("replacement", "[PHONE]"))
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = re.sub(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", replacement, transformed[key])
        elif kind == "fill_missing":
            value = config.get("value", "")
            for key in config.get("fields") or []:
                if transformed.get(key) is None or transformed.get(key) == "":
                    transformed[key] = value
        elif kind == "hash_field":
            field = str(config.get("field") or "")
            if field in transformed:
                salt = str(config.get("salt") or "")
                transformed[field] = hashlib.sha256((salt + str(transformed[field])).encode("utf-8")).hexdigest()
        elif kind == "truncate_text":
            maximum = max(0, int(config.get("maximum") or 0))
            for key in _fields(operation, transformed):
                if maximum and isinstance(transformed.get(key), str):
                    transformed[key] = transformed[key][:maximum]
        elif kind in {"prepend_text", "append_text"}:
            value = str(config.get("value") or "")
            for key in _fields(operation, transformed):
                if isinstance(transformed.get(key), str):
                    transformed[key] = value + transformed[key] if kind == "prepend_text" else transformed[key] + value
        elif kind == "remove_empty_records":
            required = config.get("fields") or list(transformed)
            if any(transformed.get(key) is None or (isinstance(transformed.get(key), str) and not transformed[key].strip()) for key in required):
                return None
        elif kind == "filter_length":
            field = config.get("field")
            length = len(str(transformed.get(field, "")))
            if config.get("minimum") is not None and length < int(config["minimum"]):
                return None
            if config.get("maximum") is not None and length > int(config["maximum"]):
                return None
        elif kind == "filter_numeric":
            field = config.get("field")
            try:
                number = float(transformed.get(field))
            except (TypeError, ValueError):
                return None
            if config.get("minimum") is not None and number < float(config["minimum"]):
                return None
            if config.get("maximum") is not None and number > float(config["maximum"]):
                return None
        elif kind == "filter_regex":
            field = str(config.get("field") or "")
            pattern = str(config.get("pattern") or "")
            if len(pattern) > 300:
                raise DatasetError("Regular-expression patterns are limited to 300 characters.")
            matched = bool(re.search(pattern, str(transformed.get(field, ""))))
            if bool(config.get("exclude")) == matched:
                return None
        elif kind == "map_labels":
            field = config.get("field")
            mapping = {str(key): value for key, value in (config.get("mapping") or {}).items()}
            if field in transformed and str(transformed[field]) in mapping:
                transformed[field] = mapping[str(transformed[field])]
        elif kind == "convert_type":
            field = config.get("field")
            target = config.get("target")
            value = transformed.get(field)
            try:
                if target == "integer":
                    transformed[field] = int(value)
                elif target == "float":
                    transformed[field] = float(value)
                elif target == "boolean":
                    transformed[field] = str(value).strip().lower() in {"1", "true", "yes", "on"}
                elif target == "string":
                    transformed[field] = "" if value is None else str(value)
                else:
                    raise DatasetError("convert_type target must be string, integer, float, or boolean.")
            except (TypeError, ValueError) as exc:
                raise DatasetError(f"Could not convert field {field!r} to {target}.") from exc
        elif kind in {"remove_duplicates", "shuffle", "limit_examples"}:
            continue
        else:
            raise DatasetError(f"Unsupported transformation operation: {kind!r}.")
    return transformed


def map_record(row: dict[str, Any], output_schema: str, mapping: dict[str, str], template: str = "") -> dict[str, Any]:
    if output_schema not in SCHEMA_REQUIRED_FIELDS:
        raise DatasetError(f"Unsupported output schema: {output_schema}")
    if output_schema == "chat":
        system = row.get(mapping.get("system", ""), "") if mapping.get("system") else ""
        prompt = row.get(mapping.get("prompt", ""), "")
        response = row.get(mapping.get("response", ""), "")
        messages = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.extend([{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}])
        return {"messages": messages}
    if output_schema == "tabular_supervised":
        feature_sources = mapping.get("features") or []
        if isinstance(feature_sources, str):
            feature_sources = [item.strip() for item in feature_sources.split(",") if item.strip()]
        return {
            "features": {str(source): row.get(str(source)) for source in feature_sources},
            "target": row.get(str(mapping.get("target") or "")),
        }
    output = {target: row.get(source) for target, source in mapping.items() if target in SCHEMA_REQUIRED_FIELDS[output_schema]}
    if output_schema == "text_completion" and template:
        try:
            output["text"] = template.format(**row)
        except KeyError as exc:
            raise DatasetError(f"The prompt template references a missing field: {exc}") from exc
    return output


def _record_findings(record: dict[str, Any], schema: str, row_number: int) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for field in SCHEMA_REQUIRED_FIELDS.get(schema, []):
        value = record.get(field)
        if value is None or value == "" or value == []:
            findings.append({"severity": "error", "code": "missing_required_field", "message": f"Required field {field!r} is empty.", "row": row_number, "field": field, "suggested_fix": "Map a non-empty source field."})
    if schema == "chat" and record.get("messages"):
        allowed_roles = {"system", "user", "assistant"}
        for message in record["messages"]:
            if not isinstance(message, dict) or message.get("role") not in allowed_roles or not str(message.get("content", "")).strip():
                findings.append({"severity": "error", "code": "invalid_chat_message", "message": "Chat messages require system/user/assistant roles and non-empty content.", "row": row_number, "suggested_fix": "Correct the role and content mappings."})
                break
    if schema == "image_classification":
        image = str(record.get("image") or "").strip()
        if Path(image).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            findings.append({"severity": "error", "code": "invalid_image_format", "message": "Image classification rows require PNG, JPEG, WebP, or BMP archive members.", "row": row_number, "field": "image", "suggested_fix": "Remove unsupported files from the image archive."})
    if schema == "tabular_supervised":
        features = record.get("features")
        if not isinstance(features, dict) or not features:
            findings.append({"severity": "error", "code": "missing_features", "message": "Tabular rows require at least one selected feature column.", "row": row_number, "field": "features", "suggested_fix": "Select one or more predictor columns."})
    return findings


def preview_transform(dataset: RawDataset, operations: list[dict[str, Any]], output_schema: str, mapping: dict[str, str], template: str = "", limit: int = 20) -> dict[str, Any]:
    rows = []
    removed = 0
    seen: set[str] = set()
    remove_duplicates = any(operation.get("type") == "remove_duplicates" for operation in operations)
    for source in iter_records(_source_path(dataset), dataset.file_format):
        transformed = transform_record(source, operations)
        if transformed is None:
            removed += 1
            continue
        mapped = map_record(transformed, output_schema, mapping, template)
        canonical = json.dumps(mapped, sort_keys=True, ensure_ascii=False)
        if remove_duplicates and canonical in seen:
            removed += 1
            continue
        seen.add(canonical)
        rows.append({"source": source, "output": mapped, "findings": _record_findings(mapped, output_schema, len(rows) + 1)})
        if len(rows) >= limit:
            break
    return {"rows": rows, "removed_in_preview": removed, "output_schema": output_schema}


def save_dataset_version(dataset: RawDataset, owner: User, operations: list[dict[str, Any]], output_schema: str, mapping: dict[str, str], template: str = "", limit: int = 0, shuffle: bool = False, shuffle_seed: int = 42) -> DatasetVersion:
    if dataset.owner_id != owner.id:
        raise DatasetError("You do not own this dataset.")
    missing_mappings = [field for field in SCHEMA_REQUIRED_FIELDS.get(output_schema, []) if output_schema != "chat" and field not in mapping and not (output_schema == "text_completion" and field == "text" and template)]
    if missing_mappings:
        raise DatasetError(f"Missing field mappings: {', '.join(missing_mappings)}")
    next_number = (db.session.query(db.func.max(DatasetVersion.version_number)).filter_by(raw_dataset_id=dataset.id).scalar() or 0) + 1
    version_id = new_id()
    directory = allocate_directory("datasets", "processed", str(owner.id), dataset.id, version_id)
    output_path = directory / "data.jsonl"
    invalid_path = directory / "invalid_rows.jsonl"
    findings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] | None = [] if shuffle else None
    seen: set[str] = set()
    remove_duplicates = any(operation.get("type") == "remove_duplicates" for operation in operations)
    valid_count = 0
    invalid_count = 0
    token_estimate = 0

    def transformed_rows() -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
        for source_number, source in enumerate(iter_records(_source_path(dataset), dataset.file_format), 1):
            transformed = transform_record(source, operations)
            if transformed is None:
                continue
            mapped = map_record(transformed, output_schema, mapping, template)
            yield source_number, source, mapped

    try:
        iterator = transformed_rows()
        if shuffle:
            materialized = list(iterator)
            random.Random(shuffle_seed).shuffle(materialized)
            iterator = iter(materialized)
        digest = hashlib.sha256()
        with output_path.open("xb") as output, invalid_path.open("x", encoding="utf-8", newline="\n") as invalid_stream:
            for source_number, source, mapped in iterator:
                canonical = json.dumps(mapped, sort_keys=True, ensure_ascii=False)
                if remove_duplicates and canonical in seen:
                    continue
                seen.add(canonical)
                row_findings = _record_findings(mapped, output_schema, source_number)
                errors = [finding for finding in row_findings if finding["severity"] == "error"]
                findings.extend(row_findings[:1000])
                if errors:
                    invalid_count += 1
                    invalid_stream.write(json.dumps({"row": source_number, "findings": errors, "source": source}, ensure_ascii=False, default=str) + "\n")
                    continue
                line = (json.dumps(mapped, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode("utf-8")
                output.write(line)
                digest.update(line)
                valid_count += 1
                token_estimate += max(1, len(line) // 4)
                if limit and valid_count >= limit:
                    break
        if valid_count == 0:
            findings.append({"severity": "error", "code": "no_valid_examples", "message": "No valid examples remained after transformation.", "suggested_fix": "Review mappings and filters."})
        elif valid_count < 5:
            findings.append({"severity": "warning", "code": "small_dataset", "message": f"Only {valid_count} valid examples remain.", "suggested_fix": "Use at least five examples for a meaningful smoke fine-tune."})
        errors = [finding for finding in findings if finding["severity"] == "error"]
        warnings = [finding for finding in findings if finding["severity"] == "warning"]
        status = "invalid" if errors else ("warning" if warnings else "valid")
        validation = {"status": status, "errors": errors, "warnings": warnings, "valid_rows": valid_count, "invalid_rows": invalid_count}
        statistics = {"row_count": valid_count, "invalid_row_count": invalid_count, "token_estimate": token_estimate, "output_bytes": output_path.stat().st_size}
        normalized_operations = operations + ([{"type": "shuffle", "config": {"seed": shuffle_seed}}] if shuffle else []) + ([{"type": "limit_examples", "config": {"limit": limit}}] if limit else [])
        atomic_write_json(directory / "schema.json", {"name": output_schema, "required_fields": SCHEMA_REQUIRED_FIELDS[output_schema], "field_mapping": mapping})
        atomic_write_json(directory / "transformation.json", {"operations": normalized_operations, "template": template})
        atomic_write_json(directory / "statistics.json", statistics)
        atomic_write_json(directory / "validation.json", validation)
        version = DatasetVersion(
            id=version_id,
            raw_dataset=dataset,
            owner=owner,
            version_number=next_number,
            output_format=output_schema,
            storage_path=str(output_path),
            transformation_config=normalized_operations,
            field_mapping=mapping,
            validation_status=status,
            validation_json=validation,
            row_count=valid_count,
            invalid_row_count=invalid_count,
            token_estimate=token_estimate,
            sha256=digest.hexdigest(),
        )
        db.session.add(version)
        db.session.flush()
        for index, operation in enumerate(normalized_operations):
            db.session.add(DatasetTransformation(version=version, operation_order=index, operation_type=operation.get("type", "unknown"), configuration=operation.get("config") or {}, result_summary={}))
        db.session.commit()
        return version
    except Exception:
        db.session.rollback()
        if directory.exists():
            shutil.rmtree(directory)
        raise


def validate_jsonl_file(path: Path, schema: str) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    row_count = 0
    if schema not in SCHEMA_REQUIRED_FIELDS:
        return {"status": "invalid", "errors": [{"severity": "error", "code": "unsupported_schema", "message": f"Unsupported schema: {schema}"}], "warnings": [], "row_count": 0}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for row_count, line in enumerate(stream, 1):
                if not line.strip():
                    findings.append({"severity": "error", "code": "empty_record", "message": "An empty JSONL line was found.", "row": row_count})
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    findings.append({"severity": "error", "code": "invalid_record", "message": "A JSONL record is not an object.", "row": row_count})
                    continue
                findings.extend(_record_findings(value, schema, row_count))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        findings.append({"severity": "error", "code": "file_integrity", "message": str(exc)})
    if row_count == 0:
        findings.append({"severity": "error", "code": "no_examples", "message": "The dataset contains no examples."})
    errors = [finding for finding in findings if finding["severity"] == "error"]
    warnings = [finding for finding in findings if finding["severity"] == "warning"]
    return {"status": "invalid" if errors else ("warning" if warnings else "valid"), "errors": errors, "warnings": warnings, "row_count": row_count}


def revalidate_version(version: DatasetVersion) -> dict[str, Any]:
    report = validate_jsonl_file(Path(version.storage_path), version.output_format)
    version.validation_status = report["status"]
    version.validation_json = report
    db.session.commit()
    return report
