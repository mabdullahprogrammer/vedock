from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from flask import current_app

from vedock.extensions import db
from vedock.models import MergeRecord, ModelRecord, ModelVersion, User, new_id

from .model_registry import latest_version
from .paths import allocate_directory, atomic_write_json


class MergeError(ValueError):
    pass


def record_failed_merge_attempt(
    version_a: ModelVersion,
    version_b: ModelVersion,
    method: str,
    weights: list[float],
    owner: User,
    report: dict[str, Any],
    error: str,
) -> MergeRecord:
    """Persist an experimental attempt without producing a corrupt artifact."""
    record = MergeRecord(
        owner=owner,
        source_versions_json=[version_a.id, version_b.id],
        method=str(method or "auto")[:40],
        weights_json=weights,
        configuration_json={"attempted": True, "error": str(error)[:5000]},
        compatibility_json=report,
        status="failed",
    )
    db.session.add(record)
    db.session.commit()
    return record


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_signature(path: Path) -> tuple[dict[str, list[int]], str | None]:
    weights = path / "model.safetensors"
    if not weights.is_file():
        return {}, None
    from safetensors import safe_open

    with safe_open(weights, framework="pt", device="cpu") as stream:
        signature = {key: list(stream.get_slice(key).get_shape()) for key in stream.keys()}
    return signature, str(weights)


def _adapter_signature(path: Path) -> dict[str, Any] | None:
    config = _json(path / "adapter_config.json")
    if not config:
        return None
    return {
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "target_modules": sorted(config.get("target_modules") or []),
        "r": config.get("r"),
    }


def _adapter_tensor_signature(path: Path) -> tuple[dict[str, list[int]], str | None]:
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        return {}, None
    from safetensors import safe_open

    with safe_open(weights, framework="pt", device="cpu") as stream:
        signature = {key: list(stream.get_slice(key).get_shape()) for key in stream.keys()}
    return signature, str(weights)


def compatibility_report(version_a: ModelVersion, version_b: ModelVersion) -> dict[str, Any]:
    path_a = Path(version_a.storage_path)
    path_b = Path(version_b.storage_path)
    config_a = _json(path_a / "config.json")
    config_b = _json(path_b / "config.json")
    adapter_a = _adapter_signature(path_a)
    adapter_b = _adapter_signature(path_b)
    adapter_tensors_a, adapter_weights_a = _adapter_tensor_signature(path_a)
    adapter_tensors_b, adapter_weights_b = _adapter_tensor_signature(path_b)
    tensors_a, weights_a = _tensor_signature(path_a)
    tensors_b, weights_b = _tensor_signature(path_b)

    architecture_match = bool(config_a and config_b and config_a.get("model_type") == config_b.get("model_type") and config_a.get("architectures") == config_b.get("architectures"))
    vocabulary_match = bool(config_a and config_b and config_a.get("vocab_size") == config_b.get("vocab_size"))
    precision_match = bool(config_a and config_b and config_a.get("torch_dtype") == config_b.get("torch_dtype"))
    tensor_names_match = bool(tensors_a and tensors_a.keys() == tensors_b.keys())
    tensor_shapes_match = bool(tensors_a and tensors_a == tensors_b)

    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
    tokenizer_checks = {}
    for filename in tokenizer_files:
        hash_a = _file_hash(path_a / filename)
        hash_b = _file_hash(path_b / filename)
        tokenizer_checks[filename] = {"hash_a": hash_a, "hash_b": hash_b, "match": bool(hash_a and hash_b and hash_a == hash_b)}
    tokenizer_match = all(check["match"] for check in tokenizer_checks.values())

    adapter_match = bool(adapter_a and adapter_b and adapter_a["base_model_name_or_path"] == adapter_b["base_model_name_or_path"] and adapter_a["peft_type"] == adapter_b["peft_type"] and adapter_a["task_type"] == adapter_b["task_type"] and adapter_a["target_modules"] == adapter_b["target_modules"] and adapter_a["r"] == adapter_b["r"])
    adapter_tensor_match = bool(adapter_tensors_a and adapter_tensors_a == adapter_tensors_b)
    sizes = [(Path(path).stat().st_size if path else 0) for path in [weights_a, weights_b]]
    adapter_sizes = [(Path(path).stat().st_size if path else 0) for path in [adapter_weights_a, adapter_weights_b]]
    required_memory = sum(sizes) + max(sizes or [0]) + 512 * 1024 * 1024
    required_storage = max(sizes or [0]) * 2 + 256 * 1024 * 1024
    adapter_required_memory = sum(adapter_sizes) + max(adapter_sizes or [0]) + 64 * 1024 * 1024
    adapter_required_storage = max(adapter_sizes or [0]) * 2 + 16 * 1024 * 1024
    try:
        import psutil

        available_memory = psutil.virtual_memory().available
    except Exception:
        available_memory = 0
    available_storage = shutil.disk_usage(current_app.config["STORAGE_ROOT"]).free
    memory_ok = available_memory >= required_memory
    storage_ok = available_storage >= required_storage
    adapter_memory_ok = available_memory >= adapter_required_memory
    adapter_storage_ok = available_storage >= adapter_required_storage

    checks = [
        {"name": "architecture", "passed": architecture_match, "detail": "Model type and architecture list must match."},
        {"name": "tensor_names", "passed": tensor_names_match, "detail": f"Compared {len(tensors_a)} and {len(tensors_b)} tensors."},
        {"name": "tensor_shapes", "passed": tensor_shapes_match, "detail": "Every shared tensor shape must match."},
        {"name": "vocabulary_size", "passed": vocabulary_match, "detail": "Vocabulary sizes must match."},
        {"name": "tokenizer", "passed": tokenizer_match, "detail": "Tokenizer, vocabulary, merges, and special-token metadata must match exactly."},
        {"name": "precision", "passed": precision_match, "detail": "Saved model precision metadata must match."},
        {"name": "memory", "passed": memory_ok, "detail": f"Estimated {required_memory} bytes required; {available_memory} bytes available."},
        {"name": "storage", "passed": storage_ok, "detail": f"Estimated {required_storage} bytes required; {available_storage} bytes available."},
    ]
    linear_safe = all(check["passed"] for check in checks)
    lora_safe = bool(adapter_match and adapter_tensor_match and adapter_memory_ok and adapter_storage_ok)
    blockers = [check["name"] for check in checks if not check["passed"]]
    warnings = ["License compatibility is unknown and must be reviewed manually."]
    if tensor_names_match and tensor_shapes_match and not tokenizer_match:
        warnings.append("Weights are structurally compatible, but tokenizer artifacts differ; a blind merge is blocked.")
    return {
        "model_a": version_a.to_dict(),
        "model_b": version_b.to_dict(),
        "checks": checks,
        "tokenizer_files": tokenizer_checks,
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "adapter_tensor_count": [len(adapter_tensors_a), len(adapter_tensors_b)],
        "adapter_weights": [adapter_weights_a, adapter_weights_b],
        "adapter_checks": {
            "configuration": adapter_match,
            "tensor_names_and_shapes": adapter_tensor_match,
            "memory": adapter_memory_ok,
            "storage": adapter_storage_ok,
        },
        "adapter_resource_estimate": {
            "required_memory": adapter_required_memory,
            "available_memory": available_memory,
            "required_storage": adapter_required_storage,
            "available_storage": available_storage,
        },
        "linear_safe": linear_safe,
        "lora_safe": lora_safe,
        "allowed_methods": (["linear"] if linear_safe else []) + (["weighted_adapter"] if lora_safe else []),
        "blockers": blockers,
        "warnings": warnings,
    }


def _combined_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest.update(file.relative_to(path).as_posix().encode("utf-8"))
            digest.update(_file_hash(file).encode("ascii"))
    return digest.hexdigest()


def execute_linear_merge(version_a: ModelVersion, version_b: ModelVersion, weight_a: float, weight_b: float, owner: User, output_name: str) -> tuple[MergeRecord, ModelVersion]:
    report = compatibility_report(version_a, version_b)
    if not report["linear_safe"]:
        raise MergeError(f"Linear merge is blocked by: {', '.join(report['blockers'])}")
    if weight_a < 0 or weight_b < 0 or abs((weight_a + weight_b) - 1.0) > 1e-6:
        raise MergeError("Linear merge weights must be non-negative and sum to 1.0.")
    model_id = new_id()
    version_id = new_id()
    directory = allocate_directory("models", str(owner.id), model_id, version_id)
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        path_a = Path(version_a.storage_path)
        path_b = Path(version_b.storage_path)
        tensors = {}
        with safe_open(path_a / "model.safetensors", framework="pt", device="cpu") as first, safe_open(path_b / "model.safetensors", framework="pt", device="cpu") as second:
            for key in first.keys():
                tensors[key] = first.get_tensor(key).to(torch.float32).mul(weight_a).add(second.get_tensor(key).to(torch.float32), alpha=weight_b)
        save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
        for filename in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
            source = path_a / filename
            if source.is_file():
                shutil.copy2(source, directory / filename)
        metadata = {
            "method": "linear",
            "source_versions": [version_a.id, version_b.id],
            "weights": [weight_a, weight_b],
            "compatibility": report,
        }
        atomic_write_json(directory / "merge_metadata.json", metadata)
        output_hash = _combined_hash(directory)
        model = ModelRecord(
            id=model_id,
            owner=owner,
            slug=f"merged-{model_id[:8]}",
            name=output_name[:160],
            description="Vedock linear model merge",
            task_type=version_a.model.task_type,
            runtime_key="transformers_text",
            source_type="merge",
            source_path=str(directory),
        )
        output_version = ModelVersion(
            id=version_id,
            model=model,
            version_number=1,
            label="Linear merge",
            storage_path=str(directory),
            status="completed",
            config_json={"merge_method": "linear", "weights": [weight_a, weight_b]},
            metadata_json=metadata,
            sha256=output_hash,
        )
        merge = MergeRecord(
            owner=owner,
            source_versions_json=[version_a.id, version_b.id],
            method="linear",
            weights_json=[weight_a, weight_b],
            compatibility_json=report,
            output_model_version=output_version,
            status="completed",
            output_hash=output_hash,
        )
        db.session.add_all([model, output_version, merge])
        db.session.commit()
        return merge, output_version
    except Exception:
        db.session.rollback()
        if directory.exists():
            shutil.rmtree(directory)
        raise


def execute_weighted_adapter_merge(version_a: ModelVersion, version_b: ModelVersion, weight_a: float, weight_b: float, owner: User, output_name: str) -> tuple[MergeRecord, ModelVersion]:
    report = compatibility_report(version_a, version_b)
    if not report["lora_safe"]:
        failed = [name for name, passed in report.get("adapter_checks", {}).items() if not passed]
        raise MergeError(f"Weighted LoRA merge is blocked by: {', '.join(failed) or 'adapter compatibility'}")
    if weight_a < 0 or weight_b < 0 or abs((weight_a + weight_b) - 1.0) > 1e-6:
        raise MergeError("Adapter merge weights must be non-negative and sum to 1.0.")
    model_id = new_id()
    version_id = new_id()
    directory = allocate_directory("models", str(owner.id), model_id, version_id)
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        path_a = Path(version_a.storage_path)
        path_b = Path(version_b.storage_path)
        tensors = {}
        with safe_open(path_a / "adapter_model.safetensors", framework="pt", device="cpu") as first, safe_open(path_b / "adapter_model.safetensors", framework="pt", device="cpu") as second:
            for key in first.keys():
                tensors[key] = first.get_tensor(key).to(torch.float32).mul(weight_a).add(second.get_tensor(key).to(torch.float32), alpha=weight_b)
        save_file(tensors, directory / "adapter_model.safetensors", metadata={"format": "pt"})
        for filename in ["adapter_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
            source = path_a / filename
            if source.is_file():
                shutil.copy2(source, directory / filename)
        adapter_config = _json(directory / "adapter_config.json") or {}
        metadata = {
            "method": "weighted_adapter",
            "source_versions": [version_a.id, version_b.id],
            "weights": [weight_a, weight_b],
            "compatibility": report,
        }
        atomic_write_json(directory / "merge_metadata.json", metadata)
        output_hash = _combined_hash(directory)
        model = ModelRecord(
            id=model_id,
            owner=owner,
            slug=f"merged-adapter-{model_id[:8]}",
            name=output_name[:160],
            description="Vedock weighted LoRA adapter merge",
            task_type=version_a.model.task_type,
            runtime_key=version_a.model.runtime_key,
            source_type="merge",
            source_path=str(directory),
        )
        output_version = ModelVersion(
            id=version_id,
            model=model,
            version_number=1,
            label="Weighted LoRA adapter merge",
            storage_path=str(directory),
            base_model_path=adapter_config.get("base_model_name_or_path"),
            status="completed",
            config_json={"merge_method": "weighted_adapter", "weights": [weight_a, weight_b]},
            metadata_json=metadata,
            sha256=output_hash,
        )
        merge = MergeRecord(
            owner=owner,
            source_versions_json=[version_a.id, version_b.id],
            method="weighted_adapter",
            weights_json=[weight_a, weight_b],
            compatibility_json=report,
            output_model_version=output_version,
            status="completed",
            output_hash=output_hash,
        )
        db.session.add_all([model, output_version, merge])
        db.session.commit()
        return merge, output_version
    except Exception:
        db.session.rollback()
        if directory.exists():
            shutil.rmtree(directory)
        raise


def resolve_latest_pair(model_a: ModelRecord, model_b: ModelRecord) -> tuple[ModelVersion, ModelVersion]:
    first = latest_version(model_a)
    second = latest_version(model_b)
    if not first or not second:
        raise MergeError("Both models require at least one completed version.")
    return first, second
