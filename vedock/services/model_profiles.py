from __future__ import annotations

from copy import deepcopy
from typing import Any

from vedock.models import ModelFork, ModelRecord, ModelVersion, ModelWorkspaceState


PLAIN_OUTPUT_PATTERN = "{prompt}{response}"
STORYMAKER_OUTPUT_PATTERN = (
    "<|start_of_input|>{prompt}<|end_of_input|>\n\n"
    "<|start_of_response|>{response}<|end_of_response|>"
)


def validate_output_pattern(value: str) -> str:
    pattern = str(value or "").strip()
    if not pattern:
        raise ValueError("The input/output pattern cannot be empty.")
    if "{prompt}" not in pattern:
        raise ValueError("The input/output pattern must contain {prompt}.")
    if "{response}" not in pattern:
        raise ValueError("The input/output pattern must contain {response} so Vedock can separate the inference prefix from generated output.")
    if len(pattern) > 8_000:
        raise ValueError("The input/output pattern is limited to 8,000 characters.")
    return pattern


def publisher_defaults(version: ModelVersion | None) -> dict[str, Any]:
    """Return safe publication-time defaults attached to one model version."""
    if not version:
        return {"inference_parameters": {}, "chat": {}, "allow_user_overrides": True}
    stored = dict((version.metadata_json or {}).get("publisher_defaults") or {})
    inference = stored.get("inference_parameters")
    chat = stored.get("chat")
    return {
        "inference_parameters": dict(inference) if isinstance(inference, dict) else {},
        "chat": dict(chat) if isinstance(chat, dict) else {},
        "allow_user_overrides": bool(stored.get("allow_user_overrides", True)),
    }


def set_publisher_defaults(
    version: ModelVersion,
    inference_parameters: dict[str, Any],
    chat: dict[str, Any] | None = None,
    *,
    allow_user_overrides: bool = True,
) -> dict[str, Any]:
    defaults = {
        "inference_parameters": dict(inference_parameters or {}),
        "chat": dict(chat or {}),
        "allow_user_overrides": bool(allow_user_overrides),
    }
    metadata = dict(version.metadata_json or {})
    metadata["publisher_defaults"] = defaults
    version.metadata_json = metadata
    return defaults


def model_output_pattern(model: ModelRecord, version: ModelVersion | None, owner_id: int | None = None) -> str:
    if owner_id is not None:
        state = ModelWorkspaceState.query.filter_by(owner_id=owner_id, model_id=model.id).first()
        configured = str(((state.configuration_json if state else {}) or {}).get("output_pattern") or "")
        if configured:
            return configured
    origin = ModelFork.query.filter_by(child_model_id=model.id).first()
    if origin:
        configured = str((origin.configuration_json or {}).get("output_pattern") or "")
        if configured:
            return configured
    if version:
        configured = str(publisher_defaults(version)["inference_parameters"].get("output_pattern") or "")
        if configured:
            return configured
        for container in [version.config_json or {}, version.metadata_json or {}]:
            configured = str(container.get("output_pattern") or ((container.get("parameters") or {}).get("output_pattern")) or "")
            if configured:
                return configured
    return STORYMAKER_OUTPUT_PATTERN if model.runtime_key == "storymaker" else PLAIN_OUTPUT_PATTERN


def schema_with_model_defaults(
    schema: list[dict[str, Any]], model: ModelRecord, version: ModelVersion | None, owner_id: int | None = None
) -> list[dict[str, Any]]:
    copied = deepcopy(schema)
    pattern = model_output_pattern(model, version, owner_id)
    published = publisher_defaults(version)["inference_parameters"]
    for field in copied:
        if field["name"] in published:
            field["default"] = deepcopy(published[field["name"]])
        if field["name"] == "output_pattern":
            field["default"] = pattern
    return copied


def submitted_with_model_defaults(
    submitted: dict[str, Any], model: ModelRecord, version: ModelVersion | None, owner_id: int | None = None
) -> dict[str, Any]:
    values = dict(submitted or {})
    if not str(values.get("output_pattern") or "").strip():
        values["output_pattern"] = model_output_pattern(model, version, owner_id)
    return values
