from __future__ import annotations

import json
from typing import Any


class ParameterValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("Invalid runtime parameters")
        self.errors = errors


def parameter(
    name: str,
    label: str,
    description: str,
    value_type: str,
    default: Any,
    group: str,
    *,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    step: float | int | None = None,
    required: bool = False,
    choices: list[Any] | None = None,
    advanced: bool = False,
    runtime_compatibility: list[str] | None = None,
    hardware_warning: str | None = None,
    depends_on: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "description": description,
        "type": value_type,
        "default": default,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "required": required,
        "choices": choices,
        "group": group,
        "advanced": advanced,
        "runtime_compatibility": runtime_compatibility or ["transformers_text", "storymaker"],
        "hardware_warning": hardware_warning,
        "validation_rules": {
            "minimum": minimum,
            "maximum": maximum,
            "choices": choices,
        },
        "depends_on": depends_on or {},
    }


def _coerce(value: Any, field: dict[str, Any]) -> Any:
    value_type = field["type"]
    if value is None or value == "":
        return field.get("default")
    if value_type == "boolean":
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError("must be true or false")
    if value_type == "integer":
        return int(value)
    if value_type == "float":
        return float(value)
    if value_type == "list":
        if isinstance(value, list):
            return value
        stripped = str(value).strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError("must be a JSON array or comma-separated list")
            return parsed
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return str(value)


def validate_parameters(
    submitted: dict[str, Any], schema: list[dict[str, Any]], *, include_defaults: bool = True
) -> dict[str, Any]:
    allowed = {field["name"]: field for field in schema}
    errors: dict[str, str] = {}
    output: dict[str, Any] = {}
    unknown = sorted(set(submitted) - set(allowed))
    if unknown:
        errors["_unknown"] = f"Unsupported parameters: {', '.join(unknown)}"

    dependency_names = {
        dependency
        for field in schema
        for dependency in (field.get("depends_on") or {})
    }
    dependency_values: dict[str, Any] = {}
    for name in dependency_names:
        controller = allowed.get(name)
        if controller:
            try:
                dependency_values[name] = _coerce(submitted.get(name, controller.get("default")), controller)
            except (TypeError, ValueError, json.JSONDecodeError):
                # The controller's own validation below reports the useful error.
                pass

    for name, field in allowed.items():
        dependencies = field.get("depends_on") or {}
        if dependencies and not all(dependency_values.get(key) == expected for key, expected in dependencies.items()):
            # Inactive controls are intentionally ignored. This prevents hidden
            # LoRA defaults from invalidating full/scratch training forms.
            continue
        if name not in submitted and not include_defaults:
            continue
        raw = submitted.get(name, field.get("default"))
        try:
            value = _coerce(raw, field)
            if field.get("required") and (value is None or value == ""):
                raise ValueError("is required")
            choices = field.get("choices")
            if choices is not None and value not in choices:
                raise ValueError(f"must be one of: {', '.join(map(str, choices))}")
            output[name] = value
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors[name] = str(exc)

    if errors:
        raise ParameterValidationError(errors)
    return output


def schema_groups(schema: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for field in schema:
        group = field.get("group", "general")
        if group not in grouped:
            order.append(group)
            grouped[group] = []
        grouped[group].append(field)
    return [(group, grouped[group]) for group in order]
