from __future__ import annotations

import json
from typing import Any

from vedock.runtimes.base import RuntimeAdapter


INPUT_TYPES = {"text", "textarea", "number", "integer", "boolean", "select", "json", "image", "file", "date"}
OUTPUT_TYPES = {"text", "metric", "probabilities", "table", "series", "embedding", "image", "images", "json"}


class RunnerValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("Invalid model inputs")
        self.errors = errors


def runner_contract(runtime: RuntimeAdapter, model_path: str | None) -> dict[str, Any]:
    contract = dict(runtime.get_runner_schema(model_path) or {})
    inputs = list(contract.get("inputs") or [])
    if not inputs:
        raise ValueError("The runtime did not declare any inference inputs.")
    seen: set[str] = set()
    normalized_inputs = []
    for field in inputs:
        item = dict(field)
        name = str(item.get("name") or "").strip()
        kind = str(item.get("type") or "text")
        if not name or name in seen:
            raise ValueError("Runtime input names must be non-empty and unique.")
        if kind not in INPUT_TYPES:
            raise ValueError(f"Unsupported runtime input type: {kind}")
        seen.add(name)
        item.update({"name": name, "type": kind, "label": str(item.get("label") or name.replace("_", " ").title()), "required": bool(item.get("required"))})
        normalized_inputs.append(item)
    outputs = []
    for declared in contract.get("outputs") or [{"type": "json", "label": "Result"}]:
        item = dict(declared)
        kind = str(item.get("type") or "json")
        if kind not in OUTPUT_TYPES:
            raise ValueError(f"Unsupported runtime output type: {kind}")
        item.update({"type": kind, "label": str(item.get("label") or "Result")})
        outputs.append(item)
    contract.update({
        "interaction": str(contract.get("interaction") or "structured"),
        "title": str(contract.get("title") or "Run model"),
        "description": str(contract.get("description") or "Supply the inputs expected by this model."),
        "submit_label": str(contract.get("submit_label") or "Run model"),
        "inputs": normalized_inputs,
        "outputs": outputs,
    })
    return contract


def validate_runner_inputs(submitted: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    fields = {str(item["name"]): item for item in contract.get("inputs") or []}
    errors: dict[str, str] = {}
    unknown = sorted(set(submitted) - set(fields))
    if unknown:
        errors["_unknown"] = f"Unsupported inputs: {', '.join(unknown)}"
    output: dict[str, Any] = {}
    for name, field in fields.items():
        value = submitted.get(name, field.get("default"))
        if field.get("required") and (value is None or value == ""):
            errors[name] = "is required"
            continue
        if value is None or value == "":
            output[name] = value
            continue
        try:
            kind = field["type"]
            if kind == "number":
                value = float(value)
            elif kind == "integer":
                value = int(value)
            elif kind == "boolean":
                if isinstance(value, bool):
                    pass
                elif str(value).strip().lower() in {"1", "true", "yes", "on"}:
                    value = True
                elif str(value).strip().lower() in {"0", "false", "no", "off"}:
                    value = False
                else:
                    raise ValueError("must be true or false")
            elif kind == "json":
                value = value if isinstance(value, (dict, list)) else json.loads(str(value))
            elif kind not in {"image", "file"}:
                value = str(value)
            choices = field.get("choices")
            if choices and value not in choices and not field.get("allow_custom"):
                raise ValueError(f"must be one of: {', '.join(map(str, choices))}")
            output[name] = value
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors[name] = str(exc)
    if errors:
        raise RunnerValidationError(errors)
    return output


def normalize_runtime_result(result: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Return safe display blocks without exposing runtime filesystem details."""
    blocks = result.get("outputs")
    if not isinstance(blocks, list):
        blocks = []
        if result.get("predictions"):
            blocks.append({"type": "probabilities", "label": "Predictions", "items": result["predictions"]})
        if result.get("prediction") is not None:
            blocks.append({"type": "metric", "label": str(result.get("prediction_label") or "Prediction"), "value": result["prediction"], "unit": result.get("unit")})
        if result.get("series"):
            blocks.append({"type": "series", "label": str(result.get("series_label") or "Forecast"), "points": result["series"]})
        if result.get("table"):
            table = result["table"]
            blocks.append({"type": "table", "label": str(result.get("table_label") or "Results"), "columns": table.get("columns", []), "rows": table.get("rows", [])})
        if result.get("embedding") is not None:
            blocks.append({"type": "embedding", "label": "Embedding", "values": result["embedding"]})
        if result.get("images"):
            blocks.append({"type": "images", "label": "Generated images", "items": result["images"]})
        if result.get("text") not in {None, ""} and not result.get("predictions"):
            blocks.append({"type": "text", "label": "Output", "value": result["text"]})
        if not blocks:
            blocks.append({"type": "json", "label": "Raw output", "value": {key: value for key, value in result.items() if "path" not in key.lower()}})
    safe_blocks = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") not in OUTPUT_TYPES:
            continue
        safe_blocks.append({key: value for key, value in block.items() if "path" not in str(key).lower()})
    response = {
        "interaction": contract["interaction"],
        "outputs": safe_blocks,
        "parameters": result.get("parameters") or {},
        "device": result.get("device"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }
    for legacy in ("text", "predictions", "sequences", "prediction"):
        if legacy in result:
            response[legacy] = result[legacy]
    return response
