from __future__ import annotations

import gc
import json
import math
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vedock.services.model_references import parse_model_reference

from .base import RuntimeAdapter
from .parameters import parameter, validate_parameters


class TabularPredictionRuntime(RuntimeAdapter):
    """Safe JSON-backed regression/classification runtime for structured data."""

    key = "tabular_prediction"
    display_name = "Tabular prediction"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded_path: str | None = None
        self._predictor: dict[str, Any] | None = None

    @staticmethod
    def _metadata(model_path: str | None) -> dict[str, Any]:
        if not model_path:
            return {}
        reference = parse_model_reference(model_path)
        if reference.kind != "local":
            return {}
        path = Path(reference.source) / "predictor.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def get_model_capabilities(self, model_path: str | None = None) -> dict[str, Any]:
        metadata = self._metadata(model_path)
        objective = str(metadata.get("objective") or "regression")
        interaction = "numeric_prediction" if objective == "regression" else "structured_classification"
        return {
            "runtime": self.key,
            "tasks": [f"tabular_{objective}"],
            "modality": "tabular",
            "interaction": interaction,
            "input_schema": {"type": "structured", "fields": [item.get("name") for item in metadata.get("features", [])]},
            "output_schema": {"type": "prediction", "objective": objective, "probabilities": objective == "classification"},
            "streaming": False,
            "stoppable_generation": False,
            "training_methods": ["linear_fit", "logistic_fit"],
            "devices": ["cpu"],
            "precisions": ["float64"],
            "quantization": False,
            "qlora": False,
            "loaded_model_path": self._loaded_path,
            "validation": self.validate_model(model_path) if model_path else None,
            "runner": self.get_runner_schema(model_path),
        }

    def get_runner_schema(self, model_path: str | None = None) -> dict[str, Any]:
        metadata = self._metadata(model_path)
        objective = str(metadata.get("objective") or "regression")
        inputs = []
        for feature in metadata.get("features") or []:
            kind = "number" if feature.get("kind") == "numeric" else "select"
            item = {
                "name": str(feature["name"]),
                "label": str(feature.get("label") or feature["name"]).replace("_", " ").title(),
                "description": f"{feature.get('kind', 'Input').title()} predictor used by the fitted model.",
                "type": kind,
                "required": True,
            }
            if kind == "select":
                item["choices"] = list(feature.get("categories") or [])
                item["allow_custom"] = True
            inputs.append(item)
        if not inputs:
            inputs = [{"name": "features", "label": "Feature values", "description": "JSON object of feature names and values. A fitted version replaces this with individual controls.", "type": "json", "required": True}]
        target = str(metadata.get("target_name") or "Prediction")
        return {
            "interaction": "numeric_prediction" if objective == "regression" else "structured_classification",
            "title": f"Predict {target.replace('_', ' ')}" if objective == "regression" else f"Classify {target.replace('_', ' ')}",
            "description": "Enter one structured record. Vedock validates and encodes each field exactly as it was prepared during fitting.",
            "submit_label": "Calculate prediction" if objective == "regression" else "Classify record",
            "inputs": inputs,
            "outputs": [{"type": "metric", "label": target}] if objective == "regression" else [{"type": "probabilities", "label": "Class probabilities"}],
        }

    def get_training_parameter_schema(self) -> list[dict[str, Any]]:
        return [
            parameter("output_model_name", "Output model name", "Name of the immutable fitted predictor.", "string", "tabular-predictor", "general", required=True),
            parameter("training_method", "Training method", "Fits a safe portable numeric predictor; no executable pickle is stored.", "string", "linear_fit", "general", choices=["linear_fit", "logistic_fit"]),
            parameter("objective", "Objective", "Regression predicts a number; classification predicts a category.", "string", "regression", "model", choices=["regression", "classification"]),
            parameter("regularization", "Regularization", "L2 penalty that stabilizes fitted coefficients.", "float", 0.001, "model", minimum=0.0, maximum=1_000_000.0),
            parameter("learning_rate", "Classification learning rate", "Update size for multinomial logistic fitting.", "float", 0.05, "model", minimum=0.000001, maximum=10.0, advanced=True),
            parameter("iterations", "Classification iterations", "Maximum full-batch optimizer updates.", "integer", 500, "model", minimum=1, maximum=100_000, advanced=True),
            parameter("maximum_categories", "Categories per feature", "Most frequent categorical values retained before using an OTHER bucket.", "integer", 50, "preprocessing", minimum=2, maximum=10_000),
            parameter("target_transform", "Regression target transform", "Log1p can help non-negative skewed sales or demand targets.", "string", "none", "preprocessing", choices=["none", "log1p"]),
            parameter("target_unit", "Prediction unit", "Optional unit displayed beside numeric predictions, such as USD or units.", "string", "", "output"),
            parameter("maximum_examples", "Maximum examples", "Zero uses every valid prepared record.", "integer", 0, "data", minimum=0, maximum=10_000_000),
            parameter("test_split", "Validation fraction", "Fraction held out for an honest pre-publication metric.", "float", 0.2, "evaluation", minimum=0.05, maximum=0.5),
            parameter("seed", "Random seed", "Controls the reproducible train/validation split.", "integer", 42, "evaluation", minimum=0, maximum=2_147_483_647),
        ]

    def get_inference_parameter_schema(self) -> list[dict[str, Any]]:
        return [parameter("return_details", "Show input details", "Include the normalized feature snapshot with the result.", "boolean", True, "output")]

    def get_dataset_schema(self) -> list[dict[str, Any]]:
        return [{"name": "tabular_supervised", "required_fields": ["features", "target"], "task": "tabular_prediction"}]

    def validate_model(self, model_path: str | None) -> dict[str, Any]:
        if not model_path:
            return {"valid": False, "errors": ["Model path is required"], "warnings": []}
        reference = parse_model_reference(model_path)
        if reference.kind == "scratch":
            return {"valid": True, "errors": [], "warnings": ["This predictor definition must be fitted before inference."], "reference_type": "scratch"}
        path = Path(reference.source) / "predictor.json"
        errors: list[str] = []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") != "vedock.tabular.v1":
                errors.append("predictor.json does not use the Vedock tabular v1 format")
            if not payload.get("features") or not payload.get("weights"):
                errors.append("predictor.json is missing fitted features or weights")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"predictor.json could not be read: {exc}")
        return {"valid": not errors, "errors": errors, "warnings": [], "reference_type": "local"}

    def validate_dataset(self, dataset_path: str, schema: str) -> dict[str, Any]:
        from vedock.services.datasets import validate_jsonl_file

        return validate_jsonl_file(Path(dataset_path), schema)

    def load_model(self, model_path: str, **kwargs: Any) -> dict[str, Any]:
        reference = parse_model_reference(model_path)
        normalized = reference.source
        with self._lock:
            if self._loaded_path == normalized and self._predictor is not None:
                return self._predictor
            validation = self.validate_model(model_path)
            if not validation["valid"]:
                raise ValueError("; ".join(validation["errors"]))
            predictor = json.loads((Path(normalized) / "predictor.json").read_text(encoding="utf-8"))
            self._loaded_path = normalized
            self._predictor = predictor
            return predictor

    def unload_model(self) -> None:
        with self._lock:
            self._loaded_path = None
            self._predictor = None
            gc.collect()

    @staticmethod
    def _vector(inputs: dict[str, Any], predictor: dict[str, Any]) -> np.ndarray:
        vector = [1.0]
        for feature in predictor["features"]:
            name = feature["name"]
            value = inputs.get(name)
            if feature["kind"] == "numeric":
                try:
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise ValueError
                except (TypeError, ValueError):
                    numeric = float(feature["median"])
                vector.append((numeric - float(feature["mean"])) / max(float(feature["scale"]), 1e-12))
            else:
                value = str(value if value is not None and value != "" else "[MISSING]")
                categories = list(feature["categories"])
                vector.extend(1.0 if value == category else 0.0 for category in categories)
                vector.append(0.0 if value in categories else 1.0)
        return np.asarray(vector, dtype=np.float64)

    def run(self, model_path: str, inputs: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_parameters(parameters, self.get_inference_parameter_schema())
        predictor = self.load_model(model_path)
        vector = self._vector(inputs, predictor)
        weights = np.asarray(predictor["weights"], dtype=np.float64)
        if predictor["objective"] == "regression":
            value = float(vector @ weights)
            if predictor.get("target_transform") == "log1p":
                value = float(np.expm1(value))
            result: dict[str, Any] = {
                "prediction": value,
                "prediction_label": predictor.get("target_name") or "Prediction",
                "unit": predictor.get("target_unit") or None,
                "outputs": [{"type": "metric", "label": predictor.get("target_name") or "Prediction", "value": value, "unit": predictor.get("target_unit") or None}],
            }
        else:
            scores = vector @ weights
            scores = scores - np.max(scores)
            probabilities = np.exp(scores) / np.exp(scores).sum()
            ranked = sorted(({"label": label, "score": float(score)} for label, score in zip(predictor["labels"], probabilities)), key=lambda item: item["score"], reverse=True)
            result = {"text": ranked[0]["label"], "predictions": ranked, "outputs": [{"type": "probabilities", "label": predictor.get("target_name") or "Class", "items": ranked}]}
        if normalized["return_details"]:
            result["outputs"].append({"type": "table", "label": "Input record", "columns": ["Feature", "Value"], "rows": [[feature["name"], inputs.get(feature["name"])] for feature in predictor["features"]]})
        result.update({"parameters": normalized, "device": "cpu", "elapsed_seconds": 0.0})
        return result

    def infer(self, model_path: str, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            inputs = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise ValueError("Tabular inference expects a JSON object of feature values.") from exc
        if not isinstance(inputs, dict):
            raise ValueError("Tabular inference expects a JSON object of feature values.")
        return self.run(model_path, inputs, parameters)

    def stream_infer(self, model_path: str, prompt: str, parameters: dict[str, Any]) -> Iterable[str]:
        yield json.dumps(self.infer(model_path, prompt, parameters), ensure_ascii=False)
