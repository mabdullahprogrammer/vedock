from __future__ import annotations

import gc
import json
import random
import re
import threading
from pathlib import Path
from typing import Any, Iterable

from vedock.services.model_references import parse_model_reference

from .base import RuntimeAdapter
from .parameters import parameter, validate_parameters


class PatternSequenceRuntime(RuntimeAdapter):
    key = "pattern_sequence"
    display_name = "N-gram pattern sequence"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded_path: str | None = None
        self._model: dict[str, Any] | None = None

    def get_model_capabilities(self, model_path: str | None = None) -> dict[str, Any]:
        return {
            "runtime": self.key,
            "tasks": ["pattern_sequence"],
            "modality": "text",
            "interaction": "sequence_completion",
            "input_schema": {"type": "text", "field": "prompt"},
            "output_schema": {"type": "text", "streaming": True},
            "streaming": True,
            "stoppable_generation": False,
            "training_methods": ["pattern_fit"],
            "devices": ["cpu"],
            "precisions": ["integer_counts"],
            "quantization": False,
            "qlora": False,
            "loaded_model_path": self._loaded_path,
            "validation": self.validate_model(model_path) if model_path else None,
            "runner": self.get_runner_schema(model_path),
        }

    def get_runner_schema(self, model_path: str | None = None) -> dict[str, Any]:
        return {
            "interaction": "sequence_completion",
            "title": "Continue a sequence",
            "description": "Give this fitted pattern model a starting sequence. It predicts the most likely continuation; it does not maintain a chat conversation.",
            "submit_label": "Predict continuation",
            "inputs": [{"name": "sequence", "label": "Starting sequence", "description": "Include enough tokens to cover the fitted pattern order.", "type": "textarea", "required": True, "placeholder": "Enter a sequence to continue..."}],
            "outputs": [{"type": "text", "label": "Predicted continuation"}],
        }

    def get_training_parameter_schema(self) -> list[dict[str, Any]]:
        return [
            parameter("output_model_name", "Output model name", "Name of the immutable fitted pattern model.", "string", "pattern-model", "general", required=True),
            parameter("training_method", "Training method", "Counts token transitions; no neural network or GPU is involved.", "string", "pattern_fit", "general", choices=["pattern_fit"]),
            parameter("order", "Pattern order", "Number of previous tokens used as the prediction state.", "integer", 2, "model", minimum=1, maximum=8),
            parameter("lowercase", "Lowercase tokens", "Normalize text to lowercase before counting patterns.", "boolean", True, "data"),
            parameter("token_pattern", "Token regular expression", "Regular expression used to extract sequence tokens.", "string", r"\w+|[^\w\s]", "data", advanced=True),
            parameter("maximum_examples", "Maximum examples", "Zero uses every validated row.", "integer", 0, "data", minimum=0, maximum=10_000_000),
            parameter("seed", "Seed", "Controls reproducible weighted predictions.", "integer", 42, "model", minimum=0, maximum=2_147_483_647),
        ]

    def get_inference_parameter_schema(self) -> list[dict[str, Any]]:
        return [
            parameter("max_new_tokens", "Maximum new tokens", "Maximum sequence tokens generated after the supplied state.", "integer", 30, "length", minimum=1, maximum=1000),
            parameter("deterministic", "Use most frequent transition", "Choose the most frequent next token instead of weighted sampling.", "boolean", False, "sampling"),
            parameter("seed", "Seed", "Reproduces weighted transition sampling.", "integer", 42, "sampling", minimum=0, maximum=2_147_483_647),
            parameter("streaming", "Stream output", "Emit predicted tokens as they are selected.", "boolean", True, "runtime"),
        ]

    def get_dataset_schema(self) -> list[dict[str, Any]]:
        return [
            {"name": "text_completion", "required_fields": ["text"], "task": "pattern_sequence"},
            {"name": "prompt_response", "required_fields": ["prompt", "response"], "task": "pattern_sequence"},
        ]

    def validate_model(self, model_path: str | None) -> dict[str, Any]:
        if not model_path:
            return {"valid": False, "errors": ["Model path is required"], "warnings": []}
        reference = parse_model_reference(model_path)
        if reference.kind == "scratch":
            return {"valid": True, "errors": [], "warnings": ["This definition must be fitted before inference."], "reference_type": "scratch"}
        path = Path(reference.source)
        errors = [] if (path / "pattern_model.json").is_file() else ["pattern_model.json was not found"]
        return {"valid": not errors, "errors": errors, "warnings": [], "reference_type": "local"}

    def validate_dataset(self, dataset_path: str, schema: str) -> dict[str, Any]:
        from vedock.services.datasets import validate_jsonl_file

        return validate_jsonl_file(Path(dataset_path), schema)

    def load_model(self, model_path: str, **kwargs: Any) -> dict[str, Any]:
        reference = parse_model_reference(model_path)
        normalized = reference.source
        with self._lock:
            if self._loaded_path == normalized and self._model is not None:
                return self._model
            validation = self.validate_model(model_path)
            if not validation["valid"]:
                raise ValueError("; ".join(validation["errors"]))
            model = json.loads((Path(normalized) / "pattern_model.json").read_text(encoding="utf-8"))
            self._loaded_path = normalized
            self._model = model
            return model

    def unload_model(self) -> None:
        with self._lock:
            self._loaded_path = None
            self._model = None
            gc.collect()

    @staticmethod
    def _tokens(text: str, pattern: str, lowercase: bool) -> list[str]:
        value = text.lower() if lowercase else text
        return re.findall(pattern, value)

    def infer(self, model_path: str, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_parameters(parameters, self.get_inference_parameter_schema())
        model = self.load_model(model_path)
        order = int(model["order"])
        tokens = self._tokens(prompt, model["token_pattern"], bool(model["lowercase"]))
        if len(tokens) < order:
            raise ValueError(f"Provide at least {order} tokens so the model has a complete pattern state.")
        generated: list[str] = []
        state = tokens[-order:]
        randomizer = random.Random(normalized["seed"])
        transitions = model["transitions"]
        for _ in range(normalized["max_new_tokens"]):
            options = transitions.get("\u001f".join(state))
            if not options:
                break
            if normalized["deterministic"]:
                next_token = max(options, key=options.get)
            else:
                next_token = randomizer.choices(list(options), weights=list(options.values()), k=1)[0]
            generated.append(next_token)
            state = (state + [next_token])[-order:]
        text = " ".join(generated)
        return {"text": text, "sequences": [text], "parameters": normalized, "device": "cpu", "elapsed_seconds": 0.0}

    def stream_infer(self, model_path: str, prompt: str, parameters: dict[str, Any]) -> Iterable[str]:
        values = dict(parameters)
        values.pop("_generation_id", None)
        result = self.infer(model_path, prompt, values)
        for index, token in enumerate(result["text"].split()):
            yield ("" if index == 0 else " ") + token
