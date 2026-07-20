from __future__ import annotations

import gc
import json
import threading
from pathlib import Path
from typing import Any, Iterable

from vedock.services.model_references import parse_model_reference

from .base import RuntimeAdapter
from .parameters import parameter, validate_parameters


class SklearnImageClassificationRuntime(RuntimeAdapter):
    key = "sklearn_image"
    display_name = "Fast image classification"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded_path: str | None = None
        self._bundle: dict[str, Any] | None = None

    def get_model_capabilities(self, model_path: str | None = None) -> dict[str, Any]:
        return {
            "runtime": self.key,
            "tasks": ["image_classification"],
            "modality": "image",
            "interaction": "image_classification",
            "input_schema": {"type": "image", "field": "image"},
            "output_schema": {"type": "classification", "probabilities": True},
            "streaming": False,
            "stoppable_generation": False,
            "training_methods": ["classifier_fit"],
            "devices": ["cpu"],
            "precisions": ["float64"],
            "quantization": False,
            "qlora": False,
            "loaded_model_path": self._loaded_path,
            "validation": self.validate_model(model_path) if model_path else None,
            "runner": self.get_runner_schema(model_path),
        }

    def get_runner_schema(self, model_path: str | None = None) -> dict[str, Any]:
        return {
            "interaction": "image_classification",
            "title": "Classify an image",
            "description": "Upload an image and receive ranked labels with confidence scores.",
            "submit_label": "Classify image",
            "inputs": [{"name": "image", "label": "Image", "description": "PNG, JPEG, WebP, or BMP image.", "type": "image", "required": True, "accept": [".png", ".jpg", ".jpeg", ".webp", ".bmp"]}],
            "outputs": [{"type": "probabilities", "label": "Ranked labels"}],
        }

    def get_training_parameter_schema(self) -> list[dict[str, Any]]:
        return [
            parameter("output_model_name", "Output model name", "Name of the immutable image classifier.", "string", "image-classifier", "general", required=True),
            parameter("training_method", "Training method", "Fits a CPU classifier to normalized image pixels.", "string", "classifier_fit", "general", choices=["classifier_fit"]),
            parameter("algorithm", "Classifier", "Portable classifier stored as data-only JSON, safe to publish for hosted inference.", "string", "linear_softmax", "model", choices=["linear_softmax", "nearest_centroid"]),
            parameter("image_width", "Image width", "Every processed image is resized to this width.", "integer", 64, "preprocessing", minimum=16, maximum=512),
            parameter("image_height", "Image height", "Every processed image is resized to this height.", "integer", 64, "preprocessing", minimum=16, maximum=512),
            parameter("color_mode", "Color mode", "RGB preserves color; grayscale is smaller and faster.", "string", "rgb", "preprocessing", choices=["rgb", "grayscale"]),
            parameter("maximum_examples", "Maximum examples", "Zero uses every valid image in the archive.", "integer", 0, "data", minimum=0, maximum=1_000_000),
            parameter("test_split", "Validation fraction", "Fraction held out for an accuracy report.", "float", 0.2, "evaluation", minimum=0.05, maximum=0.5, step=0.01),
            parameter("seed", "Seed", "Controls the split and compatible estimators.", "integer", 42, "evaluation", minimum=0, maximum=2_147_483_647),
        ]

    def get_inference_parameter_schema(self) -> list[dict[str, Any]]:
        return [
            parameter("top_k", "Top results", "Number of ranked labels to return.", "integer", 3, "output", minimum=1, maximum=100),
            parameter("device", "Device", "Fast classifiers run on CPU.", "string", "cpu", "runtime", choices=["cpu"]),
        ]

    def get_dataset_schema(self) -> list[dict[str, Any]]:
        return [{"name": "image_classification", "required_fields": ["image", "label"], "task": "image_classification"}]

    def validate_model(self, model_path: str | None) -> dict[str, Any]:
        if not model_path:
            return {"valid": False, "errors": ["Model path is required"], "warnings": []}
        reference = parse_model_reference(model_path)
        if reference.kind == "scratch":
            return {"valid": True, "errors": [], "warnings": ["This classifier definition must be fitted before inference."], "reference_type": "scratch"}
        path = Path(reference.source)
        errors = []
        if not (path / "classifier.json").is_file() and not (path / "classifier.joblib").is_file():
            errors.append("classifier.json was not found")
        if not (path / "metadata.json").is_file():
            errors.append("metadata.json was not found")
        return {"valid": not errors, "errors": errors, "warnings": [], "reference_type": "local"}

    def validate_dataset(self, dataset_path: str, schema: str) -> dict[str, Any]:
        from vedock.services.datasets import validate_jsonl_file

        return validate_jsonl_file(Path(dataset_path), schema)

    def load_model(self, model_path: str, **kwargs: Any) -> dict[str, Any]:
        reference = parse_model_reference(model_path)
        normalized = reference.source
        with self._lock:
            if self._loaded_path == normalized and self._bundle is not None:
                return self._bundle
            validation = self.validate_model(model_path)
            if not validation["valid"]:
                raise ValueError("; ".join(validation["errors"]))
            path = Path(normalized)
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            portable = path / "classifier.json"
            if portable.is_file():
                bundle = {"portable": json.loads(portable.read_text(encoding="utf-8")), "metadata": metadata}
            else:
                # Backward compatibility for local models created before the
                # portable format. Untrusted uploads never admit joblib files.
                import joblib

                bundle = {"classifier": joblib.load(path / "classifier.joblib"), "metadata": metadata}
            self._loaded_path = normalized
            self._bundle = bundle
            return bundle

    def unload_model(self) -> None:
        with self._lock:
            self._loaded_path = None
            self._bundle = None
            gc.collect()

    @staticmethod
    def preprocess_image(path: Path, metadata: dict[str, Any]) -> Any:
        import numpy as np
        from PIL import Image

        mode = "RGB" if metadata["color_mode"] == "rgb" else "L"
        with Image.open(path) as image:
            resized = image.convert(mode).resize((int(metadata["image_width"]), int(metadata["image_height"])))
            return np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0

    def infer(self, model_path: str, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        normalized = validate_parameters(parameters, self.get_inference_parameter_schema())
        image_path = Path(prompt).resolve()
        if not image_path.is_file():
            raise ValueError("Choose an existing local image file.")
        bundle = self.load_model(model_path)
        sample = self.preprocess_image(image_path, bundle["metadata"]).reshape(1, -1)
        if "portable" in bundle:
            portable = bundle["portable"]
            labels = [str(value) for value in portable["labels"]]
            if portable["kind"] == "linear_softmax":
                weights = np.asarray(portable["weights"], dtype=np.float64)
                intercept = np.asarray(portable["intercept"], dtype=np.float64)
                decision = np.asarray(sample @ weights.T + intercept).reshape(-1)
                if len(decision) == 1 and len(labels) == 2:
                    decision = np.array([-decision[0], decision[0]])
            else:
                centroids = np.asarray(portable["centroids"], dtype=np.float64)
                decision = -np.mean((centroids - sample[0]) ** 2, axis=1)
            shifted = np.exp(decision - np.max(decision))
            scores = shifted / shifted.sum()
        else:
            classifier = bundle["classifier"]
            labels = [str(value) for value in classifier.classes_]
            if hasattr(classifier, "predict_proba"):
                scores = classifier.predict_proba(sample)[0]
            elif hasattr(classifier, "decision_function"):
                decision = np.asarray(classifier.decision_function(sample)).reshape(-1)
                if len(decision) == 1 and len(labels) == 2:
                    decision = np.array([-decision[0], decision[0]])
                shifted = np.exp(decision - np.max(decision))
                scores = shifted / shifted.sum()
            else:
                predicted = str(classifier.predict(sample)[0])
                scores = np.array([1.0 if label == predicted else 0.0 for label in labels])
        ranked = sorted(({"label": label, "score": float(score)} for label, score in zip(labels, scores)), key=lambda item: item["score"], reverse=True)
        top = ranked[: normalized["top_k"]]
        return {"text": top[0]["label"] if top else "", "predictions": top, "parameters": normalized, "device": "cpu", "elapsed_seconds": 0.0}

    def stream_infer(self, model_path: str, prompt: str, parameters: dict[str, Any]) -> Iterable[str]:
        yield json.dumps(self.infer(model_path, prompt, parameters), ensure_ascii=False)
