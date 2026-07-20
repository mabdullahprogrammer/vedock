from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class RuntimeAdapter(ABC):
    key = "base"
    display_name = "Base runtime"

    @abstractmethod
    def get_model_capabilities(self, model_path: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_training_parameter_schema(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_inference_parameter_schema(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_schema(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def validate_model(self, model_path: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def validate_dataset(self, dataset_path: str, schema: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(self, model_path: str, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def get_runner_schema(self, model_path: str | None = None) -> dict[str, Any]:
        """Describe the no-code interaction this runtime needs.

        Older adapters expose a single ``input_schema`` and ``output_schema`` in
        their capabilities.  The default keeps those adapters usable while new
        runtimes can declare multiple typed inputs and purpose-built results.
        """
        capabilities = self.get_model_capabilities(model_path)
        source = capabilities.get("input_schema") or {"type": "text", "field": "prompt"}
        source_type = str(source.get("type") or "text")
        field_type = "textarea" if source_type == "text" else source_type
        field_name = str(source.get("field") or "input")
        output = capabilities.get("output_schema") or {"type": "json"}
        return {
            "interaction": str(capabilities.get("interaction") or "structured"),
            "title": "Run model",
            "description": "Supply the input expected by this model runtime.",
            "submit_label": "Run model",
            "inputs": [
                {
                    "name": field_name,
                    "label": field_name.replace("_", " ").title(),
                    "description": "Primary model input.",
                    "type": field_type,
                    "required": True,
                }
            ],
            "outputs": [
                {
                    "type": str(output.get("type") or "json"),
                    "label": "Result",
                }
            ],
        }

    def run(self, model_path: str, inputs: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        """Run typed inputs. Multi-input runtimes should override this method."""
        runner = self.get_runner_schema(model_path)
        fields = runner.get("inputs") or []
        if len(fields) != 1:
            raise NotImplementedError("This multi-input runtime must implement run().")
        name = str(fields[0]["name"])
        return self.infer(model_path, str(inputs.get(name, "")), parameters)

    @abstractmethod
    def stream_infer(
        self, model_path: str, prompt: str, parameters: dict[str, Any]
    ) -> Iterable[str]:
        raise NotImplementedError

    def prepare_training(self, configuration: dict[str, Any]) -> dict[str, Any]:
        return configuration

    def train(self, configuration: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def cancel(self, job_id: str) -> bool:
        return False

    def evaluate(self, model_path: str, dataset_path: str, parameters: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def save(self, destination: str) -> str:
        raise NotImplementedError

    def export(self, model_path: str, destination: str) -> str:
        raise NotImplementedError
