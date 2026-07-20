from __future__ import annotations

from .base import RuntimeAdapter
from .pattern import PatternSequenceRuntime
from .sklearn_image import SklearnImageClassificationRuntime
from .tabular import TabularPredictionRuntime
from .transformers_text import StoryMakerRuntime, TransformersTextRuntime


_runtime_types: dict[str, type[RuntimeAdapter]] = {
    TransformersTextRuntime.key: TransformersTextRuntime,
    StoryMakerRuntime.key: StoryMakerRuntime,
    PatternSequenceRuntime.key: PatternSequenceRuntime,
    SklearnImageClassificationRuntime.key: SklearnImageClassificationRuntime,
    TabularPredictionRuntime.key: TabularPredictionRuntime,
}
_instances: dict[str, RuntimeAdapter] = {}


def get_runtime(key: str) -> RuntimeAdapter:
    if key not in _runtime_types:
        raise KeyError(f"Unknown runtime: {key}")
    if key not in _instances:
        _instances[key] = _runtime_types[key]()
    return _instances[key]


def list_runtime_keys() -> list[str]:
    return sorted(_runtime_types)


def unload_all() -> None:
    for runtime in _instances.values():
        runtime.unload_model()
