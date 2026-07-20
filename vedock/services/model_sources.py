from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from flask import current_app

from vedock.extensions import db
from vedock.models import ModelRecord, ModelVersion, User, new_id
from vedock.runtimes import get_runtime

from .model_references import make_huggingface_reference, make_scratch_reference
from .training import re_safe_slug


class ModelSourceError(ValueError):
    pass


PRETRAINED_MODEL_CATALOG = [
    {"id": "distilgpt2", "name": "DistilGPT-2", "parameters": "82M", "note": "Smallest GPT-2-family starting point."},
    {"id": "gpt2", "name": "GPT-2 Small", "parameters": "124M", "note": "Original compact GPT-2 architecture."},
    {"id": "gpt2-medium", "name": "GPT-2 Medium", "parameters": "355M", "note": "Requires substantially more RAM than GPT-2 Small."},
    {"id": "gpt2-large", "name": "GPT-2 Large", "parameters": "774M", "note": "Not recommended on the currently detected CPU-only hardware."},
    {"id": "gpt2-xl", "name": "GPT-2 XL", "parameters": "1.5B", "note": "Requires high-memory hardware."},
]


SCRATCH_PRESETS = {
    "tiny": {"name": "Tiny GPT-2", "n_layer": 4, "n_head": 4, "n_embd": 256, "n_positions": 512},
    "small": {"name": "GPT-2 Small architecture", "n_layer": 12, "n_head": 12, "n_embd": 768, "n_positions": 1024},
    "medium": {"name": "GPT-2 Medium architecture", "n_layer": 24, "n_head": 16, "n_embd": 1024, "n_positions": 1024},
    "custom": {"name": "Custom GPT-2 architecture", "n_layer": 6, "n_head": 8, "n_embd": 512, "n_positions": 1024},
}


TASK_OPTIONS = [
    {"id": "causal_lm", "name": "Causal language model", "description": "Predicts the next token. This is the objective used by GPT-style decoder models.", "runtime": "transformers_text", "available": True},
    {"id": "pattern_sequence", "name": "Pattern sequence model", "description": "Learns fast n-gram transition counts for lightweight sequence prediction and generation.", "runtime": "pattern_sequence", "available": True},
    {"id": "image_classification", "name": "Image classification", "description": "Fits a fast CPU classifier to labeled image folders stored in a ZIP dataset.", "runtime": "sklearn_image", "available": True},
    {"id": "tabular_regression", "name": "Tabular regression", "description": "Predicts a numeric target such as sales, demand, weight, price, or risk from structured feature columns.", "runtime": "tabular_prediction", "available": True},
    {"id": "tabular_classification", "name": "Tabular classification", "description": "Predicts a category and ranked probabilities from structured feature columns.", "runtime": "tabular_prediction", "available": True},
    {"id": "masked_lm", "name": "Masked language model", "description": "Reconstructs hidden tokens for encoder models such as BERT.", "runtime": None, "available": False},
    {"id": "seq2seq_lm", "name": "Sequence-to-sequence LM", "description": "Maps an input sequence to an output sequence using encoder-decoder models.", "runtime": None, "available": False},
    {"id": "sequence_classification", "name": "Sequence classification", "description": "Assigns one or more labels to complete text sequences.", "runtime": None, "available": False},
    {"id": "token_classification", "name": "Token classification", "description": "Assigns labels to individual tokens, including NER and tagging tasks.", "runtime": None, "available": False},
    {"id": "text_embedding", "name": "Text embedding", "description": "Produces numeric vectors for semantic search, retrieval, and clustering.", "runtime": None, "available": False},
    {"id": "text_to_image", "name": "Text-to-image diffusion", "description": "Generates image pixels or latent representations from text conditioning.", "runtime": None, "available": False},
    {"id": "image_captioning", "name": "Vision-language captioning", "description": "Generates text conditioned on an image and optional instruction.", "runtime": None, "available": False},
]


BUILD_MODES = [
    {"id": "fine_tune", "name": "Fine-tune a model", "description": "Use LoRA or full weight updates on a validated dataset."},
    {"id": "continue_pretraining", "name": "Continue pretraining", "description": "Continue causal-language training from existing weights."},
    {"id": "scratch", "name": "Build from scratch", "description": "Create a new GPT-2 architecture with randomly initialized weights."},
    {"id": "inference_only", "name": "Import or run a model", "description": "Register local or Hugging Face weights without training."},
    {"id": "merge", "name": "Combine models", "description": "Check compatibility, assign weights, and create an immutable merge."},
]


_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def _accessible(model: ModelRecord, owner: User) -> bool:
    return model.owner_id == owner.id or model.visibility == "public"


def _existing_reference(reference: str, owner: User) -> ModelRecord | None:
    version = ModelVersion.query.filter_by(storage_path=reference).first()
    if not version:
        return None
    if not _accessible(version.model, owner):
        raise ModelSourceError("That model reference is already registered by another local account.")
    return version.model


def _new_model(
    owner: User | None,
    name: str,
    task_type: str,
    runtime_key: str,
    source_type: str,
    reference: str,
    status: str,
    configuration: dict[str, Any] | None = None,
) -> ModelRecord:
    model_id = new_id()
    slug = f"{re_safe_slug(name)}-{model_id[:8]}"
    model = ModelRecord(
        id=model_id,
        owner=owner,
        slug=slug,
        name=name[:160],
        description=f"{source_type.replace('_', ' ').title()} model registered with {current_app.config['APP_NAME']}.",
        task_type=task_type,
        runtime_key=runtime_key,
        source_type=source_type,
        source_path=reference,
        visibility="public" if owner is None else "private",
    )
    version = ModelVersion(
        model=model,
        version_number=1,
        label="Base definition" if source_type == "scratch_definition" else "Imported base",
        storage_path=reference,
        base_model_path=reference,
        status=status,
        config_json=configuration or {},
        metadata_json={"source_type": source_type, "registered_without_loading": True},
    )
    db.session.add_all([model, version])
    db.session.flush()
    return model


def resolve_model_source(owner: User, task_type: str, source_type: str, values: dict[str, Any]) -> ModelRecord:
    if task_type in {"pattern_sequence", "image_classification", "tabular_regression", "tabular_classification"}:
        runtime_key = {
            "pattern_sequence": "pattern_sequence",
            "image_classification": "sklearn_image",
            "tabular_regression": "tabular_prediction",
            "tabular_classification": "tabular_prediction",
        }[task_type]
        if source_type not in {"scratch", "fast"}:
            raise ModelSourceError("Fast pattern and image classifiers start from a new fitted definition; select the scratch workflow.")
        default_name = {
            "pattern_sequence": "Pattern sequence model",
            "image_classification": "Fast image classifier",
            "tabular_regression": "Numeric predictor",
            "tabular_classification": "Category predictor",
        }[task_type]
        model_name = str(values.get("model_name") or default_name).strip()
        reference = make_scratch_reference(new_id())
        configuration = {
            "architecture_family": runtime_key,
            "randomly_initialized": True,
            "requires_fit": True,
            "source_assessment": "Clean Vedock adapter based on the read-only IRModule and pattern predictor concepts.",
            "objective": "classification" if task_type == "tabular_classification" else "regression" if task_type == "tabular_regression" else task_type,
        }
        return _new_model(owner, model_name, task_type, runtime_key, "scratch_definition", reference, "definition", configuration)

    if task_type != "causal_lm":
        raise ModelSourceError("The selected task runtime is not installed yet.")
    runtime_key = "transformers_text"

    if source_type == "existing":
        model = db.session.get(ModelRecord, values.get("base_model_id"))
        if not model or not _accessible(model, owner):
            raise ModelSourceError("Select an accessible registered model.")
        return model

    if source_type in {"catalog", "huggingface"}:
        repository = str(values.get("repository") or "").strip()
        if source_type == "catalog" and repository not in {item["id"] for item in PRETRAINED_MODEL_CATALOG}:
            raise ModelSourceError("Select a model from the supported catalog.")
        if not _REPOSITORY_PATTERN.fullmatch(repository):
            raise ModelSourceError("Enter a Hugging Face repository such as gpt2-medium or owner/model-name.")
        revision = str(values.get("revision") or "main").strip()
        reference = make_huggingface_reference(repository, revision)
        existing = _existing_reference(reference, owner)
        if existing:
            return existing
        catalog_entry = next((item for item in PRETRAINED_MODEL_CATALOG if item["id"] == repository), None)
        name = str(values.get("model_name") or (catalog_entry or {}).get("name") or repository).strip()
        configuration = {
            "repository": repository,
            "revision": revision,
            "parameter_count_label": (catalog_entry or {}).get("parameters"),
            "network_required_if_not_cached": True,
        }
        return _new_model(None, name, task_type, runtime_key, "community_online", reference, "available", configuration)

    if source_type in {"local", "checkpoint"}:
        raw_path = str(values.get("local_path") or "").strip().strip('"')
        if not raw_path:
            raise ModelSourceError("Enter a local model directory.")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise ModelSourceError(f"Local model directory does not exist: {path}")
        validation = get_runtime(runtime_key).validate_model(str(path))
        if not validation["valid"]:
            raise ModelSourceError("; ".join(validation["errors"]))
        reference = str(path)
        existing = _existing_reference(reference, owner)
        if existing:
            return existing
        name = str(values.get("model_name") or path.name).strip()
        kind = "checkpoint_read_only" if source_type == "checkpoint" else "local_read_only"
        return _new_model(owner, name, task_type, runtime_key, kind, reference, "completed", {"validation": validation})

    if source_type == "scratch":
        architecture_family = str(values.get("architecture_family") or "gpt2").strip().lower()
        if architecture_family != "gpt2":
            raise ModelSourceError("Only the GPT-2 decoder-only architecture factory is installed for scratch creation.")
        preset_name = str(values.get("scratch_preset") or "tiny")
        preset = SCRATCH_PRESETS.get(preset_name)
        if not preset:
            raise ModelSourceError("Select a supported scratch architecture preset.")
        scratch = dict(preset)
        if preset_name == "custom":
            for field, minimum, maximum in [
                ("n_layer", 1, 48),
                ("n_head", 1, 32),
                ("n_embd", 64, 4096),
                ("n_positions", 64, 4096),
            ]:
                try:
                    scratch[field] = int(values.get(field) or scratch[field])
                except (TypeError, ValueError) as exc:
                    raise ModelSourceError(f"{field} must be an integer.") from exc
                if not minimum <= scratch[field] <= maximum:
                    raise ModelSourceError(f"{field} must be between {minimum} and {maximum}.")
        if scratch["n_embd"] % scratch["n_head"]:
            raise ModelSourceError("Embedding size must be divisible by the number of attention heads.")
        model_name = str(values.get("model_name") or scratch["name"]).strip()
        reference = make_scratch_reference(new_id())
        tokenizer_repository = str(values.get("tokenizer_repository") or "gpt2").strip()
        if not _REPOSITORY_PATTERN.fullmatch(tokenizer_repository):
            raise ModelSourceError("Enter a valid tokenizer repository, such as gpt2.")
        configuration = {
            "scratch_config": {
                "vocab_size": 50257,
                "n_layer": scratch["n_layer"],
                "n_head": scratch["n_head"],
                "n_embd": scratch["n_embd"],
                "n_positions": scratch["n_positions"],
                "n_ctx": scratch["n_positions"],
                "bos_token_id": 50256,
                "eos_token_id": 50256,
            },
            "tokenizer_reference": make_huggingface_reference(tokenizer_repository),
            "randomly_initialized": True,
            "preset": preset_name,
            "architecture_family": architecture_family,
        }
        return _new_model(owner, model_name, task_type, runtime_key, "scratch_definition", reference, "definition", configuration)

    raise ModelSourceError("Select how the base model should be loaded.")
