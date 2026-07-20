from __future__ import annotations

from pathlib import Path

from flask import current_app

from vedock.extensions import db
from vedock.models import ModelFork, ModelRecord, ModelVersion, ModelWorkspaceState

from .model_profiles import STORYMAKER_OUTPUT_PATTERN


def _register_model(slug: str, name: str, description: str, path: Path) -> ModelRecord | None:
    if not path.is_dir():
        return None
    model = ModelRecord.query.filter_by(slug=slug).first()
    if model is None:
        model = ModelRecord(
            slug=slug,
            name=name,
            description=description,
            task_type="causal_lm",
            runtime_key="storymaker",
            source_type="legacy_read_only",
            source_path=str(path),
            visibility="public",
        )
        db.session.add(model)
        db.session.flush()
    else:
        model.source_path = str(path)
        model.task_type = "causal_lm"
        model.visibility = "public"
    if not model.versions:
        db.session.add(
            ModelVersion(
                model=model,
                version_number=1,
                label="Legacy source",
                storage_path=str(path),
                base_model_path=str(path),
                status="completed",
                config_json={"read_only": True, "source": "StoryMaker", "output_pattern": STORYMAKER_OUTPUT_PATTERN},
                metadata_json={"registered_path": str(path)},
            )
        )
    for version in model.versions:
        configuration = dict(version.config_json or {})
        configuration.setdefault("output_pattern", STORYMAKER_OUTPUT_PATTERN)
        version.config_json = configuration
    return model


def register_legacy_models() -> list[ModelRecord]:
    models = [
        _register_model(
            "storymaker-final",
            "StoryMaker Final",
            "The final prompt-to-story GPT-2 model from the protected legacy project.",
            Path(current_app.config["STORYMAKER_FINAL_PATH"]),
        ),
        _register_model(
            "storymaker-finetuned",
            "StoryMaker Fine-tuned",
            "The earlier fine-tuned StoryMaker GPT-2 checkpoint.",
            Path(current_app.config["STORYMAKER_FINETUNED_PATH"]),
        ),
    ]
    ModelRecord.query.filter(ModelRecord.task_type.in_(["text_generation", "story_generation", "chat_model"])).update(
        {ModelRecord.task_type: "causal_lm"}, synchronize_session=False
    )
    db.session.commit()
    return [model for model in models if model is not None]


def visible_models(owner_id: int | None) -> list[ModelRecord]:
    query = ModelRecord.query
    if owner_id is None:
        return query.filter(ModelRecord.visibility == "public").order_by(ModelRecord.created_at.desc()).all()
    archived_ids = db.session.query(ModelWorkspaceState.model_id).filter_by(owner_id=owner_id, archived=True)
    return (
        query.filter(db.or_(ModelRecord.visibility == "public", ModelRecord.owner_id == owner_id))
        .filter(~ModelRecord.id.in_(archived_ids))
        .order_by(ModelRecord.created_at.desc())
        .all()
    )


def latest_version(model: ModelRecord) -> ModelVersion | None:
    if model.versions:
        return max(model.versions, key=lambda version: version.version_number)
    origin = ModelFork.query.filter_by(child_model_id=model.id).first()
    return origin.source_version if origin else None


def fork_count(model: ModelRecord) -> int:
    return (
        db.session.query(ModelFork.owner_id)
        .filter(ModelFork.source_model_id == model.id)
        .distinct()
        .count()
    )


def runnable_models(owner_id: int | None) -> list[ModelRecord]:
    runnable = []
    for model in visible_models(owner_id):
        version = latest_version(model)
        if version and version.status in {"completed", "available"} and model.source_type != "scratch_definition":
            runnable.append(model)
    return runnable


def recent_chat_models(owner_id: int) -> list[ModelRecord]:
    from vedock.models import Conversation

    available = {model.id: model for model in runnable_models(owner_id)}
    ordered: list[ModelRecord] = []
    seen: set[str] = set()
    conversations = Conversation.query.filter_by(owner_id=owner_id).order_by(Conversation.updated_at.desc()).all()
    for conversation in conversations:
        model = conversation.chat_model
        if model.id in available and model.id not in seen:
            ordered.append(available[model.id])
            seen.add(model.id)
    ordered.extend(model for model in available.values() if model.id not in seen)
    return ordered
