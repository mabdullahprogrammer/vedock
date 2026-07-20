from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)


class User(UserMixin, TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class ApiToken(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False, default="CLI")
    prefix = db.Column(db.String(16), nullable=False, index=True)
    token_hash = db.Column(db.String(64), nullable=False, unique=True)
    revoked_at = db.Column(db.DateTime(timezone=True))
    last_used_at = db.Column(db.DateTime(timezone=True))
    user = db.relationship("User", backref=db.backref("api_tokens", lazy=True, cascade="all, delete-orphan"))

    @classmethod
    def issue(cls, user: User, name: str = "CLI") -> tuple["ApiToken", str]:
        plain = "vdk_" + secrets.token_urlsafe(32)
        record = cls(
            user=user,
            name=name[:100],
            prefix=plain[:12],
            token_hash=hashlib.sha256(plain.encode("utf-8")).hexdigest(),
        )
        return record, plain

    @classmethod
    def authenticate(cls, plain: str | None) -> "ApiToken | None":
        if not plain or not plain.startswith("vdk_"):
            return None
        digest = hashlib.sha256(plain.encode("utf-8")).hexdigest()
        token = cls.query.filter_by(token_hash=digest, revoked_at=None).first()
        if token:
            token.last_used_at = utcnow()
            db.session.commit()
        return token


class RawDataset(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    name = db.Column(db.String(160), nullable=False)
    description = db.Column(db.Text, default="")
    source_type = db.Column(db.String(20), nullable=False)
    source_url = db.Column(db.Text)
    original_filename = db.Column(db.String(255), nullable=False)
    storage_path = db.Column(db.Text, nullable=False, unique=True)
    file_format = db.Column(db.String(20), nullable=False)
    mime_type = db.Column(db.String(120))
    size_bytes = db.Column(db.BigInteger, nullable=False)
    sha256 = db.Column(db.String(64), nullable=False, index=True)
    inspection_status = db.Column(db.String(30), nullable=False, default="pending")
    detected_schema_json = db.Column(db.JSON, default=dict)
    statistics_json = db.Column(db.JSON, default=dict)
    row_count = db.Column(db.Integer)
    owner = db.relationship("User", backref=db.backref("raw_datasets", lazy=True))

    def to_dict(self, include_inspection: bool = True) -> dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "original_filename": self.original_filename,
            "file_format": self.file_format,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "inspection_status": self.inspection_status,
            "row_count": self.row_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_inspection:
            data["schema"] = self.detected_schema_json or {}
            data["statistics"] = self.statistics_json or {}
        return data


class DatasetVersion(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    raw_dataset_id = db.Column(db.String(36), db.ForeignKey("raw_dataset.id"), nullable=False, index=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    version_number = db.Column(db.Integer, nullable=False)
    output_format = db.Column(db.String(30), nullable=False, default="prompt_response")
    storage_path = db.Column(db.Text, nullable=False, unique=True)
    transformation_config = db.Column(db.JSON, default=list)
    field_mapping = db.Column(db.JSON, default=dict)
    validation_status = db.Column(db.String(30), nullable=False, default="pending")
    validation_json = db.Column(db.JSON, default=dict)
    row_count = db.Column(db.Integer, nullable=False, default=0)
    invalid_row_count = db.Column(db.Integer, nullable=False, default=0)
    token_estimate = db.Column(db.Integer, nullable=False, default=0)
    sha256 = db.Column(db.String(64), nullable=False)
    raw_dataset = db.relationship("RawDataset", backref=db.backref("versions", lazy=True, order_by="DatasetVersion.version_number"))
    owner = db.relationship("User")
    __table_args__ = (db.UniqueConstraint("raw_dataset_id", "version_number", name="uq_dataset_version_number"),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "raw_dataset_id": self.raw_dataset_id,
            "version_number": self.version_number,
            "output_format": self.output_format,
            "validation_status": self.validation_status,
            "validation": self.validation_json or {},
            "row_count": self.row_count,
            "invalid_row_count": self.invalid_row_count,
            "token_estimate": self.token_estimate,
            "sha256": self.sha256,
            "field_mapping": self.field_mapping or {},
            "transformations": self.transformation_config or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DatasetTransformation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_version_id = db.Column(db.String(36), db.ForeignKey("dataset_version.id", ondelete="CASCADE"), nullable=False)
    operation_order = db.Column(db.Integer, nullable=False)
    operation_type = db.Column(db.String(80), nullable=False)
    configuration = db.Column(db.JSON, default=dict)
    result_summary = db.Column(db.JSON, default=dict)
    version = db.relationship("DatasetVersion", backref=db.backref("transformation_records", cascade="all, delete-orphan"))


class ModelRecord(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), index=True)
    slug = db.Column(db.String(160), unique=True, nullable=False, index=True)
    name = db.Column(db.String(160), nullable=False)
    description = db.Column(db.Text, default="")
    task_type = db.Column(db.String(50), nullable=False, default="text_generation")
    runtime_key = db.Column(db.String(80), nullable=False, default="storymaker")
    source_type = db.Column(db.String(40), nullable=False, default="vedock")
    source_path = db.Column(db.Text, nullable=False)
    visibility = db.Column(db.String(16), nullable=False, default="private", index=True)
    cover_image_path = db.Column(db.Text)
    owner = db.relationship("User", backref=db.backref("models", lazy=True))

    @property
    def effective_versions(self) -> list["ModelVersion"]:
        if self.versions:
            return list(self.versions)
        origin = getattr(self, "fork_origin", None)
        return [origin.source_version] if origin and origin.source_version else []

    def to_dict(self, include_versions: bool = False) -> dict[str, Any]:
        data = {
            "id": self.id,
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "runtime": self.runtime_key,
            "source_type": self.source_type,
            "visibility": self.visibility,
            "cover_image": f"/model-media/{self.slug}/cover",
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_versions:
            data["versions"] = [version.to_dict() for version in self.effective_versions]
        return data


class ModelVersion(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    model_id = db.Column(db.String(36), db.ForeignKey("model_record.id"), nullable=False, index=True)
    version_number = db.Column(db.Integer, nullable=False)
    label = db.Column(db.String(160), nullable=False)
    storage_path = db.Column(db.Text, nullable=False, unique=True)
    base_model_path = db.Column(db.Text)
    status = db.Column(db.String(30), nullable=False, default="completed")
    config_json = db.Column(db.JSON, default=dict)
    metadata_json = db.Column(db.JSON, default=dict)
    sha256 = db.Column(db.String(64))
    model = db.relationship("ModelRecord", backref=db.backref("versions", lazy=True, order_by="ModelVersion.version_number"))
    __table_args__ = (db.UniqueConstraint("model_id", "version_number", name="uq_model_version_number"),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "version_number": self.version_number,
            "label": self.label,
            "status": self.status,
            "config": self.config_json or {},
            "metadata": self.metadata_json or {},
            "sha256": self.sha256,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ModelFork(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    child_model_id = db.Column(db.String(36), db.ForeignKey("model_record.id", ondelete="CASCADE"), nullable=False, unique=True)
    source_model_id = db.Column(db.String(36), db.ForeignKey("model_record.id"), nullable=False, index=True)
    source_version_id = db.Column(db.String(36), db.ForeignKey("model_version.id"), nullable=False, index=True)
    configuration_json = db.Column(db.JSON, default=dict)
    owner = db.relationship("User")
    child_model = db.relationship("ModelRecord", foreign_keys=[child_model_id], backref=db.backref("fork_origin", uselist=False, cascade="all, delete-orphan"))
    source_model = db.relationship("ModelRecord", foreign_keys=[source_model_id], backref=db.backref("direct_forks", lazy=True))
    source_version = db.relationship("ModelVersion", foreign_keys=[source_version_id], backref=db.backref("forks", lazy=True))


class ModelWorkspaceState(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    model_id = db.Column(db.String(36), db.ForeignKey("model_record.id", ondelete="CASCADE"), nullable=False, index=True)
    archived = db.Column(db.Boolean, nullable=False, default=False)
    configuration_json = db.Column(db.JSON, default=dict)
    owner = db.relationship("User")
    model = db.relationship("ModelRecord", backref=db.backref("workspace_states", lazy=True, cascade="all, delete-orphan"))
    __table_args__ = (db.UniqueConstraint("owner_id", "model_id", name="uq_model_workspace_owner"),)


class ModelReaction(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    model_id = db.Column(db.String(36), db.ForeignKey("model_record.id", ondelete="CASCADE"), nullable=False, index=True)
    value = db.Column(db.String(8), nullable=False)
    owner = db.relationship("User")
    model = db.relationship("ModelRecord", backref=db.backref("reactions", lazy=True, cascade="all, delete-orphan"))
    __table_args__ = (db.UniqueConstraint("owner_id", "model_id", name="uq_model_reaction_owner"),)


class ModelProject(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    name = db.Column(db.String(160), nullable=False)
    task_type = db.Column(db.String(50), nullable=False)
    base_model_id = db.Column(db.String(36), db.ForeignKey("model_record.id"), nullable=False)
    dataset_version_id = db.Column(db.String(36), db.ForeignKey("dataset_version.id"), nullable=False)
    training_method = db.Column(db.String(30), nullable=False)
    status = db.Column(db.String(30), nullable=False, default="draft")
    config_json = db.Column(db.JSON, default=dict)
    owner = db.relationship("User")
    base_model = db.relationship("ModelRecord")
    dataset_version = db.relationship("DatasetVersion")


class TrainingRecipe(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    name = db.Column(db.String(160), nullable=False)
    runtime_key = db.Column(db.String(80), nullable=False)
    config_json = db.Column(db.JSON, nullable=False, default=dict)
    owner = db.relationship("User")


class Job(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    job_type = db.Column(db.String(40), nullable=False)
    status = db.Column(db.String(30), nullable=False, default="queued", index=True)
    progress = db.Column(db.Integer, nullable=False, default=0)
    current_stage = db.Column(db.String(80), nullable=False, default="queued")
    config_json = db.Column(db.JSON, nullable=False, default=dict)
    logs_path = db.Column(db.Text, nullable=False)
    result_model_version_id = db.Column(db.String(36), db.ForeignKey("model_version.id"))
    error_message = db.Column(db.Text)
    cancel_requested = db.Column(db.Boolean, nullable=False, default=False)
    worker_pid = db.Column(db.Integer)
    claimed_by_device = db.Column(db.String(120), index=True)
    device_name = db.Column(db.String(160))
    last_heartbeat_at = db.Column(db.DateTime(timezone=True))
    started_at = db.Column(db.DateTime(timezone=True))
    finished_at = db.Column(db.DateTime(timezone=True))
    owner = db.relationship("User")
    result_model_version = db.relationship("ModelVersion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.job_type,
            "status": self.status,
            "progress": self.progress,
            "stage": self.current_stage,
            "config": self.config_json or {},
            "result_model_version_id": self.result_model_version_id,
            "error": self.error_message,
            "cancel_requested": self.cancel_requested,
            "claimed_by_device": self.claimed_by_device,
            "device_name": self.device_name,
            "last_heartbeat_at": self.last_heartbeat_at.isoformat() if self.last_heartbeat_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class Conversation(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    model_version_id = db.Column(db.String(36), db.ForeignKey("model_version.id"), nullable=False)
    model_id = db.Column(db.String(36), db.ForeignKey("model_record.id"), index=True)
    title = db.Column(db.String(200), nullable=False)
    parameters_json = db.Column(db.JSON, default=dict)
    updated_at = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    owner = db.relationship("User")
    model_version = db.relationship("ModelVersion")
    selected_model = db.relationship("ModelRecord")

    @property
    def chat_model(self) -> ModelRecord:
        return self.selected_model or self.model_version.model

    def to_dict(self, include_messages: bool = False) -> dict[str, Any]:
        data = {
            "id": self.id,
            "title": self.title,
            "model_version_id": self.model_version_id,
            "model_id": self.model_id,
            "parameters": self.parameters_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_messages:
            data["messages"] = [message.to_dict() for message in self.messages]
        return data


class Message(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), db.ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    conversation = db.relationship(
        "Conversation",
        backref=db.backref("messages", lazy=True, cascade="all, delete-orphan", order_by="Message.id"),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MergeRecord(TimestampMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True, default=new_id)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    source_versions_json = db.Column(db.JSON, nullable=False)
    method = db.Column(db.String(40), nullable=False)
    weights_json = db.Column(db.JSON, nullable=False)
    configuration_json = db.Column(db.JSON, default=dict)
    compatibility_json = db.Column(db.JSON, nullable=False)
    output_model_version_id = db.Column(db.String(36), db.ForeignKey("model_version.id"))
    status = db.Column(db.String(30), nullable=False, default="checked")
    output_hash = db.Column(db.String(64))
    owner = db.relationship("User")
    output_model_version = db.relationship("ModelVersion")
