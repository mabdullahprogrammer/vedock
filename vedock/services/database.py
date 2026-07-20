from __future__ import annotations

from sqlalchemy import inspect, text

from vedock.extensions import db


def ensure_schema_compatibility() -> None:
    """Apply the tiny additive SQLite migrations needed by source distributions.

    The project deliberately avoids a migration-server dependency for the local-node
    build. New installations receive the full SQLAlchemy schema; existing SQLite
    nodes receive only additive columns with safe defaults.
    """

    inspector = inspect(db.engine)
    if "model_record" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("model_record")}
    statements = []
    if "visibility" not in columns:
        statements.append("ALTER TABLE model_record ADD COLUMN visibility VARCHAR(16) NOT NULL DEFAULT 'private'")
    if "cover_image_path" not in columns:
        statements.append("ALTER TABLE model_record ADD COLUMN cover_image_path TEXT")
    with db.engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
        connection.execute(text("UPDATE model_record SET visibility='public' WHERE owner_id IS NULL"))
        connection.execute(text("UPDATE model_record SET visibility='private' WHERE visibility IS NULL OR visibility NOT IN ('public','private')"))

    inspector = inspect(db.engine)
    conversation_columns = {column["name"] for column in inspector.get_columns("conversation")} if "conversation" in inspector.get_table_names() else set()
    if "model_id" not in conversation_columns:
        with db.engine.begin() as connection:
            connection.execute(text("ALTER TABLE conversation ADD COLUMN model_id VARCHAR(36)"))
            connection.execute(
                text(
                    "UPDATE conversation SET model_id=(SELECT model_version.model_id FROM model_version WHERE model_version.id=conversation.model_version_id) WHERE model_id IS NULL"
                )
            )

    inspector = inspect(db.engine)
    fork_columns = {column["name"] for column in inspector.get_columns("model_fork")} if "model_fork" in inspector.get_table_names() else set()
    if "source_model_id" not in fork_columns:
        with db.engine.begin() as connection:
            connection.execute(text("ALTER TABLE model_fork ADD COLUMN source_model_id VARCHAR(36)"))
            connection.execute(
                text(
                    "UPDATE model_fork SET source_model_id=(SELECT model_version.model_id FROM model_version WHERE model_version.id=model_fork.source_version_id) WHERE source_model_id IS NULL"
                )
            )

    inspector = inspect(db.engine)
    job_columns = {column["name"] for column in inspector.get_columns("job")} if "job" in inspector.get_table_names() else set()
    job_statements = []
    if "claimed_by_device" not in job_columns:
        job_statements.append("ALTER TABLE job ADD COLUMN claimed_by_device VARCHAR(120)")
    if "device_name" not in job_columns:
        job_statements.append("ALTER TABLE job ADD COLUMN device_name VARCHAR(160)")
    if "last_heartbeat_at" not in job_columns:
        job_statements.append("ALTER TABLE job ADD COLUMN last_heartbeat_at DATETIME")
    if job_statements:
        with db.engine.begin() as connection:
            for statement in job_statements:
                connection.execute(text(statement))
