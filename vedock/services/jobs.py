from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from flask import current_app

from vedock.extensions import db
from vedock.models import DatasetVersion, Job, ModelRecord, ModelVersion, User, new_id, utcnow
from vedock.runtimes import get_runtime
from vedock.runtimes.parameters import validate_parameters

from .model_registry import latest_version
from .paths import allocate_directory


class JobError(ValueError):
    pass


def assert_training_enabled() -> None:
    if not current_app.config.get("MODEL_TRAINING_ENABLED", True):
        raise JobError("Model training is disabled on this Vedock installation.")


def assert_local_training_node() -> None:
    assert_training_enabled()
    if current_app.config.get("NODE_MODE") != "local_compute":
        raise JobError("This task must be claimed by the owner's Vedock CLI or desktop app.")


def append_job_log(job: Job, message: str, **data: Any) -> None:
    entry = {"time": utcnow().isoformat(), "message": message, **data}
    path = Path(job.logs_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def read_job_logs(job: Job, limit: int = 500) -> list[dict[str, Any]]:
    path = Path(job.logs_path)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(limit, 5000)):]
    output = []
    for line in lines:
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError:
            output.append({"message": line})
    return output


def _launch_worker(job: Job) -> None:
    if not current_app.config.get("LAUNCH_JOBS", True):
        return
    python = Path(current_app.config["RUNTIME_PYTHON"])
    if not python.is_file():
        raise JobError(f"Vedock worker Python was not found: {python}")
    project_root = Path(__file__).resolve().parents[2]
    command = [str(python), str(project_root / "worker.py"), job.id]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if current_app.config.get("OFFLINE_MODE", True):
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        creationflags = subprocess.CREATE_NO_WINDOW
    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        startupinfo=startupinfo,
        creationflags=creationflags,
        close_fds=True,
    )
    job.worker_pid = process.pid
    db.session.commit()
    append_job_log(job, "Worker process launched", worker_pid=process.pid)


def enqueue_training(owner: User, model: ModelRecord, dataset: DatasetVersion, submitted: dict[str, Any], task_type: str | None = None) -> Job:
    assert_training_enabled()
    if model.owner_id not in {None, owner.id}:
        raise JobError("You do not have access to this model.")
    if dataset.owner_id != owner.id:
        raise JobError("You do not own this dataset version.")
    if dataset.validation_status == "invalid":
        raise JobError("Critical dataset validation errors must be fixed before training.")
    base_version = latest_version(model)
    if not base_version or base_version.status not in {"completed", "available", "definition"}:
        raise JobError("The selected model has no usable base version or architecture definition.")
    runtime = get_runtime(model.runtime_key)
    parameters = validate_parameters(submitted, runtime.get_training_parameter_schema())
    job_id = new_id()
    job_directory = allocate_directory("jobs", job_id)
    job = Job(
        id=job_id,
        owner=owner,
        job_type="training",
        status="queued" if current_app.config.get("NODE_MODE") == "local_compute" else "awaiting_device",
        progress=0,
        current_stage="queued" if current_app.config.get("NODE_MODE") == "local_compute" else "waiting_for_device",
        logs_path=str(job_directory / "job.jsonl"),
        config_json={
            "model_id": model.id,
            "base_model_version_id": base_version.id,
            "dataset_version_id": dataset.id,
            "runtime": model.runtime_key,
            "task_type": task_type or model.task_type,
            "parameters": parameters,
        },
    )
    db.session.add(job)
    db.session.commit()
    append_job_log(job, "Training task created", base_model=model.slug, dataset_version=dataset.id, parameters=parameters, execution_location="local_device")
    if current_app.config.get("NODE_MODE") != "local_compute":
        append_job_log(job, "Waiting for an authenticated Vedock CLI or desktop app to claim this task")
        return job
    try:
        _launch_worker(job)
    except Exception as exc:
        job.status = "failed"
        job.current_stage = "failed"
        job.error_message = str(exc)
        job.finished_at = utcnow()
        db.session.commit()
        append_job_log(job, "Worker launch failed", error=str(exc))
        raise
    return job


def _new_job(owner: User, job_type: str, configuration: dict[str, Any]) -> Job:
    job_id = new_id()
    job_directory = allocate_directory("jobs", job_id)
    job = Job(
        id=job_id,
        owner=owner,
        job_type=job_type,
        status="queued",
        progress=0,
        current_stage="queued",
        logs_path=str(job_directory / "job.jsonl"),
        config_json=configuration,
    )
    db.session.add(job)
    db.session.commit()
    append_job_log(job, f"{job_type.replace('_', ' ').title()} job queued", configuration=configuration)
    try:
        _launch_worker(job)
    except Exception as exc:
        job.status = "failed"
        job.current_stage = "failed"
        job.error_message = str(exc)
        job.finished_at = utcnow()
        db.session.commit()
        append_job_log(job, "Worker launch failed", error=str(exc))
        raise
    return job


def enqueue_dataset_inspection(owner: User, dataset: Any) -> Job:
    if dataset.owner_id != owner.id:
        raise JobError("You do not own this dataset.")
    return _new_job(owner, "dataset_inspection", {"dataset_id": dataset.id})


def enqueue_dataset_transform(
    owner: User,
    dataset: Any,
    operations: list[dict[str, Any]],
    output_schema: str,
    mapping: dict[str, str],
    template: str,
    limit: int,
    shuffle: bool,
    shuffle_seed: int,
) -> Job:
    if dataset.owner_id != owner.id:
        raise JobError("You do not own this dataset.")
    return _new_job(
        owner,
        "dataset_transform",
        {
            "dataset_id": dataset.id,
            "operations": operations,
            "output_schema": output_schema,
            "field_mapping": mapping,
            "template": template,
            "limit_examples": limit,
            "shuffle": shuffle,
            "shuffle_seed": shuffle_seed,
        },
    )


def run_job(job_id: str) -> bool:
    job = db.session.get(Job, job_id)
    if not job:
        return False
    if job.status != "queued":
        append_job_log(job, "Worker refused job because it is not queued", status=job.status)
        return False
    job.status = "running"
    job.current_stage = "starting"
    job.started_at = utcnow()
    job.worker_pid = os.getpid()
    db.session.commit()
    append_job_log(job, "Worker claimed job", worker_pid=os.getpid(), job_type=job.job_type)
    try:
        if job.job_type == "training":
            from .training import run_training

            run_training(job)
        elif job.job_type == "dataset_inspection":
            from vedock.models import RawDataset
            from .datasets import inspect_dataset

            dataset = db.session.get(RawDataset, job.config_json["dataset_id"])
            if not dataset or dataset.owner_id != job.owner_id:
                raise JobError("The dataset no longer exists or is not owned by the job owner.")
            job.current_stage = "inspecting"
            job.progress = 10
            db.session.commit()
            inspect_dataset(dataset)
            append_job_log(job, "Dataset inspection completed", rows=dataset.row_count)
        elif job.job_type == "dataset_transform":
            from vedock.models import RawDataset
            from .datasets import save_dataset_version

            configuration = job.config_json
            dataset = db.session.get(RawDataset, configuration["dataset_id"])
            if not dataset or dataset.owner_id != job.owner_id:
                raise JobError("The dataset no longer exists or is not owned by the job owner.")
            job.current_stage = "transforming"
            job.progress = 10
            db.session.commit()
            version = save_dataset_version(
                dataset,
                job.owner,
                configuration.get("operations") or [],
                configuration["output_schema"],
                configuration.get("field_mapping") or {},
                configuration.get("template") or "",
                int(configuration.get("limit_examples") or 0),
                bool(configuration.get("shuffle")),
                int(configuration.get("shuffle_seed") or 42),
            )
            updated = dict(configuration)
            updated["result_dataset_version_id"] = version.id
            job.config_json = updated
            db.session.commit()
            append_job_log(job, "Immutable dataset version saved", dataset_version_id=version.id, validation_status=version.validation_status)
        else:
            raise JobError(f"Unsupported job type: {job.job_type}")
        db.session.expire(job)
        if job.cancel_requested:
            job.status = "cancelled"
            job.current_stage = "cancelled"
        else:
            job.status = "completed"
            job.current_stage = "completed"
            job.progress = 100
        job.finished_at = utcnow()
        db.session.commit()
        append_job_log(job, f"Job {job.status}")
        return job.status == "completed"
    except InterruptedError as exc:
        db.session.rollback()
        job = db.session.get(Job, job_id)
        job.status = "cancelled"
        job.current_stage = "cancelled"
        job.error_message = str(exc)
        job.finished_at = utcnow()
        db.session.commit()
        append_job_log(job, "Job cancelled", detail=str(exc))
        return False
    except Exception as exc:
        db.session.rollback()
        job = db.session.get(Job, job_id)
        job.status = "failed"
        job.current_stage = "failed"
        job.error_message = str(exc)
        job.finished_at = utcnow()
        db.session.commit()
        append_job_log(job, "Job failed", error=str(exc), traceback=traceback.format_exc())
        return False


def request_cancellation(job: Job, owner: User) -> Job:
    if job.owner_id != owner.id:
        raise JobError("You do not own this job.")
    if job.status not in {"awaiting_device", "claimed", "queued", "running"}:
        raise JobError(f"A {job.status} job cannot be cancelled.")
    job.cancel_requested = True
    if job.status in {"awaiting_device", "claimed", "queued"} and not job.worker_pid:
        job.status = "cancelled"
        job.current_stage = "cancelled"
        job.finished_at = utcnow()
    db.session.commit()
    append_job_log(job, "Cancellation requested")
    return job


def resume_job(job: Job, owner: User) -> Job:
    """Return a terminal training task to a manual queue without executing it."""
    if job.owner_id != owner.id:
        raise JobError("You do not own this job.")
    if job.status not in {"failed", "cancelled"}:
        raise JobError("Only a failed or cancelled task can be resumed.")
    configuration = job.config_json or {}
    if job.job_type == "training":
        base_version_id = configuration.get("base_model_version_id")
        dataset_version_id = configuration.get("dataset_version_id")
        if not base_version_id or not db.session.get(ModelVersion, base_version_id):
            raise JobError("The base model version no longer exists.")
        if not dataset_version_id or not db.session.get(DatasetVersion, dataset_version_id):
            raise JobError("The dataset version no longer exists.")
    job.status = "queued" if current_app.config.get("NODE_MODE") == "local_compute" else "awaiting_device"
    job.current_stage = "queued" if job.status == "queued" else "waiting_for_device"
    job.progress = 0
    job.error_message = None
    job.cancel_requested = False
    job.worker_pid = None
    job.claimed_by_device = None
    job.device_name = None
    job.last_heartbeat_at = None
    job.started_at = None
    job.finished_at = None
    db.session.commit()
    append_job_log(job, "Task resumed and returned to the manual queue; no compute was started")
    return job


def delete_job(job: Job, owner: User) -> str:
    if job.owner_id != owner.id:
        raise JobError("You do not own this job.")
    if job.status not in {"failed", "cancelled", "completed"}:
        raise JobError("Cancel an active task before deleting it.")
    job_id = job.id
    log_path = Path(job.logs_path)
    db.session.delete(job)
    db.session.commit()
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        pass
    return job_id
