from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import click


def compute_root() -> Path:
    base = Path(os.getenv("LOCALAPPDATA", "")) if os.name == "nt" else Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    root = (base if str(base) else Path.home()) / "Vedock" / "compute"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _project_file(name: str) -> Path | None:
    candidates = [Path(__file__).resolve().parents[1] / name, Path(sys.prefix) / "share" / "vedock" / name]
    return next((item for item in candidates if item.is_file()), None)


def _missing_modules(runtime: str) -> list[str]:
    required = {
        "transformers_text": ["torch", "transformers", "datasets", "accelerate", "peft", "safetensors"],
        "storymaker": ["torch", "transformers", "datasets", "accelerate", "peft", "safetensors"],
        "pattern_sequence": [],
        "sklearn_image": ["numpy", "PIL", "sklearn", "joblib"],
        "tabular_prediction": ["numpy"],
    }.get(runtime, [])
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def ensure_runtime(runtime: str) -> None:
    core_modules = ["flask", "flask_sqlalchemy", "flask_login", "sqlalchemy", "dotenv", "PIL"]
    missing_core = []
    for module in core_modules:
        try:
            __import__(module)
        except ImportError:
            missing_core.append(module)
    assume_yes = os.getenv("VEDOCK_ASSUME_YES") == "1"
    if missing_core:
        click.secho("Local training engine required", fg="yellow", bold=True)
        if not assume_yes and not click.confirm("Download the Vedock local training engine now?", default=True):
            raise click.ClickException("Training cannot start without the local engine.")
        requirements = _project_file("requirements-local-core.txt")
        command = [sys.executable, "-m", "pip", "install"] + (["-r", str(requirements)] if requirements else missing_core)
        if subprocess.run(command).returncode:
            raise click.ClickException("Local training engine installation failed. Nothing was trained.")
    missing = _missing_modules(runtime)
    if not missing:
        click.secho("✓ Required training runtime is installed", fg="green", bold=True)
        return
    click.secho("Training runtime required", fg="yellow", bold=True)
    click.echo("Missing: " + ", ".join(missing))
    if not assume_yes and not click.confirm("Download and install the required packages now?", default=True):
        raise click.ClickException("Training cannot start until the required runtime is installed.")
    requirements = _project_file("requirements-text.txt" if runtime in {"transformers_text", "storymaker"} else "requirements-fast-ml.txt")
    command = [sys.executable, "-m", "pip", "install"]
    if requirements:
        command.extend(["-r", str(requirements)])
    else:
        command.extend(missing)
    click.secho("Installing runtime in the background environment…", fg="blue")
    result = subprocess.run(command)
    if result.returncode:
        raise click.ClickException("Runtime installation failed. Nothing was trained.")


def _download(client: Any, path: str, output: Path, device_id: str) -> None:
    response = client.request("GET", path, raw=True, headers={"X-Vedock-Device": device_id}, stream=True, timeout=3600)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("wb") as stream:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                stream.write(chunk)
    temporary.replace(output)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise click.ClickException("The downloaded model artifact contains an unsafe path.")
            target = (destination / Path(*relative.parts)).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise click.ClickException("The downloaded model artifact escapes the job workspace.")
        archive.extractall(destination)


def _local_app(database: Path, storage: Path):
    from vedock import create_app

    return create_app(
        {
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{database.as_posix()}",
            "STORAGE_ROOT": storage,
            "NODE_MODE": "local_compute",
            "MODEL_TRAINING_ENABLED": True,
            "LAUNCH_JOBS": False,
            "OFFLINE_MODE": False,
            "PROTECTED_ROOTS": (),
        },
        register_legacy=False,
    )


def prepare_local_job(client: Any, remote_job_id: str, manifest: dict[str, Any], device_id: str) -> tuple[Path, Path, str, Path]:
    from vedock.extensions import db
    from vedock.models import DatasetVersion, Job, ModelRecord, ModelVersion, RawDataset, User, new_id

    workspace = compute_root() / "jobs" / remote_job_id
    storage = workspace / "storage"
    database = workspace / "vedock-local.db"
    dataset_info = manifest["dataset"]
    if dataset_info.get("device_resource_id"):
        from vedock_cli.resources import resolve_local_resource

        if dataset_info.get("required_device_id") != device_id:
            raise click.ClickException("This private dataset belongs to another connected device.")
        dataset_path = resolve_local_resource(client, str(dataset_info["device_resource_id"]), "dataset")
    else:
        dataset_path = workspace / "inputs" / "dataset.jsonl"
        _download(client, f"/jobs/{remote_job_id}/dataset", dataset_path, device_id)
    expected_hash = str(manifest["dataset"].get("sha256") or "")
    if expected_hash:
        import hashlib

        hasher = hashlib.sha256()
        with dataset_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != expected_hash:
            raise click.ClickException("The downloaded dataset hash does not match the task manifest.")

    model_info = manifest["model"]
    if model_info.get("device_resource_id"):
        from vedock_cli.resources import resolve_local_resource

        if model_info.get("required_device_id") != device_id:
            raise click.ClickException("This private base model belongs to another connected device.")
        model_reference = str(resolve_local_resource(client, str(model_info["device_resource_id"]), "model"))
    elif model_info.get("artifact_required"):
        archive = workspace / "inputs" / "base-model.zip"
        base_path = workspace / "inputs" / "base-model"
        _download(client, f"/jobs/{remote_job_id}/base-model", archive, device_id)
        _safe_extract(archive, base_path)
        model_reference = str(base_path)
    else:
        model_reference = str(model_info.get("reference") or "")

    app = _local_app(database, storage)
    with app.app_context():
        user = User.query.filter_by(username="local-device-owner").first()
        if not user:
            user = User(username="local-device-owner", email=f"local-{new_id()[:8]}@device.invalid")
            user.set_password(new_id())
            db.session.add(user)
            db.session.flush()
        model = ModelRecord(
            owner=user,
            slug=f"remote-base-{remote_job_id[:8]}",
            name=model_info["name"],
            description="Authenticated base model mirrored for one local training task.",
            task_type=model_info["task_type"],
            runtime_key=model_info["runtime"],
            source_type="remote_task_cache",
            source_path=model_reference,
            visibility="private",
        )
        version = ModelVersion(
            model=model,
            version_number=1,
            label="Remote task base",
            storage_path=model_reference,
            status="completed",
            config_json=model_info.get("version_config") or {},
        )
        raw = RawDataset(
            owner=user,
            name=manifest["dataset"]["name"],
            source_type="remote_task",
            original_filename="dataset.jsonl",
            storage_path=str(dataset_path.with_name(f"raw-{new_id()}.jsonl")),
            file_format="jsonl",
            size_bytes=dataset_path.stat().st_size,
            sha256=expected_hash or "0" * 64,
            inspection_status="completed",
            row_count=int(manifest["dataset"].get("rows") or 0),
        )
        Path(raw.storage_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset_path, raw.storage_path)
        dataset = DatasetVersion(
            raw_dataset=raw,
            owner=user,
            version_number=1,
            output_format=manifest["dataset"]["schema"],
            storage_path=str(dataset_path),
            validation_status="valid",
            validation_json={"status": "valid", "source": "authenticated_remote_task"},
            row_count=int(manifest["dataset"].get("rows") or 0),
            invalid_row_count=0,
            token_estimate=0,
            sha256=expected_hash or "0" * 64,
        )
        local_job_id = new_id()
        log_path = workspace / "local-job.jsonl"
        job = Job(
            id=local_job_id,
            owner=user,
            job_type="training",
            status="queued",
            progress=0,
            current_stage="queued",
            logs_path=str(log_path),
            config_json={
                "remote_job_id": remote_job_id,
                "model_id": model.id,
                "base_model_version_id": version.id,
                "dataset_version_id": dataset.id,
                "runtime": manifest["runtime"],
                "task_type": manifest["task_type"],
                "parameters": manifest["parameters"],
            },
        )
        db.session.add_all([model, version, raw, dataset, job])
        db.session.commit()
    return database, storage, local_job_id, log_path


def _status(database: Path, job_id: str) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    row = connection.execute("SELECT status, progress, current_stage, error_message, result_model_version_id FROM job WHERE id=?", (job_id,)).fetchone()
    connection.close()
    return dict(row) if row else {}


def _publish_files(output: Path, archive: Path) -> tuple[int, int, list[str]]:
    from vedock.services.remote_jobs import _allowed_final_file

    files = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for item in sorted(output.rglob("*")):
            if not item.is_file():
                continue
            relative = PurePosixPath(item.relative_to(output).as_posix())
            if _allowed_final_file(relative):
                bundle.write(item, relative.as_posix())
                files.append(relative.as_posix())
    return len(files), sum((output / Path(name)).stat().st_size for name in files), files


def run_claimed_job(client: Any, remote_job_id: str, manifest: dict[str, Any], device_id: str, publish: bool | None) -> dict[str, Any]:
    database, storage, local_job_id, log_path = prepare_local_job(client, remote_job_id, manifest, device_id)
    client.request("POST", f"/jobs/{remote_job_id}/events", json={"device_id": device_id, "status": "running", "stage": "starting_local_worker", "progress": 1, "message": "Local training worker started"})
    command = [sys.executable, "-m", "vedock_cli.local_worker", local_job_id, "--database", database.as_posix(), "--storage", storage.as_posix()]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, creationflags=flags)
    seen = 0
    last_state: tuple[Any, ...] | None = None
    poll_count = 0
    cancelled_remotely = False
    while process.poll() is None:
        poll_count += 1
        state = _status(database, local_job_id)
        key = (state.get("status"), state.get("progress"), state.get("current_stage"))
        entries = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.is_file() else []
        for line in entries[seen:]:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entry = {"message": line}
            client.request("POST", f"/jobs/{remote_job_id}/events", json={"device_id": device_id, "status": "running", "stage": state.get("current_stage") or "training", "progress": state.get("progress") or 0, "message": entry.get("message") or "Local training update", "metrics": entry.get("metrics")})
        seen = len(entries)
        if key != last_state:
            click.secho(f"{int(state.get('progress') or 0):3d}%  {str(state.get('current_stage') or 'starting').replace('_', ' ')}", fg="blue")
            last_state = key
        if poll_count % 3 == 0:
            remote_state = client.request("GET", f"/jobs/{remote_job_id}")
            if remote_state.get("cancel_requested") or remote_state.get("status") == "cancelled":
                connection = sqlite3.connect(database)
                connection.execute("UPDATE job SET cancel_requested=1 WHERE id=?", (local_job_id,))
                connection.commit()
                connection.close()
                click.secho("Cancellation received from Vedock. Stopping the local worker…", fg="yellow")
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
                cancelled_remotely = True
                break
        time.sleep(1)
    if cancelled_remotely:
        client.request("POST", f"/jobs/{remote_job_id}/events", json={"device_id": device_id, "status": "cancelled", "stage": "cancelled", "progress": _status(database, local_job_id).get("progress") or 0, "message": "Local training stopped after owner cancellation"})
        click.secho("Task cancelled. No model was published.", fg="yellow", bold=True)
        return {"published": False, "cancelled": True}
    state = _status(database, local_job_id)
    if process.returncode or state.get("status") != "completed":
        message = state.get("error_message") or "Local training worker failed."
        client.request("POST", f"/jobs/{remote_job_id}/events", json={"device_id": device_id, "status": "failed", "stage": "failed", "progress": state.get("progress") or 0, "message": message, "error": message})
        raise click.ClickException(message)

    connection = sqlite3.connect(database)
    row = connection.execute("SELECT storage_path, model_id FROM model_version WHERE id=?", (state["result_model_version_id"],)).fetchone()
    model_row = connection.execute("SELECT name, description FROM model_record WHERE id=?", (row[1],)).fetchone() if row else None
    connection.close()
    if not row or not Path(row[0]).is_dir():
        raise click.ClickException("Training completed but the finalized model directory is missing.")
    output = Path(row[0])
    archive = output.parent / f"{output.name}-publish.zip"
    file_count, byte_count, files = _publish_files(output, archive)
    click.secho("\nFinalize and publish", fg="blue", bold=True)
    click.echo(f"  Model       {model_row[0] if model_row else 'Finalized model'}")
    click.echo(f"  Runtime     {manifest['runtime']}")
    click.echo(f"  Method      {manifest['parameters'].get('training_method')}")
    click.echo(f"  Dataset     {manifest['dataset']['name']} ({manifest['dataset']['rows']} rows)")
    click.echo(f"  Files       {file_count} necessary inference/edit files")
    click.echo(f"  Upload      {byte_count / (1024 * 1024):.1f} MiB")
    for name in files[:12]:
        click.echo(f"    · {name}")
    if len(files) > 12:
        click.echo(f"    · … and {len(files) - 12} more")
    should_publish = publish if publish is not None else click.confirm("Upload this finalized model to Vedock and make it public?", default=False)
    if not should_publish:
        client.request("POST", f"/jobs/{remote_job_id}/events", json={"device_id": device_id, "status": "awaiting_publish", "stage": "finalized_on_device", "progress": 99, "message": "Training completed locally; publication is waiting for owner confirmation"})
        click.secho(f"Kept locally: {output}", fg="yellow")
        return {"published": False, "output": str(output), "archive": str(archive)}
    with archive.open("rb") as stream:
        result = client.request(
            "POST",
            f"/jobs/{remote_job_id}/finalize",
            files={"artifact": (archive.name, stream, "application/zip")},
            data={"device_id": device_id, "metadata": json.dumps({"name": model_row[0] if model_row else "Finalized model", "description": model_row[1] if model_row else "", "publish": True})},
            timeout=7200,
        )
    click.secho("✓ Finalized model published", fg="green", bold=True)
    return result
