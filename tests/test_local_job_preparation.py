from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from vedock_cli import local_jobs


def test_remote_job_preparation_assigns_ids_and_recovers_retry(monkeypatch, tmp_path: Path):
    content = b'{"prompt":"Question","response":"Answer"}\n'
    digest = hashlib.sha256(content).hexdigest()

    def fake_download(client, path: str, output: Path, device_id: str) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)

    monkeypatch.setattr(local_jobs, "compute_root", lambda: tmp_path / "compute")
    monkeypatch.setattr(local_jobs, "_download", fake_download)
    manifest = {
        "runtime": "transformers_text",
        "task_type": "causal_lm",
        "parameters": {"training_method": "full"},
        "model": {
            "name": "Tiny QA",
            "task_type": "causal_lm",
            "runtime": "transformers_text",
            "reference": "scratch://tiny-qa",
            "artifact_required": False,
            "version_config": {"architecture_family": "gpt2"},
        },
        "dataset": {
            "name": "Test data",
            "schema": "prompt_response",
            "rows": 1,
            "sha256": digest,
        },
    }

    first = local_jobs.prepare_local_job(object(), "remote-job-12345678", manifest, "device-1")
    database, _, first_job_id, _ = first
    with sqlite3.connect(database) as connection:
        config = json.loads(connection.execute("SELECT config_json FROM job WHERE id=?", (first_job_id,)).fetchone()[0])
        assert config["model_id"]
        assert config["base_model_version_id"]
        assert config["dataset_version_id"]

        # Reproduce the half-created state from clients released before this fix.
        connection.execute(
            "UPDATE job SET status='failed', error_message='old failure', config_json=? WHERE id=?",
            (json.dumps({**config, "model_id": None, "base_model_version_id": None, "dataset_version_id": None}), first_job_id),
        )
        connection.commit()

    second = local_jobs.prepare_local_job(object(), "remote-job-12345678", manifest, "device-1")
    assert second[2] == first_job_id
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT status, error_message, config_json FROM job WHERE id=?", (first_job_id,)).fetchone()
        repaired = json.loads(row[2])
        assert row[:2] == ("queued", None)
        assert all(repaired[key] for key in ("model_id", "base_model_version_id", "dataset_version_id"))
        assert connection.execute("SELECT COUNT(*) FROM model_record").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM dataset_version").fetchone()[0] == 1

