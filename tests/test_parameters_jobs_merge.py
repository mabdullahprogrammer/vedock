from __future__ import annotations

from pathlib import Path

import pytest

from vedock.extensions import db
from vedock.models import DatasetVersion, ModelRecord, ModelVersion, RawDataset, User
from vedock.runtimes import get_runtime
from vedock.runtimes.parameters import ParameterValidationError, validate_parameters
from vedock.services.jobs import enqueue_training
from vedock.services.merges import compatibility_report, execute_weighted_adapter_merge, resolve_latest_pair
from vedock.services.paths import UnsafePathError, assert_writable_path


def test_server_parameter_validation():
    schema = get_runtime("storymaker").get_inference_parameter_schema()
    normalized = validate_parameters({"temperature": "0.75", "max_new_tokens": "32", "do_sample": "true"}, schema)
    assert normalized["temperature"] == 0.75
    assert normalized["max_new_tokens"] == 32
    with pytest.raises(ParameterValidationError):
        validate_parameters({"temperature": 99, "made_up_parameter": 1}, schema)


def test_inactive_lora_parameters_are_ignored_instead_of_rejected():
    schema = get_runtime("storymaker").get_training_parameter_schema()
    normalized = validate_parameters(
        {
            "output_model_name": "Full model",
            "training_method": "full",
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.2,
        },
        schema,
    )
    assert normalized["training_method"] == "full"
    assert "lora_r" not in normalized
    assert "lora_alpha" not in normalized


def test_protected_and_outside_paths_are_rejected(app):
    with app.app_context():
        with pytest.raises(UnsafePathError):
            assert_writable_path(Path(r"D:\LLM\new-llm\LLM-2025\StoryMaker\new-output"))
        with pytest.raises(UnsafePathError):
            assert_writable_path(Path(r"D:\LLM\unrelated-output"))


def test_training_is_enqueued_not_run_in_request(app, tmp_path):
    with app.app_context():
        user = User(username="workeruser", email="worker@example.com")
        user.set_password("password123")
        db.session.add(user)
        model = ModelRecord.query.filter_by(slug="storymaker-final").one()
        raw_file = tmp_path / "raw.csv"
        raw_file.write_text("prompt,story\na,b\n", encoding="utf-8")
        raw = RawDataset(owner=user, name="Raw", source_type="upload", original_filename="raw.csv", storage_path=str(raw_file), file_format="csv", size_bytes=raw_file.stat().st_size, sha256="0" * 64, inspection_status="completed")
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"prompt":"a","response":"b"}\n', encoding="utf-8")
        version = DatasetVersion(raw_dataset=raw, owner=user, version_number=1, output_format="prompt_response", storage_path=str(data_file), validation_status="valid", validation_json={"status": "valid"}, row_count=1, invalid_row_count=0, token_estimate=8, sha256="1" * 64)
        db.session.add_all([raw, version])
        db.session.commit()
        job = enqueue_training(user, model, version, {"output_model_name": "Queued model", "max_steps": 1})
        assert job.status == "queued"
        assert job.worker_pid is None
        assert Path(job.logs_path).is_file()


def test_hosted_training_task_waits_for_owner_device(app, tmp_path):
    from vedock.services.remote_jobs import claim_job, edit_waiting_job

    with app.app_context():
        app.config["NODE_MODE"] = "hosted_inference"
        user = User(username="remoteowner", email="remote@example.com")
        user.set_password("password123")
        db.session.add(user)
        model = ModelRecord.query.filter_by(slug="storymaker-final").one()
        raw_file = tmp_path / "remote-raw.csv"
        raw_file.write_text("prompt,story\na,b\n", encoding="utf-8")
        raw = RawDataset(owner=user, name="Remote raw", source_type="upload", original_filename="remote-raw.csv", storage_path=str(raw_file), file_format="csv", size_bytes=raw_file.stat().st_size, sha256="2" * 64, inspection_status="completed")
        data_file = tmp_path / "remote-data.jsonl"
        data_file.write_text('{"prompt":"a","response":"b"}\n', encoding="utf-8")
        version = DatasetVersion(raw_dataset=raw, owner=user, version_number=1, output_format="prompt_response", storage_path=str(data_file), validation_status="valid", validation_json={"status": "valid"}, row_count=1, invalid_row_count=0, token_estimate=8, sha256="3" * 64)
        db.session.add_all([raw, version])
        db.session.commit()
        job = enqueue_training(user, model, version, {"output_model_name": "Owner device model", "max_steps": 1})
        assert job.status == "awaiting_device"
        assert job.worker_pid is None
        edited = edit_waiting_job(job, {"training_method": "full", "lora_r": 64})
        assert edited.config_json["parameters"]["training_method"] == "full"
        assert "lora_r" not in edited.config_json["parameters"]
        claimed = claim_job(job, "device-123", "Owner laptop")
        assert claimed.status == "claimed"
        assert claimed.claimed_by_device == "device-123"


def test_legacy_merge_is_blocked_on_tokenizer_mismatch(app):
    with app.app_context():
        first = ModelRecord.query.filter_by(slug="storymaker-final").one()
        second = ModelRecord.query.filter_by(slug="storymaker-finetuned").one()
        version_a, version_b = resolve_latest_pair(first, second)
        report = compatibility_report(version_a, version_b)
        assert any(check["name"] == "tensor_shapes" and check["passed"] for check in report["checks"])
        assert any(check["name"] == "tokenizer" and not check["passed"] for check in report["checks"])
        assert report["linear_safe"] is False


def test_compatible_lora_adapters_can_be_weighted_without_loading_a_model(app, tmp_path):
    import json
    import torch
    from safetensors.torch import load_file, save_file

    with app.app_context():
        user = User(username="mergeuser", email="merge@example.com")
        user.set_password("password123")
        db.session.add(user)
        directories = [tmp_path / "adapter-a", tmp_path / "adapter-b"]
        configuration = {
            "base_model_name_or_path": "gpt2",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "target_modules": ["c_attn"],
            "r": 2,
        }
        models = []
        for index, directory in enumerate(directories, start=1):
            directory.mkdir()
            (directory / "adapter_config.json").write_text(json.dumps(configuration), encoding="utf-8")
            save_file({"layer.lora_A.weight": torch.full((2, 2), float(index))}, directory / "adapter_model.safetensors")
            model = ModelRecord(owner=user, slug=f"adapter-{index}", name=f"Adapter {index}", task_type="text_generation", runtime_key="transformers_text", source_type="training", source_path=str(directory))
            model.versions.append(ModelVersion(version_number=1, label="LoRA", storage_path=str(directory), status="completed"))
            models.append(model)
        db.session.add_all(models)
        db.session.commit()
        report = compatibility_report(models[0].versions[0], models[1].versions[0])
        assert report["lora_safe"] is True
        _, output = execute_weighted_adapter_merge(models[0].versions[0], models[1].versions[0], 0.25, 0.75, user, "Weighted adapter")
        merged = load_file(str(Path(output.storage_path) / "adapter_model.safetensors"))["layer.lora_A.weight"]
        assert torch.allclose(merged, torch.full((2, 2), 1.75))
