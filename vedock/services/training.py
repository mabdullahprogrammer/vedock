from __future__ import annotations

import gc
import hashlib
import inspect
import json
import random
import shutil
from pathlib import Path
from typing import Any

from flask import current_app

from vedock.extensions import db
from vedock.models import DatasetVersion, Job, ModelRecord, ModelVersion, new_id

from .model_references import parse_model_reference
from .model_profiles import PLAIN_OUTPUT_PATTERN, STORYMAKER_OUTPUT_PATTERN, validate_output_pattern
from .paths import allocate_directory, atomic_write_json


class TrainingError(ValueError):
    pass


def _log(job: Job, message: str, **data: Any) -> None:
    from .jobs import append_job_log

    append_job_log(job, message, **data)


def _stage(job: Job, name: str, progress: int) -> None:
    job.current_stage = name
    job.progress = progress
    db.session.commit()


def _load_examples(version: DatasetVersion, maximum: int, shuffle: bool, seed: int) -> list[dict[str, Any]]:
    examples = []
    with Path(version.storage_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                examples.append(json.loads(line))
            if maximum and len(examples) >= maximum:
                break
    if shuffle:
        random.Random(seed).shuffle(examples)
    if not examples:
        raise TrainingError("The processed dataset contains no valid examples.")
    return examples


def _format_example(example: dict[str, Any], schema: str, eos: str, output_pattern: str = PLAIN_OUTPUT_PATTERN) -> str:
    if schema == "prompt_response":
        pattern = validate_output_pattern(output_pattern)
        return (
            pattern.replace("{prompt}", str(example["prompt"]).strip())
            .replace("{response}", str(example["response"]).strip())
            .replace("{history}", "")
            .replace("$sep", "\n\n")
        )
    if schema == "text_completion":
        return f"{example['text'].strip()}{eos}"
    if schema == "instruction":
        instruction = str(example.get("instruction", "")).strip()
        input_text = str(example.get("input", "")).strip()
        output = str(example.get("output", "")).strip()
        return f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}{eos}"
    if schema == "chat":
        return "\n".join(f"{message['role'].title()}: {message['content']}" for message in example["messages"]) + eos
    raise TrainingError(f"The runtime cannot train dataset schema {schema!r}.")


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        with file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _preflight(base_version: ModelVersion, method: str) -> dict[str, Any]:
    reference = parse_model_reference(base_version.storage_path)
    weight_file = None
    if reference.kind == "local":
        base_path = Path(reference.source)
        weight_file = next((base_path / name for name in ["model.safetensors", "pytorch_model.bin"] if (base_path / name).is_file()), None)
    if weight_file:
        model_bytes = weight_file.stat().st_size
    elif reference.kind == "scratch":
        scratch = (base_version.config_json or {}).get("scratch_config") or {}
        layers = int(scratch.get("n_layer", 4))
        width = int(scratch.get("n_embd", 256))
        vocab = int(scratch.get("vocab_size", 50257))
        approximate_parameters = vocab * width + layers * (12 * width * width)
        model_bytes = approximate_parameters * 4
    else:
        size_label = str((base_version.config_json or {}).get("parameter_count_label") or "500M")
        size_match = __import__("re").search(r"([\d.]+)\s*([MB])", size_label, __import__("re").IGNORECASE)
        count = float(size_match.group(1)) if size_match else 500.0
        scale = 1_000_000 if not size_match or size_match.group(2).upper() == "M" else 1_000_000_000
        model_bytes = int(count * scale * 4)
    multiplier = 1.2 if method == "lora" else 5.5
    required = int(model_bytes * multiplier + 256 * 1024 * 1024)
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:
        available = 0
    return {"model_bytes": model_bytes, "estimated_memory_bytes": required, "available_memory_bytes": available, "safe": available >= required}


def run_training(job: Job) -> ModelVersion:
    configuration = job.config_json
    if configuration.get("runtime") == "pattern_sequence":
        from .fast_training import run_pattern_training

        return run_pattern_training(job)
    if configuration.get("runtime") == "sklearn_image":
        from .fast_training import run_image_classifier_training

        return run_image_classifier_training(job)
    if configuration.get("runtime") == "tabular_prediction":
        from .fast_training import run_tabular_training

        return run_tabular_training(job)
    params = configuration["parameters"]
    task_type = configuration.get("task_type") or "causal_lm"
    format_profile = "storymaker" if (configuration.get("runtime") == "storymaker") else "plain"
    output_pattern = validate_output_pattern(
        params.get("output_pattern")
        or (STORYMAKER_OUTPUT_PATTERN if format_profile == "storymaker" else PLAIN_OUTPUT_PATTERN)
    )
    dataset = db.session.get(DatasetVersion, configuration["dataset_version_id"])
    base_version = db.session.get(ModelVersion, configuration["base_model_version_id"])
    if not dataset or dataset.owner_id != job.owner_id:
        raise TrainingError("The selected dataset version no longer exists or is not owned by the job owner.")
    if dataset.validation_status == "invalid":
        raise TrainingError("Training is blocked because the dataset has critical validation errors.")
    if not base_version or base_version.status != "completed":
        raise TrainingError("The selected base model version is unavailable.")
    base_reference = base_version.storage_path
    parsed_reference = parse_model_reference(base_reference)
    _stage(job, "hardware_review", 2)
    preflight = _preflight(base_version, params["training_method"])
    _log(job, "Hardware preflight complete", **preflight)
    if not preflight["safe"]:
        raise TrainingError(
            f"Estimated memory requirement is {preflight['estimated_memory_bytes']} bytes, but only "
            f"{preflight['available_memory_bytes']} bytes are available. Free memory or use a smaller base model."
        )

    output_model_id = new_id()
    output_version_id = new_id()
    output_directory = allocate_directory("models", str(job.owner_id), output_model_id, output_version_id)
    checkpoints = output_directory / "checkpoints"
    checkpoints.mkdir()
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            GPT2Config,
            GPT2LMHeadModel,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )

        set_seed(params["seed"])
        offline = bool(current_app.config.get("OFFLINE_MODE", True))
        _stage(job, "loading_tokenizer", 5)
        method = params["training_method"]
        if parsed_reference.kind == "scratch":
            scratch_configuration = base_version.config_json or {}
            tokenizer_reference = parse_model_reference(scratch_configuration.get("tokenizer_reference", "hf://gpt2?revision=main"))
            if tokenizer_reference.kind == "scratch":
                raise TrainingError("A scratch model needs an existing tokenizer source.")
            tokenizer_source = tokenizer_reference.source
            tokenizer_revision = tokenizer_reference.revision
        else:
            tokenizer_source = parsed_reference.source
            tokenizer_revision = parsed_reference.revision
        _log(job, "Loading tokenizer", reference=tokenizer_source, revision=tokenizer_revision)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=tokenizer_revision,
            local_files_only=offline,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        added_tokens = 0
        if format_profile == "storymaker":
            added_tokens = tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        "<|start_of_input|>",
                        "<|end_of_input|>",
                        "<|start_of_response|>",
                        "<|end_of_response|>",
                    ]
                }
            )
        tokenizer.padding_side = params["padding_side"]
        _stage(job, "loading_model", 10)
        if parsed_reference.kind == "scratch":
            if method != "scratch":
                raise TrainingError("A scratch architecture must use the scratch training method.")
            scratch_config = dict((base_version.config_json or {}).get("scratch_config") or {})
            if not scratch_config:
                raise TrainingError("The scratch architecture configuration is missing.")
            model = GPT2LMHeadModel(GPT2Config(**scratch_config))
            _log(job, "Created randomly initialized GPT-2 architecture", configuration=scratch_config)
        else:
            _log(job, "Loading base model lazily inside worker", reference=parsed_reference.source, revision=parsed_reference.revision)
            model = AutoModelForCausalLM.from_pretrained(
                parsed_reference.source,
                revision=parsed_reference.revision,
                local_files_only=offline,
            )
        if added_tokens:
            model.resize_token_embeddings(len(tokenizer))
            _log(job, "Extended tokenizer and model embeddings", added_special_tokens=added_tokens)

        if method == "lora":
            from peft import LoraConfig, get_peft_model

            lora_kwargs = {
                "r": params["lora_r"],
                "lora_alpha": params["lora_alpha"],
                "lora_dropout": params["lora_dropout"],
                "bias": params["lora_bias"],
                "target_modules": params["target_modules"],
                "modules_to_save": params["modules_to_save"] or None,
                "task_type": "CAUSAL_LM",
                "use_rslora": params["use_rslora"],
                "use_dora": params["use_dora"],
            }
            accepted = inspect.signature(LoraConfig).parameters
            model = get_peft_model(model, LoraConfig(**{key: value for key, value in lora_kwargs.items() if key in accepted}))
            trainable, total = model.get_nb_trainable_parameters()
            _log(job, "LoRA adapter prepared", trainable_parameters=trainable, total_parameters=total)
        elif method not in {"full", "continue_pretraining", "scratch"}:
            raise TrainingError(f"Unsupported training method: {method}")

        if params["gradient_checkpointing"]:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        _stage(job, "preparing_dataset", 20)
        examples = _load_examples(dataset, params["max_examples"], params["shuffle"], params["shuffle_seed"])
        texts = [_format_example(example, dataset.output_format, tokenizer.eos_token or "", output_pattern) for example in examples]
        _log(job, "Formatting training examples", examples=len(texts), schema=dataset.output_format, output_pattern=output_pattern)

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            encoded = tokenizer(
                batch["text"],
                truncation=params["truncation"],
                max_length=params["max_seq_length"],
                padding="max_length" if params["padding"] == "max_length" else False,
                add_special_tokens=params["add_special_tokens"],
            )
            return encoded

        training_dataset = Dataset.from_dict({"text": texts}).map(
            tokenize,
            batched=True,
            batch_size=params["tokenization_batch_size"],
            remove_columns=["text"],
            num_proc=params["preprocessing_workers"],
        )

        argument_values: dict[str, Any] = {
            "output_dir": str(checkpoints),
            "overwrite_output_dir": False,
            "num_train_epochs": params["num_train_epochs"],
            "max_steps": params["max_steps"] if params["max_steps"] > 0 else -1,
            "per_device_train_batch_size": params["per_device_train_batch_size"],
            "per_device_eval_batch_size": params["per_device_eval_batch_size"],
            "gradient_accumulation_steps": params["gradient_accumulation_steps"],
            "learning_rate": params["learning_rate"],
            "lr_scheduler_type": params["lr_scheduler_type"],
            "warmup_steps": params["warmup_steps"],
            "warmup_ratio": params["warmup_ratio"],
            "weight_decay": params["weight_decay"],
            "optim": params["optim"],
            "adam_beta1": params["adam_beta1"],
            "adam_beta2": params["adam_beta2"],
            "adam_epsilon": params["adam_epsilon"],
            "max_grad_norm": params["max_grad_norm"],
            "label_smoothing_factor": params["label_smoothing_factor"],
            "logging_strategy": params["logging_strategy"],
            "logging_steps": params["logging_steps"],
            "log_level": params["log_level"],
            "save_strategy": params["save_strategy"],
            "save_steps": params["save_steps"],
            "save_total_limit": params["save_total_limit"],
            "save_safetensors": params["save_safetensors"],
            "report_to": "none",
            "remove_unused_columns": False,
            "seed": params["seed"],
            "data_seed": params["shuffle_seed"],
            "use_cpu": params["device"] == "cpu",
            "fp16": params["precision"] == "float16",
            "bf16": params["precision"] == "bfloat16",
            "tf32": False,
            "disable_tqdm": True,
            "dataloader_num_workers": 0,
        }
        signature = inspect.signature(TrainingArguments).parameters
        evaluation_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
        argument_values[evaluation_key] = params["evaluation_strategy"]
        argument_values["eval_steps"] = params["eval_steps"]
        training_args = TrainingArguments(**{key: value for key, value in argument_values.items() if key in signature})

        class VedockCallback(TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
                if logs:
                    _log(job, "Training metrics", step=state.global_step, metrics=logs)

            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                db.session.expire(job)
                if job.cancel_requested:
                    control.should_training_stop = True
                if state.max_steps:
                    job.progress = min(95, int((state.global_step / state.max_steps) * 90) + 5)
                    job.current_stage = "training"
                    db.session.commit()
                return control

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[VedockCallback()],
        )
        _stage(job, "training", 30)
        _log(job, "Starting real fine-tuning", method=method, maximum_steps=params["max_steps"])
        resume = params["resume_from_checkpoint"] or None
        trainer.train(resume_from_checkpoint=resume)
        db.session.expire(job)
        if job.cancel_requested:
            raise InterruptedError("Training was cancelled by the user.")
        job.current_stage = "saving"
        job.progress = 96
        db.session.commit()
        model.save_pretrained(output_directory, safe_serialization=params["save_safetensors"])
        tokenizer.save_pretrained(output_directory)
        metadata = {
            "runtime": base_version.model.runtime_key,
            "training_method": method,
            "base_model_version_id": base_version.id,
            "base_model_path": base_reference,
            "dataset_version_id": dataset.id,
            "dataset_hash": dataset.sha256,
            "parameters": params,
            "output_pattern": output_pattern,
            "metrics": trainer.state.log_history,
        }
        atomic_write_json(output_directory / "vedock_training.json", metadata)
        model_card = (
            f"# {params['output_model_name']}\n\n"
            f"Generated by {current_app.config['APP_NAME']} — {current_app.config['APP_TAGLINE']}\n\n"
            f"- Runtime: StoryMaker / Transformers text generation\n"
            f"- Method: {method}\n"
            f"- Base version: {base_version.id}\n"
            f"- Dataset version: {dataset.id}\n"
        )
        (output_directory / "README.md").write_text(model_card, encoding="utf-8", newline="\n")
        output_hash = _directory_hash(output_directory)
        output_model = ModelRecord(
            id=output_model_id,
            owner_id=job.owner_id,
            slug=f"{re_safe_slug(params['output_model_name'])}-{output_model_id[:8]}",
            name=params["output_model_name"][:160],
            description=f"Built from {base_version.model.name} with {current_app.config['APP_NAME']}.",
            task_type=task_type,
            runtime_key=base_version.model.runtime_key,
            source_type="training",
            source_path=str(output_directory),
        )
        output_version = ModelVersion(
            id=output_version_id,
            model=output_model,
            version_number=1,
            label={"scratch": "Scratch training", "continue_pretraining": "Continued pretraining"}.get(method, f"{method.upper()} fine-tune"),
            storage_path=str(output_directory),
            base_model_path=base_reference,
            status="completed",
            config_json=params,
            metadata_json=metadata,
            sha256=output_hash,
        )
        db.session.add_all([output_model, output_version])
        db.session.flush()
        job.result_model_version_id = output_version.id
        db.session.commit()
        _log(job, "Model version saved", model_version_id=output_version.id, output_hash=output_hash)
        return output_version
    except Exception:
        if output_directory.exists():
            shutil.rmtree(output_directory)
        raise
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def re_safe_slug(value: str) -> str:
    import re

    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:80] or "model"
