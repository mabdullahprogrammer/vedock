from __future__ import annotations

import gc
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from flask import current_app, has_app_context

from vedock.services.model_references import parse_model_reference
from vedock.services.model_profiles import PLAIN_OUTPUT_PATTERN, STORYMAKER_OUTPUT_PATTERN, validate_output_pattern

from .base import RuntimeAdapter
from .parameters import parameter, validate_parameters


class TransformersTextRuntime(RuntimeAdapter):
    key = "transformers_text"
    display_name = "Transformers text generation"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded_path: str | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._precision = "float32"
        self._stop_events: dict[str, threading.Event] = {}

    def get_model_capabilities(self, model_path: str | None = None) -> dict[str, Any]:
        cuda = False
        bf16 = False
        try:
            import torch

            cuda = torch.cuda.is_available()
            bf16 = bool(cuda and torch.cuda.is_bf16_supported())
        except Exception:
            pass
        validation = self.validate_model(model_path) if model_path else None
        return {
            "runtime": self.key,
            "tasks": ["causal_lm"],
            "modality": "text",
            "interaction": "chat",
            "input_schema": {"type": "text", "field": "prompt"},
            "output_schema": {"type": "text", "streaming": True},
            "streaming": True,
            "stoppable_generation": True,
            "training_methods": ["full", "lora", "continue_pretraining", "scratch"],
            "devices": ["cpu"] + (["cuda"] if cuda else []),
            "precisions": ["float32"] + (["float16"] if cuda else []) + (["bfloat16"] if bf16 else []),
            "quantization": False,
            "qlora": False,
            "loaded_model_path": self._loaded_path,
            "validation": validation,
            "runner": self.get_runner_schema(model_path),
        }

    def get_runner_schema(self, model_path: str | None = None) -> dict[str, Any]:
        return {
            "interaction": "chat",
            "title": "Chat",
            "description": "Continue a conversation with optional history and a controllable context window.",
            "submit_label": "Send",
            "inputs": [{"name": "prompt", "label": "Message", "description": "The next message sent to the model.", "type": "textarea", "required": True}],
            "outputs": [{"type": "text", "label": "Response", "streaming": True}],
        }

    def get_inference_parameter_schema(self) -> list[dict[str, Any]]:
        devices = ["cpu"]
        precisions = ["float32"]
        try:
            import torch

            if torch.cuda.is_available():
                devices.append("cuda")
                precisions.append("float16")
                if torch.cuda.is_bf16_supported():
                    precisions.append("bfloat16")
        except Exception:
            pass
        return [
            parameter("system_prompt", "System prompt", "Optional instruction placed before the prompt.", "string", "", "prompt"),
            parameter("output_pattern", "Input/output pattern", "The exact sequence format used during training. It must contain {prompt} and {response}; during inference Vedock sends only the prefix before {response}.", "string", PLAIN_OUTPUT_PATTERN, "prompt"),
            parameter("temperature", "Temperature", "Higher values produce more varied text.", "float", 0.9, "sampling", minimum=0.01, maximum=2.0, step=0.01),
            parameter("top_p", "Top P", "Limits sampling to a probability mass.", "float", 0.9, "sampling", minimum=0.01, maximum=1.0, step=0.01),
            parameter("top_k", "Top K", "Limits sampling to the most likely tokens; zero disables it.", "integer", 50, "sampling", minimum=0, maximum=500, step=1),
            parameter("typical_p", "Typical P", "Typical decoding probability mass.", "float", 1.0, "sampling", minimum=0.01, maximum=1.0, step=0.01, advanced=True),
            parameter("min_p", "Minimum P", "Relative minimum token probability.", "float", 0.0, "sampling", minimum=0.0, maximum=1.0, step=0.01, advanced=True),
            parameter("max_new_tokens", "Maximum new tokens", "Maximum generated tokens after the prompt.", "integer", 120, "length", minimum=1, maximum=512, step=1),
            parameter("min_new_tokens", "Minimum new tokens", "Minimum generated tokens before EOS is accepted.", "integer", 0, "length", minimum=0, maximum=256, step=1, advanced=True),
            parameter("repetition_penalty", "Repetition penalty", "Values above one discourage repetition.", "float", 1.05, "penalties", minimum=0.5, maximum=2.0, step=0.01),
            parameter("no_repeat_ngram_size", "No-repeat n-gram", "Blocks repeated n-grams; zero disables the rule.", "integer", 0, "penalties", minimum=0, maximum=12, step=1, advanced=True),
            parameter("num_beams", "Beams", "Beam count. Keep one for normal sampling.", "integer", 1, "search", minimum=1, maximum=8, step=1, advanced=True),
            parameter("num_beam_groups", "Beam groups", "Groups used for diverse beam search.", "integer", 1, "search", minimum=1, maximum=8, step=1, advanced=True),
            parameter("diversity_penalty", "Diversity penalty", "Encourages different beam groups.", "float", 0.0, "search", minimum=0.0, maximum=5.0, step=0.1, advanced=True),
            parameter("length_penalty", "Length penalty", "Beam-search length normalization.", "float", 1.0, "search", minimum=-2.0, maximum=3.0, step=0.1, advanced=True),
            parameter("early_stopping", "Early stopping", "Stop beam search when enough candidates finish.", "boolean", False, "search", advanced=True),
            parameter("do_sample", "Sampling enabled", "Sample tokens instead of deterministic greedy decoding.", "boolean", True, "sampling"),
            parameter("num_return_sequences", "Returned sequences", "Number of alternatives returned.", "integer", 1, "output", minimum=1, maximum=4, step=1, advanced=True),
            parameter("seed", "Seed", "Reproduces the random sampling stream.", "integer", 42, "output", minimum=0, maximum=2_147_483_647, step=1),
            parameter("stop_sequences", "Stop sequences", "Comma-separated text sequences that trim the result.", "list", [], "output", advanced=True),
            parameter("bad_words", "Bad words", "Comma-separated phrases the generator must avoid.", "list", [], "output", advanced=True),
            parameter("use_cache", "Use cache", "Cache attention states for faster autoregressive generation.", "boolean", True, "runtime", advanced=True),
            parameter("streaming", "Stream output", "Show tokens as the model produces them instead of waiting for the complete response.", "boolean", True, "runtime"),
            parameter("device", "Device", "Execution device allowed by the current hardware.", "string", "cpu", "runtime", choices=devices),
            parameter("precision", "Precision", "Model numeric precision.", "string", "float32", "runtime", choices=precisions),
        ]

    def get_training_parameter_schema(self) -> list[dict[str, Any]]:
        hosted_task_definition = has_app_context() and current_app.config.get("NODE_MODE") == "hosted_inference"
        devices = ["cpu", "cuda"] if hosted_task_definition else ["cpu"]
        precisions = ["float32", "float16", "bfloat16"] if hosted_task_definition else ["float32"]
        try:
            import torch

            if not hosted_task_definition and torch.cuda.is_available():
                devices.append("cuda")
                precisions.append("float16")
                if torch.cuda.is_bf16_supported():
                    precisions.append("bfloat16")
        except Exception:
            pass
        return [
            parameter("output_model_name", "Output model name", "Name for the immutable Vedock model version.", "string", "model-output", "general", required=True),
            parameter("training_method", "Training method", "Choose adapter tuning, full tuning, continued pretraining, or a randomly initialized scratch architecture.", "string", "lora", "general", choices=["lora", "full", "continue_pretraining", "scratch"]),
            parameter("output_pattern", "Input/output pattern", "Exact serialized training example. Use {prompt} and {response}. For StoryMaker this includes its four special marker tokens.", "string", PLAIN_OUTPUT_PATTERN, "general", required=True),
            parameter("device", "Device", "Training device supported by detected hardware.", "string", "cpu", "general", choices=devices),
            parameter("precision", "Precision", "Training numeric precision.", "string", "float32", "general", choices=precisions),
            parameter("seed", "Random seed", "Seed for data order and weight updates.", "integer", 42, "general", minimum=0, maximum=2_147_483_647),
            parameter("resume_from_checkpoint", "Resume checkpoint", "Optional Vedock-owned checkpoint path.", "string", "", "general", advanced=True),
            parameter("max_examples", "Maximum examples", "Bound the processed dataset used by this job; zero uses all.", "integer", 200, "dataset", minimum=0, maximum=1_000_000),
            parameter("shuffle", "Shuffle", "Shuffle examples before training.", "boolean", True, "dataset"),
            parameter("shuffle_seed", "Shuffle seed", "Seed used only for dataset shuffling.", "integer", 42, "dataset", minimum=0, maximum=2_147_483_647, advanced=True),
            parameter("max_seq_length", "Maximum sequence length", "Token length after prompt/response formatting.", "integer", 128, "tokenization", minimum=32, maximum=1024, step=8),
            parameter("truncation", "Truncation", "Trim examples beyond maximum sequence length.", "boolean", True, "tokenization"),
            parameter("padding", "Padding", "Pad batches to their longest sequence.", "string", "longest", "tokenization", choices=["longest", "max_length"], advanced=True),
            parameter("padding_side", "Padding side", "Side used for tokenizer padding.", "string", "right", "tokenization", choices=["left", "right"], advanced=True),
            parameter("add_special_tokens", "Add special tokens", "Allow tokenizer special tokens during encoding.", "boolean", True, "tokenization", advanced=True),
            parameter("preprocessing_workers", "Preprocessing workers", "CPU workers used to tokenize examples.", "integer", 1, "tokenization", minimum=1, maximum=8, advanced=True),
            parameter("tokenization_batch_size", "Tokenization batch size", "Examples processed together by the tokenizer.", "integer", 100, "tokenization", minimum=1, maximum=10_000, advanced=True),
            parameter("num_train_epochs", "Epochs", "Complete passes through the training dataset.", "float", 1.0, "optimization", minimum=0.01, maximum=100.0, step=0.1),
            parameter("max_steps", "Maximum steps", "Positive value overrides epochs; use zero for epoch-based training.", "integer", 1, "optimization", minimum=0, maximum=1_000_000),
            parameter("per_device_train_batch_size", "Training batch size", "Examples per device per optimizer micro-step.", "integer", 1, "optimization", minimum=1, maximum=128),
            parameter("per_device_eval_batch_size", "Evaluation batch size", "Examples per device during evaluation.", "integer", 1, "optimization", minimum=1, maximum=128, advanced=True),
            parameter("gradient_accumulation_steps", "Gradient accumulation", "Micro-steps accumulated before an update.", "integer", 1, "optimization", minimum=1, maximum=1024),
            parameter("learning_rate", "Learning rate", "Size of each optimizer update.", "float", 0.0002, "optimization", minimum=0.0000001, maximum=0.1, step=0.000001),
            parameter("lr_scheduler_type", "LR scheduler", "Learning-rate schedule across training.", "string", "linear", "optimization", choices=["linear", "cosine", "constant", "constant_with_warmup", "polynomial"], advanced=True),
            parameter("warmup_steps", "Warmup steps", "Fixed number of warmup updates.", "integer", 0, "optimization", minimum=0, maximum=100_000, advanced=True),
            parameter("warmup_ratio", "Warmup ratio", "Fraction of total steps used for warmup.", "float", 0.0, "optimization", minimum=0.0, maximum=1.0, step=0.01, advanced=True),
            parameter("weight_decay", "Weight decay", "L2-style parameter regularization.", "float", 0.01, "optimization", minimum=0.0, maximum=1.0, step=0.001, advanced=True),
            parameter("optim", "Optimizer", "Transformers optimizer implementation.", "string", "adamw_torch", "optimization", choices=["adamw_torch", "adamw_hf", "adafactor", "sgd"], advanced=True),
            parameter("adam_beta1", "Adam beta 1", "First-moment decay.", "float", 0.9, "optimization", minimum=0.0, maximum=0.9999, step=0.0001, advanced=True),
            parameter("adam_beta2", "Adam beta 2", "Second-moment decay.", "float", 0.999, "optimization", minimum=0.0, maximum=0.99999, step=0.0001, advanced=True),
            parameter("adam_epsilon", "Adam epsilon", "Numerical stability constant.", "float", 0.00000001, "optimization", minimum=1e-12, maximum=0.01, advanced=True),
            parameter("max_grad_norm", "Maximum gradient norm", "Clips gradients above this norm; zero disables clipping.", "float", 1.0, "optimization", minimum=0.0, maximum=100.0, step=0.1, advanced=True),
            parameter("label_smoothing_factor", "Label smoothing", "Smooths token targets.", "float", 0.0, "optimization", minimum=0.0, maximum=0.5, step=0.01, advanced=True),
            parameter("gradient_checkpointing", "Gradient checkpointing", "Trades compute for lower activation memory.", "boolean", True, "memory", advanced=True),
            parameter("logging_strategy", "Logging strategy", "When Trainer emits metrics.", "string", "steps", "logging", choices=["no", "steps", "epoch"], advanced=True),
            parameter("logging_steps", "Logging steps", "Steps between metric events.", "integer", 1, "logging", minimum=1, maximum=100_000),
            parameter("log_level", "Log level", "Transformers logging detail.", "string", "info", "logging", choices=["debug", "info", "warning", "error", "passive"], advanced=True),
            parameter("evaluation_strategy", "Evaluation strategy", "Evaluation is disabled until this dataset version includes a validation split.", "string", "no", "evaluation", choices=["no"], advanced=True),
            parameter("eval_steps", "Evaluation steps", "Steps between evaluations.", "integer", 100, "evaluation", minimum=1, maximum=100_000, advanced=True),
            parameter("save_strategy", "Save strategy", "Checkpoint schedule inside the job output.", "string", "no", "saving", choices=["no", "steps", "epoch"], advanced=True),
            parameter("save_steps", "Save steps", "Steps between checkpoints.", "integer", 500, "saving", minimum=1, maximum=100_000, advanced=True),
            parameter("save_total_limit", "Checkpoint limit", "Old checkpoint count retained inside this new version.", "integer", 2, "saving", minimum=1, maximum=20, advanced=True),
            parameter("save_safetensors", "Save safetensors", "Use the safer safetensors format.", "boolean", True, "saving"),
            parameter("lora_r", "LoRA rank", "Adapter rank; higher values add capacity and memory use.", "integer", 4, "lora", minimum=1, maximum=256, depends_on={"training_method": "lora"}),
            parameter("lora_alpha", "LoRA alpha", "Adapter scaling value.", "integer", 8, "lora", minimum=1, maximum=1024, depends_on={"training_method": "lora"}),
            parameter("lora_dropout", "LoRA dropout", "Dropout applied within LoRA adapters.", "float", 0.05, "lora", minimum=0.0, maximum=0.9, step=0.01, depends_on={"training_method": "lora"}),
            parameter("lora_bias", "LoRA bias", "Bias parameters included in adapter training.", "string", "none", "lora", choices=["none", "all", "lora_only"], advanced=True, depends_on={"training_method": "lora"}),
            parameter("target_modules", "Target modules", "Comma-separated module names for GPT-2 adapters.", "list", ["c_attn", "c_proj"], "lora", advanced=True, depends_on={"training_method": "lora"}),
            parameter("modules_to_save", "Modules to save", "Additional trainable modules stored with the adapter.", "list", [], "lora", advanced=True, depends_on={"training_method": "lora"}),
            parameter("use_rslora", "Use RSLoRA", "Use rank-stabilized LoRA scaling.", "boolean", False, "lora", advanced=True, depends_on={"training_method": "lora"}),
            parameter("use_dora", "Use DoRA", "Use weight-decomposed LoRA where PEFT supports it.", "boolean", False, "lora", advanced=True, depends_on={"training_method": "lora"}),
        ]

    def get_dataset_schema(self) -> list[dict[str, Any]]:
        return [
            {"name": "text_completion", "required_fields": ["text"], "task": "causal_lm"},
            {"name": "prompt_response", "required_fields": ["prompt", "response"], "task": "causal_lm"},
            {"name": "instruction", "required_fields": ["instruction", "input", "output"], "task": "causal_lm"},
            {"name": "chat", "required_fields": ["messages"], "task": "causal_lm"},
        ]

    def validate_model(self, model_path: str | None) -> dict[str, Any]:
        if not model_path:
            return {"valid": False, "errors": ["Model path is required"], "warnings": []}
        reference = parse_model_reference(model_path)
        errors: list[str] = []
        warnings: list[str] = []
        if reference.kind == "huggingface":
            if current_app.config.get("OFFLINE_MODE", True):
                warnings.append("Offline mode is enabled; this repository must already exist in the Hugging Face cache.")
            else:
                warnings.append("Weights will download lazily on first inference or inside a training worker.")
            return {
                "valid": True,
                "errors": [],
                "warnings": warnings,
                "reference_type": "huggingface",
                "repository": reference.source,
                "revision": reference.revision,
            }
        if reference.kind == "scratch":
            return {
                "valid": True,
                "errors": [],
                "warnings": ["This is an architecture definition. It cannot run inference until it has been trained."],
                "reference_type": "scratch",
            }
        path = Path(reference.source)
        if not path.is_dir():
            errors.append("Model directory does not exist")
            return {"valid": False, "errors": errors, "warnings": warnings}
        config_path = path / "config.json"
        adapter_path = path / "adapter_config.json"
        if not config_path.exists() and not adapter_path.exists():
            errors.append("Neither config.json nor adapter_config.json was found")
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if config.get("model_type") != "gpt2":
                    warnings.append(f"Model type {config.get('model_type')!r} has not been verified in this MVP")
            except Exception as exc:
                errors.append(f"Invalid config.json: {exc}")
        if not any((path / name).exists() for name in ["model.safetensors", "pytorch_model.bin", "adapter_model.safetensors"]):
            errors.append("No supported model or adapter weights were found")
        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def validate_dataset(self, dataset_path: str, schema: str) -> dict[str, Any]:
        from vedock.services.datasets import validate_jsonl_file

        return validate_jsonl_file(Path(dataset_path), schema)

    def _resolve_base_and_adapter(self, model_path: str) -> tuple[str, str | None, str | None]:
        reference = parse_model_reference(model_path)
        if reference.kind == "huggingface":
            return reference.source, None, reference.revision
        if reference.kind == "scratch":
            raise ValueError("A scratch architecture must be trained before inference.")
        path = Path(reference.source)
        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            data = json.loads(adapter_config.read_text(encoding="utf-8"))
            base = data.get("base_model_name_or_path")
            if not base:
                raise ValueError("Adapter is missing base_model_name_or_path")
            return str(base), str(path), data.get("revision")
        return str(path), None, None

    def load_model(self, model_path: str, **kwargs: Any) -> tuple[Any, Any, str]:
        reference = parse_model_reference(model_path)
        normalized = model_path if reference.kind != "local" else reference.source
        with self._lock:
            precision = kwargs.get("precision", "float32")
            device = kwargs.get("device", "cpu")
            if self._loaded_path == normalized and self._model is not None and self._device == device and self._precision == precision:
                return self._tokenizer, self._model, self._device
            self.unload_model()
            validation = self.validate_model(normalized)
            if not validation["valid"]:
                raise ValueError("; ".join(validation["errors"]))

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base_path, adapter_path, revision = self._resolve_base_and_adapter(normalized)
            local_model_path = Path(reference.source) if reference.kind == "local" else None
            tokenizer_source = normalized if local_model_path and (local_model_path / "tokenizer.json").exists() else base_path
            offline = bool(current_app.config.get("OFFLINE_MODE", True))
            load_options = {"local_files_only": offline, "revision": revision, "use_fast": True}
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **load_options)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            if precision != "float32" and device != "cuda":
                raise ValueError(f"{precision} inference requires a supported CUDA device")
            dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]
            model = AutoModelForCausalLM.from_pretrained(
                base_path,
                local_files_only=offline,
                revision=revision,
                torch_dtype=dtype,
            )
            if adapter_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
            embeddings = model.get_input_embeddings()
            if embeddings is not None and len(tokenizer) > int(embeddings.num_embeddings):
                # Some locally saved tokenizers contain task markers that were
                # not reflected in the base config. Resize in memory only; source
                # model directories remain read-only.
                model.resize_token_embeddings(len(tokenizer))
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is not available")
            model.to(torch.device(device))
            model.eval()
            self._loaded_path = normalized
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._precision = precision
            return tokenizer, model, device

    def unload_model(self) -> None:
        with self._lock:
            self._tokenizer = None
            self._model = None
            self._loaded_path = None
            self._device = "cpu"
            self._precision = "float32"
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _format_prompt(prompt: str, parameters: dict[str, Any]) -> str:
        system_prompt = parameters.get("system_prompt", "").strip()
        combined = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        pattern = validate_output_pattern(parameters.get("output_pattern") or PLAIN_OUTPUT_PATTERN)
        inference_prefix = pattern.split("{response}", 1)[0]
        return inference_prefix.replace("{prompt}", combined).replace("{history}", "").replace("$sep", "\n\n")

    @staticmethod
    def _pattern_suffix(parameters: dict[str, Any]) -> str:
        pattern = validate_output_pattern(parameters.get("output_pattern") or PLAIN_OUTPUT_PATTERN)
        return pattern.split("{response}", 1)[1].replace("$sep", "\n\n")

    @staticmethod
    def _context_window(tokenizer: Any, model: Any) -> int:
        candidates = [
            getattr(model.config, "max_position_embeddings", None),
            getattr(model.config, "n_positions", None),
            getattr(model.config, "n_ctx", None),
            getattr(tokenizer, "model_max_length", None),
        ]
        usable = [int(value) for value in candidates if isinstance(value, (int, float)) and 1 < int(value) < 1_000_000]
        return min(usable) if usable else 1024

    def _encode_for_generation(self, tokenizer: Any, model: Any, formatted: str, parameters: dict[str, Any]) -> dict[str, Any]:
        context_window = self._context_window(tokenizer, model)
        requested_tokens = int(parameters["max_new_tokens"])
        if requested_tokens >= context_window:
            raise ValueError(
                f"Maximum new tokens ({requested_tokens}) must be smaller than this model's {context_window}-token context window."
            )
        if int(parameters["min_new_tokens"]) > requested_tokens:
            raise ValueError("Minimum new tokens cannot be greater than maximum new tokens.")
        maximum_input = max(1, context_window - requested_tokens)
        encoded = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=maximum_input)
        embeddings = model.get_input_embeddings()
        if embeddings is not None and encoded["input_ids"].numel():
            largest_token = int(encoded["input_ids"].max().item())
            if largest_token >= int(embeddings.num_embeddings):
                raise ValueError(
                    "The tokenizer produced a token ID outside this model's vocabulary. Re-save the model with its matching tokenizer or remix it with the correct tokenizer."
                )
        return encoded

    def _generation_kwargs(self, tokenizer: Any, parameters: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": parameters["max_new_tokens"],
            "min_new_tokens": parameters["min_new_tokens"],
            "do_sample": parameters["do_sample"],
            "repetition_penalty": parameters["repetition_penalty"],
            "no_repeat_ngram_size": parameters["no_repeat_ngram_size"],
            "num_beams": parameters["num_beams"],
            "num_beam_groups": parameters["num_beam_groups"],
            "diversity_penalty": parameters["diversity_penalty"],
            "length_penalty": parameters["length_penalty"],
            "early_stopping": parameters["early_stopping"],
            "num_return_sequences": parameters["num_return_sequences"],
            "use_cache": parameters["use_cache"],
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if parameters["do_sample"]:
            kwargs.update(
                temperature=parameters["temperature"],
                top_p=parameters["top_p"],
                top_k=parameters["top_k"],
                typical_p=parameters["typical_p"],
                min_p=parameters["min_p"],
            )
        bad_words = parameters.get("bad_words") or []
        if bad_words:
            kwargs["bad_words_ids"] = [
                tokenizer(item, add_special_tokens=False).input_ids for item in bad_words if item
            ]
        return kwargs

    def infer(self, model_path: str, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
        import torch

        normalized = validate_parameters(parameters, self.get_inference_parameter_schema())
        normalized["output_pattern"] = validate_output_pattern(normalized["output_pattern"])
        tokenizer, model, device = self.load_model(model_path, device=normalized["device"], precision=normalized["precision"])
        formatted = self._format_prompt(prompt, normalized)
        encoded = self._encode_for_generation(tokenizer, model, formatted, normalized)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        torch.manual_seed(normalized["seed"])
        if device == "cuda":
            torch.cuda.manual_seed_all(normalized["seed"])
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **self._generation_kwargs(tokenizer, normalized),
            )
        input_length = encoded["input_ids"].shape[-1]
        outputs = [tokenizer.decode(sequence[input_length:], skip_special_tokens=True) for sequence in generated]
        stops = list(normalized.get("stop_sequences") or [])
        suffix = self._pattern_suffix(normalized)
        if suffix:
            stops.append(suffix)
        for index, output in enumerate(outputs):
            cut = min([output.find(stop) for stop in stops if stop and output.find(stop) >= 0] or [len(output)])
            outputs[index] = output[:cut].strip()
        return {
            "text": outputs[0] if outputs else "",
            "sequences": outputs,
            "parameters": normalized,
            "device": device,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def stream_infer(self, model_path: str, prompt: str, parameters: dict[str, Any]) -> Iterable[str]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        request_parameters = dict(parameters)
        generation_id = str(request_parameters.pop("_generation_id", "") or uuid.uuid4())
        normalized = validate_parameters(request_parameters, self.get_inference_parameter_schema())
        normalized["output_pattern"] = validate_output_pattern(normalized["output_pattern"])
        normalized["num_return_sequences"] = 1
        tokenizer, model, device = self.load_model(model_path, device=normalized["device"], precision=normalized["precision"])
        formatted = self._format_prompt(prompt, normalized)
        encoded = self._encode_for_generation(tokenizer, model, formatted, normalized)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        stop_event = threading.Event()
        self._stop_events[generation_id] = stop_event

        class EventStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return stop_event.is_set()

        kwargs = {
            **encoded,
            **self._generation_kwargs(tokenizer, normalized),
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([EventStoppingCriteria()]),
        }
        torch.manual_seed(normalized["seed"])
        if device == "cuda":
            torch.cuda.manual_seed_all(normalized["seed"])
        generation_errors: list[BaseException] = []

        def generate() -> None:
            try:
                model.generate(**kwargs)
            except BaseException as exc:
                generation_errors.append(exc)
                try:
                    streamer.on_finalized_text("", stream_end=True)
                except Exception:
                    pass

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
        stops = [item for item in list(normalized.get("stop_sequences") or []) + [self._pattern_suffix(normalized)] if item]
        longest_stop = max((len(item) for item in stops), default=0)
        pending = ""
        try:
            for chunk in streamer:
                pending += chunk
                positions = [pending.find(stop) for stop in stops if pending.find(stop) >= 0]
                if positions:
                    cut = min(positions)
                    if cut:
                        yield pending[:cut]
                    pending = ""
                    stop_event.set()
                    break
                safe_length = len(pending) - max(0, longest_stop - 1)
                if safe_length > 0:
                    yield pending[:safe_length]
                    pending = pending[safe_length:]
            if pending and not stop_event.is_set():
                yield pending
        finally:
            stop_event.set()
            thread.join(timeout=5)
            self._stop_events.pop(generation_id, None)
        if generation_errors:
            raise RuntimeError(str(generation_errors[0]))

    def cancel(self, job_id: str) -> bool:
        event = self._stop_events.get(str(job_id))
        if not event:
            return False
        event.set()
        return True


class StoryMakerRuntime(TransformersTextRuntime):
    key = "storymaker"
    display_name = "StoryMaker adapter"

    def get_model_capabilities(self, model_path: str | None = None) -> dict[str, Any]:
        capabilities = super().get_model_capabilities(model_path)
        capabilities["runtime"] = self.key
        capabilities["legacy_profile"] = "prompt_to_story"
        return capabilities

    def get_inference_parameter_schema(self) -> list[dict[str, Any]]:
        schema = super().get_inference_parameter_schema()
        for field in schema:
            if field["name"] == "output_pattern":
                field["default"] = STORYMAKER_OUTPUT_PATTERN
        return schema

    def get_training_parameter_schema(self) -> list[dict[str, Any]]:
        schema = super().get_training_parameter_schema()
        for field in schema:
            if field["name"] == "output_pattern":
                field["default"] = STORYMAKER_OUTPUT_PATTERN
        return schema
