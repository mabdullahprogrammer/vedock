# Dependency Report

Assessment date: 2026-07-19

## Existing runtime

The existing Python environment is `D:\LLM\cuda` with Python 3.11.0.

| Package | Detected version | MVP use | Status |
| --- | --- | --- | --- |
| PyTorch | 2.3.0+cu118 | model load, generation, training | Installed; CUDA unavailable at runtime |
| Transformers | 4.55.0 | GPT-2 runtime, generation, Trainer | Installed |
| Datasets | 4.8.5 | legacy CSV loading | Installed |
| Safetensors | 0.7.0 | safe model inspection/load/merge | Installed |
| Pandas | 2.2.2 | optional tabular helpers | Installed |
| PyArrow | 24.0.0 | Parquet capability | Installed |
| Requests | 2.32.3 | HTTP client and URL import | Installed |
| Click | 8.4.1 | CLI | Installed |
| psutil | present | hardware/process inspection | Installed |
| accelerate | absent | Transformers Trainer device setup | Blocking training |
| PEFT | absent | LoRA training/adapter merge | Blocking LoRA |
| bitsandbytes | absent | 4-bit/8-bit QLoRA | Unsupported on current machine |
| Flask | absent | web application | Required |
| Flask-SQLAlchemy | absent | database integration | Required |
| SQLAlchemy | absent | persistence | Required |
| Flask-Login | absent | browser authentication | Required |
| Waitress | absent | optional Windows serving | Recommended |

Attempting to construct `TrainingArguments` produced:

```text
ImportError: Using the Trainer with PyTorch requires accelerate>=0.26.0
```

## Hardware result

PyTorch was compiled for CUDA 11.8, but `torch.cuda.is_available()` returned false, the CUDA device count was zero, and `nvidia-smi` was unavailable. Therefore CUDA, FP16-on-GPU, BF16-on-GPU, bitsandbytes quantization, and QLoRA must not be offered as working choices.

The host had approximately 7.73 GiB physical RAM and only 0.41 GiB free during assessment. CPU model loading succeeded through virtual memory but is slow and makes full-model fine-tuning high risk.

## Vedock-owned dependency strategy

The protected StoryMaker directory and existing `D:\LLM\cuda` environment will not be modified. Vedock creates its own environment under:

```text
D:\LLM\vedock\.venv
```

The Vedock environment may reference the read-only legacy site-packages path for the large existing PyTorch/Transformers stack while installing web and training additions only beneath `D:\LLM\vedock`. This avoids duplicating the 2+ GiB CUDA stack and respects the new-file boundary.

Required direct dependencies for the MVP are pinned by compatible ranges in `pyproject.toml` and a lock-style installed-version report after setup:

```text
Flask
Flask-SQLAlchemy
Flask-Login
SQLAlchemy
python-dotenv
requests
click
waitress
accelerate
peft
```

PyTorch, Transformers, Datasets, Safetensors, Pandas, PyArrow, and psutil are supplied by the inspected runtime unless a clean installation is deliberately requested later.

## Capability gates

- `transformers_text`: PyTorch + Transformers + Safetensors import and model validates.
- `training_full`: Transformers + Accelerate import and sufficient memory/disk estimate.
- `training_lora`: `training_full` + PEFT + supported target modules.
- `training_qlora`: LoRA + CUDA + supported bitsandbytes; currently false.
- `parquet_dataset`: PyArrow import; currently true.
- `streaming`: Transformers iterator streamer support.
- `linear_merge`: Safetensors + exact tensor/config/tokenizer policy match + memory/disk budget.

The backend capability report is the source of truth. Unsupported fields are omitted or disabled in generated forms and rejected server-side if submitted manually.

## Reproducibility outputs

Vedock will keep:

- `.env.example` without secrets;
- `pyproject.toml` and `requirements.txt` for direct dependencies;
- a doctor command showing executable and package versions;
- job metadata recording relevant package and hardware versions;
- model cards recording the runtime adapter and normalized parameters.
