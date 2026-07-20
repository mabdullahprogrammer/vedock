# Existing Code Assessment

Assessment date: 2026-07-19 (Asia/Karachi)

## Executive result

The legacy StoryMaker assets are usable for a real Vedock text-generation vertical slice. The requested path `D:\LLM\StoryMaker` does not exist on this machine. The matching project was discovered at:

```text
D:\LLM\new-llm\LLM-2025\StoryMaker
```

Vedock treats the discovered directory as protected and read-only in addition to the requested path. No application artifact will be written there.

Real inference from `gpt-storygen-final` was executed successfully on CPU. The current machine does not expose a usable NVIDIA GPU to PyTorch, and the legacy environment is missing `accelerate`, so the legacy `Trainer` path cannot run until dependencies are added outside the protected project.

## Location reconciliation

| Item | Requested location | Discovered location | Status |
| --- | --- | --- | --- |
| StoryMaker root | `D:\LLM\StoryMaker` | `D:\LLM\new-llm\LLM-2025\StoryMaker` | Requested path absent; discovered path protected |
| Training script | `D:\LLM\StoryMaker\train_gptprompt2story.py` | `D:\LLM\new-llm\LLM-2025\StoryMaker\train_gptprompt2story.py` | Found |
| Fine-tuned model | `D:\LLM\StoryMaker\gpt2fintuned_storymaker` | `D:\LLM\new-llm\LLM-2025\StoryMaker\gpt2fintuned_storymaker` | Found |
| Final model | `D:\LLM\StoryMaker\gpt-storygen-final` | `D:\LLM\new-llm\LLM-2025\StoryMaker\gpt-storygen-final` | Found |
| Python environment | `D:\LLM\cuda` | `D:\LLM\cuda` | Found |

The legacy root contained 67 files totaling 5,755,180,858 bytes at the start of work. The pre-build preservation fingerprints are:

```text
metadata SHA-256:           30634041E79B6BA4C54211728531BD9A634D69922A6370175EC0A6BBB587D67E
small-file content SHA-256: B09AF488B18BC9536E2038DA2B8E02EFA995E3CB68D49F6CCF0233FC0AC93251
```

The metadata fingerprint covers every relative path, byte length, and UTC modification timestamp. The small-file content fingerprint covers every legacy file smaller than 10 MiB. These fingerprints will be recomputed after implementation.

## Legacy script behavior

The script is a single-file GPT-2 prompt-to-story trainer and inference program built on PyTorch, Hugging Face Datasets, and Transformers.

### Training flow

1. Load one CSV using `datasets.load_dataset("csv")`.
2. Require exactly the `prompt` and `story` columns.
3. Load a tokenizer from `--pretrained_model`.
4. Add four custom prompt/response marker tokens.
5. Convert each example to a fixed-length token sequence.
6. Mask prompt tokens in `labels` with `-100`.
7. Load a pretrained causal LM for fine-tuning or create a GPT-2 architecture from scratch.
8. Train through `transformers.Trainer`.
9. Save model, tokenizer, and `training_args.json` to the selected output directory.

Supported legacy training CLI arguments are:

```text
--csv
--mode finetune|scratch
--pretrained_model
--output_dir
--epochs
--per_device_train_batch_size
--lr
--max_length
--seed
--n_layer
--n_head
--n_embd
--config
```

Hard-coded `TrainingArguments` values are weight decay `0.01`, logging every 100 steps, saving every 500 steps, a three-checkpoint limit, automatic FP16 only when CUDA is available, no report destination, and no evaluation dataset.

### Inference flow

The script loads the tokenizer and model from `--output_dir`, selects CUDA when PyTorch reports it available, and otherwise uses CPU. Generation uses sampling and supports:

```text
--prompt_text
--max_length
--temperature
--top_k
--top_p
--stream
--seed
```

The runtime needs attention-mask and padding handling corrected. The verified legacy command emits warnings because the script passes only `input_ids`, uses the same token for padding and EOS, and does not pass `attention_mask` or `pad_token_id` explicitly.

### Script defects and migration cautions

- `train_text.replace(...)` is called without assigning the returned string, so custom training templates do not substitute fields.
- The generated architecture config uses `max_len`, while the parser expects `max_length`.
- The generated architecture config uses `device_count_for_train_batch`, while the parser expects `per_device_train_batch_size`.
- `build_architecture()` derives its name from a path and writes `model_architecture.json` in the current directory; Vedock must not invoke this helper in the protected tree.
- Generated config templates use hyphenated markers such as `<|start-of-input|>`, while training code adds underscore markers such as `<|start_of_input|>`.
- Inference adds four new special tokens and resizes embeddings every load, even though the saved final tokenizer did not record those tokens.
- `max_length` controls total prompt-plus-output length. Vedock should expose `max_new_tokens` and translate safely.
- Streaming uses `TextStreamer`, which writes to standard output and is not suitable for browser SSE without an iterator streamer.
- The script does not perform evaluation, checkpoint resume validation, cancellation, immutable output versioning, or collision checks.
- The script can overwrite an existing `output_dir`; Vedock must allocate a new version directory before training.

## Model assets

Both primary models identify as 12-layer GPT-2 causal language models with 768 hidden units, 12 attention heads, 1,024-token context, 50,257-token vocabulary, and Float32 weights.

| Model | Weight file | Approximate weight size | Tensor count | Load status |
| --- | --- | ---: | ---: | --- |
| `gpt2fintuned_storymaker` | `model.safetensors` | 497,774,208 bytes | 148 | Structurally inspected |
| `gpt-storygen-final` | `model.safetensors` | 497,774,208 bytes | 148 | Loaded and inferred successfully |

The two safetensor files have identical tensor names and shapes. Their model `config.json`, `generation_config.json`, `vocab.json`, and `merges.txt` match. Their `tokenizer.json` and special-token metadata differ, and the older model lacks `tokenizer_config.json`. Therefore a linear weight merge is structurally possible but must be blocked by default until the user explicitly selects a tokenizer policy and a full compatibility check succeeds.

## Verified real inference

The following prompt was executed against the protected final model using `D:\LLM\cuda\Scripts\python.exe` with offline model loading:

```text
A clockmaker discovered that midnight had stopped.
```

The model produced:

```text
A clockmaker discovered that midnight had stopped. It was midnight and he was alone in his studio, staring at the ceiling, his phone vibrating with the click of his fingers. His phone was on speaker. He stared
```

The process completed successfully in approximately 38 seconds on CPU for a 44-token total sequence. This proves model loading and generation, but it is not evidence of GPU inference or acceptable multi-user latency.

## Dataset assessment

The script-compatible datasets use CSV with `prompt` and `story` fields. Inspected examples include:

| File | Size | Columns |
| --- | ---: | --- |
| `writingprompts.csv` | 858,543,630 bytes | `prompt`, `story` |
| `writingprompts_sampled.csv` | 468,314,101 bytes | `prompt`, `story` |
| `dataset_trimmed.csv` | 31,706,134 bytes | `prompt`, `story` |
| `train.csv` | 841,439,979 bytes | `prompt`, `story` |
| `test.csv` | 17,103,665 bytes | `prompt`, `story` |
| `gkreddit_sampled_sampled.csv` | 422,264 bytes | `prompt`, `completion` |

The large CSVs contain multiline quoted text. Dataset inspection and transformation must use a real CSV parser and streaming/sample-based analysis rather than line splitting or loading an entire file into memory.

## Environment and hardware

```text
Python:               3.11.0
Environment:          D:\LLM\cuda
PyTorch:              2.3.0+cu118
Compiled CUDA:        11.8
Transformers:         4.55.0
Datasets:             4.8.5
Safetensors:          0.7.0
Pandas:               2.2.2
PyArrow:              24.0.0
CUDA available:       false
CUDA device count:    0
nvidia-smi:           not found
Physical RAM:         7.73 GiB
Free RAM at check:    0.41 GiB
Free D: storage:      46.17 GiB
```

`accelerate`, `peft`, `bitsandbytes`, Flask, SQLAlchemy, Flask-Login, and Waitress were absent. Instantiating `TrainingArguments` failed with the explicit requirement for `accelerate>=0.26.0`.

## MVP decisions from the assessment

- Register the actual legacy paths through environment configuration; never copy or modify them automatically.
- Load models lazily on the first inference call and provide explicit unload support.
- Use a Transformers runtime adapter with a StoryMaker prompt-template profile.
- Run inference and training only with local paths unless the user deliberately enables an online base model.
- Allocate all datasets, logs, recipes, jobs, outputs, and merge artifacts inside `D:\LLM\vedock\storage`.
- Add missing web/training dependencies only inside a Vedock-owned virtual environment.
- Treat LoRA and QLoRA as unavailable until `peft` and compatible hardware/package checks pass.
- Expose CPU inference now, clearly flag it as slow, and report CUDA as unavailable rather than implying acceleration.
- Execute all training in a separate worker process and reject training when dataset validation has critical errors.
