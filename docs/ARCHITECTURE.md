# Vedock Architecture

## Architectural objective

The two-day build is a modular Flask monolith with separate runtime and worker boundaries. It is intentionally small enough to demo locally while preserving the contracts needed for additional model families later.

## Process view

```text
Browser / HTMX / CLI
          |
          v
Flask application on :5464
  - authentication and ownership
  - HTML blueprints
  - versioned JSON API
  - schema validation
  - database transactions
          |
          +--> SQLite metadata
          +--> immutable file storage
          +--> lazy inference runtime
          +--> worker subprocess launcher
                         |
                         v
                 dataset / train / merge worker
```

Models are not loaded while the Flask app is created. The runtime registry stores metadata and factories only. A model is loaded on the first inference request and can be explicitly unloaded.

## Source layout

```text
vedock/
├── run.py
├── worker.py
├── pyproject.toml
├── requirements.txt
├── .env.example
├── docs/
├── instance/
├── storage/
├── tests/
├── vedock/
│   ├── __init__.py
│   ├── config.py
│   ├── extensions.py
│   ├── models.py
│   ├── auth/
│   ├── main/
│   ├── api/
│   ├── datasets/
│   ├── studio/
│   ├── jobs/
│   ├── models_ui/
│   ├── playground/
│   ├── conversations/
│   ├── merges/
│   ├── system/
│   ├── runtimes/
│   ├── services/
│   ├── templates/
│   └── static/
└── vedock_cli/
```

## Configuration and branding

Environment variables are loaded into one configuration object. Required defaults are:

```env
APP_NAME=Vedock
APP_SHORT_NAME=Vedock
CLI_NAME=vedock
APP_TAGLINE=Build any AI. No code. Full control.
APP_HOST=0.0.0.0
APP_PORT=5464
```

Templates receive a branding object from an application context processor. API metadata, generated model cards, CLI help, error messages, and documentation helpers read from the same settings. No visible product name is used as an authorization or database key.

Additional path settings point to the Vedock storage root, legacy runtime Python, and discovered legacy model directories. Path validation forbids any output destination beneath a protected StoryMaker root.

## Persistence

SQLAlchemy models target SQLite for the MVP and avoid SQLite-specific query APIs so a PostgreSQL URL can replace it later.

Core entities are:

- `User` and API tokens;
- `RawDataset`, `DatasetVersion`, and `DatasetTransformation`;
- `ModelProject`, `ModelRecord`, and immutable `ModelVersion`;
- `TrainingRecipe` and `Job`;
- `Conversation` and `Message`;
- `MergeRecord`.

Database rows store metadata and ownership. Dataset bytes, model weights, logs, reports, and exports live in the file store. Paths in the database are normalized absolute paths under configured allowed roots.

## Authentication

Browser authentication uses Flask-Login, Werkzeug password hashing, CSRF tokens, secure session cookies, and server-side ownership checks. The landing page is public; dashboard, datasets, models, jobs, conversations, settings, and mutation endpoints require login.

The CLI exchanges username/password for a revocable random API token, stores it in the user profile configuration with restricted local permissions where Windows permits, and sends it as a Bearer token. API authorization is checked server-side for every resource.

## Runtime interface

All model families implement the same abstract contract:

```text
get_model_capabilities
get_training_parameter_schema
get_inference_parameter_schema
get_dataset_schema
validate_model
validate_dataset
load_model
unload_model
infer
stream_infer
prepare_training
train
cancel
evaluate
save
export
```

Runtimes also expose `get_runner_schema(model_path)` and typed `run(model_path, inputs, parameters)`. This separates interaction design from model identity: chat, sequence completion, image upload, numeric prediction, category prediction, time series, embeddings, galleries, and hybrid output can each declare the fields and presentation they actually need. The canonical cross-model endpoint is `POST /api/v1/models/<model>/run`; legacy text/image endpoints remain compatible.

`TransformersTextRuntime` implements GPT-2 causal generation. `StoryMakerRuntime` supplies the legacy prompt/response template and registration defaults. `PatternSequenceRuntime` provides non-chat sequence prediction, `SklearnImageClassificationRuntime` provides image-upload classification, and `TabularPredictionRuntime` provides numeric regression and category classification from prepared feature columns. Future generation, captioning, embedding, forecasting, and hybrid runtimes can register their typed contracts without changing the universal runner.

## Parameter schemas

Every field declares name, label, description, type, default, bounds or choices, step, required flag, basic/advanced group, runtime compatibility, warnings, validation constraints, and dependencies. The same schema drives:

1. HTML form rendering;
2. CLI option discovery and documentation;
3. JSON request validation;
4. normalized recipe storage;
5. worker configuration.

Hardware/package filters remove unsupported choices. For example, BF16, FP16, CUDA, QLoRA, and bitsandbytes settings are hidden or disabled when their capability checks fail.

## Web surfaces

The navigation contains Landing, Dashboard, Create Model, Datasets, Training Jobs, Models, Playground, Conversations, Merge Models, Developer / CLI, Settings, and System. Available objectives are selectable and unavailable adapters are explicitly labeled rather than presented as working.

HTMX handles compact form/result updates. Alpine.js provides local form toggles and ordered transformation editing. Server-Sent Events stream generation and job-log updates where supported, with ordinary polling as fallback. The MVP ships local CSS rather than requiring a production network dependency.

## API

The JSON API is rooted at `/api/v1` and returns a consistent envelope with request-safe error codes. Important routes cover authentication, system/doctor, model list/details/inference/capabilities, conversations, datasets/import/inspection/preview/version/validation, recipes, training jobs/logs/cancel, versions, merges/compatibility/execute, and export.

HTML and CLI both call the same service layer; critical validation is not implemented only in the browser.

## Jobs

HTTP handlers create a job row and launch a Vedock-owned worker subprocess, then return immediately. The worker claims the queued job, writes line-oriented logs, persists progress and terminal status, and allocates immutable output directories. Cancellation sets a database flag; workers check it at safe boundaries and terminate training cooperatively.

Only one training job runs by default on this low-memory machine. A file/database lock prevents duplicate claims. On app restart, jobs left in active states are marked interrupted and may be retried explicitly.

## Model versions and export

Every training or merge produces a new model-version directory using a generated identifier. A version becomes visible as completed only after artifacts, metadata, hashes, and validation are written successfully. Existing completed directories are never reused.

Exports are generated into a separate Vedock export directory. Legacy source model directories remain references and are never altered or bundled in place.

## Merge safety

Compatibility inspection reads model configs, tokenizer metadata, and safetensor tensor names/shapes before any weights. A merge is blocked unless architecture, names, shapes, vocabulary, tokenizer policy, precision, RAM, disk, and known license constraints pass.

The MVP supports compatibility reports unconditionally. Linear Float32 safetensor merge is capability-gated by memory and tokenizer compatibility. LoRA adapter merging appears only when PEFT adapters with the same base model and compatible target modules are detected.

## Security boundaries

- Protected StoryMaker roots are read-only and excluded from all output allocators.
- URL imports block server-side request forgery targets and enforce streaming size/time limits.
- Upload names never determine storage directories.
- ZIP support, when enabled, rejects absolute paths, traversal, symlinks, excessive files, and expansion bombs.
- JSON request sizes, prompt sizes, generation limits, regular expressions, and job parameters are bounded.
- Secrets and password hashes never enter logs or model cards.
- `trust_remote_code` defaults to false and requires an explicit warning acknowledgement.
