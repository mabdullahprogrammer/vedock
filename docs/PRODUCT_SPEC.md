# Vedock — Two-Day MVP Product Specification

## Product name

```text
Vedock
```

## CLI command

```text
vedock
```

## Tagline

```text
Build any AI. No code. Full control.
```

## Core idea

Vedock is a no-code AI development environment where users can:

* Import data
* Clean and structure data
* Convert data into training-ready formats
* Select or import a base model
* Configure every supported training parameter
* Train or fine-tune models
* Test models
* Compare model versions
* Run models through the web
* Run models through the CLI
* Merge compatible models
* Export completed models

The primary objective is not community or social features.

The primary objective is:

> Give non-programmers access to the same important controls that AI developers normally use in code.

The application must simplify AI development without removing technical control.

---

# 1. Strict Two-Day Scope

Only two days remain.

The first version must be a minimal but real working product.

Do not spend time on:

* Followers
* Comments
* Likes
* Stars
* Public activity feeds
* Social notifications
* Complex organization management
* Public discussions
* Model ranking
* Billing
* Subscription plans
* Kubernetes
* Multi-cloud deployment
* Large-scale distributed training
* Elaborate model marketplaces

The two-day project must focus on:

1. Application structure
2. User login
3. Dataset import
4. Dataset cleaning
5. Dataset formatting
6. LLM project creation
7. Editable training parameters
8. Fine-tuning
9. Inference
10. Saved models
11. Model version records
12. CLI access
13. Minimal model merging
14. Architecture for future image models

The first model type to make fully functional is:

```text
Text generation / LLM
```

Only after the LLM flow works should image functionality be added.

---

# 2. Protected Existing Project

The existing project is located at:

```text
D:\LLM\StoryMaker
```

The principal working file is:

```text
D:\LLM\StoryMaker\train_gptprompt2story.py
```

Existing models:

```text
D:\LLM\StoryMaker\gpt2fintuned_storymaker
D:\LLM\StoryMaker\gpt-storygen-final
```

Existing environment:

```text
D:\LLM\cuda
```

All new development must be outside `StoryMaker`.

Recommended new application location:

```text
D:\LLM\vedock
```

Do not modify, move, rename, reformat, or generate files inside:

```text
D:\LLM\StoryMaker
```

Use the existing project only as:

* A read-only implementation reference
* A source of working models
* A source of reusable training logic
* A source of reusable inference logic

---

# 3. Branding Configuration

The application name and CLI name must be environment-controlled.

Use:

```env
APP_NAME=Vedock
APP_SHORT_NAME=Vedock
CLI_NAME=vedock
APP_TAGLINE=Build any AI. No code. Full control.
APP_HOST=0.0.0.0
APP_PORT=5464
```

The visible application name must update throughout the interface when `APP_NAME` changes.

This includes:

* Browser title
* Header
* Sidebar
* Landing page
* Login pages
* Dashboard
* Studio
* Playground
* Footer
* Generated documentation
* Generated model cards
* CLI documentation
* API metadata
* Toasts and error messages

---

# 4. Fundamental Product Principle

Vedock must provide:

```text
No code
+
Full functionality
+
Editable configuration
```

“No code” must not mean that Vedock hides all model settings.

The system should provide:

## Simple mode

For beginners:

* Fast
* Balanced
* High Quality
* Low Memory
* Custom

## Advanced mode

For developers and advanced users:

* Every training parameter supported by the selected runtime
* Every inference parameter supported by the selected model
* Dataset transformation configuration
* Model architecture options where safely supported
* Adapter configuration
* Checkpoint behavior
* Evaluation configuration
* Export configuration

The interface should explain advanced settings using labels, descriptions, warnings, and recommended ranges.

---

# 5. Universal Parameter Editing

Every model runtime must declare which parameters it supports.

The frontend must dynamically build configuration forms from a parameter schema.

Do not hardcode one static form for every model type.

Each parameter definition should include:

* Internal name
* Display name
* Description
* Data type
* Default value
* Minimum value
* Maximum value
* Step
* Required status
* Supported values
* Advanced or basic classification
* Runtime compatibility
* Hardware warning
* Validation rules
* Dependency rules

Example schema:

```json
{
  "name": "learning_rate",
  "label": "Learning Rate",
  "description": "Controls the size of each training update.",
  "type": "float",
  "default": 0.0002,
  "minimum": 0.0000001,
  "maximum": 0.1,
  "step": 0.000001,
  "group": "optimization",
  "advanced": false
}
```

The backend remains the source of truth.

All parameters must be validated again on the server.

---

# 6. Editable LLM Training Parameters

For supported LLM fine-tuning workflows, expose as many applicable parameters as the runtime supports.

## General

* Project name
* Output model name
* Base model
* Base model revision
* Task type
* Training method
* Device
* Precision
* Random seed
* Trust remote code
* Resume from checkpoint

## Dataset

* Dataset source
* Training split
* Validation split
* Test split
* Text field
* Prompt field
* Input field
* Output field
* Conversation field
* Role field
* Content field
* Formatting template
* Maximum examples
* Shuffle
* Shuffle seed
* Remove duplicates
* Remove empty records
* Minimum text length
* Maximum text length
* Dataset filtering rules

## Tokenization

* Tokenizer
* Maximum sequence length
* Truncation
* Padding
* Padding side
* Add special tokens
* Beginning-of-sequence token
* End-of-sequence token
* Unknown token
* Packing
* Group by length
* Number of preprocessing workers
* Tokenization batch size

## Core training

* Number of epochs
* Maximum steps
* Per-device training batch size
* Per-device evaluation batch size
* Gradient accumulation steps
* Learning rate
* Learning-rate scheduler
* Warmup steps
* Warmup ratio
* Weight decay
* Optimizer
* Adam beta 1
* Adam beta 2
* Adam epsilon
* Maximum gradient norm
* Label smoothing
* Gradient checkpointing
* Mixed precision
* BF16
* FP16
* TF32 where supported

## Logging

* Logging strategy
* Logging steps
* Log level
* Report destination
* Loss display interval

## Evaluation

* Evaluation strategy
* Evaluation steps
* Evaluation dataset
* Metrics
* Generate evaluation samples
* Number of evaluation samples
* Early stopping
* Early-stopping patience
* Early-stopping threshold

## Saving

* Save strategy
* Save steps
* Save total limit
* Save safetensors
* Load best model at end
* Metric for best model
* Greater-is-better setting

## LoRA

* LoRA rank
* LoRA alpha
* LoRA dropout
* LoRA bias
* Target modules
* Modules to save
* Task type
* Use RSLoRA where supported
* Use DoRA where supported

## Quantization and QLoRA

* Enable quantization
* Load in 4-bit
* Load in 8-bit
* Quantization type
* Compute dtype
* Double quantization
* CPU offloading
* Device map

Only display options supported by installed packages and hardware.

---

# 7. Editable Inference Parameters

Users must be able to infer models through the browser and CLI.

Expose supported controls such as:

* Prompt
* System prompt
* Chat template
* Temperature
* Top-p
* Top-k
* Typical-p
* Minimum-p
* Maximum new tokens
* Minimum new tokens
* Maximum total length
* Repetition penalty
* Encoder repetition penalty
* Frequency penalty
* Presence penalty
* No-repeat n-gram size
* Number of beams
* Beam groups
* Diversity penalty
* Length penalty
* Early stopping
* Sampling enabled
* Number of returned sequences
* Seed
* Stop sequences
* Bad words
* Forced beginning token
* Forced ending token
* Use cache
* Streaming
* Device
* Precision

The UI must not show parameters that the selected model runtime does not support.

---

# 8. Dataset Management Is a Core Feature

Raw datasets cannot always be sent directly into a training script.

Vedock must include a proper dataset preparation environment.

A user must be able to:

1. Import raw data.
2. Preview its structure.
3. Select relevant columns.
4. Rename columns.
5. Map fields.
6. Remove invalid records.
7. Clean text.
8. Convert the dataset into a supported training schema.
9. Validate the final structure.
10. Save the processed dataset.
11. Reuse the processed dataset in training jobs.

The system must maintain a distinction between:

```text
Raw dataset
Processed dataset
Training-ready dataset version
```

Never overwrite the raw dataset.

---

# 9. Dataset Sources

Users must be able to import datasets from:

## Local upload

* CSV
* JSON
* JSONL
* TXT
* ZIP containing supported files
* Parquet where installed
* Images in directories or archives
* Caption files
* Tabular metadata

## Online URL

The user can enter a direct URL.

The system should:

1. Validate the URL.
2. Restrict unsafe network targets.
3. Download using streaming.
4. Enforce size limits.
5. Detect the file type.
6. Save it as a raw dataset.
7. Calculate a hash.
8. Preserve source information.
9. Continue into the dataset inspection process.

Possible sources:

* Direct file URL
* Public JSON endpoint
* Public CSV URL
* Supported model or dataset repository URL
* Public archive URL

For the two-day MVP, prioritize direct downloadable URLs.

Do not attempt to support every cloud-storage provider immediately.

---

# 10. Dataset Storage Structure

Use a structured application storage system.

Recommended layout:

```text
D:\LLM\vedock\storage\datasets\
├── raw\
│   └── {user_id}\
│       └── {dataset_id}\
│           ├── source_file
│           └── source_metadata.json
│
├── processed\
│   └── {user_id}\
│       └── {dataset_id}\
│           └── {version_id}\
│               ├── data.jsonl
│               ├── schema.json
│               ├── transformation.json
│               ├── statistics.json
│               └── validation.json
│
└── temporary\
```

Never place application datasets inside `StoryMaker`.

Every processed dataset version should be immutable.

When a user changes the cleaning or mapping configuration, create a new processed version.

---

# 11. Dataset Database Records

Create separate entities.

## RawDataset

Fields:

* ID
* Owner
* Name
* Description
* Source type
* Source URL
* Original filename
* Storage path
* File format
* MIME type
* Size
* Hash
* Upload time
* Inspection status
* Detected schema
* Row count

## DatasetVersion

Fields:

* ID
* Raw dataset ID
* Owner
* Version number
* Output format
* Storage path
* Transformation configuration
* Field mapping
* Validation status
* Row count
* Invalid-row count
* Token estimate
* Hash
* Created time

## DatasetTransformation

Fields:

* ID
* Dataset version ID
* Operation order
* Operation type
* Configuration
* Result summary

---

# 12. Dataset Inspection

After import, inspect and display:

* File type
* Encoding
* File size
* Row count
* Column names
* Detected data types
* Null count
* Empty values
* Duplicate count
* Minimum text length
* Maximum text length
* Average text length
* Sample rows
* Possible target fields
* Possible prompt fields
* Possible label fields
* Possible image paths
* Possible caption fields

For large datasets, inspect samples without loading the entire file into memory.

---

# 13. Visual Dataset Builder

Create a no-code dataset preparation interface.

The interface should allow the user to configure ordered transformation steps.
Possible operations:

* Select columns
* Rename columns
* Remove columns
* Join columns
* Split a column
* Add constant field
* Trim whitespace
* Normalize Unicode
* Remove HTML
* Remove URLs
* Remove control characters
* Replace text
* Apply regular expression replacement
* Convert to lowercase
* Remove empty records
* Remove duplicates
* Filter by minimum length
* Filter by maximum length
* Filter by numeric value
* Map labels
* Convert data types
* Shuffle
* Limit examples
* Create training-validation split
* Create prompt templates
* Create chat-message structures
* Add beginning/end tokens where appropriate
* Link image files to captions

Each transformation must be:

* Previewable
* Reversible before saving
* Recorded in the final dataset version
* Reproducible

---

# 14. Training-Ready Dataset Formats

Vedock must convert datasets into model-specific formats.

## Text completion

```json
{
  "text": "Complete training example text."
}
```

## Prompt and response

```json
{
  "prompt": "Write a story about the moon.",
  "response": "The moon watched silently..."
}
```

## Instruction format

```json
{
  "instruction": "Write a short mystery.",
  "input": "It takes place in an abandoned station.",
  "output": "The final train arrived..."
}
```

## Chat format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful story-writing assistant."
    },
    {
      "role": "user",
      "content": "Write a story about a lost robot."
    },
    {
      "role": "assistant",
      "content": "The robot woke beneath a silent tower..."
    }
  ]
}
```

## Classification

```json
{
  "text": "Example content",
  "label": "category_name"
}
```

## Image captioning

```json
{
  "image": "images/example-001.png",
  "caption": "A small rabbit sitting in green grass."
}
```

## Image generation

```json
{
  "image": "images/example-001.png",
  "prompt": "A futuristic city at sunset."
}
```

The runtime adapter must specify its accepted dataset schemas.

---

# 15. Dataset Validation

Before training, validate:

* Required fields
* File integrity
* Valid JSON
* Consistent schema
* Empty records
* Null records
* Duplicate records
* Invalid image paths
* Unsupported image formats
* Invalid labels
* Excessively long sequences
* Insufficient examples
* Tokenization failures
* Missing train split
* Missing validation split where required
* Encoding problems
* Invalid chat roles
* Missing prompt-response pairs

Display:

* Errors
* Warnings
* Recommended fixes
* Rows affected
* Downloadable invalid-row report

Training must not begin if critical validation errors remain.

---

# 16. Dataset Transformation Worker

Dataset processing can be expensive.

Create a background dataset-processing job.

States:

```text
queued
downloading
inspecting
transforming
validating
saving
completed
failed
cancelled
```

Do not process large datasets inside ordinary Flask requests.

For very small datasets, a development synchronous mode is acceptable.

---

# 17. Model Creation Studio

The model creation interface must be based on structured stages.

## Stage 1: Model task

Choose:

* Text generation
* Chat model
* Story generation
* Classification
* Embeddings
* Image generation
* Image captioning
* Advanced custom model

For the MVP, only text generation and story generation must work completely.

Other tasks may appear as:

```text
Coming next
```

Do not present incomplete tasks as working.

## Stage 2: Base model

Select:

* Existing StoryMaker model
* Existing Vedock model
* Local model directory
* Supported online model
* Blank supported small architecture
* Existing checkpoint

## Stage 3: Dataset

Select:

* Existing processed dataset
* Import local dataset
* Import from URL
* Build a dataset manually
* Continue as inference-only model

## Stage 4: Data preparation

Open the dataset builder.

The user must finish validation before training.

## Stage 5: Training method

Choose:

* Full fine-tuning
* LoRA
* QLoRA where supported
* Continue pretraining
* Train from scratch for a supported small architecture

For the MVP, prioritize LoRA and the existing StoryMaker-compatible method.

## Stage 6: Parameters

Provide:

* Simple preset mode
* Full advanced parameter mode

All runtime-supported parameters must remain editable.

## Stage 7: Hardware review

Show:

* GPU
* CUDA
* VRAM
* RAM
* Disk
* Estimated memory requirement
* Estimated output size
* Compatibility warnings

## Stage 8: Review and train

Show the complete generated configuration before execution.

Allow the user to save the configuration as a reusable recipe.

---

# 18. Training Recipes

Users should be able to save model-development configurations.

A recipe should include:

* Task type
* Base model
* Dataset schema
* Dataset transformation steps
* Training method
* Hyperparameters
* Tokenization settings
* Evaluation settings
* Saving settings
* Export settings

Recipes allow users to repeat a model-development workflow without coding.

---

# 19. Model Runtime Interface

Create a runtime plugin architecture.

Each runtime must expose:

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

Initial runtime:

```text
Transformers text generation
```

Legacy runtime adapter:

```text
StoryMaker adapter
```

Future runtimes:

```text
Diffusers image generation
Vision-language image captioning
Transformers classification
Sentence-transformers embeddings
```

---

# 20. Multimodal Architecture

Vedock is not limited to LLMs.

The architecture must support multiple AI task families.

However, only implement tasks sequentially.

## Phase 1

```text
Text generation
Story generation
```

## Phase 2

```text
Image generation
```

## Phase 3

```text
Image captioning
```

## Phase 4

```text
Classification
Embeddings
```

Do not attempt to complete all model types simultaneously.

---

# 21. Image Dataset Preparation

Future image workflows must support:

* Image upload
* ZIP image dataset
* Image folder scan
* Image-caption pairing
* CSV metadata linking
* JSON metadata linking
* Missing-image detection
* Duplicate-image detection
* Invalid-image detection
* Image dimension analysis
* Resize configuration
* Crop configuration
* Aspect-ratio filtering
* Format conversion
* Caption cleaning
* Caption generation assistance
* Train-validation split

Store processed images outside raw image storage.

Never overwrite original images.

---

# 22. Model Combining

Developers should be able to combine two or more compatible models.

This feature should be minimal in the two-day build.

## MVP combining support

Prioritize:

* Merging compatible LoRA adapters
* Weighted adapter combination
* Linear weight merge only when architectures and tensor shapes match

Possible later methods:

* SLERP
* TIES
* DARE
* Task arithmetic

## User workflow

```text
Select two or more models
→ verify compatibility
→ select merge method
→ assign weights
→ review memory and storage requirements
→ execute merge
→ save as a new model
→ test result
```

Example weights:

```text
Model A: 0.7
Model B: 0.3
```

## Compatibility validation

Before merging, validate:

* Same architecture
* Same tensor names
* Compatible tensor shapes
* Compatible tokenizer
* Compatible vocabulary size
* Compatible precision
* Compatible model configuration
* Sufficient RAM
* Sufficient storage
* Compatible licenses where known

If models cannot be safely merged, block the operation and explain why.

Do not attempt to “join weights” blindly.

## Merge metadata

Save:

* Source models
* Source versions
* Merge method
* Merge weights
* Configuration
* Compatibility report
* Output hash
* Output path
* Creation time

Merged models become normal Vedock model versions and can be inferred like other models.

---

# 23. Inference for All Supported Models

Every completed model should have a task-specific inference page.

## Text

* Chat
* Prompt completion
* Story generation
* Streaming
* Parameter controls
* Saved conversations

## Image generation

* Prompt
* Negative prompt
* Seed
* Width
* Height
* Steps
* Guidance
* Scheduler
* Image count
* Gallery

## Image captioning

* Image upload
* Caption prompt
* Maximum output tokens
* Alternative captions
* Saved output

## Classification

* Text or image input
* Label probabilities
* Top results
* Raw output

Inference controls must be generated from each runtime’s capability schema.

---

# 24. Minimal Interface Pages

The MVP needs only these primary pages:

```text
Landing
Login
Register
Dashboard
Create Model
Datasets
Dataset Builder
Training Jobs
Models
Model Details
Playground
Conversations
Merge Models
Developer / CLI
Settings
System
```

Do not build unnecessary social pages.

---

# 25. Flask Server

Required:

```text
Python
Flask
Port 5464
```

Run using:

```bash
python run.py
```

Configuration:

```env
APP_HOST=0.0.0.0
APP_PORT=5464
```

Use Flask application-factory architecture.

Use blueprints.

Do not put the entire server in one file.

Recommended UI stack:

* Flask
* Jinja
* HTMX
* Alpine.js
* Tailwind CSS
* Server-Sent Events

Recommended storage:

* SQLite during the two-day MVP
* File-based model and dataset artifacts
* PostgreSQL-ready SQLAlchemy architecture

Recommended jobs:

* RQ and Redis if operational on Windows
* Otherwise a controlled local worker process
* Temporary synchronous mode only for small testing jobs

---

# 26. CLI

The CLI is:

```bash
vedock
```

Required MVP commands:

```bash
vedock doctor
vedock login
vedock whoami

vedock models list
vedock models info MODEL
vedock models use MODEL
vedock chat MODEL

vedock datasets list
vedock datasets inspect PATH_OR_URL
vedock datasets validate DATASET
vedock datasets transform DATASET

vedock train MODEL --dataset DATASET
vedock jobs list
vedock jobs show JOB_ID
vedock jobs logs JOB_ID
vedock jobs cancel JOB_ID

vedock merge MODEL_A MODEL_B
vedock versions list MODEL
vedock export MODEL
```

Example:

```bash
vedock models use storymaker-final \
  --prompt "Write a story about a machine that learns to dream." \
  --temperature 0.9 \
  --top-p 0.95 \
  --max-new-tokens 300
```

Training example:

```bash
vedock train storymaker-final \
  --dataset my-clean-story-dataset \
  --method lora \
  --epochs 2 \
  --learning-rate 0.0002 \
  --batch-size 2
```

The CLI should communicate with:

```text
http://127.0.0.1:5464/api/v1
```

---

# 27. Two-Day Build Order

## Step 1: Existing code inspection

Inspect:

```text
D:\LLM\StoryMaker\train_gptprompt2story.py
D:\LLM\StoryMaker\gpt2fintuned_storymaker
D:\LLM\StoryMaker\gpt-storygen-final
D:\LLM\cuda
```

Confirm:

* Model loading
* Inference
* Training
* Dataset format
* Supported parameters
* CUDA status
* Dependencies

## Step 2: Vedock foundation

Build:

* Flask application
* Configuration
* Branding
* Database
* Login
* Main layout
* Model registry
* Hardware detection
* Error handling

## Step 3: Existing LLM inference

Build:

* StoryMaker adapter
* Register existing models
* Playground
* Text generation
* Editable inference settings
* Saved conversations

Verify real output.

## Step 4: Dataset system

Build:

* Local upload
* URL import
* Raw dataset records
* Inspection
* Field mapping
* Basic cleaning
* Conversion to JSONL
* Validation
* Processed dataset versions

## Step 5: Fine-tuning

Build:

* Create model project
* Select model
* Select processed dataset
* Editable parameters
* Training job
* Logs
* Output model version
* Test new model

## Step 6: CLI

Build:

* Doctor
* Login
* List models
* Use model
* List datasets
* Launch training
* View job logs

## Step 7: Minimal model merge

Build:

* Compatibility check
* LoRA merge if available
* Linear merge for compatible model weights if safely implementable
* Save result
* Infer result

## Step 8: UI polish and demo

Build:

* Responsive layout
* Loading states
* Error states
* Demo data
* Documentation
* Three-minute demo flow

## Step 9: Image architecture only

After all previous steps work:

* Add runtime interfaces
* Add dataset schema definitions
* Add disabled or experimental image project type

Only implement real image generation if sufficient time remains.

---

# 28. Definition of MVP Completion

The MVP is complete when a judge can:

1. Open Vedock on port 5464.
2. Register or log in.
3. View detected CUDA hardware.
4. Open the StoryMaker Final model.
5. Change inference parameters.
6. Generate a real story.
7. Save and reopen the conversation.
8. Import a dataset from a local file.
9. Import a dataset from a direct URL.
10. Inspect dataset columns.
11. Clean the dataset.
12. Map fields.
13. Convert it into a training-ready format.
14. Save the processed dataset.
15. Create a model project.
16. Select a base model.
17. Select the processed dataset.
18. Edit training parameters.
19. Start a real fine-tuning job.
20. View real training logs.
21. Receive a saved model version.
22. Run the new model.
23. Use it through `vedock`.
24. Attempt a compatible model merge.
25. Receive a clear compatibility result.
26. Confirm that `StoryMaker` was not modified.

---

# 29. Codex Immediate Instruction

Save this file as:

```text
D:\LLM\vedock\docs\PRODUCT_SPEC.md
```

Then give Codex:

```text
Read D:\LLM\vedock\docs\PRODUCT_SPEC.md completely.

The protected legacy project is located at:
D:\LLM\StoryMaker

Do not modify any file or directory inside StoryMaker.

The main working legacy file is:
D:\LLM\StoryMaker\train_gptprompt2story.py

The existing model directories are:
D:\LLM\StoryMaker\gpt2fintuned_storymaker
D:\LLM\StoryMaker\gpt-storygen-final

The existing Python and CUDA environment is:
D:\LLM\cuda

All new files must be created under:
D:\LLM\vedock

The application is named Vedock.
The CLI command is vedock.
Both must be controlled through environment settings.

Only two days remain.

Do not build community features.

Begin by inspecting the legacy script, model files, environment, dependencies, dataset structure, supported training arguments, supported inference arguments, and CUDA availability.

Then create:
docs/EXISTING_CODE_ASSESSMENT.md
docs/DATASET_PIPELINE.md
docs/ARCHITECTURE.md
docs/TWO_DAY_IMPLEMENTATION_PLAN.md
docs/DEPENDENCY_REPORT.md
docs/RISK_REGISTER.md

After assessment, build the smallest complete flow in this order:

1. Flask foundation
2. StoryMaker model inference
3. Editable inference parameters
4. Saved conversations
5. Dataset local upload
6. Dataset URL import
7. Dataset inspection
8. Dataset cleaning and field mapping
9. Training-ready JSONL conversion
10. Dataset validation
11. Editable training configuration
12. Real fine-tuning job
13. Saved output version
14. CLI inference
15. Minimal compatible model merging

Do not claim that a feature works unless it has been executed and verified.

Do not load models during Flask startup.

Do not run training inside an HTTP request handler.

Do not overwrite raw datasets.

Do not overwrite completed model versions.

Do not blindly merge incompatible model weights.

Do not begin image functionality until the full LLM vertical slice works.
```
