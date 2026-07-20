# Functional Repair Report

Date: 2026-07-19

## Reported problems and resolutions

### Model creation was limited to existing StoryMaker models

Resolved. The Model Studio now supports:

- Existing Vedock and StoryMaker models
- A built-in catalog containing DistilGPT-2, GPT-2 Small, GPT-2 Medium, GPT-2 Large, and GPT-2 XL
- Arbitrary Hugging Face repository identifiers and revisions
- Read-only local model directories
- Read-only checkpoint directories
- New GPT-2 architectures with Tiny, Small, Medium, or fully custom layer/head/embedding/context settings

Online weights are registered without loading and are downloaded lazily only when inference or training is explicitly started. Local sources are validated and never overwritten.

### Model creation meant only immediate LLM fine-tuning

Resolved for the installed Transformers text runtime. The Studio now separates task, workflow, source, dataset, and runtime parameters. Its working workflows are:

- Import or run a model without training
- LoRA fine-tuning
- Full fine-tuning
- Continued pretraining
- Randomly initialized scratch GPT-2 training
- Safe linear full-weight merging
- Safe weighted LoRA-adapter merging

Creating a configuration saves a draft `ModelProject` and optional recipe. Training is not started from the creation form. The separate project review page is the final execution gate.

The selectable installed task is the real `causal_lm` objective. Story writing and chat are uses of that objective, not fake model types. Masked LM, sequence-to-sequence LM, classification, embedding, text-to-image, and image captioning use their real ML names and remain explicitly unavailable until their runtime adapter exists.

### Dataset actions returned HTTP 500

Resolved. The dataset-builder template contained an invalid Jinja condition in its field-suggestion loop. The expression was corrected and expanded into testable statements. Dataset inspection, transformation, preview, and validation routes also convert expected file/data failures into visible validation messages.

### Application startup was unclear

Resolved. `start-vedock.cmd` is available for double-click startup, and `scripts/start.ps1` starts the Windows Waitress server and prints the URL and shutdown instruction. `python run.py` remains available as the required development entry point.

## Verification without training

- 17 automated tests passed.
- Every Jinja template compiled.
- Dataset upload opened the builder successfully.
- Dataset transformation preview completed successfully.
- GPT-2 Medium was registered from the catalog without loading weights.
- A scratch project remained a draft and created zero training jobs.
- Compatible synthetic LoRA adapters were merged without loading a base model.
- Live server checks returned HTTP 200 for Model Studio, model registration, Dataset Builder, and transformation preview.
- No worker process was active and no training was started during this repair.

## Second correction pass

- `/playground` accepts both GET and POST, eliminating the reported HTTP 405.
- The main interaction is now a capability-neutral persistent chat with a complete local model directory, neutral Send action, streamed output, saved context, and optional runtime parameters.
- Browser and CLI streaming use the same `/api/v1/models/{model}/stream` local API. Interactive CLI chat retains the returned conversation ID for follow-up messages.
- The runtime caches its loaded model by reference, device, and precision instead of reloading it for every message.
- GPT-2 catalog entries, arbitrary Hugging Face causal-LM repositories, local model directories, checkpoints, and scratch GPT-2 architectures are available as source types.
- Float inputs use browser `step="any"`; the server remains the authoritative bounds validator, so values such as `0.0002` are accepted.
- Dataset pages show the raw local path, each immutable processed-version path, hashes, validation status, and download links.
- Expected merge failures return compatibility details or visible experimental errors without producing HTTP 500 pages.
- `MODEL_TRAINING_ENABLED=false` now blocks model-training submission from both web and API/CLI until the requested final phase.
- The read-only AI source trees were assessed in `READ_ONLY_AI_CODE_ASSESSMENT.md`; unsafe train-on-import experiments were not copied into the production runtime.
