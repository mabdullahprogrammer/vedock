# Two-Day Implementation Plan

## Completion strategy

The project is judged as a vertical slice, not by the number of visible placeholders. Every working label must have a verified backend path. Text/story generation is the only fully enabled model family. Other task families remain architecture entries marked “Coming next.”

## Day 1 — usable inference and data

### Block 1: foundation and guardrails

- Create the Flask application factory, configuration, blueprints, error handling, database, and storage allocators.
- Add environment-controlled branding and paths.
- Add registration, login, logout, API tokens, ownership checks, and CSRF protection.
- Register the discovered legacy models without loading them.
- Add hardware and dependency detection.

Exit gate: app starts on port 5464; a user can register/login; dashboard and system page show actual CPU/CUDA status; protected-path write checks pass.

### Block 2: StoryMaker inference

- Implement the runtime interface and StoryMaker adapter.
- Implement lazy model loading and explicit unloading.
- Generate inference controls from the runtime schema.
- Add prompt completion, story generation, streaming interface, and conversation/message persistence.
- Add model list/details and playground pages plus API routes.

Exit gate: real text is generated from `gpt-storygen-final` through the service/API path with changed parameters, then saved and reopened as a conversation.

### Block 3: dataset vertical slice

- Implement local CSV/JSON/JSONL/TXT upload and direct URL import defenses.
- Store raw files immutably with hashes and source metadata.
- Inspect schemas and streaming statistics.
- Build field mapping and the priority cleaning transformations.
- Preview changes, save immutable JSONL versions, validate, and expose invalid-row reports.

Exit gate: one local fixture and one controlled HTTP URL fixture import successfully; raw hashes remain unchanged; transformed prompt/response JSONL validates and can be selected by the studio.

## Day 2 — training, CLI, merge, demo

### Block 4: studio and jobs

- Build the staged text/story model project form.
- Render simple presets and advanced runtime-supported training fields from schemas.
- Show hardware/package review and normalized configuration preview.
- Save reusable recipes.
- Launch training only through the worker process and stream/poll logs.

Exit gate: HTTP request returns while a separate process owns training; cancellation and failure are represented correctly; critical dataset validation blocks submission.

### Block 5: real minimal fine-tune and versions

- Add the missing `accelerate` dependency in a Vedock-owned environment.
- Run a tiny, bounded CPU smoke fine-tune using the legacy-compatible formatting path.
- Save to a fresh Vedock model-version directory.
- Hash/register the version and run inference from it.

Exit gate: a real training step completes on a tiny fixture, logs are visible, the immutable model version is registered, and it generates output. If machine memory cannot safely execute the step, report the job as a verified resource failure rather than claiming training success.

### Block 6: CLI

- Implement doctor/login/whoami.
- Implement model list/info/use/chat and version list/export.
- Implement dataset list/inspect/validate/transform.
- Implement train and jobs list/show/logs/cancel.
- Implement merge compatibility and submission.

Exit gate: CLI authenticates to `http://127.0.0.1:5464/api/v1`, lists models, and performs real inference.

### Block 7: merge and hardening

- Implement config/tokenizer/tensor compatibility reports.
- Add PEFT adapter merge only when compatible adapters exist.
- Add linear merge only when all safety and memory gates pass.
- Store merge metadata as a normal model version.
- Run API, ownership, immutability, SSRF, path, dataset, runtime, and CLI tests.

Exit gate: the two legacy models return a clear evidence-backed compatibility result; unsafe merges are blocked; successful merges, if any, use new Vedock paths only.

### Block 8: demo and handoff

- Finalize error/loading/empty states and responsive layout.
- Add concise local setup, CLI, and three-minute demo documentation.
- Recompute protected-tree fingerprints.
- Record an exact verified/unverified feature matrix.

## Three-minute judge flow

1. Register and open System to see CPU, RAM, disk, and CUDA detection.
2. Open StoryMaker Final, adjust temperature/top-p/output length, and generate a story.
3. Save and reopen the conversation.
4. Upload a small prompt/story CSV, inspect it, trim/filter/map fields, preview, and save a validated JSONL version.
5. Create a story model project, select the dataset, choose Low Memory, adjust advanced fields, and submit a training job.
6. Open live job logs and the resulting model version.
7. Run the version from the CLI.
8. Compare the two legacy models and show why tokenizer policy prevents a blind merge.

## Non-goals during the two days

No community features, billing, organizations, Kubernetes, multi-cloud, marketplace, distributed training, production email recovery, public model downloads, real image generation, or incomplete task types presented as working.

## Stop/go rules

- Do not polish disabled future task cards until the LLM flow is complete.
- Do not add QLoRA when CUDA/bitsandbytes capability checks fail.
- Do not spend time optimizing giant legacy datasets before the small import-transform-train path works.
- Do not perform a merge merely because shapes match; tokenizer and resource gates remain mandatory.
- Do not report real training success without a completed job, saved artifacts, and a post-training inference check.
