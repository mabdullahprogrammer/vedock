# Vedock

> Build any AI. No code. Full control.

Vedock is a no-code AI development environment with a Flask web control plane, a system-wide `vedock` CLI, and a small connected desktop client. Users prepare datasets and configure jobs on the web; training runs only after the authenticated owner starts it on their connected computer. Final inference artifacts can then be kept private or published for browser use.

Vedock treats model objectives as real runtime capabilities—not as personalities. The implemented vertical slices are:

- causal language-model inference and training, including GPT-2-family scratch architectures, full tuning, continued pretraining, and LoRA;
- StoryMaker-compatible persistent chat with streaming, stop, context, history, and saved input/output patterns;
- lightweight pattern-sequence fitting and sequence completion;
- image-folder preparation, portable image-classifier fitting, and image-upload inference;
- tabular regression for sales, demand, price, weight, and other numeric targets;
- tabular classification with ranked probabilities;
- immutable dataset versions, transformations, validation reports, and JSONL/JSON/CSV/XLSX/TXT export;
- safe model compatibility checks and supported merges.

## Inference is capability-driven

Every runtime publishes a typed runner contract. The web, API, CLI, and desktop client use that contract to build the correct experience:

```text
chat model            → conversation + context + generation controls
pattern model         → sequence input + predicted continuation
image classifier      → image picker + ranked labels
numeric predictor     → feature form + highlighted metric/unit
category predictor    → feature form + probability bars
forecast runtime      → time-series inputs + chart output
embedding runtime     → source input + vector visualization
image generator       → prompt controls + image gallery
hybrid runtime        → multiple typed inputs + mixed output blocks
```

New runtimes declare fields and output presentations instead of being forced through a `prompt` or “Generate story” screen. See [the inference runtime contract](docs/INFERENCE_RUNTIME_CONTRACT.md).

## Quick start

Requires Python 3.11 or newer.

Windows PowerShell:

```powershell
git clone https://github.com/mabdullahprogrammer/vedock.git
Set-Location vedock
& .\scripts\setup-portable.ps1
& .\scripts\start.ps1
```

Linux:

```bash
git clone https://github.com/mabdullahprogrammer/vedock.git
cd vedock
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python serve.py
```

Open `http://127.0.0.1:5464`. Models are registered without loading their weights; loading happens lazily on the first compatible inference request.

Copy `.env.example` to `.env` before deployment. Replace `SECRET_KEY`, choose `NODE_MODE`, and set storage/database paths appropriate for the installation.

## CLI

```text
vedock login
vedock doctor
vedock models list
vedock models add-local D:\models\my-model
vedock chat MODEL
vedock models run SALES_MODEL --input ad_spend=1200 --input region=north
vedock models run IMAGE_MODEL --file image=sample.png --parameter top_k=5
vedock datasets list
vedock datasets add-local D:\data\training.csv --schema prompt_response
vedock jobs list
vedock jobs run JOB_ID
vedock jobs resume JOB_ID
vedock jobs delete JOB_ID
```

`vedock chat` is specifically for chat-capable language models. `vedock models run` is universal and reads the selected model's typed contract.

## Compute and storage boundary

- A hosted node serves authentication, community metadata, and browser inference.
- Creating a hosted training task does not execute it on the server.
- Only the owner's authenticated CLI or desktop client can claim that task.
- Runtime readiness is checked before claim; installed components are reused.
- A path entered on the hosted web is resolved by the selected connected device, never by the hosted server.
- Private local models and processed datasets are represented by opaque `device://` references; their actual paths live only in the installed client's configuration.
- The matching client validates local resources, reports safe metadata and hashes, and reads them directly during training.
- Hosted/public base artifacts may still be downloaded by the connected client when a task needs them.
- The owner explicitly chooses whether the finalized inference artifact is uploaded and published.
- Hosted APIs redact server hardware and filesystem paths.

No training job is started during Flask startup or from an ordinary request handler.

## Repository safety

The repository contains source, tests, documentation, and safe application assets. It intentionally excludes `.env`, databases, user datasets, conversations, logs, model weights, trained artifacts, build folders, and installer binaries.

## Test

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

The tests do not start queued user training jobs. The verification boundary is recorded in [docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md).
