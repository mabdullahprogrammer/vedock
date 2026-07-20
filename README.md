# Vedock

> Build any AI. No code. Full control.

Vedock is a no-code AI development environment with a Flask web control plane, a system-wide `vedock` CLI, and a small connected desktop client. Users prepare datasets and configure jobs on the web; training runs only after the authenticated owner starts it on their connected computer. Final inference artifacts can then be kept private or published for browser use.

## Built with Codex

Codex was a vital development partner in the creation of Vedock. It helped transform a collection of working Python experiments, model scripts, dataset utilities, and terminal-based workflows into one cohesive AI-development product.

With Codex assisting in architecture, implementation, debugging, testing, documentation, and packaging, Vedock grew into a polished system with:

- a hosted website for discovering, testing, publishing, forking, and remixing models;
- a connected desktop application for local datasets, models, hardware, runtimes, and training tasks;
- a global CLI for developers and automation;
- capability-driven interfaces for language, image, pattern, and tabular models;
- immutable dataset preparation and validation workflows;
- a community experience that encourages people to share useful models while keeping private data and compute under their control;
- connected-compute support designed for Windows and Linux workflows.

Codex did not replace the original ideas or working model code. It helped preserve that foundation, connect the pieces, and turn terminal-oriented prototypes into the higher-end Vedock application and web platform presented in this repository.

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

## Install and use Vedock

### Windows 10 or 11

1. Download the [Vedock installer](https://vedock.ecorims.com/downloads/vedock-installer.exe).
2. Run the installer and choose an installation folder.
3. Keep **Create desktop shortcut** selected, then press **Install Vedock**.
4. Open **Vedock** from the desktop or Start menu and sign in with the same account used on the website.
5. Open a new terminal and verify the connection:

```powershell
vedock doctor
vedock login
```

The installer is intentionally small. The CLI and desktop controller are installed first; large ML runtimes are downloaded only when a selected task actually needs them.

### Linux

```bash
git clone https://github.com/mabdullahprogrammer/vedock.git
cd vedock
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-client.txt
.venv/bin/python -m pip install -e .
.venv/bin/vedock login
.venv/bin/vedock doctor
```

### Run the Vedock server from source

Python 3.11 or newer is required:

```powershell
git clone https://github.com/mabdullahprogrammer/vedock.git
Set-Location vedock
& .\scripts\setup-portable.ps1
& .\scripts\start.ps1
```

Open `http://127.0.0.1:5464`. Copy `.env.example` to `.env` before deployment, replace `SECRET_KEY`, choose `NODE_MODE`, and configure storage/database paths for the installation. Models are registered without loading their weights; loading happens lazily for compatible inference requests.

## Train a model — short version

1. Sign in on [vedock.ecorims.com](https://vedock.ecorims.com), prepare a dataset, create a model, choose the training settings, and save the task.
2. Open the Vedock desktop app on the computer that will perform the training and sign in to the same account.
3. Select the waiting task and press **Run on this computer**.

Or run it from any terminal:

```text
vedock jobs list
vedock jobs run JOB_ID
```

Vedock checks the required runtime before claiming the task. Training starts only after the owner explicitly runs it, compute stays on that connected computer, and publishing the finalized inference artifact is optional.

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
