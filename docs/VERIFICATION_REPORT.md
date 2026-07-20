# Verification Report

Verification date: 2026-07-20 (Asia/Karachi)

## Current result

Vedock now separates hosted control/public inference from user-owned training compute. Web training submissions produce inert `awaiting_device` records. The connected Windows/Linux CLI and native desktop controller can preview, edit, claim and explicitly execute a task locally, stream status back, respond to hosted cancellation, and finalize either locally or through filtered publication.

No model training was started during this implementation and verification pass.

## Automated checks

```text
41 passed
```

The suite covers branding, authentication, protected pages, hosted hardware/path privacy, no-load legacy registration, API tokens, chat routing, streaming context and stop dispatch, chat deletion, dataset inspection recommendations, transformation/immutability, JSON/JSONL/CSV/XLSX/TXT export, schema-driven parameters, recoverable device task release, ignored inactive LoRA fields, model ownership/remix, covers, archive generation, protected paths, safe model merging, typed runtime contracts, tabular dataset mapping, numeric-prediction API/UI rendering, sequence-completion routing, and portable image-classifier inference.

All Python sources in `vedock`, `vedock_cli`, `installer` and `tests` also passed `compileall`.

## Hosted-node checks

```text
Landing page:                  HTTP 200
API node mode:                hosted_inference
API storage location:         private_host_storage
Windows installer route:      HTTP 200
Windows installer size:       14,518,328 bytes
Windows installer SHA-256:    B68C7CCE861923837A0DBC1A96F537C164FBA20EA8C14217F92349D17C46D846
Connected client ZIP:         1,864,168 bytes
Database integrity:           ok
Active training tasks:        0
Python training workers:      0
```

The rebuilt installer was launched in a bounded smoke check and remained healthy for six seconds before the exact test process was stopped. The connected-client ZIP is generated from the current source and includes the desktop application, typed model runner, local job runner, logo, client requirements and on-demand runtime requirements.

## Storage migration

The SQLite database was copied to `E:\Vedock\instance\vedock.db`; the original D: database remains as a recovery source, and `E:\Vedock\instance\vedock.pre_e_migration.db` preserves the pre-path-migration copy.

Database schema migration added remote-device claim fields and the model reaction table. All internal dataset, job log and Vedock model paths were repointed from D: storage to E: storage.

The two existing StoryMaker models were copied without modifying their source and registered at:

```text
E:\Vedock\storage\models\published\gpt-storygen-final
E:\Vedock\storage\models\published\gpt2fintuned_storymaker
```

Source and destination file counts and byte totals matched for both copies. Both live model records now resolve to the E: directories.

## Installer and client behavior

- Fixed control plane: `https://vedock.ecorims.com`.
- No port/host question.
- User-selectable install location.
- Global `vedock` command through user PATH on Windows and `~/.local/bin` on Linux.
- Native pywebview desktop controller; it does not open the hosted site in a browser.
- Python 3.11 detection and quiet winget install on Windows when absent.
- Small base install; heavy ML runtimes download only for a task that needs them.
- Real desktop pages for training tasks, models, datasets, this device and installed runtimes.
- Runtime readiness is checked before claim; failed setup cannot strand a task in `claimed_by_local_device`.
- Claimed-but-not-running tasks can be resumed or released by the claiming device.
- Existing installs and runtimes are detected and are not downloaded twice.
- Vedock ICO branding is applied to the installer, Start Menu shortcut, desktop shortcut and desktop window.
- Desktop task list, device/precision controls, local logs and publish-or-keep-local choice.
- Readable colored CLI tables and final artifact review.

## Inference repair

The Transformers text runtime now derives prompt capacity from the model's actual context window and reserves space for new tokens. It uses left truncation, validates contradictory token settings and reports tokenizer/model vocabulary mismatch clearly. This addresses the previous GPT-2 `index out of range in self` failure without modifying protected model files.

Real inference was rerun against the published read-only copy of `gpt-storygen-final` with a bounded 20-token CPU generation. It completed successfully without an embedding or position index error. No training was started.

Inference is now capability-driven across the web, API, CLI, and connected desktop app. Chat models retain their conversation interface. Pattern models use sequence completion, image classifiers use image upload and ranked labels, and fitted tabular models generate feature-specific forms with metric or probability output. The universal result renderer also supports tables, series, embeddings, image galleries, text, and mixed blocks for future runtimes. Tabular and newly fitted image artifacts use data-only JSON formats; remote publication rejects executable joblib files.

## Honest remaining operational boundaries

- The rebuilt installer and local download route match. During the final check, the public Cloudflare endpoint served an older cached installer while uncached origin requests returned HTTP 522. Restore the independent tunnel/origin path or purge/revalidate its cache before claiming the new installer is externally deployed. A complete clean-machine installation remains a separate environment test.
- Public browser inference still runs on this host PC and needs queue/rate/VRAM controls before very large traffic.
- Image runtime packages are on-demand architecture, not a newly verified end-to-end image training run.
- No training workload was launched for verification.
