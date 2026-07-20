# Risk Register

Assessment date: 2026-07-19

| ID | Risk | Likelihood | Impact | Evidence | Mitigation / decision |
| --- | --- | --- | --- | --- | --- |
| R1 | The specified legacy path is wrong | Certain | High | `D:\LLM\StoryMaker` is absent; matching assets exist under `D:\LLM\new-llm\LLM-2025\StoryMaker` | Configure the discovered path, protect both path forms, show the resolved path in System, never auto-move assets |
| R2 | No usable CUDA device | Certain now | High | PyTorch reports zero devices; `nvidia-smi` absent | Expose CPU only, hide CUDA-only parameters, keep bounded generations, retest through doctor when drivers change |
| R3 | Very low available RAM | High | High | 7.73 GiB total and 0.41 GiB free during inspection | Lazy-load one model, offer unload, serialize training, use tiny smoke datasets, preflight estimates, avoid in-memory merges |
| R4 | Legacy training cannot start | Certain until fixed | High | `accelerate` missing and `TrainingArguments` raises ImportError | Install Accelerate only in Vedock-owned environment; verify one real worker job |
| R5 | LoRA unavailable | Certain until fixed | Medium | PEFT missing | Install PEFT under Vedock and capability-gate LoRA; fall back to bounded legacy-compatible full fine-tune only if safe |
| R6 | QLoRA not viable on current Windows/hardware | High | Medium | No CUDA and no bitsandbytes | Mark QLoRA unavailable; do not spend two-day scope on it |
| R7 | CPU inference is slow | High | Medium | 44-token real run took about 38 seconds including load | Cache one lazily loaded model, stream tokens, cap defaults, provide load status, document CPU latency |
| R8 | Legacy prompt token handling is inconsistent | High | Medium | Saved tokenizer lacks added four tokens; script resizes on every load; hyphen/underscore markers disagree | StoryMaker adapter owns one explicit template policy; do not mutate source tokenizers; persist chosen policy per model version |
| R9 | Legacy generation warnings reduce reliability | Certain in legacy path | Medium | Missing attention mask and ambiguous pad/EOS warnings | Vedock runtime passes attention mask, explicit pad token, and `max_new_tokens` |
| R10 | Raw datasets are huge | High | High | Multiple CSV files are 0.4–0.86 GiB and contain multiline text | Stream parsing, bounded previews, byte/row limits, background processing, never load whole CSV into memory |
| R11 | URL import can reach internal services | Medium | High | Product accepts arbitrary URLs | Validate schemes/DNS/redirects, block private/reserved IPs, enforce size and timeouts, omit credentials |
| R12 | Transformation regex can hang | Medium | Medium | User-editable regex is requested | Bound pattern/input sizes, limit supported regex operations, process in worker, allow cancellation |
| R13 | Partial artifacts appear as completed versions | Medium | High | Training and transformation are interruptible | Write to temporary Vedock directory, fsync/close, validate/hash, then atomically publish and commit completed status |
| R14 | HTTP process performs long training | Medium | High | Easy Flask implementation trap | HTTP only creates job and launches worker subprocess; integration test checks prompt response and worker PID |
| R15 | Existing versions are overwritten | Medium | High | Legacy script accepts arbitrary `output_dir` | Server allocates UUID version paths exclusively; reject existing destinations and protected roots |
| R16 | Blind merge corrupts a model | Medium | High | Legacy weights share shapes but tokenizers differ | Block by default on tokenizer mismatch; require exact compatibility report and selected tokenizer policy; stream tensor merge if enabled |
| R17 | Linear merge exceeds RAM/disk | High | High | Two 498 MB models, low free RAM, 46 GiB disk | Preflight memory and disk, merge one tensor at a time, write only under new version, fail safely before execution |
| R18 | Authentication/ownership gaps expose files | Medium | High | Multi-user login plus file artifacts | Server-side owner filters, opaque IDs, no direct user paths, CSRF on browser writes, Bearer tokens for CLI |
| R19 | Dynamic forms accept unsupported parameters | Medium | High | Universal editing can drift from runtime | Backend schema is authoritative; normalize and validate types, bounds, choices, dependencies, and capability conditions again server-side |
| R20 | Scope expands into unfinished multimodal/social work | Medium | High | Broad long-term specification | Enable only text/story; mark future tasks “Coming next”; no social, billing, org, marketplace, or image execution work |
| R21 | Protected legacy content is changed accidentally | Low with guardrails | Critical | 5.76 GB protected tree is used as source | Read-only code path, output allocator denylist, before/after fingerprints, never run legacy training with a legacy output directory |
| R22 | Online model loading stalls or executes remote code | Medium | High | Transformers accepts remote repositories and `trust_remote_code` | Local-only defaults, offline legacy loading, explicit online opt-in, `trust_remote_code=false`, warning acknowledgement and validation |
| R23 | SQLite concurrency causes locked jobs | Medium | Medium | Flask and workers share one SQLite file | WAL mode, short transactions, retry with bounded backoff, one training worker, PostgreSQL-ready data layer |
| R24 | Two-day build claims unverified features | Medium | High | Large requested surface | Maintain test evidence and a verified feature matrix; labels distinguish verified, available but untested, and coming next |

## Highest-priority gates

1. Preserve the discovered StoryMaker tree.
2. Keep models lazy and bound memory use.
3. Make inference real before dataset/training polish.
4. Install Accelerate and verify training in a separate process before claiming fine-tuning.
5. Block merge on the observed tokenizer mismatch unless an explicit, validated policy resolves it.
