# Three-Minute Vedock Demo

## 0:00–0:25 — real system state

1. Open `http://127.0.0.1:5464` and log in or register.
2. Open **System**.
3. Point out the detected Python packages, CPU-only runtime, unavailable CUDA, RAM/disk values, and protected StoryMaker path.

Message: Vedock shows what this machine can actually run and does not offer QLoRA or CUDA controls here.

## 0:25–1:00 — existing model inference

1. Open **Models → StoryMaker Final → Open playground**.
2. Use a short prompt such as `A clockmaker discovers that midnight has stopped.`
3. Change temperature, top-p, and maximum new tokens.
4. Keep **Save this generation** enabled and generate.
5. Open **Conversations** to reopen the prompt, output, model version, and parameters.

Message: the model loads only on first use; controls come from the runtime schema and are revalidated by the server.

## 1:00–1:50 — raw data to immutable JSONL

1. Open **Datasets**.
2. Upload `D:\LLM\vedock\demo\story_prompts.csv`.
3. Show detected `prompt` and `story` fields, row count, length statistics, and raw SHA-256.
4. Open the builder, map prompt → prompt and story → response.
5. Keep trim, Unicode normalization, empty removal, and duplicate removal enabled.
6. Click **Preview only**, then **Save immutable version**.
7. Show validation state, row counts, token estimate, and output hash.

Message: raw bytes remain unchanged; a different cleaning recipe creates another version.

## 1:50–2:25 — configuration and jobs

1. Open **Create Model**.
2. Show active text/story tasks and disabled “Coming next” task families.
3. Select StoryMaker Final and the validated dataset.
4. Choose LoRA and inspect preset versus advanced fields.
5. Do not start another training job during a short demo; open **Training Jobs** and select the already completed smoke job to show real persisted metrics and its saved model version.

Message: jobs run in a separate worker, not inside an HTTP request. The completed smoke evidence is recorded in `VERIFICATION_REPORT.md`.

## 2:25–2:50 — CLI

```powershell
$env:VEDOCK_CLI_CONFIG='D:\LLM\vedock\storage\cli\config.json'
vedock doctor
vedock models list
vedock versions list vedock-dreamer-smoke-0ff3207d
```

Optionally run a short inference with the completed adapter if memory allows; do not start training.

## 2:50–3:00 — safe merge result

```powershell
vedock merge storymaker-final storymaker-finetuned --weight-a 0.7 --weight-b 0.3
```

Show that architecture, tensor names, shapes, vocabulary, and precision pass, while tokenizer metadata and current memory fail. Vedock blocks the operation instead of joining weights blindly.
