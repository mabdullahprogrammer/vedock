# Dataset Pipeline

## Goal

Vedock preserves raw inputs, produces reproducible immutable versions, and gives the training runtime only validated, training-ready artifacts. The MVP supports text/tabular sources plus ZIP image-folder datasets for the real image-classification runtime. Text-to-image and image-captioning datasets remain future adapters and are not presented as working training features.

## Storage contract

```text
storage/datasets/
├── raw/{user_id}/{dataset_id}/
│   ├── source_file
│   └── source_metadata.json
├── processed/{user_id}/{dataset_id}/{version_id}/
│   ├── data.jsonl
│   ├── schema.json
│   ├── transformation.json
│   ├── statistics.json
│   ├── validation.json
│   └── invalid_rows.jsonl
└── temporary/
```

Raw files are never edited. A new transformation configuration always creates a new version directory with an exclusive identifier. Saving fails if that destination already exists.

## Data records

### RawDataset

Stores ownership, display metadata, source type and URL, original filename, safe storage path, detected format and MIME type, byte length, SHA-256, upload time, inspection status, detected schema, row count, and inspection summary.

### DatasetVersion

Stores the raw dataset reference, owner, monotonically increasing version number, schema type, immutable storage path, ordered transformations, field mapping, validation state, row counts, token estimate, content hash, and creation time.

### DatasetTransformation

Stores one ordered operation, its validated configuration, and its result summary. This makes the transformation recipe reproducible and inspectable.

## Import flow

### Local upload

1. Authenticate and verify the configured upload limit.
2. Normalize the user filename with a safe basename.
3. Stream the upload into a newly allocated temporary file while calculating SHA-256 and byte count.
4. Detect format from extension plus content checks.
5. Atomically move the completed temporary file into a new raw dataset directory.
6. Write source metadata and commit the database record.
7. Inspect through the dataset service.

The MVP accepts CSV, JSON, JSONL, and TXT. ZIP and Parquet are capability-gated: ZIP extraction requires path traversal, count, expanded-size, and extension defenses; Parquet is enabled only when PyArrow imports successfully.

### Direct URL

1. Require `http` or `https`.
2. Resolve every hostname and reject loopback, link-local, private, multicast, unspecified, and reserved addresses.
3. Limit redirects and repeat the target validation after every redirect.
4. Stream with connect/read timeouts and a strict maximum byte count.
5. Reject unsafe or unsupported content.
6. Store the final URL, response content type, filename, hash, and size with the raw artifact.

Credentials in URLs are rejected. URL import never sends local cookies or authentication headers.

## Inspection

Readers share a row iterator abstraction. CSV parsing honors quoted multiline fields. JSON accepts an array of objects for MVP-sized files. JSONL is processed incrementally. TXT yields one `text` record per nonempty line.

Inspection returns:

- format, encoding, size, row count, and sampled/full-scan status;
- columns and inferred scalar types;
- null and empty counts;
- duplicate count based on canonical row hashes;
- per-text-column minimum, maximum, and average character lengths;
- sample rows with bounded text sizes;
- likely prompt, response/story, text, label, image, and caption fields.

Large files use a bounded preview sample and an incremental statistics pass. They are never materialized as one in-memory table.

## Transformation model

The client submits an ordered JSON list. The server validates each operation against a transformation schema before preview or save. The same pure row-transform functions are used for preview and final processing.

MVP operations are:

- select columns;
- rename columns;
- remove columns;
- join columns;
- add a constant field;
- trim whitespace;
- normalize Unicode;
- remove HTML;
- remove URLs;
- remove control characters;
- plain text replacement;
- bounded regular-expression replacement;
- lowercase;
- remove empty records;
- remove duplicates;
- filter by minimum or maximum text length;
- shuffle with a recorded seed;
- limit examples;
- map source fields to an output schema;
- apply a prompt template.

Preview runs against a bounded row sample and does not save artifacts. The browser can reorder, edit, or remove operations before saving. Saved versions are never mutated; “reversing” a saved version means creating another version from the raw dataset or a selected source version.

## Output schemas

The MVP writes UTF-8 JSONL in one of these server-defined forms:

```json
{"text":"Complete training example text."}
```

```json
{"prompt":"Write a story about the moon.","response":"The moon watched silently..."}
```

```json
{"instruction":"Write a short mystery.","input":"It takes place in an abandoned station.","output":"The final train arrived..."}
```

```json
{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"}]}
```

```json
{"text":"Example content","label":"category_name"}
```

The StoryMaker runtime accepts prompt/response JSONL. Its adapter maps `response` to the legacy concept of `story` without changing the immutable dataset artifact.

## Validation

Validation emits structured findings with severity, code, message, suggested fix, and affected row numbers where practical.

Critical errors block training:

- unreadable or invalid file;
- inconsistent JSON object structure;
- missing required output fields;
- empty required values;
- invalid chat message structure or roles;
- no valid examples;
- missing training split when required;
- tokenizer failure;
- every example exceeding the configured usable sequence length.

Warnings include duplicates, small dataset size, isolated long examples, missing optional validation split, high empty-row removal rates, replacement-character encoding symptoms, and likely train/validation leakage.

`invalid_rows.jsonl` is downloadable when invalid rows exist. It contains the row number, finding codes, and a bounded copy of the source record; it never changes the raw artifact.

## Processing jobs

Dataset job states are:

```text
queued → downloading → inspecting → transforming → validating → saving → completed
```

Any active state may transition to `failed` or `cancelled`. Small development uploads may be inspected synchronously, but transformations that create a version use the controlled worker interface. Job logs and progress are persisted so page refreshes do not lose status.

Cancellation is cooperative between records and before final save. A cancelled or failed job may leave only a Vedock-owned temporary directory, which cleanup can remove. It never leaves a partially published version directory.

## Reproducibility

Each version records:

- raw SHA-256;
- exact ordered transformation JSON;
- output schema and mapping;
- shuffle seed and limits;
- software/runtime version information;
- row and invalid-row counts;
- output SHA-256;
- validation report.

Applying the same normalized transformation configuration to the same raw hash must produce the same output bytes unless an operation explicitly records a nondeterministic seed.
