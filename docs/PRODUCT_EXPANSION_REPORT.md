# Product Expansion Report

## Implemented in this pass

- Persistent model-specific `output_pattern` configuration for training serialization and inference prefix/suffix handling. The StoryMaker default matches the protected legacy script's exact input/response markers.
- Runtime-backed stop strings plus an explicit generation ID and Stop endpoint/button for streamed text generation.
- A model-first authenticated directory with creators, fork counts, shared models, personal models, search, run controls, editable owned metadata, recoverable archive/restore, and local forks that preserve source provenance.
- Newest-first local datasets and locally importable community starter datasets.
- ZIP image-folder inspection and immutable `image_classification` JSONL versions without overwriting source archives.
- Advanced ordered dataset operations including column selection/rename/remove/join/split, constants, cleaning, regex replacement, numeric filtering, label mapping, type conversion, shuffle, and limits.
- Real `pattern_sequence` and `sklearn_image_classification` runtime adapters with schema-driven training and inference parameters.
- Browser image-classification runner and matching API/CLI commands.
- One-file Windows installer wizard with selectable core, LLM, fast-ML, and developer components; `vedock ui`; OpenAPI JSON; and a safe live GET explorer.
- Public/private model ownership, owner-only edits, model covers, distinct-person remix counts, and public community discovery.
- Unified chat history, start-new-chat, history toggle, context budget, and non-destructive context override inside the model chat.
- Model-aware dataset modification that recommends a runtime schema and field mappings before immutable version creation.

## Explicit availability boundary

No model was trained in this pass. `MODEL_TRAINING_ENABLED=true` now permits an owner to start one reviewed project manually. Saving a project and starting the application do not launch training, and old queued records are not automatically resumed. Pattern fitting and image-classifier fitting are implemented behind the controlled worker interface, but they have not been executed in this pass and are not claimed as new trained outputs.

The inspected image-captioning and image-completion experiments are not production artifacts: they train on import or execution, contain hard-coded paths, require unavailable TensorFlow/OpenCV packages, and have no reusable matching weights. Vedock therefore does not label text-to-image or captioning as working. Those task names remain capability-gated until an adapter and artifact pass real execution tests.

## Source protection

All inspected legacy and experimental trees were read-only. Source remains under `D:\LLM\vedock`; live Vedock storage and distributions use `E:\Vedock`, while connected users' private model and dataset artifacts remain on their own registered devices.
