# Read-Only AI Code Assessment

## Scope inspected

The following source trees were treated as read-only:

- `D:\PycharmProjects\PythonProjects\Programes`
- `D:\PycharmProjects\PythonProjects\PyLang`
- `D:\PycharmProjects\PythonProjects\PyLang\IRModule`

Only files relevant to model loading, inference, image tasks, and streaming were inspected.

## Relevant implementations

### Image captioning

`Programes\Image Captioning\captioner.py` contains a TensorFlow image-captioning experiment using an InceptionV3 CNN encoder, Transformer encoder/decoder layers, masked caption loss, and autoregressive caption decoding.

Useful ideas retained:

- Image preprocessing belongs to the runtime adapter.
- Captioning inference has a different input/output schema from text chat.
- The model should be loaded once and reused for repeated images.
- Caption decoding parameters must be declared by the caption runtime.

The file was not copied directly because it trains immediately during import, contains hard-coded external paths, mixes dataset preparation/training/inference in one module, and has no matching saved weight artifact in that directory. Importing it into a production server would unexpectedly start training.

### Pixel image completion

`Programes\Image Completer\complete.py` contains a PyTorch PixelCNN-style experiment with explicit train and autoregressive generate functions.

Useful ideas retained:

- Image generation/completion is a separate runtime interaction, never a text-chat label.
- Training and generation must be isolated methods.
- Generated images are artifacts, not chat strings.

It was not copied directly because its example downloads MNIST and trains when run, uses a fixed demonstration workflow, and does not include reusable saved weights or model metadata.

### Image recognition/classification

`Programes\Image Recognition\model.py` contains a small OpenCV/scikit-learn camera classifier.

`PyLang\IRModule\model.py`, `easyRecognition.py`, and `app.py` contain related OpenCV/scikit-learn image-recognition experiments, fixed-size image flattening, persisted estimators, and camera/file inference paths.

Useful ideas retained:

- Classification output should expose labels and scores.
- Camera/image preprocessing must be part of a declared runtime schema.

Those modules were not copied because they write temporary files to relative paths, assume fixed dimensions/classes, mix camera UI and model logic, and depend on OpenCV, which is not installed in the Vedock environment. Their useful behavior was reimplemented as a clean `sklearn_image_classification` runtime: safe ZIP folder datasets, Pillow preprocessing, declared dimensions/classifier parameters, job-compatible fitting, persisted estimator metadata, and ranked label scores.

### Fast pattern prediction

The `NextwordPrediction` and predictor examples in `Programes` use substring matching or small Markov transition tables. Vedock implements the reusable idea as a separate `pattern_sequence` runtime with configurable n-gram order, deterministic seed support, JSON artifacts, and streaming-compatible generation. It is intentionally not mislabeled as an LLM.

### Text generation

`Programes\1chtgpt_\collected_conversation\load.py` and `Programes\ML\chatbot\bot.py` contain earlier text-generation examples. The protected StoryMaker script remains the stronger source for Transformers generation, prompt markers, streaming, and scratch GPT-2 construction.

## Production decision

No source file was modified. No experimental module was copied wholesale. The reusable concepts were incorporated into Vedock's runtime contract and local-node architecture. The production adapters are clean modules under `D:\LLM\vedock`, with no training on import, no hard-coded user paths, lazy loading, schema-driven inputs/outputs, and structured errors.
