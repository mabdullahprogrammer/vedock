# Inference runtime contract

Vedock does not assume that every model is a chatbot. Each runtime adapter declares a runner schema and implements typed execution.

## Contract

`get_runner_schema(model_path)` returns:

```json
{
  "interaction": "numeric_prediction",
  "title": "Predict monthly sales",
  "description": "Enter one structured business record.",
  "submit_label": "Calculate prediction",
  "inputs": [
    {
      "name": "ad_spend",
      "label": "Ad spend",
      "description": "Numeric predictor used during fitting.",
      "type": "number",
      "required": true
    },
    {
      "name": "region",
      "label": "Region",
      "type": "select",
      "choices": ["north", "south"],
      "allow_custom": true,
      "required": true
    }
  ],
  "outputs": [{"type": "metric", "label": "Monthly sales"}]
}
```

Supported inputs are `text`, `textarea`, `number`, `integer`, `boolean`, `select`, `json`, `image`, `file`, and `date`. Supported output blocks are `text`, `metric`, `probabilities`, `table`, `series`, `embedding`, `image`, `images`, and `json`.

The backend validates both model inputs and runtime parameters. Runtimes with one legacy input remain compatible through `RuntimeAdapter.run`; runtimes with multiple inputs override `run(model_path, inputs, parameters)`.

## Surfaces

- Web: `/playground/<model>` selects chat, image classification, or the universal structured runner from the capability contract.
- API: `POST /api/v1/models/<model>/run` accepts JSON typed inputs or multipart file/image inputs.
- CLI: `vedock models run MODEL --input FIELD=VALUE --file IMAGE_FIELD=PATH`.
- Desktop: model cards open the same generated controls and use the native OS file picker for file/image fields.

Legacy text and image endpoints remain available for compatibility.

## Portable artifacts

Tabular predictors are saved as `predictor.json`. Newly fitted image classifiers are saved as `classifier.json`. Pattern models use `pattern_model.json`. These are data-only formats; a connected user cannot publish executable pickle content to the hosted inference process. Older locally trusted image classifiers can still be opened from `classifier.joblib`, but the remote artifact allowlist never accepts joblib files.

## Adding a runtime

Implement the `RuntimeAdapter` interface, declare its runner and parameter schemas, normalize output into supported blocks, validate its artifact without loading it, and register it in `vedock/runtimes/registry.py`. The interface then renders without adding a task-specific hardcoded chat form.
