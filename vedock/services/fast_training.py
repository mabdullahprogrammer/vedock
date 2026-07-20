from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from flask import current_app
from PIL import Image

from vedock.extensions import db
from vedock.models import DatasetVersion, Job, ModelRecord, ModelVersion, new_id
from vedock.runtimes import get_runtime

from .jobs import append_job_log
from .paths import allocate_directory, atomic_write_json
from .training import TrainingError, re_safe_slug


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest.update(file.relative_to(path).as_posix().encode("utf-8"))
            digest.update(file.read_bytes())
    return digest.hexdigest()


def _rows(version: DatasetVersion, maximum: int = 0) -> list[dict[str, Any]]:
    output = []
    with Path(version.storage_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                output.append(json.loads(line))
            if maximum and len(output) >= maximum:
                break
    return output


def _save_output(
    job: Job,
    directory: Path,
    model_id: str,
    version_id: str,
    name: str,
    task_type: str,
    runtime_key: str,
    metadata: dict[str, Any],
) -> ModelVersion:
    output_hash = _directory_hash(directory)
    model = ModelRecord(
        id=model_id,
        owner_id=job.owner_id,
        slug=f"{re_safe_slug(name)}-{model_id[:8]}",
        name=name[:160],
        description=f"Built locally with the {runtime_key} runtime in {current_app.config['APP_NAME']}.",
        task_type=task_type,
        runtime_key=runtime_key,
        source_type="training",
        source_path=str(directory),
    )
    version = ModelVersion(
        id=version_id,
        model=model,
        version_number=1,
        label="Fitted model",
        storage_path=str(directory),
        status="completed",
        config_json=metadata.get("parameters") or {},
        metadata_json=metadata,
        sha256=output_hash,
    )
    db.session.add_all([model, version])
    db.session.flush()
    job.result_model_version_id = version.id
    db.session.commit()
    append_job_log(job, "Fast model version saved", model_version_id=version.id, output_hash=output_hash)
    return version


def run_pattern_training(job: Job) -> ModelVersion:
    configuration = job.config_json
    params = get_runtime("pattern_sequence").get_training_parameter_schema()
    from vedock.runtimes.parameters import validate_parameters

    values = validate_parameters(configuration.get("parameters") or {}, params)
    dataset = db.session.get(DatasetVersion, configuration["dataset_version_id"])
    if not dataset or dataset.owner_id != job.owner_id:
        raise TrainingError("The pattern dataset is unavailable.")
    if dataset.output_format not in {"text_completion", "prompt_response"}:
        raise TrainingError("Pattern models require text-completion or prompt-response JSONL.")
    job.current_stage = "counting_patterns"
    job.progress = 20
    db.session.commit()
    transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    pattern = str(values["token_pattern"])
    try:
        re.compile(pattern)
    except re.error as exc:
        raise TrainingError(f"Invalid token regular expression: {exc}") from exc
    order = int(values["order"])
    examples = _rows(dataset, int(values["maximum_examples"]))
    token_count = 0
    for index, row in enumerate(examples, 1):
        db.session.refresh(job)
        if job.cancel_requested:
            raise InterruptedError("Pattern fitting was cancelled.")
        text = str(row.get("text") or f"{row.get('prompt', '')} {row.get('response', '')}")
        if values["lowercase"]:
            text = text.lower()
        tokens = re.findall(pattern, text)
        token_count += len(tokens)
        for offset in range(len(tokens) - order):
            transitions[tuple(tokens[offset:offset + order])][tokens[offset + order]] += 1
        if index % 1000 == 0:
            job.progress = min(85, 20 + int((index / max(1, len(examples))) * 65))
            db.session.commit()
    if not transitions:
        raise TrainingError("The dataset did not contain enough tokens for the selected pattern order.")
    model_id = new_id()
    version_id = new_id()
    directory = allocate_directory("models", str(job.owner_id), model_id, version_id)
    payload = {
        "format": "vedock.pattern.v1",
        "order": order,
        "lowercase": bool(values["lowercase"]),
        "token_pattern": pattern,
        "transitions": {"\u001f".join(state): dict(counts) for state, counts in transitions.items()},
    }
    atomic_write_json(directory / "pattern_model.json", payload)
    metadata = {"runtime": "pattern_sequence", "parameters": values, "dataset_version_id": dataset.id, "examples": len(examples), "tokens": token_count, "states": len(transitions)}
    atomic_write_json(directory / "metadata.json", metadata)
    return _save_output(job, directory, model_id, version_id, values["output_model_name"], "pattern_sequence", "pattern_sequence", metadata)


def run_image_classifier_training(job: Job) -> ModelVersion:
    configuration = job.config_json
    runtime = get_runtime("sklearn_image")
    from vedock.runtimes.parameters import validate_parameters

    values = validate_parameters(configuration.get("parameters") or {}, runtime.get_training_parameter_schema())
    dataset = db.session.get(DatasetVersion, configuration["dataset_version_id"])
    if not dataset or dataset.owner_id != job.owner_id:
        raise TrainingError("The image dataset is unavailable.")
    if dataset.output_format != "image_classification" or dataset.raw_dataset.file_format != "zip":
        raise TrainingError("Fast image classification requires an image_classification version created from a ZIP archive.")
    rows = _rows(dataset, int(values["maximum_examples"]))
    job.current_stage = "decoding_images"
    job.progress = 10
    db.session.commit()
    width, height = int(values["image_width"]), int(values["image_height"])
    mode = "RGB" if values["color_mode"] == "rgb" else "L"
    samples: list[np.ndarray] = []
    labels: list[str] = []
    with zipfile.ZipFile(dataset.raw_dataset.storage_path) as archive:
        names = set(archive.namelist())
        for index, row in enumerate(rows, 1):
            db.session.refresh(job)
            if job.cancel_requested:
                raise InterruptedError("Image classifier fitting was cancelled.")
            member = str(row.get("image") or "")
            if member not in names:
                continue
            try:
                with Image.open(BytesIO(archive.read(member))) as image:
                    vector = np.asarray(image.convert(mode).resize((width, height)), dtype=np.float32).reshape(-1) / 255.0
            except Exception:
                continue
            samples.append(vector)
            labels.append(str(row.get("label") or ""))
            if index % 100 == 0:
                job.progress = min(50, 10 + int((index / max(1, len(rows))) * 40))
                db.session.commit()
    counts = Counter(labels)
    if len(samples) < 4 or len(counts) < 2 or min(counts.values(), default=0) < 2:
        raise TrainingError("Use at least four valid images, at least two class folders, and at least two images in every class.")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestCentroid
    features = np.stack(samples)
    targets = np.asarray(labels)
    class_count = len(counts)
    requested_test_count = max(class_count, int(round(len(samples) * float(values["test_split"]))))
    test_count = min(len(samples) - class_count, requested_test_count)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_count,
        random_state=values["seed"],
        stratify=targets,
    )
    job.current_stage = "fitting_classifier"
    job.progress = 60
    db.session.commit()
    classifier = LogisticRegression(max_iter=1000, random_state=values["seed"]) if values["algorithm"] == "linear_softmax" else NearestCentroid()
    classifier.fit(x_train, y_train)
    accuracy = float(accuracy_score(y_test, classifier.predict(x_test)))
    model_id = new_id()
    version_id = new_id()
    directory = allocate_directory("models", str(job.owner_id), model_id, version_id)
    if values["algorithm"] == "linear_softmax":
        portable_classifier = {
            "format": "vedock.image_classifier.v1",
            "kind": "linear_softmax",
            "labels": [str(value) for value in classifier.classes_],
            "weights": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
        }
    else:
        portable_classifier = {
            "format": "vedock.image_classifier.v1",
            "kind": "nearest_centroid",
            "labels": [str(value) for value in classifier.classes_],
            "centroids": classifier.centroids_.tolist(),
        }
    atomic_write_json(directory / "classifier.json", portable_classifier)
    metadata = {
        "format": "vedock.sklearn_image.v1",
        "runtime": "sklearn_image",
        "parameters": values,
        "dataset_version_id": dataset.id,
        "image_width": width,
        "image_height": height,
        "color_mode": values["color_mode"],
        "labels": sorted(set(labels)),
        "examples": len(samples),
        "validation_accuracy": accuracy,
    }
    atomic_write_json(directory / "metadata.json", metadata)
    return _save_output(job, directory, model_id, version_id, values["output_model_name"], "image_classification", "sklearn_image", metadata)


def _tabular_feature_schema(rows: list[dict[str, Any]], maximum_categories: int) -> list[dict[str, Any]]:
    names: list[str] = []
    for row in rows:
        for name in (row.get("features") or {}):
            if name not in names:
                names.append(str(name))
    schema: list[dict[str, Any]] = []
    for name in names:
        values = [(row.get("features") or {}).get(name) for row in rows]
        numeric_values: list[float] = []
        for value in values:
            try:
                number = float(value)
                if np.isfinite(number):
                    numeric_values.append(number)
            except (TypeError, ValueError):
                pass
        nonempty_count = sum(value is not None and value != "" for value in values)
        if numeric_values and len(numeric_values) >= max(1, int(nonempty_count * 0.9)):
            median = float(np.median(numeric_values))
            mean = float(np.mean(numeric_values))
            scale = float(np.std(numeric_values)) or 1.0
            schema.append({"name": name, "kind": "numeric", "median": median, "mean": mean, "scale": scale})
        else:
            counts = Counter(str(value if value is not None and value != "" else "[MISSING]") for value in values)
            categories = [value for value, _count in counts.most_common(maximum_categories)]
            schema.append({"name": name, "kind": "categorical", "categories": categories})
    return schema


def _tabular_matrix(rows: list[dict[str, Any]], schema: list[dict[str, Any]]) -> np.ndarray:
    vectors: list[list[float]] = []
    for row in rows:
        source = row.get("features") or {}
        vector = [1.0]
        for feature in schema:
            value = source.get(feature["name"])
            if feature["kind"] == "numeric":
                try:
                    numeric = float(value)
                    if not np.isfinite(numeric):
                        raise ValueError
                except (TypeError, ValueError):
                    numeric = float(feature["median"])
                vector.append((numeric - float(feature["mean"])) / max(float(feature["scale"]), 1e-12))
            else:
                normalized = str(value if value is not None and value != "" else "[MISSING]")
                categories = feature["categories"]
                vector.extend(1.0 if normalized == category else 0.0 for category in categories)
                vector.append(0.0 if normalized in categories else 1.0)
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float64)


def _fit_linear(features: np.ndarray, targets: np.ndarray, regularization: float) -> np.ndarray:
    penalty = np.eye(features.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    return np.linalg.pinv(features.T @ features + penalty) @ features.T @ targets


def _fit_softmax(features: np.ndarray, targets: np.ndarray, class_count: int, learning_rate: float, iterations: int, regularization: float) -> np.ndarray:
    weights = np.zeros((features.shape[1], class_count), dtype=np.float64)
    encoded = np.eye(class_count, dtype=np.float64)[targets]
    previous = float("inf")
    for _ in range(iterations):
        scores = features @ weights
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = features.T @ (probabilities - encoded) / len(features)
        gradient[1:] += regularization * weights[1:]
        weights -= learning_rate * gradient
        loss = float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean())
        if abs(previous - loss) < 1e-9:
            break
        previous = loss
    return weights


def run_tabular_training(job: Job) -> ModelVersion:
    configuration = job.config_json
    runtime = get_runtime("tabular_prediction")
    from vedock.runtimes.parameters import validate_parameters

    values = validate_parameters(configuration.get("parameters") or {}, runtime.get_training_parameter_schema())
    objective = str(values["objective"])
    task_type = str(configuration.get("task_type") or f"tabular_{objective}")
    expected_objective = "classification" if task_type == "tabular_classification" else "regression"
    if objective != expected_objective:
        raise TrainingError(f"The {task_type.replace('_', ' ')} task requires objective={expected_objective!r}.")
    if values["training_method"] != ("logistic_fit" if objective == "classification" else "linear_fit"):
        raise TrainingError("The selected training method does not match the tabular objective.")
    dataset = db.session.get(DatasetVersion, configuration["dataset_version_id"])
    if not dataset or dataset.owner_id != job.owner_id:
        raise TrainingError("The tabular dataset is unavailable.")
    if dataset.output_format != "tabular_supervised":
        raise TrainingError("Tabular predictors require a tabular_supervised JSONL version.")
    rows = _rows(dataset, int(values["maximum_examples"]))
    if len(rows) < 5:
        raise TrainingError("Use at least five valid tabular records.")
    if any(not isinstance(row.get("features"), dict) or not row.get("features") for row in rows):
        raise TrainingError("Every tabular row needs a non-empty features object.")
    job.current_stage = "encoding_features"
    job.progress = 20
    db.session.commit()
    randomizer = np.random.default_rng(int(values["seed"]))
    indices = randomizer.permutation(len(rows))
    validation_count = max(1, min(len(rows) - 2, int(round(len(rows) * float(values["test_split"])))))
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]
    feature_schema = _tabular_feature_schema([rows[index] for index in training_indices], int(values["maximum_categories"]))
    if not feature_schema:
        raise TrainingError("No usable predictor columns remained after preparation.")
    features = _tabular_matrix(rows, feature_schema)
    x_train, x_validation = features[training_indices], features[validation_indices]
    job.current_stage = "fitting_predictor"
    job.progress = 60
    db.session.commit()
    target_name = str((dataset.field_mapping or {}).get("target") or "target")
    predictor: dict[str, Any] = {
        "format": "vedock.tabular.v1",
        "objective": objective,
        "target_name": target_name,
        "target_unit": values["target_unit"],
        "target_transform": values["target_transform"] if objective == "regression" else "none",
        "features": feature_schema,
    }
    if objective == "regression":
        try:
            targets = np.asarray([float(row.get("target")) for row in rows], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TrainingError("Regression targets must all be numeric.") from exc
        if not np.isfinite(targets).all():
            raise TrainingError("Regression targets cannot contain infinity or NaN.")
        if values["target_transform"] == "log1p":
            if np.any(targets < 0):
                raise TrainingError("The log1p target transform requires non-negative targets.")
            fitted_targets = np.log1p(targets)
        else:
            fitted_targets = targets
        validation_weights = _fit_linear(x_train, fitted_targets[training_indices], float(values["regularization"]))
        predictions = x_validation @ validation_weights
        if values["target_transform"] == "log1p":
            predictions = np.expm1(predictions)
        truth = targets[validation_indices]
        mae = float(np.mean(np.abs(predictions - truth)))
        rmse = float(np.sqrt(np.mean((predictions - truth) ** 2)))
        denominator = float(np.sum((truth - truth.mean()) ** 2))
        r2 = float(1.0 - np.sum((truth - predictions) ** 2) / denominator) if denominator else 0.0
        final_weights = _fit_linear(features, fitted_targets, float(values["regularization"]))
        predictor["weights"] = final_weights.tolist()
        metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    else:
        labels = sorted({str(row.get("target")) for row in rows})
        if len(labels) < 2:
            raise TrainingError("Classification requires at least two target classes.")
        label_index = {label: index for index, label in enumerate(labels)}
        targets = np.asarray([label_index[str(row.get("target"))] for row in rows], dtype=np.int64)
        validation_weights = _fit_softmax(x_train, targets[training_indices], len(labels), float(values["learning_rate"]), int(values["iterations"]), float(values["regularization"]))
        predicted = np.argmax(x_validation @ validation_weights, axis=1)
        accuracy = float(np.mean(predicted == targets[validation_indices]))
        final_weights = _fit_softmax(features, targets, len(labels), float(values["learning_rate"]), int(values["iterations"]), float(values["regularization"]))
        predictor.update({"weights": final_weights.tolist(), "labels": labels})
        metrics = {"accuracy": accuracy}
    model_id, version_id = new_id(), new_id()
    directory = allocate_directory("models", str(job.owner_id), model_id, version_id)
    atomic_write_json(directory / "predictor.json", predictor)
    metadata = {
        "format": "vedock.tabular.v1",
        "runtime": "tabular_prediction",
        "parameters": values,
        "dataset_version_id": dataset.id,
        "examples": len(rows),
        "features": len(feature_schema),
        "encoded_dimensions": int(features.shape[1]),
        "validation_examples": validation_count,
        "metrics": metrics,
    }
    atomic_write_json(directory / "metadata.json", metadata)
    return _save_output(job, directory, model_id, version_id, values["output_model_name"], task_type, "tabular_prediction", metadata)
