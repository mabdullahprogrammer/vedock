from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from vedock.extensions import db
from vedock.models import ModelRecord, ModelVersion, User
from vedock.runtimes.registry import get_runtime
from vedock.services.datasets import map_record
from vedock.services.inference import normalize_runtime_result, runner_contract, validate_runner_inputs
from vedock.services.remote_jobs import _allowed_final_file
from pathlib import PurePosixPath


def _predictor(path: Path) -> Path:
    path.mkdir(parents=True)
    payload = {
        "format": "vedock.tabular.v1",
        "objective": "regression",
        "target_name": "monthly_sales",
        "target_unit": "USD",
        "target_transform": "none",
        "features": [
            {"name": "ad_spend", "kind": "numeric", "median": 100.0, "mean": 100.0, "scale": 20.0},
            {"name": "region", "kind": "categorical", "categories": ["north", "south"]},
        ],
        # intercept + one numeric + north/south/other
        "weights": [1000.0, 200.0, 50.0, -50.0, 0.0],
    }
    (path / "predictor.json").write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_runtime_contract_supports_structured_prediction(tmp_path):
    runtime = get_runtime("tabular_prediction")
    model_path = _predictor(tmp_path / "predictor")
    contract = runner_contract(runtime, str(model_path))
    assert contract["interaction"] == "numeric_prediction"
    assert [field["name"] for field in contract["inputs"]] == ["ad_spend", "region"]
    inputs = validate_runner_inputs({"ad_spend": "120", "region": "north"}, contract)
    result = normalize_runtime_result(runtime.run(str(model_path), inputs, {"return_details": True}), contract)
    assert result["prediction"] == 1250.0
    assert result["outputs"][0] == {"type": "metric", "label": "monthly_sales", "value": 1250.0, "unit": "USD"}
    assert result["outputs"][1]["type"] == "table"


def test_tabular_dataset_schema_keeps_features_separate_from_target():
    record = map_record(
        {"ad_spend": "120", "region": "north", "sales": "1250"},
        "tabular_supervised",
        {"features": ["ad_spend", "region"], "target": "sales"},
    )
    assert record == {"features": {"ad_spend": "120", "region": "north"}, "target": "1250"}


def test_api_and_web_render_predictor_instead_of_chat(registered_client, app, tmp_path):
    model_path = _predictor(tmp_path / "api-predictor")
    with app.app_context():
        user = User.query.filter_by(username="tester").one()
        model = ModelRecord(
            owner=user,
            slug="sales-predictor",
            name="Monthly sales predictor",
            description="Predicts monthly sales from structured business inputs.",
            task_type="tabular_regression",
            runtime_key="tabular_prediction",
            source_type="training",
            source_path=str(model_path),
            visibility="private",
        )
        db.session.add(model)
        db.session.add(ModelVersion(model=model, version_number=1, label="Fitted", storage_path=str(model_path), status="completed"))
        db.session.commit()
    page = registered_client.get("/playground/sales-predictor")
    assert page.status_code == 200
    assert b"Predict monthly sales" in page.data
    assert b'name="ad_spend"' in page.data
    assert b"chat-messages" not in page.data
    response = registered_client.post(
        "/api/v1/models/sales-predictor/run",
        json={"inputs": {"ad_spend": 120, "region": "north"}, "parameters": {"return_details": False}},
    )
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["interaction"] == "numeric_prediction"
    assert data["prediction"] == 1250.0


def test_pattern_model_is_sequence_completion_not_chat():
    capabilities = get_runtime("pattern_sequence").get_model_capabilities()
    assert capabilities["interaction"] == "sequence_completion"
    assert capabilities["runner"]["submit_label"] == "Predict continuation"


def test_portable_image_classifier_runs_without_unpickling(tmp_path):
    model_path = tmp_path / "image-classifier"
    model_path.mkdir()
    (model_path / "classifier.json").write_text(
        json.dumps({"format": "vedock.image_classifier.v1", "kind": "nearest_centroid", "labels": ["dark", "light"], "centroids": [[0.0], [1.0]]}),
        encoding="utf-8",
    )
    (model_path / "metadata.json").write_text(
        json.dumps({"image_width": 1, "image_height": 1, "color_mode": "grayscale"}),
        encoding="utf-8",
    )
    image_path = tmp_path / "white.png"
    output = BytesIO()
    Image.new("L", (1, 1), 255).save(output, format="PNG")
    image_path.write_bytes(output.getvalue())
    result = get_runtime("sklearn_image").infer(str(model_path), str(image_path), {"top_k": 2, "device": "cpu"})
    assert result["predictions"][0]["label"] == "light"
    assert _allowed_final_file(PurePosixPath("classifier.json")) is True
    assert _allowed_final_file(PurePosixPath("classifier.joblib")) is False
